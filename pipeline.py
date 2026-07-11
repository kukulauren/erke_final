import logging
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
from ultralytics import YOLO

from app.helper_functions import analytics_step, predict_frame, render_frame
from app.pose_estimator import PoseEstimator
from app.retail_analytics import RetailAnalytics
from app.variables import (
    FRAME_QUEUE_SIZE,
    POSE_ENABLED,
    POSE_MODEL_PATH,
    TRAINING_CLIP_EVERY_N,
    TRAINING_DATA_DIR,
)
from app.video_writer import create_video_writer

logger = logging.getLogger(__name__)

STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")

_SENTINEL = object()  # end-of-stream marker on the frame queue


class Prediction:
    """Video analytics pipeline.

    Two threads decoupled by a bounded frame queue:
      - capture thread: reads frames from the file/stream (and reconnects
        streams); never blocked by inference, so RTSP frames don't go stale.
      - worker thread: YOLO tracking (BoT-SORT + ReID), pose estimation,
        temporal action recognition, rendering, and recording.
    """

    def __init__(self, MODEL_PATH, VIDEO_PATH, confidence=0.7, target_fps=10):
        # Validate model path early to provide clear errors
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        if not VIDEO_PATH:
            raise ValueError("VIDEO_PATH is empty – set it in .env or app/variables.py")

        self.model = YOLO(MODEL_PATH)
        self.pose = PoseEstimator(POSE_MODEL_PATH) if POSE_ENABLED else None
        self.confidence = confidence
        self.video_source = VIDEO_PATH
        # Live streams get wall-clock time and infinite reconnects; files get
        # frame-based time and stop at end of file.
        self.is_stream = str(VIDEO_PATH).lower().startswith(STREAM_PREFIXES)

        self.running = False
        self._lock = threading.Lock()
        self.capture_thread = None
        self.worker_thread = None
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.analytics = RetailAnalytics()

        # Event used to request the pipeline to stop
        self.stop_event = threading.Event()

        self.cap = None
        self.source_fps = 25
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self._open_capture()

        self.target_fps = target_fps  # FPS at which to process frames
        self.frame_skip_interval = max(1, int(round(self.source_fps / self.target_fps)))

        self.suspicious = False
        self.frame = None
        self.out = None
        self.temp_video_path = None
        self.frame_count = 0
        self.frames_dropped = 0
        self.recording_enabled = False
        self._loop_started_at = None
        self._clean_clip_counter = 0  # for training-data sampling

        # Event set when the writer has been safely flushed and closed
        self._recording_flushed = threading.Event()
        self._recording_flushed.set()  # Initially "flushed" (nothing to flush)

    # ── Capture management ───────────────────────────────────────────────────

    def _open_capture(self) -> bool:
        """Open (or re-open) the video source, releasing any previous handle."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning("Error releasing previous capture: %s", e)

        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            return False

        self.source_fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return True

    def _reconnect(self, attempts=0, delay=2.0) -> bool:
        """Try to re-open the source. attempts=0 means retry forever (streams)."""
        attempt = 0
        while not self.stop_event.is_set():
            attempt += 1
            if self._open_capture():
                logger.info("Reconnected to video source on attempt %d", attempt)
                return True
            if attempts and attempt >= attempts:
                break
            logger.warning(
                "Failed to open video source (attempt %d%s), retrying in %.1fs...",
                attempt, f"/{attempts}" if attempts else "", delay
            )
            # Sleep in small chunks so stop_event can interrupt promptly
            deadline = time.time() + delay
            while time.time() < deadline and not self.stop_event.is_set():
                time.sleep(0.2)
        return False

    def set_target_fps(self, target_fps):
        """Update the target FPS for frame processing"""
        with self._lock:
            self.target_fps = target_fps
            self.frame_skip_interval = max(1, int(round(self.source_fps / self.target_fps)))
            logger.info(
                "Target FPS updated to %s (processing every %d frame(s))",
                target_fps, self.frame_skip_interval
            )

    # ── Recording ────────────────────────────────────────────────────────────

    def enable_recording(self, output_dir=None):
        """Start recording video to a temp file on the same drive as output_dir."""
        with self._lock:
            if self.recording_enabled:
                return True

            try:
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    self.temp_video_path = os.path.join(
                        output_dir,
                        f"txn_{int(time.time() * 1000)}.mp4"
                    )
                else:
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    self.temp_video_path = tmp.name
                    tmp.close()

                self.out = create_video_writer(
                    self.temp_video_path, self.source_fps, (self.width, self.height)
                )
                if self.out is None:
                    return False

                self.recording_enabled = True
                self._recording_flushed.clear()  # Mark as "not yet flushed"
                logger.info("Recording started: %s", self.temp_video_path)
                return True

            except Exception as e:
                logger.exception("Error enabling recording: %s", e)
                return False

    def disable_recording(self, wait_for_flush=True, timeout=10.0):
        """
        Signal the recording to stop.  The actual writer release is done by
        the worker thread at the end of its next iteration so that no frame
        is written after the writer has been closed.

        If wait_for_flush=True (the default) this call blocks until the writer
        has been fully released, which is required before the caller tries to
        move/delete the file.
        """
        with self._lock:
            if self.recording_enabled:
                # Tell the worker to stop writing; it closes the writer itself.
                self.recording_enabled = False

        if wait_for_flush:
            flushed = self._recording_flushed.wait(timeout=timeout)
            if not flushed:
                logger.warning("Recording flush timed out; forcing writer close")
                self._force_close_writer()

    def _force_close_writer(self):
        """Release the writer from outside the worker (fallback path only)."""
        with self._lock:
            out = self.out
            self.out = None
            self.recording_enabled = False
        if out is not None:
            try:
                out.release()
            except Exception as e:
                logger.warning("Error releasing video writer: %s", e)
        self._recording_flushed.set()

    def finalize_recording(self, voucher_number: str, output_dir: str) -> Tuple[bool, Dict]:
        """
        Move the temp recording to its final name if the transaction was
        suspicious, otherwise delete it.  Also resets recording state.
        Returns (video_saved, debug_info).
        """
        import getpass

        is_suspicious = self.suspicious
        temp_path = self.temp_video_path

        debug = {
            "suspicious": is_suspicious,
            "temp_path": temp_path,
            "temp_exists": bool(temp_path) and os.path.exists(temp_path),
            "output_dir_exists": os.path.exists(output_dir),
            "cwd": os.getcwd(),
            "running_user": getpass.getuser(),
            "error": None
        }

        video_saved = False

        if is_suspicious:
            if not temp_path:
                debug["error"] = "temp_video_path is None"
            elif not os.path.exists(temp_path):
                debug["error"] = (
                    "temp video file does not exist after flush – "
                    "recording may have been too short or writer failed to open"
                )
            else:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    safe_voucher = re.sub(r'[\\/:*?"<>|]', "_", voucher_number)
                    output_path = os.path.join(output_dir, f"{safe_voucher}.mp4")
                    os.replace(temp_path, output_path)
                    video_saved = True
                    logger.info("Suspicious recording saved: %s", output_path)
                except Exception as e:
                    debug["error"] = f"{type(e).__name__}: {e}"
                    logger.error("Failed to save recording: %s", debug["error"])
        else:
            # Not suspicious → keep every Nth clip for future model training,
            # delete the rest.
            if temp_path and os.path.exists(temp_path):
                self._clean_clip_counter += 1
                keep_for_training = (
                    TRAINING_CLIP_EVERY_N > 0
                    and self._clean_clip_counter % TRAINING_CLIP_EVERY_N == 0
                )
                try:
                    if keep_for_training:
                        train_dir = os.path.join(TRAINING_DATA_DIR, "clean")
                        os.makedirs(train_dir, exist_ok=True)
                        safe_voucher = re.sub(r'[\\/:*?"<>|]', "_", voucher_number)
                        train_path = os.path.join(
                            train_dir, f"{safe_voucher}_{int(time.time())}.mp4"
                        )
                        os.replace(temp_path, train_path)
                        debug["training_clip"] = train_path
                        logger.info("Clean clip kept for training: %s", train_path)
                    else:
                        os.remove(temp_path)
                        logger.info("Non-suspicious temp recording deleted")
                except Exception as e:
                    debug["error"] = f"Cleanup error: {e}"

        self.temp_video_path = None
        self.suspicious = False
        return video_saved, debug

    def reset_analytics(self):
        """Fresh analytics state for the next transaction."""
        with self._lock:
            self.analytics = RetailAnalytics()

    # ── Capture thread ───────────────────────────────────────────────────────

    def _capture_loop(self):
        """Read frames as fast as the source delivers them and enqueue them.

        Streams: if the worker falls behind and the queue fills up, the oldest
        frame is dropped so processing stays real-time instead of drifting
        seconds behind the camera.
        Files: the put blocks (backpressure) so no frame is skipped.
        """
        try:
            while self.running and not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    if self.is_stream:
                        logger.warning("Lost stream, attempting to reconnect...")
                        if self._reconnect(attempts=0):
                            continue
                    break  # end of file or unrecoverable stream

                if self.is_stream:
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()  # drop oldest
                            self.frames_dropped += 1
                        except queue.Empty:
                            pass
                        try:
                            self.frame_queue.put_nowait(frame)
                        except queue.Full:
                            self.frames_dropped += 1
                else:
                    while not self.stop_event.is_set():
                        try:
                            self.frame_queue.put(frame, timeout=0.5)
                            break
                        except queue.Full:
                            continue
        except Exception as e:
            logger.exception("Error in capture loop: %s", e)
        finally:
            try:
                if self.cap:
                    self.cap.release()
            except Exception as e:
                logger.warning("Error releasing video capture: %s", e)
            # Wake the worker so it can exit
            try:
                self.frame_queue.put(_SENTINEL, timeout=1.0)
            except queue.Full:
                pass
            logger.info("Capture loop exited (%d frames dropped)", self.frames_dropped)

    # ── Worker thread ────────────────────────────────────────────────────────

    def _current_time(self) -> float:
        """Transaction-relative time in seconds.

        Streams use wall-clock time (frame positions are unreliable on RTSP);
        files use frame index / fps so timing is correct even when processing
        faster than real time.
        """
        if self.is_stream:
            return time.time() - self._loop_started_at
        return self.frame_count / self.source_fps

    def _worker_loop(self):
        """Inference, analytics, rendering and recording."""
        self._loop_started_at = time.time()
        self.frame_count = 0

        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    if not self.running:
                        break
                    continue
                if frame is _SENTINEL:
                    break

                try:
                    self.frame = frame
                    should_process = (self.frame_count % self.frame_skip_interval) == 0

                    if should_process:
                        current_time = self._current_time()
                        detections = predict_frame(self.model, frame)

                        wrists = []
                        if self.pose is not None and detections['cashier']:
                            wrists = self.pose.cashier_wrists(
                                frame, [c['box'] for c in detections['cashier']]
                            )

                        with self._lock:
                            analytics = self.analytics
                        events = analytics_step(analytics, detections, current_time, wrists)
                        render_frame(
                            frame, detections, analytics, events,
                            current_time, self.width, self.height, wrists
                        )
                        for event in events:
                            logger.info("[%.1fs] %s", current_time, event)

                    # Write frame under lock; also detect if recording was
                    # disabled and flush the writer in THIS thread (its owner).
                    should_signal = False
                    with self._lock:
                        if self.out is not None:
                            self.out.write(frame)

                        if not self.recording_enabled and self.out is not None:
                            try:
                                self.out.release()
                            except Exception as e:
                                logger.warning("Error releasing writer in worker: %s", e)
                            finally:
                                self.out = None
                            should_signal = True

                    if should_signal:
                        self._recording_flushed.set()
                        logger.info("Recording flushed inside worker loop")

                    self.frame_count += 1

                except Exception as e:
                    logger.exception("Error processing frame: %s", e)
                    continue

        except Exception as e:
            logger.exception("Error in worker loop: %s", e)
        finally:
            # Make sure writer is always closed when the worker exits
            self._force_close_writer()
            self.running = False
            logger.info("Worker loop exited")

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start_prediction(self) -> bool:
        with self._lock:
            if self.running:
                return True

            self.stop_event.clear()
            self.running = True

        # Re-open the source if the previous run released it (or never opened)
        if self.cap is None or not self.cap.isOpened():
            if not self._reconnect(attempts=3):
                self.running = False
                return False

        # Drain any frames left over from a previous run
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=False)
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=False)
        self.capture_thread.start()
        self.worker_thread.start()
        return True

    def stop_prediction(self):
        """Stop the pipeline entirely (e.g. on server shutdown)."""
        self.stop_event.set()
        self.running = False

        for thread in (self.capture_thread, self.worker_thread):
            if thread and thread.is_alive():
                thread.join(timeout=10)

        # Force-close writer if the worker didn't manage to do it
        self._force_close_writer()

        self.stop_event.clear()
        logger.info("Prediction stopped cleanly")

    # ── Output ───────────────────────────────────────────────────────────────

    def status(self) -> Dict:
        """Lightweight state snapshot for a health endpoint."""
        with self._lock:
            analytics = self.analytics
        return {
            "running": self.running,
            "source": self.video_source,
            "is_stream": self.is_stream,
            "source_fps": self.source_fps,
            "target_fps": self.target_fps,
            "frames_processed": self.frame_count,
            "frames_dropped": self.frames_dropped,
            "queue_depth": self.frame_queue.qsize(),
            "pose_enabled": self.pose is not None,
            "recording": self.recording_enabled,
            "temp_video_path": self.temp_video_path,
            "items_scanned": len(analytics.scanned_items),
            "payments": len(analytics.completed_payments),
            "customer_visits": len(analytics.customer_visits),
        }

    def print_output(self, pos_wallet: bool = False, pos_member: bool = False) -> Tuple[Dict, Dict]:
        with self._lock:
            analytics = self.analytics

        # Observed CV signals (previously hardcoded to True)
        output = {
            "items_scanned": len(analytics.scanned_items) > 0,
            "cashier": analytics.cashier_seen(),
            "scanner_moving": analytics.scanner_ever_moved,
            "pos_member": pos_member,
            "suspicious_activity": False,
            "customer_paid_wallet": pos_wallet,
            "customer_paid_cash": not pos_wallet,
            "purchasing_customer": False,
            "member_use": False
        }

        developer_message = {}
        self.suspicious = False

        # WALLET PAYMENT (POS CONFIRMED – NO CASH REQUIRED)
        if pos_wallet:
            output.update({
                "purchasing_customer": True,
                "member_use": pos_member
            })
            return output, developer_message

        # CASH FLOW
        if pos_member:
            has_customer = bool(analytics.customer_visits)
            has_cash = bool(analytics.cash_detected)
            has_member_scan = len(analytics.completed_payments) > 0

            if has_customer and has_cash and has_member_scan:
                output["purchasing_customer"] = True
                output["member_use"] = True
            else:
                output["suspicious_activity"] = True
                self.suspicious = True

                if not has_customer:
                    developer_message["customer_detection"] = "POSM1-MODELC0"
                if not has_cash:
                    developer_message["cash_detection"] = "POSM1-MODELB0"
                if not has_member_scan:
                    developer_message["member_detection"] = "POSM1-MODELM0"

        # NON-MEMBER CASH TRANSACTION → DO NOT ENFORCE CV
        else:
            output["purchasing_customer"] = True
            output["member_use"] = False

        return output, developer_message
