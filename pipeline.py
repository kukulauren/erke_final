from ultralytics import YOLO

from app.helper_functions import preprocess_frame, predict_frame, analytics_step, debug_step, render_frame

import cv2
from app.retail_analytics import RetailAnalytics
import threading
import time
import shutil
import os
import tempfile
from pathlib import Path
from typing import Tuple, Dict


class Prediction:

    def __init__(self, MODEL_PATH, VIDEO_PATH, confidence=0.7, target_fps=10):
        # Validate model path early to provide clear errors
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        self.model = YOLO(MODEL_PATH)
        self.confidence = confidence
        self.rtsp_path = VIDEO_PATH
        self.running = False
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        self._lock = threading.Lock()
        self.thread = None
        self.analytics = RetailAnalytics()

        # Event used to request the prediction loop to stop
        self.stop_event = threading.Event()

        self.source_fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25
        self.target_fps = target_fps  # FPS at which to process frames
        self.frame_skip_interval = max(1, int(round(self.source_fps / self.target_fps)))

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.suspicious = False
        self.frame = None
        self.out = None
        self.temp_video_path = None
        self.frame_count = 0
        self.recording_enabled = False

        # Event set when the writer has been safely flushed and closed
        self._recording_flushed = threading.Event()
        self._recording_flushed.set()  # Initially "flushed" (nothing to flush)

    def set_target_fps(self, target_fps):
        """Update the target FPS for frame processing"""
        with self._lock:
            self.target_fps = target_fps
            self.frame_skip_interval = max(1, int(round(self.source_fps / self.target_fps)))
            print(f"Target FPS updated to {target_fps} (processing every {self.frame_skip_interval} frame(s))")

    def capture_video(self, reconnect_attempts=3, reconnect_delay=2.0):
        attempt = 0
        while attempt < reconnect_attempts:
            self.cap = cv2.VideoCapture(self.rtsp_path)
            if self.cap.isOpened():
                return True
            attempt += 1
            print(f"Failed to open RTSP stream (attempt {attempt}/{reconnect_attempts}), retrying in {reconnect_delay}s...")
            try:
                self.cap.release()
            except Exception as e:
                print(f"Error releasing video capture: {e}")
            time.sleep(reconnect_delay)

        # explicit failure
        try:
            if self.cap:
                self.cap.release()
        except Exception as e:
            print(f"Error releasing video capture on final cleanup: {e}")
        return False

    def enable_recording(self, output_dir=None):
        """Start recording video to temp file on the same drive as output_dir."""
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
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    self.temp_video_path = tmp.name
                    tmp.close()

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out = cv2.VideoWriter(
                    self.temp_video_path,
                    fourcc,
                    self.source_fps,
                    (self.width, self.height)
                )

                if not self.out.isOpened():
                    print("Error: Failed to initialize video writer")
                    self.out = None
                    return False

                self.recording_enabled = True
                self.frame_count = 0
                self._recording_flushed.clear()  # Mark as "not yet flushed"
                print(f"✓ Recording started: {self.temp_video_path}")
                return True

            except Exception as e:
                print(f"Error enabling recording: {e}")
                return False

    def _flush_and_close_writer(self):
        """
        Internal: release the VideoWriter under lock and signal completion.
        Must only be called from within the prediction thread or with no
        concurrent writers active.
        """
        with self._lock:
            if self.out is not None:
                try:
                    if self.out.isOpened():
                        self.out.release()
                except Exception as e:
                    print(f"Error releasing VideoWriter: {e}")
                finally:
                    self.out = None
            self.recording_enabled = False
        self._recording_flushed.set()
        print("✓ Recording flushed and closed")

    def disable_recording(self, wait_for_flush=True, timeout=10.0):
        """
        Signal the recording to stop.  The actual writer release is done by
        the prediction loop thread at the end of its next iteration so that
        no frame is written after the writer has been closed.

        If wait_for_flush=True (the default) this call blocks until the writer
        has been fully released, which is required before the caller tries to
        move/delete the file.
        """
        with self._lock:
            if not self.recording_enabled:
                # Nothing to do – but still wait in case a flush is in progress
                pass
            else:
                # Tell the loop to stop writing and close the writer.
                # The loop checks this flag and calls _flush_and_close_writer.
                self.recording_enabled = False

        if wait_for_flush:
            flushed = self._recording_flushed.wait(timeout=timeout)
            if not flushed:
                print("Warning: recording flush timed out; forcing writer close")
                self._flush_and_close_writer()

    def _run_prediction_loop(self):
        """Thread target: runs the prediction loop (continuous monitoring)"""
        try:
            self.frame_count = 0

            while self.running and not self.stop_event.is_set():
                try:
                    self.frame, meta = preprocess_frame(self.cap, self.source_fps)
                    if self.frame is None:
                        break

                    should_process = (self.frame_count % self.frame_skip_interval) == 0

                    if should_process:
                        current_time = meta["current_time"]
                        detections = predict_frame(self.model, self.frame)
                        events = analytics_step(self.analytics, detections, current_time)
                        render_frame(
                            self.frame,
                            detections,
                            self.analytics,
                            events,
                            current_time,
                            self.width,
                            self.height
                        )

                    # Write frame under lock; also detect if recording was disabled
                    # and flush the writer in THIS thread (owner of the writer).
                    with self._lock:
                        if self.out is not None and self.out.isOpened() and self.frame is not None:
                            self.out.write(self.frame.copy())

                        # If recording was disabled externally, flush now.
                        if not self.recording_enabled and self.out is not None:
                            try:
                                if self.out.isOpened():
                                    self.out.release()
                            except Exception as e:
                                print(f"Error releasing VideoWriter in loop: {e}")
                            finally:
                                self.out = None
                            # Signal that the flush is done (outside the lock below)
                            should_signal = True
                        else:
                            should_signal = False

                    if should_signal:
                        self._recording_flushed.set()
                        print("✓ Recording flushed inside prediction loop")

                    self.frame_count += 1

                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue

        except Exception as e:
            print(f"Error in prediction loop: {e}")
        finally:
            try:
                if self.cap:
                    self.cap.release()
            except Exception as e:
                print(f"Error releasing video capture: {e}")

            # Make sure writer is always closed when the loop exits
            with self._lock:
                out = self.out
                self.out = None
                self.recording_enabled = False
            if out is not None:
                try:
                    if out.isOpened():
                        out.release()
                except Exception as e:
                    print(f"Error releasing VideoWriter on loop exit: {e}")
            self._recording_flushed.set()

            self.running = False
            print("✓ Prediction loop exited")

    def start_prediction(self):
        with self._lock:
            if self.running:
                return

            self.stop_event.clear()
            self.running = True

        if not self.capture_video():
            self.running = False
            return

        self.thread = threading.Thread(target=self._run_prediction_loop, daemon=False)
        self.thread.start()

    def save_video(self, OUTPUT_PATH):
        """Move the completed temp recording to OUTPUT_PATH."""
        if not self.temp_video_path or not os.path.exists(self.temp_video_path):
            print("Error: No temporary video recording found to save")
            return False

        try:
            shutil.move(self.temp_video_path, OUTPUT_PATH)
            print(f"✓ Video saved to {OUTPUT_PATH}")
            return True
        except Exception as e:
            print(f"✗ Error saving video: {e}")
            return False

    def print_output(self, pos_wallet: bool = False, pos_member: bool = False) -> Tuple[Dict, Dict]:
        output = {
            "items_scanned": True,
            "cashier": True,
            "scanner_moving": True,
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
            has_customer = bool(self.analytics.customer_visits)
            has_cash = bool(self.analytics.cash_detected)
            has_member_scan = len(self.analytics.completed_payments) > 0

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

    def stop_prediction(self):
        """Stop prediction loop entirely (e.g. on server shutdown)."""
        self.stop_event.set()
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)

        # Force-close writer if the thread didn't manage to do it
        with self._lock:
            out = self.out
            self.out = None
            self.recording_enabled = False
        if out is not None:
            try:
                if out.isOpened():
                    out.release()
            except Exception as e:
                print(f"Error releasing VideoWriter in stop_prediction: {e}")
        self._recording_flushed.set()

        self.stop_event.clear()
        print("✓ Prediction stopped cleanly")
