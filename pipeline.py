from ultralytics import YOLO
from app.helper_functions import preprocess_frame, predict_frame, analytics_step, debug_step, render_frame
import cv2
from app.retail_analytics import RetailAnalytics
import threading
import time
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
        self.analytics=RetailAnalytics()
        # Event used to request the prediction loop to stop
        self.stop_event = threading.Event()
        self.source_fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25
        self.target_fps = target_fps  # FPS at which to process frames
        self.frame_skip_interval = max(1, int(round(self.source_fps / self.target_fps)))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.suspicious=False
        self.frame= None
        self.out = None
        self.temp_video_path = None
        self.frame_count = 0
        self.recording_enabled = False


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

    def enable_recording(self):
        """Start recording video to temp file"""
        with self._lock:
            if self.recording_enabled:
                return True
            
            try:
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
                    return False
                
                self.recording_enabled = True
                self.frame_count = 0  # Reset frame counter for this sale
                print(f"✓ Recording started: {self.temp_video_path}")
                return True
            except Exception as e:
                print(f"Error enabling recording: {e}")
                return False
    
    def disable_recording(self):
        """Stop recording video and flush frames to disk"""
        with self._lock:
            if not self.recording_enabled:
                return
            
            try:
                if self.out is not None and self.out.isOpened():
                    # Flush any remaining frames to disk
                    time.sleep(0.1)
                    self.out.release()
                self.recording_enabled = False
                print("✓ Recording stopped and flushed")
            except Exception as e:
                print(f"Error disabling recording: {e}")

    def _run_prediction_loop(self):
        """Thread target: runs the prediction loop (continuous monitoring)"""
        try:
            self.frame_count = 0
            # Loop continuously while prediction is running
            while self.running and not self.stop_event.is_set():
                try:
                    self.frame, meta = preprocess_frame(self.cap, self.source_fps)
                    if self.frame is None:
                        break
                    
                    # Only process frames based on target FPS
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

                    # Only record if recording is enabled (during active sales)
                    if self.recording_enabled and self.frame is not None:
                        with self._lock:
                            if self.out is not None and self.out.isOpened():
                                self.out.write(self.frame)
                    
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
            
            # Disable recording on loop exit
            self.disable_recording()
            
            # Ensure running flag is cleared so callers know thread finished
            self.running = False

    def start_prediction(self):
        with self._lock:
            if self.running:
                return
            # Clear any previous stop request and mark running
            self.stop_event.clear()
            self.running = True

        if not self.capture_video():
            self.running = False
            return

        # Create and start prediction thread
        self.thread = threading.Thread(target=self._run_prediction_loop, daemon=False)
        self.thread.start()

    def save_video(self, OUTPUT_PATH):
        """Save the recorded video from temp location to the specified output path"""
        if not self.temp_video_path or not os.path.exists(self.temp_video_path):
            print("Error: No temporary video recording found to save")
            return False
        
        try:
            os.replace(self.temp_video_path, OUTPUT_PATH)
            print(f"Video saved to {OUTPUT_PATH}")
            return True
        except Exception as e:
            print(f"Error saving video: {e}")
            return False
    
    def print_output(self, pos_wallet: bool = False, pos_member: bool = False) -> Tuple[Dict, Dict]:

        output = {
            "items_scanned": True,
            "cashier": True,
            "scanner_moving": True,
            "pos_member": pos_member,
            "suspicious_activity": False,
            "customer_paid_wallet": False,
            "customer_paid_cash": False,
            "purchasing_customer": False,
            "member_use": False
        }

        developer_message = {}
        self.suspicious = False

        has_customer = bool(self.analytics.customer_visits)
        has_cash = bool(self.analytics.cash_detected)
        has_member_scan = len(self.analytics.completed_payments) > 0

        # WALLET PAYMENT (POS CONFIRMED – SHORT CIRCUIT)
        if pos_wallet:
            output.update({
                "customer_paid_wallet": True,
                "purchasing_customer": True,
                "member_use": pos_member
            })
            return output, developer_message

        # CASH + MEMBER DETECTION
        output["customer_paid_cash"] = has_cash

        if has_customer and has_cash and has_member_scan:
            output["purchasing_customer"] = True
            output["member_use"] = True
        else:
            output["suspicious_activity"] = True
            self.suspicious = True

            if not has_customer:
                developer_message["customer_detection"] = "POSC1-MODELC0"
            if not has_cash:
                developer_message["cash_detection"] = "POSB1-MODELB0"
            if not has_member_scan:
                developer_message["member_detection"] = "POSM1-MODELM0"

        return output, developer_message

    def stop_prediction(self, path):
        # Signal the loop to stop and clear running flag
        self.stop_event.set()
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

        if not self.temp_video_path or not os.path.exists(self.temp_video_path):
            raise FileNotFoundError(
                f"Temp recording not found: {self.temp_video_path}"
            )

        if self.suspicious:
            final_path = f"{path}.mp4"
            os.replace(self.temp_video_path, final_path)
            # Clear stop event so next run can start cleanly
            self.stop_event.clear()
            return True

        try:
            os.remove(self.temp_video_path)
        finally:
            # Clear stop event so next run can start cleanly
            self.stop_event.clear()
        return False