from ultralytics import YOLO
from app.helper_functions import preprocess_frame, predict_frame, analytics_step, debug_step, render_frame
import cv2
from app.retail_analytics import RetailAnalytics
import threading
import time
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

class Prediction:
    def __init__(self, MODEL_PATH, VIDEO_PATH, confidence=0.7):
        self.model = YOLO(MODEL_PATH)
        self.confidence = confidence
        self.rtsp_path = VIDEO_PATH
        self.running = False
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        self._lock = threading.Lock()
        self.thread = None
        self.analytics=RetailAnalytics()
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.suspicious=False
        self.frame= None


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
            except Exception:
                pass
            time.sleep(reconnect_delay)
        # explicit failure
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        return False

    def start_prediction(self):
        with self._lock:
            if self.running:
                print("Prediction already running")
                return
            self.running = True

        if not self.capture_video(reconnect_attempts=2, reconnect_delay=1.0):
            print("Unable to open video source - exiting prediction loop")
            with self._lock:
                self.running = False
            return
        while True:
            self.frame, meta = preprocess_frame(self.cap, self.fps)
            if self.frame is None:
                break

            frame_count = meta["frame_count"]
            current_time = meta["current_time"]

            detections = predict_frame(self.model, self.frame)
            events = analytics_step(self.analytics, detections, current_time)

            debug_step(frame_count, self.total_frames, detections, self.analytics)

            render_frame(
                self.frame,
                detections,
                self.analytics,
                events,
                current_time,
                self.width,
                self.height
            )
        self.cap.release()
    def save_video(self,OUTPUT_PATH):
        out = cv2.VideoWriter(fourcc, self.fps, (self.width, self.height))
        out.write(self.frame)
        out.release()

    def print_output(self,pos_wallet: bool = False,pos_member: bool = False) -> tuple[dict, dict]:

        output = {
            "items_scanned": True,
            "cashier": True,
            "scanner_moving": True,
            "pos_member": pos_member,
            "suspicious_activity":False
        }

        developer_message = {}
        self.suspicious = False

        has_customer = bool(self.analytics.customer_visits)
        has_cash = bool(self.analytics.cash_detected)
        has_phone = len(self.analytics.completed_payments) > 0  # member scan

        # WALLET PAYMENT (POS CONFIRMED – SHORT CIRCUIT)
        if pos_wallet:
            output["customer_paid_wallet"] = True
            output["purchasing_customer"] = True
            output["member_use"] = pos_member
            return output, developer_message

        # CASH + MEMBER DETECTION
        output["customer_paid_cash"] = has_cash

        if has_customer and has_cash and has_phone:
            # ✅ PERFECT CASE
            output["purchasing_customer"] = True
            output["member_use"] = True

        else:
            # Something missing
            output["suspicious_activity"] = True
            self.suspicious = True

            if not has_customer:
                developer_message["customer_detection"] = "POSC1-MODELC0"

            if not has_cash:
                developer_message["cash_detection"] = "POSB1-MODELB0"

            if not has_phone:
                developer_message["member_detection"] = "POSM1-MODELM0"

        return output, developer_message

    def stop_prediction(self,path="A100200"):
        if self.suspicious:
            self.save_video(self.frame,path)
            return True
        return False
