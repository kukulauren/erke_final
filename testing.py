"""Offline testing harness.

Processes a video file end-to-end with the SAME detection + analytics code the
server uses (app/helper_functions.py and app/retail_analytics.py), writes an
annotated video, and prints a summary report.

Usage:
    python testing.py                     # uses VIDEO_PATH / OUTPUT_PATH from .env
    python testing.py input.mp4 out.mp4   # explicit paths
"""
import logging
import sys
import time

import cv2
from ultralytics import YOLO

from app.helper_functions import analytics_step, debug_step, predict_frame, render_frame
from app.pose_estimator import PoseEstimator
from app.retail_analytics import RetailAnalytics
from app.variables import MODEL_PATH, OUTPUT_PATH, POSE_ENABLED, POSE_MODEL_PATH, VIDEO_PATH

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def process_video(model_path, video_path, output_path):
    logger.info("Loading model...")
    model = YOLO(model_path)
    pose = PoseEstimator(POSE_MODEL_PATH) if POSE_ENABLED else None

    logger.info("Opening video: %s", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    analytics = RetailAnalytics(fps=fps)

    frame_count = 0
    start_time = time.time()

    logger.info("Processing %d frames at %d FPS...", total_frames, fps)
    logger.info("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps

        detections = predict_frame(model, frame)

        wrists = []
        if pose is not None and detections['cashier']:
            wrists = pose.cashier_wrists(frame, [c['box'] for c in detections['cashier']])

        events = analytics_step(analytics, detections, current_time, wrists)
        debug_step(frame_count, total_frames, detections, analytics)

        render_frame(frame, detections, analytics, events, current_time, width, height, wrists)

        for event in events:
            logger.info("[%.1fs] %s", current_time, event)

        out.write(frame)

    cap.release()
    out.release()

    processing_time = time.time() - start_time

    logger.info("\n" + "=" * 60)
    logger.info("RETAIL ANALYTICS REPORT")
    logger.info("=" * 60)
    logger.info("  Items Scanned:    %d", len(analytics.scanned_items))
    logger.info("  Payments:         %d", len(analytics.completed_payments))
    logger.info("  Customers Served: %d", len(analytics.customer_visits))
    logger.info("  Cash Detections:  %d", len(analytics.cash_detected))
    logger.info("  Cashier Seen:     %s", analytics.cashier_seen())
    logger.info("  Scanner Moved:    %s", analytics.scanner_ever_moved)
    logger.info("Output: %s", output_path)
    logger.info("Time: %.1fs (%.1f fps)", processing_time, frame_count / max(processing_time, 0.001))
    logger.info("=" * 60)


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    output = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PATH
    process_video(MODEL_PATH, video, output)
