import logging

import cv2
import numpy as np

from app.variables import CONF_THRESHOLD, SCANNER_ITEM_DISTANCE, TRACKER_CONFIG

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: 'cashier', 1: 'customer', 2: 'scanner', 3: 'item', 4: 'phone', 5: 'cash', 6: 'counter'}
CLASS_COLORS = {
    'cashier': (0, 255, 0),
    'customer': (255, 0, 0),
    'scanner': (0, 0, 255),
    'item': (255, 255, 0),
    'phone': (255, 0, 255),
    'cash': (0, 255, 255),
    'counter': (128, 128, 128)
}


def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def boxes_overlap(box1, box2):
    """Check if two boxes overlap at all"""
    if box1[0] > box2[2] or box2[0] > box1[2]:  # No horizontal overlap
        return False
    if box1[1] > box2[3] or box2[1] > box1[3]:  # No vertical overlap
        return False
    return True


def get_box_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def read_frame(cap):
    """Read one frame from the capture. Returns the frame or None at end/failure."""
    ret, frame = cap.read()
    return frame if ret else None


def predict_frame(model, frame):
    results = model.track(
        frame, persist=True, conf=CONF_THRESHOLD, tracker=TRACKER_CONFIG, verbose=False
    )

    detections = {k: [] for k in CLASS_NAMES.values()}

    if results[0].boxes is None:
        return detections

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = CLASS_NAMES.get(cls_id, 'unknown')
        if cls_name == 'unknown':
            continue

        xyxy = box.xyxy[0].cpu().numpy()
        detections[cls_name].append({
            'box': xyxy,
            'conf': float(box.conf[0]),
            'track_id': int(box.id[0]) if box.id is not None else None,
            'center': get_center(xyxy)
        })

    return detections


def analytics_step(analytics, detections, current_time, wrists=None):
    events = []

    scanner_status = analytics.update_scanner_movement(
        detections['scanner'], current_time
    )

    # Temporal action recognition: item scans, phone payments, cash handovers
    events.extend(
        analytics.update_actions(detections, scanner_status, wrists, current_time)
    )

    # Customer presence at the counter feeds `customer_visits`, which the
    # suspicious-activity decision in Prediction.print_output depends on.
    events.extend(
        analytics.update_customer_at_counter(
            detections.get('customer', []),
            detections.get('counter', []),
            current_time
        )
    )

    # Update per-person behavior (staff/customer signals)
    try:
        events.extend(analytics.update_person_behavior(
            customers=detections.get('customer', []),
            current_time=current_time,
            cashiers=detections.get('cashier', [])
        ))
    except Exception:
        logger.exception("update_person_behavior failed")

    return events


def debug_step(frame_count, total_frames, detections, analytics):
    if frame_count % 100 != 0:
        return

    if total_frames > 0:
        progress = (frame_count / total_frames) * 100
        logger.info("Progress: %.1f%% (%d/%d)", progress, frame_count, total_frames)
    logger.info("  Scanners: %d, Items: %d", len(detections['scanner']), len(detections['item']))

    for scanner in detections['scanner']:
        sid = scanner.get('track_id') or 0
        moving = analytics.scanner_moving.get(sid, False)
        logger.info("  Scanner #%s: moving=%s", sid, moving)


def render_frame(frame, detections, analytics, events, current_time, width, height, wrists=None):
    # Cashier wrists from pose estimation
    for wx, wy in (wrists or []):
        cv2.circle(frame, (int(wx), int(wy)), 8, (0, 165, 255), -1)

    # Bounding boxes
    for cls_name, dets in detections.items():
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))
        for det in dets:
            box = det['box'].astype(int)
            track_id = det.get('track_id')

            cv2.rectangle(
                frame,
                (box[0], box[1]),
                (box[2], box[3]),
                color,
                2
            )

            label = cls_name
            if track_id is not None:
                # Append person classification label including staff source
                person_label = analytics.get_person_label(track_id)
                label = f"{cls_name} #{track_id} [{person_label}]"

            cv2.putText(
                frame,
                label,
                (box[0], box[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
    # Scanner zones
    for scanner in detections['scanner']:
        center = tuple(map(int, scanner['center']))
        sid = scanner.get('track_id') or 0
        moving = analytics.scanner_moving.get(sid, False)

        color = (0, 255, 0) if moving else (0, 255, 255)
        cv2.circle(frame, center, SCANNER_ITEM_DISTANCE, color, 2)

    # Events
    y = 30
    for event in events:
        cv2.putText(frame, event, (width - 400, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 25

    # Stats bar
    stats = analytics.get_display_stats()
    cv2.rectangle(frame, (0, height - 60), (width, height), (0, 0, 0), -1)
    x = 20
    for stat in stats:
        cv2.putText(frame, stat, (x, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        x += 220
