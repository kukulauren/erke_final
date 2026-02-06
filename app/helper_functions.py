import numpy as np
import cv2
from app.variables import CONF_THRESHOLD,SCANNER_ITEM_DISTANCE
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

def preprocess_frame(cap, fps):
    """Extract frame from video and calculate metadata"""
    ret, frame = cap.read()
    if not ret:
        return None, None

    # Get frame position automatically from video capture
    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    current_time = frame_count / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frame, {
        "frame_count": frame_count,
        "current_time": current_time,
        "total_frames": total_frames,
        "width": width,
        "height": height
    }

def predict_frame(model, frame):
    results = model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False)

    detections = {k: [] for k in CLASS_NAMES.values()}

    if results[0].boxes is None:
        return detections

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = CLASS_NAMES.get(cls_id, 'unknown')

        detections[cls_name].append({
            'box': box.xyxy[0].cpu().numpy(),
            'conf': float(box.conf[0]),
            'track_id': int(box.id[0]) if box.id is not None else None,
            'center': get_center(box.xyxy[0])
        })

    return detections

def analytics_step(analytics, detections, current_time):
    events = []

    scanner_status = analytics.update_scanner_movement(
        detections['scanner'], current_time
    )

    events.extend(
        analytics.update_item_scanning(
            detections['item'],
            detections['scanner'],
            scanner_status,
            current_time
        )
    )

    events.extend(
        analytics.update_payment_scanning(
            detections['phone'],
            detections['scanner'],
            current_time
        )
    )

    events.extend(
        analytics.detect_cash(
            detections.get('cash', []),
            detections.get('customer', []),
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
        pass

    return events

def debug_step(frame_count, total_frames, detections, analytics):
    if frame_count % 100 != 0:
        return

    progress = (frame_count / total_frames) * 100
    print(f"\nProgress: {progress:.1f}% ({frame_count}/{total_frames})")
    print(f"  Scanners: {len(detections['scanner'])}, Items: {len(detections['item'])}")

    for scanner in detections['scanner']:
        sid = scanner.get('track_id') or 0
        moving = analytics.scanner_moving.get(sid, False)
        print(f"  Scanner #{sid}: moving={moving}")

def render_frame(frame, detections, analytics, events, current_time, width, height):
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