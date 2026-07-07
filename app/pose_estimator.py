"""Cashier hand tracking via YOLO11-pose.

Wrist positions disambiguate what the cashier is actually doing: a wrist next
to the scanner means scanning, a wrist on cash means cash handling — far more
reliable than whole-body bounding-box overlap.

COCO keypoint indices used: 9 = left wrist, 10 = right wrist.
"""
import logging

from ultralytics import YOLO

from app.helper_functions import get_box_iou
from app.variables import POSE_CONF, WRIST_KEYPOINT_CONF

logger = logging.getLogger(__name__)

LEFT_WRIST, RIGHT_WRIST = 9, 10


class PoseEstimator:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        logger.info("Pose model loaded: %s", model_path)

    def cashier_wrists(self, frame, cashier_boxes):
        """Return wrist positions [(x, y), ...] for people matching cashier boxes.

        cashier_boxes: list of xyxy arrays from the detection model. A pose
        person is attributed to the cashier when its box overlaps a cashier
        box with IoU > 0.3. If no cashier boxes are given, returns [].
        """
        if not cashier_boxes:
            return []

        try:
            results = self.model.predict(frame, conf=POSE_CONF, verbose=False)
        except Exception:
            logger.exception("Pose inference failed")
            return []

        r = results[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return []

        wrists = []
        kpts_xy = r.keypoints.xy.cpu().numpy()          # (n_persons, 17, 2)
        kpts_conf = r.keypoints.conf
        kpts_conf = kpts_conf.cpu().numpy() if kpts_conf is not None else None
        boxes = r.boxes.xyxy.cpu().numpy()

        for i, person_box in enumerate(boxes):
            if not any(get_box_iou(person_box, cb) > 0.3 for cb in cashier_boxes):
                continue

            for k in (LEFT_WRIST, RIGHT_WRIST):
                x, y = kpts_xy[i][k]
                conf = kpts_conf[i][k] if kpts_conf is not None else 1.0
                # (0, 0) means the keypoint was not detected
                if conf >= WRIST_KEYPOINT_CONF and (x > 0 or y > 0):
                    wrists.append((float(x), float(y)))

        return wrists
