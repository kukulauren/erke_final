"""Per-frame detection sidecar (JSONL) written alongside raw recordings.

Recordings are saved undrawn (no boxes burned into pixels) so they stay
usable as clean model-training input. The detections that would otherwise
have been drawn onto the frame are logged here instead, one JSON line per
processed frame, so the review tool can reconstruct the overlay on demand.
"""
import json
import logging

logger = logging.getLogger(__name__)


def _serialize_box(box):
    return [round(float(v), 2) for v in box]


class DetectionLogger:

    def __init__(self, path):
        self._file = open(path, "w", encoding="utf-8")

    def log_frame(self, frame_idx, timestamp, detections, wrists, analytics):
        """Serialize one processed frame's detections to a JSON line.

        `label` mirrors what app.helper_functions.render_frame draws today
        (cls_name + track_id + analytics.get_person_label), captured now so
        the review tool doesn't need to replay RetailAnalytics state later.
        """
        dets = []
        for cls_name, cls_dets in detections.items():
            for det in cls_dets:
                record = {
                    "class": cls_name,
                    "box": _serialize_box(det["box"]),
                    "conf": round(float(det["conf"]), 3),
                    "track_id": det.get("track_id"),
                }
                track_id = det.get("track_id")
                if cls_name == "scanner":
                    sid = track_id or 0
                    record["moving"] = bool(analytics.scanner_moving.get(sid, False))
                elif track_id is not None:
                    record["label"] = f"{cls_name} #{track_id} [{analytics.get_person_label(track_id)}]"
                dets.append(record)

        line = {
            "frame_idx": frame_idx,
            "timestamp": round(float(timestamp), 3),
            "detections": dets,
            "wrists": [[round(float(x), 2), round(float(y), 2)] for x, y in (wrists or [])],
        }
        self._file.write(json.dumps(line) + "\n")

    def close(self):
        try:
            self._file.close()
        except Exception as e:
            logger.warning("Error closing detection log: %s", e)
