"""Replay a raw recording with its saved detections drawn as an overlay.

Recordings are saved raw (no boxes burned into pixels, so they stay usable
as clean model-training input) with a matching per-frame detections
sidecar (see app/detection_logger.py). This script reconstructs the review
view on demand — the same boxes/labels/wrists/scanner state that used to
be drawn permanently into the video — so you can still spot missed or
wrong detections without touching the raw pixels.

Usage:
    python review_recording.py IGS_record/videos/SUS1.mp4
    python review_recording.py IGS_record/videos/SUS1.mp4 --detections path/to/other.jsonl
    python review_recording.py IGS_record/videos/SUS1.mp4 --paused

Controls: space = pause/resume, q = quit.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2

from app.helper_functions import CLASS_COLORS
from app.variables import SCANNER_ITEM_DISTANCE


def derive_detections_path(video_path):
    """Swap a videos/ path segment for detections/ and the extension for .jsonl."""
    p = Path(video_path)
    parts = list(p.parts)
    if "videos" in parts:
        parts[parts.index("videos")] = "detections"
        return str(Path(*parts).with_suffix(".jsonl"))
    return str(p.with_suffix(".jsonl"))


def load_detections(path):
    detections_by_frame = {}
    if not path or not os.path.exists(path):
        print(f"Warning: no detections file found at {path!r} — playing raw video only.",
              file=sys.stderr)
        return detections_by_frame
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            detections_by_frame[record["frame_idx"]] = record
    return detections_by_frame


def draw_overlay(frame, record):
    for det in record.get("detections", []):
        cls_name = det["class"]
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))
        box = [int(round(v)) for v in det["box"]]
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

        if cls_name == "scanner":
            center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            moving = det.get("moving", False)
            circle_color = (0, 255, 0) if moving else (0, 255, 255)
            cv2.circle(frame, center, SCANNER_ITEM_DISTANCE, circle_color, 2)
            label = f"scanner [{'moving' if moving else 'idle'}]"
        else:
            label = det.get("label", cls_name)

        cv2.putText(frame, label, (box[0], box[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for wx, wy in record.get("wrists", []):
        cv2.circle(frame, (int(round(wx)), int(round(wy))), 8, (0, 165, 255), -1)


def review(video_path, detections_path, start_paused=False):
    detections_by_frame = load_detections(detections_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    window = "review_recording (space=pause/resume, q=quit)"
    paused = start_paused
    frame_idx = 0
    frame = None

    while True:
        if not paused:
            ret, raw = cap.read()
            if not ret:
                print("End of video.")
                break
            frame = raw
            record = detections_by_frame.get(frame_idx)
            if record:
                draw_overlay(frame, record)
            frame_idx += 1

        cv2.imshow(window, frame)
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to the raw .mp4 (e.g. IGS_record/videos/SUS1.mp4)")
    parser.add_argument("--detections",
                         help="Path to the matching .jsonl (default: derived from the video "
                              "path by swapping a videos/ segment for detections/)")
    parser.add_argument("--paused", action="store_true", help="Start paused on the first frame")
    args = parser.parse_args()

    detections_path = args.detections or derive_detections_path(args.video)
    review(args.video, detections_path, start_paused=args.paused)


if __name__ == "__main__":
    main()
