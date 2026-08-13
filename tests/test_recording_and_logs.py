"""Tests for the H.264 video writer, transaction logging and calibration."""
import json
import os

import cv2
import numpy as np
import pytest

import app.video_writer as video_writer
from app.detection_logger import DetectionLogger
from app.retail_analytics import RetailAnalytics
from app.transaction_logger import log_transaction
from calibrate import _percentile, collect_samples, suggest


def write_frames(writer, n=30, size=(320, 240)):
    for _ in range(n):
        writer.write(np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    writer.release()


class TestVideoWriter:

    def test_h264_writer_produces_readable_h264(self, tmp_path):
        pytest.importorskip("av")
        path = str(tmp_path / "clip.mp4")
        w = video_writer.create_video_writer(path, 30, (320, 240))
        assert type(w).__name__ == "_AvH264Writer"
        write_frames(w)

        cap = cv2.VideoCapture(path)
        assert cap.isOpened()
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode()
        assert fourcc in ("avc1", "h264", "H264")
        ret, frame = cap.read()
        assert ret and frame.shape == (240, 320, 3)
        cap.release()

    def test_fallback_to_mp4v_when_codec_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(video_writer, "VIDEO_CODEC", "mp4v")
        path = str(tmp_path / "clip.mp4")
        w = video_writer.create_video_writer(path, 30, (320, 240))
        assert isinstance(w, cv2.VideoWriter)
        write_frames(w)
        assert os.path.getsize(path) > 0

    def test_odd_dimensions_handled(self, tmp_path):
        pytest.importorskip("av")
        path = str(tmp_path / "odd.mp4")
        w = video_writer.create_video_writer(path, 30, (321, 241))
        write_frames(w, size=(321, 241))
        assert os.path.getsize(path) > 0


class TestDetectionLogger:

    def test_log_frame_writes_expected_fields(self, tmp_path):
        analytics = RetailAnalytics()
        path = str(tmp_path / "clip.jsonl")
        dlog = DetectionLogger(path)

        detections = {
            "cashier": [{"box": np.array([10.0, 20.0, 30.0, 40.0]), "conf": 0.91, "track_id": 3}],
            "scanner": [{"box": np.array([1.0, 2.0, 3.0, 4.0]), "conf": 0.8, "track_id": 1}],
            "customer": [],
        }
        dlog.log_frame(42, 4.2, detections, [(5.0, 6.0)], analytics)
        dlog.close()

        lines = open(path).read().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["frame_idx"] == 42
        assert rec["timestamp"] == 4.2
        assert rec["wrists"] == [[5.0, 6.0]]

        by_class = {d["class"]: d for d in rec["detections"]}
        assert by_class["cashier"]["box"] == [10.0, 20.0, 30.0, 40.0]
        assert by_class["cashier"]["track_id"] == 3
        assert "label" in by_class["cashier"]  # has track_id -> analytics.get_person_label
        assert "moving" in by_class["scanner"]  # scanner class -> scanner_moving flag
        assert "label" not in by_class["scanner"]

    def test_log_frame_no_track_id_skips_label(self, tmp_path):
        analytics = RetailAnalytics()
        path = str(tmp_path / "clip.jsonl")
        dlog = DetectionLogger(path)
        detections = {"item": [{"box": np.array([0.0, 0.0, 1.0, 1.0]), "conf": 0.5, "track_id": None}]}
        dlog.log_frame(0, 0.0, detections, [], analytics)
        dlog.close()

        rec = json.loads(open(path).read().strip())
        det = rec["detections"][0]
        assert "label" not in det
        assert det["track_id"] is None


class TestTransactionLogger:

    def test_writes_complete_record(self, tmp_path):
        a = RetailAnalytics()
        a.scanned_items.append({'action': 'item_scan', 'time': 1.0, 'distance': 55.0, 'duration': 0.6})
        a.payment_times.append(1.4)

        path = log_transaction(
            a, {"suspicious_activity": True}, {"cash_detection": "POSM1-MODELB0"},
            "V123", pos_member=True, pos_wallet=False, log_dir=str(tmp_path)
        )
        assert path and os.path.exists(path)

        rec = json.load(open(path))
        assert rec["voucher_number"] == "V123"
        assert rec["verdict"]["suspicious_activity"] is True
        assert rec["thresholds"]["SCAN_MIN_DURATION"] > 0
        assert rec["evidence"]["scanned_items"][0]["distance"] == 55.0

    def test_voucher_name_sanitized(self, tmp_path):
        a = RetailAnalytics()
        path = log_transaction(a, {}, {}, "../../evil:name", True, False, log_dir=str(tmp_path))
        assert path is not None
        assert os.path.dirname(os.path.abspath(path)) == str(tmp_path)


class TestCalibration:

    def test_percentile(self):
        vals = list(range(1, 101))
        assert _percentile(vals, 50) == pytest.approx(50.5)
        assert _percentile(vals, 0) == 1
        assert _percentile(vals, 100) == 100
        assert _percentile([], 50) is None

    def test_collect_and_suggest(self):
        records = [{
            "verdict": {"suspicious_activity": False},
            "evidence": {
                "scanned_items": [
                    {"time": 1.0, "distance": 60.0, "duration": 0.7},
                    {"time": 4.0, "distance": 80.0, "duration": 0.9},
                ],
                "payment_times": [1.5],
                "service_times": [40.0],
            },
        } for _ in range(5)]

        samples = collect_samples(records)
        assert len(samples["scan_distances"]) == 10
        assert samples["scan_gaps"] == [3.0] * 5

        suggestions = suggest(samples)
        assert "SCANNER_ITEM_DISTANCE" in suggestions
        current, suggested, n, note = suggestions["SCANNER_ITEM_DISTANCE"]
        assert n == 10
        assert suggested >= 80  # p95 of distances +10% margin

    def test_suspicious_transactions_excluded_from_dwell(self):
        records = [{
            "verdict": {"suspicious_activity": True},
            "evidence": {"scanned_items": [], "payment_times": [], "service_times": [99.0]},
        }]
        samples = collect_samples(records)
        assert samples["clean_service_times"] == []
