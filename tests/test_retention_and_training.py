"""Tests for disk retention and training-data clip sampling."""
import os
import threading
import time

import app.retention as retention
import pipeline as pipeline_module
from pipeline import Prediction


def make_file(path, age_days=0):
    with open(path, "wb") as f:
        f.write(b"x" * 16)
    if age_days:
        old = time.time() - age_days * 86400
        os.utime(path, (old, old))
    return path


class TestRetention:

    def test_old_files_deleted_new_files_kept(self, tmp_path):
        old = make_file(str(tmp_path / "old.mp4"), age_days=40)
        new = make_file(str(tmp_path / "new.mp4"), age_days=1)
        deleted = retention.cleanup_directory(str(tmp_path), 30, "*.mp4")
        assert deleted == 1
        assert not os.path.exists(old)
        assert os.path.exists(new)

    def test_pattern_respected(self, tmp_path):
        keep = make_file(str(tmp_path / "old.json"), age_days=40)
        retention.cleanup_directory(str(tmp_path), 30, "*.mp4")
        assert os.path.exists(keep)

    def test_zero_days_disables(self, tmp_path):
        f = make_file(str(tmp_path / "old.mp4"), age_days=400)
        assert retention.cleanup_directory(str(tmp_path), 0, "*.mp4") == 0
        assert os.path.exists(f)

    def test_missing_directory_is_noop(self, tmp_path):
        assert retention.cleanup_directory(str(tmp_path / "nope"), 30) == 0


def make_prediction(temp_path, suspicious=False, counter=0, detections_path=None):
    """Prediction without YOLO/video, ready for finalize_recording."""
    p = Prediction.__new__(Prediction)
    p._lock = threading.Lock()
    p.suspicious = suspicious
    p.temp_video_path = temp_path
    p.temp_detections_path = detections_path
    p._clean_clip_counter = counter
    return p


class TestTrainingClipSampling:

    def test_every_nth_clean_clip_kept(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline_module, "TRAINING_DATA_DIR", str(tmp_path / "train"))
        monkeypatch.setattr(pipeline_module, "TRAINING_CLIP_EVERY_N", 3)

        kept = 0
        for i in range(6):
            clip = make_file(str(tmp_path / f"txn_{i}.mp4"))
            detections = make_file(str(tmp_path / f"txn_{i}.jsonl"))
            p = make_prediction(clip, counter=i, detections_path=detections)  # counter becomes i+1 inside
            saved, debug = p.finalize_recording(f"V{i}", str(tmp_path / "out"))
            assert saved is False  # clean clips are never "recording_saved"
            if "training_clip" in debug:
                kept += 1
                video_path = debug["training_clip"]
                assert os.path.exists(video_path)
                assert os.path.basename(os.path.dirname(video_path)) == "videos"
                stem = os.path.splitext(os.path.basename(video_path))[0]
                detections_dest = os.path.join(
                    os.path.dirname(os.path.dirname(video_path)), "detections", f"{stem}.jsonl"
                )
                assert os.path.exists(detections_dest)
                assert not os.path.exists(clip)
                assert not os.path.exists(detections)
            else:
                assert not os.path.exists(clip)  # deleted
                assert not os.path.exists(detections)
        assert kept == 2  # counters 3 and 6 out of 1..6

    def test_sampling_disabled_deletes_everything(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline_module, "TRAINING_CLIP_EVERY_N", 0)
        clip = make_file(str(tmp_path / "txn.mp4"))
        detections = make_file(str(tmp_path / "txn.jsonl"))
        p = make_prediction(clip, counter=0, detections_path=detections)
        _, debug = p.finalize_recording("V1", str(tmp_path / "out"))
        assert "training_clip" not in debug
        assert not os.path.exists(clip)
        assert not os.path.exists(detections)

    def test_suspicious_clip_still_saved_normally(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline_module, "TRAINING_CLIP_EVERY_N", 1)
        clip = make_file(str(tmp_path / "txn.mp4"))
        detections = make_file(str(tmp_path / "txn.jsonl"))
        p = make_prediction(clip, suspicious=True, detections_path=detections)
        saved, debug = p.finalize_recording("SUS1", str(tmp_path / "out"))
        assert saved is True
        assert os.path.exists(str(tmp_path / "out" / "videos" / "SUS1.mp4"))
        assert os.path.exists(str(tmp_path / "out" / "detections" / "SUS1.jsonl"))
        assert "training_clip" not in debug

    def test_suspicious_clip_without_detections_sidecar(self, tmp_path, monkeypatch):
        """A detections file isn't guaranteed (e.g. logger failed to open) —
        the video should still be saved without erroring."""
        monkeypatch.setattr(pipeline_module, "TRAINING_CLIP_EVERY_N", 1)
        clip = make_file(str(tmp_path / "txn.mp4"))
        p = make_prediction(clip, suspicious=True, detections_path=None)
        saved, debug = p.finalize_recording("SUS2", str(tmp_path / "out"))
        assert saved is True
        assert os.path.exists(str(tmp_path / "out" / "videos" / "SUS2.mp4"))
        assert not os.path.exists(str(tmp_path / "out" / "detections"))
