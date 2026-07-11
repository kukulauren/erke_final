"""Central configuration for the retail loss-prevention pipeline.

Every value can be overridden from the .env file (or a real environment
variable) without touching code.  The defaults below match production
behaviour, so an empty .env still yields a working system.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        print(f"Warning: invalid float for {name}={value!r}, using default {default}")
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        print(f"Warning: invalid int for {name}={value!r}, using default {default}")
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# ── Paths & sources ──────────────────────────────────────────────────────────
MODEL_PATH = _env_str("MODEL_PATH", "best.pt")
VIDEO_PATH = _env_str("VIDEO_PATH", "")            # file path or rtsp:// URL
OUTPUT_DIR = _env_str("OUTPUT_DIR", r"E:\IGS_record")
OUTPUT_PATH = _env_str("OUTPUT_PATH", "output_result.mp4")  # offline testing only

# ── Detection ────────────────────────────────────────────────────────────────
CONF_THRESHOLD = _env_float("CONF_THRESHOLD", 0.5)
OVERLAP_THRESHOLD = _env_float("OVERLAP_THRESHOLD", 0.3)

# ── Analytics thresholds (pixels / seconds) ──────────────────────────────────
CUSTOMER_DWELL_TIME = _env_float("CUSTOMER_DWELL_TIME", 3.0)
SCANNER_PHONE_DISTANCE = _env_int("SCANNER_PHONE_DISTANCE", 80)
SCANNER_ITEM_DISTANCE = _env_int("SCANNER_ITEM_DISTANCE", 100)
SCANNER_MOVEMENT_THRESHOLD = _env_float("SCANNER_MOVEMENT_THRESHOLD", 3)
SCAN_COOLDOWN = _env_float("SCAN_COOLDOWN", 1.5)
PAYMENT_COMPLETE_TIME = _env_float("PAYMENT_COMPLETE_TIME", 1.0)

# ── Staff / customer differentiation ─────────────────────────────────────────
STAFF_REENTRY_THRESHOLD = _env_int("STAFF_REENTRY_THRESHOLD", 4)   # re-entries within window
REENTRY_WINDOW = _env_float("REENTRY_WINDOW", 600.0)               # 10-minute window
REENTRY_MIN_SESSIONS = _env_int("REENTRY_MIN_SESSIONS", 2)         # distinct sessions required

STAFF_CUMULATIVE_TIME = _env_float("STAFF_CUMULATIVE_TIME", 900.0)  # 15 min across sessions
STAFF_TIME_MIN_SESSIONS = _env_int("STAFF_TIME_MIN_SESSIONS", 3)    # visits required

STAFF_CONFIDENCE_THRESHOLD = _env_float("STAFF_CONFIDENCE_THRESHOLD", 0.65)
RECENT_BEHAVIOR_WINDOW = _env_float("RECENT_BEHAVIOR_WINDOW", 1800.0)
CONFIDENCE_DECAY_RATE = _env_float("CONFIDENCE_DECAY_RATE", 0.98)

# ── Tracking ─────────────────────────────────────────────────────────────────
# BoT-SORT with ReID keeps track IDs stable through occlusion, which the
# staff/customer and action logic depend on.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKER_CONFIG = _env_str("TRACKER_CONFIG", os.path.join(_APP_DIR, "botsort_reid.yaml"))

# ── Pose estimation (cashier hands) ──────────────────────────────────────────
POSE_ENABLED = _env_bool("POSE_ENABLED", True)
POSE_MODEL_PATH = _env_str("POSE_MODEL_PATH", "yolo11n-pose.pt")
POSE_CONF = _env_float("POSE_CONF", 0.35)
WRIST_KEYPOINT_CONF = _env_float("WRIST_KEYPOINT_CONF", 0.3)
WRIST_NEAR_DISTANCE = _env_int("WRIST_NEAR_DISTANCE", 60)   # px: wrist "touching" an object

# ── Temporal action recognition ──────────────────────────────────────────────
# Actions are confirmed from sustained multi-frame evidence instead of a
# single-frame proximity test.
SCAN_MIN_DURATION = _env_float("SCAN_MIN_DURATION", 0.4)    # s of evidence to confirm a scan
CASH_MIN_DURATION = _env_float("CASH_MIN_DURATION", 0.3)    # s of evidence to confirm cash
ACTION_GAP_TOLERANCE = _env_float("ACTION_GAP_TOLERANCE", 0.5)  # s of missed detections allowed

# ── Frame queue ──────────────────────────────────────────────────────────────
FRAME_QUEUE_SIZE = _env_int("FRAME_QUEUE_SIZE", 64)

# ── Recording ────────────────────────────────────────────────────────────────
VIDEO_CODEC = _env_str("VIDEO_CODEC", "h264")   # "h264" (PyAV) or "mp4v" (OpenCV)
H264_CRF = _env_int("H264_CRF", 23)
H264_PRESET = _env_str("H264_PRESET", "veryfast")

# ── Transaction logging / calibration ────────────────────────────────────────
TRANSACTION_LOG_DIR = _env_str("TRANSACTION_LOG_DIR", os.path.join("logs", "transactions"))

# ── Disk retention ───────────────────────────────────────────────────────────
RETENTION_DAYS = _env_int("RETENTION_DAYS", 30)          # suspicious clips in OUTPUT_DIR
LOG_RETENTION_DAYS = _env_int("LOG_RETENTION_DAYS", 180)  # transaction JSON logs

# ── Training data collection ─────────────────────────────────────────────────
# Every Nth clean (non-suspicious) transaction clip is kept for future model
# training instead of being deleted. 0 disables collection.
TRAINING_DATA_DIR = _env_str("TRAINING_DATA_DIR", "training_data")
TRAINING_CLIP_EVERY_N = _env_int("TRAINING_CLIP_EVERY_N", 20)

# ── Server FPS profile ───────────────────────────────────────────────────────
MONITORING_FPS = _env_int("MONITORING_FPS", 10)   # idle monitoring
TRANSACTION_FPS = _env_int("TRANSACTION_FPS", 25)  # during an active sale
