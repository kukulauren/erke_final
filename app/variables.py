MODEL_PATH="best.pt"
VIDEO_PATH=""
OVERLAP_THRESHOLD=0.3
CONF_THRESHOLD=0.5
CUSTOMER_DWELL_TIME=3.0
SCANNER_PHONE_DISTANCE=80
SCANNER_ITEM_DISTANCE=100
SCANNER_MOVEMENT_THRESHOLD=3
SCAN_COOLDOWN=1.5
PAYMENT_COMPLETE_TIME=1.0

STAFF_REENTRY_THRESHOLD=4              # Need 4+ re-entries within recent window (was 3)
REENTRY_WINDOW=600.0                   # 10-minute window for counting recent re-entries (was 1 hour)
REENTRY_MIN_SESSIONS=2                 # Must have at least 2 distinct sessions

STAFF_CUMULATIVE_TIME=900.0             # 15 minutes total time across sessions (was 10 min)
STAFF_TIME_MIN_SESSIONS=3               # Must have at least 3 visits

# Confidence scoring for secondary staff
STAFF_CONFIDENCE_THRESHOLD=0.65         # Min confidence (0-1) to classify as secondary staff
RECENT_BEHAVIOR_WINDOW=1800.0           # 30 minutes - window for evaluating recent behavior
CONFIDENCE_DECAY_RATE=0.98              # Confidence decays at 2% per check (slower decay for single-day)