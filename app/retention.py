"""Disk retention for recordings and transaction logs.

Prevents the disk from silently filling up over months of operation:
  - suspicious clips in OUTPUT_DIR older than RETENTION_DAYS are deleted
    (this also sweeps up orphaned txn_*.mp4 temp files from crashed runs)
  - transaction JSON logs older than LOG_RETENTION_DAYS are deleted
  - TRAINING_DATA_DIR is never touched — training clips are curated manually

The Flask server runs this automatically once a day in a background thread.
It can also be run manually / from Windows Task Scheduler:

    python -m app.retention
"""
import glob
import logging
import os
import threading
import time

from app.variables import (
    LOG_RETENTION_DAYS,
    OUTPUT_DIR,
    RETENTION_DAYS,
    TRANSACTION_LOG_DIR,
)

logger = logging.getLogger(__name__)


def cleanup_directory(directory, older_than_days, pattern="*"):
    """Delete files matching pattern older than N days. Returns count deleted."""
    if older_than_days <= 0 or not os.path.isdir(directory):
        return 0

    cutoff = time.time() - older_than_days * 86400
    deleted = 0
    for path in glob.glob(os.path.join(directory, pattern)):
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                deleted += 1
        except OSError as e:
            logger.warning("Could not delete %s: %s", path, e)

    if deleted:
        logger.info("Retention: deleted %d file(s) from %s (older than %d days)",
                    deleted, directory, older_than_days)
    return deleted


def run_cleanup():
    """One retention pass over recordings and logs. Returns per-target counts."""
    return {
        "recordings_deleted": cleanup_directory(OUTPUT_DIR, RETENTION_DAYS, "*.mp4"),
        "logs_deleted": cleanup_directory(TRANSACTION_LOG_DIR, LOG_RETENTION_DAYS, "*.json"),
    }


def start_retention_thread(interval_hours=24):
    """Run cleanup now and then daily in a daemon thread."""
    def _loop():
        while True:
            try:
                run_cleanup()
            except Exception:
                logger.exception("Retention pass failed")
            time.sleep(interval_hours * 3600)

    thread = threading.Thread(target=_loop, daemon=True, name="retention")
    thread.start()
    return thread


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_cleanup()
    print(f"Recordings deleted: {result['recordings_deleted']}")
    print(f"Logs deleted:       {result['logs_deleted']}")
