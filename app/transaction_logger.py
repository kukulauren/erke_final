"""Per-transaction JSON logging for threshold calibration and audits.

Every /stop_prediction writes one JSON file with the raw analytics evidence
(scan distances/durations, payment durations, dwell times, cash events), the
thresholds that were active, and the final verdict. calibrate.py aggregates
these files to suggest data-driven thresholds.
"""
import json
import logging
import os
import time

from app import variables

logger = logging.getLogger(__name__)

# Thresholds worth calibrating, snapshotted into every log
_THRESHOLD_KEYS = [
    "CONF_THRESHOLD",
    "CUSTOMER_DWELL_TIME",
    "SCANNER_PHONE_DISTANCE",
    "SCANNER_ITEM_DISTANCE",
    "SCANNER_MOVEMENT_THRESHOLD",
    "SCAN_COOLDOWN",
    "PAYMENT_COMPLETE_TIME",
    "SCAN_MIN_DURATION",
    "CASH_MIN_DURATION",
    "ACTION_GAP_TOLERANCE",
    "WRIST_NEAR_DISTANCE",
]


def log_transaction(analytics, output, developer_message, voucher_number,
                    pos_member, pos_wallet, log_dir=None):
    """Write one transaction record; returns the file path or None on failure."""
    log_dir = log_dir or variables.TRANSACTION_LOG_DIR

    record = {
        "voucher_number": voucher_number,
        "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pos_member": pos_member,
        "pos_wallet": pos_wallet,
        "verdict": output,
        "developer_message": developer_message,
        "thresholds": {k: getattr(variables, k) for k in _THRESHOLD_KEYS},
        "evidence": {
            "scanned_items": analytics.scanned_items,
            "payment_times": analytics.payment_times,
            "completed_payments": analytics.completed_payments,
            "cash_events": analytics.cash_detected,
            "customer_visits": len(analytics.customer_visits),
            "service_times": analytics.service_times,
            "scanner_ever_moved": analytics.scanner_ever_moved,
            "cashier_seen": analytics.cashier_seen(),
        },
    }

    try:
        os.makedirs(log_dir, exist_ok=True)
        safe_voucher = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(voucher_number))
        path = os.path.join(log_dir, f"{int(time.time() * 1000)}_{safe_voucher}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=float)
        return path
    except Exception:
        logger.exception("Failed to write transaction log")
        return None
