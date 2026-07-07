"""Temporal action recognition over detection + pose streams.

Replaces single-frame proximity heuristics with sustained multi-frame
evidence. An action only fires after its evidence has been observed for a
minimum duration, and brief detection dropouts (up to ACTION_GAP_TOLERANCE)
do not reset the evidence — this removes both the false positives of
"scanner happened to pass near an item for one frame" and the false negatives
of "the detector missed the phone for one frame mid-payment".

Recognized actions (each returned as a dict with an 'action' key):
  - item_scan:     item inside the scanner zone while the scanner is moving,
                   OR a cashier wrist is on the item; sustained SCAN_MIN_DURATION
  - phone_payment: phone near the scanner sustained PAYMENT_COMPLETE_TIME
  - cash_handover: cash overlapping a customer or under a cashier wrist;
                   sustained CASH_MIN_DURATION

A learned video classifier (e.g. X3D / MoViNet trained on labelled checkout
clips) can replace the rule backend via set_model(); confirmed actions keep
the same schema so downstream code does not change.
"""
import logging
from collections import defaultdict

from app.helper_functions import boxes_overlap, get_box_iou, get_distance
from app.variables import (
    ACTION_GAP_TOLERANCE,
    CASH_MIN_DURATION,
    PAYMENT_COMPLETE_TIME,
    SCAN_COOLDOWN,
    SCAN_MIN_DURATION,
    SCANNER_ITEM_DISTANCE,
    SCANNER_PHONE_DISTANCE,
    WRIST_NEAR_DISTANCE,
)

logger = logging.getLogger(__name__)

PAYMENT_COOLDOWN = 5.0  # s between confirmed payments for the same phone


class _Evidence:
    """Accumulates evidence time for one (action, key) pair with gap tolerance."""
    __slots__ = ("start", "last", "last_confirm")

    def __init__(self):
        self.start = None
        self.last = None
        self.last_confirm = -1e9

    def update(self, current_time, active):
        """Feed one observation; returns accumulated evidence duration (s)."""
        if active:
            if self.start is None or current_time - self.last > ACTION_GAP_TOLERANCE:
                self.start = current_time  # gap too long → new evidence window
            self.last = current_time
            return self.last - self.start
        if self.last is not None and current_time - self.last > ACTION_GAP_TOLERANCE:
            self.start = None
            self.last = None
        return 0.0 if self.start is None else self.last - self.start

    def confirm(self, current_time):
        self.last_confirm = current_time
        self.start = None
        self.last = None

    def in_cooldown(self, current_time, cooldown):
        return current_time - self.last_confirm < cooldown


class ActionRecognizer:

    def __init__(self):
        # action type -> key (track id) -> _Evidence
        self._evidence = defaultdict(lambda: defaultdict(_Evidence))
        self._model = None

    def set_model(self, model):
        """Plug in a learned clip classifier. It must expose
        classify(frames, detections_window) -> list of action dicts.
        Until one is trained, the temporal rule backend below is used."""
        self._model = model

    # ── Main entry ───────────────────────────────────────────────────────────

    def update(self, detections, scanner_status, wrists, current_time):
        """Feed one processed frame; returns newly confirmed action dicts."""
        if self._model is not None:
            try:
                return self._model.classify(detections, current_time)
            except Exception:
                logger.exception("Learned action model failed; using rule backend")

        wrists = wrists or []
        actions = []
        actions.extend(self._item_scans(detections, scanner_status, wrists, current_time))
        actions.extend(self._phone_payments(detections, current_time))
        actions.extend(self._cash_handovers(detections, wrists, current_time))
        return actions

    # ── Rule backend ─────────────────────────────────────────────────────────

    def _item_scans(self, detections, scanner_status, wrists, current_time):
        actions = []
        scanners = detections.get('scanner', [])
        tracks = self._evidence['item_scan']

        for item in detections.get('item', []):
            item_id = item.get('track_id') or id(item)
            evidence = tracks[item_id]

            if evidence.in_cooldown(current_time, SCAN_COOLDOWN):
                continue

            # Evidence source 1: scanner in position AND moving
            active = False
            best_scanner, best_dist = None, None
            for scanner in scanners:
                sid = scanner.get('track_id') or 0
                moving = scanner_status.get(sid, {}).get('moving', False)
                dist = get_distance(scanner['center'], item['center'])
                in_position = (
                    boxes_overlap(scanner['box'], item['box'])
                    or dist < SCANNER_ITEM_DISTANCE
                )
                if in_position and moving:
                    active = True
                    if best_dist is None or dist < best_dist:
                        best_scanner, best_dist = sid, dist

            # Evidence source 2: cashier wrist on the item (pose)
            wrist_on_item = any(
                get_distance(w, item['center']) < WRIST_NEAR_DISTANCE for w in wrists
            )
            active = active or wrist_on_item

            duration = evidence.update(current_time, active)
            if active and duration >= SCAN_MIN_DURATION:
                evidence.confirm(current_time)
                actions.append({
                    'action': 'item_scan',
                    'time': current_time,
                    'item_id': item_id,
                    'scanner_id': best_scanner,
                    'distance': float(best_dist) if best_dist is not None else None,
                    'duration': duration,
                    'wrist_evidence': wrist_on_item,
                })
        return actions

    def _phone_payments(self, detections, current_time):
        actions = []
        scanners = detections.get('scanner', [])
        tracks = self._evidence['phone_payment']

        for phone in detections.get('phone', []):
            phone_id = phone.get('track_id') or id(phone)
            evidence = tracks[phone_id]

            if evidence.in_cooldown(current_time, PAYMENT_COOLDOWN):
                continue

            dists = [get_distance(phone['center'], s['center']) for s in scanners]
            active = any(d < SCANNER_PHONE_DISTANCE for d in dists)

            duration = evidence.update(current_time, active)
            if active and duration >= PAYMENT_COMPLETE_TIME:
                evidence.confirm(current_time)
                actions.append({
                    'action': 'phone_payment',
                    'time': current_time,
                    'phone_id': phone_id,
                    'duration': duration,
                    'distance': float(min(dists)) if dists else None,
                })
        return actions

    def _cash_handovers(self, detections, wrists, current_time):
        actions = []
        customers = detections.get('customer', [])
        tracks = self._evidence['cash_handover']

        for cash in detections.get('cash', []):
            cash_id = cash.get('track_id') or id(cash)
            evidence = tracks[cash_id]

            if evidence.in_cooldown(current_time, SCAN_COOLDOWN):
                continue

            customer_id = None
            for customer in customers:
                if boxes_overlap(cash['box'], customer['box']):
                    customer_id = customer.get('track_id') or id(customer)
                    break

            wrist_on_cash = any(
                get_distance(w, cash['center']) < WRIST_NEAR_DISTANCE for w in wrists
            )
            active = customer_id is not None or wrist_on_cash

            duration = evidence.update(current_time, active)
            if active and duration >= CASH_MIN_DURATION:
                evidence.confirm(current_time)
                actions.append({
                    'action': 'cash_handover',
                    'time': current_time,
                    'cash_id': cash_id,
                    'customer_id': customer_id,
                    'duration': duration,
                    'wrist_evidence': wrist_on_cash,
                })
        return actions
