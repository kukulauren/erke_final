"""Tests for the temporal action recognizer — the core anti-fraud logic."""
import numpy as np
import pytest

from app.action_recognizer import ActionRecognizer, _Evidence, PAYMENT_COOLDOWN
from app.variables import (
    ACTION_GAP_TOLERANCE,
    PAYMENT_COMPLETE_TIME,
    SCAN_COOLDOWN,
    SCAN_MIN_DURATION,
)


def make_det(track_id, cx, cy, size=50):
    half = size / 2
    return {
        'track_id': track_id,
        'box': np.array([cx - half, cy - half, cx + half, cy + half]),
        'center': (cx, cy),
        'conf': 0.9,
    }


def empty_detections(**overrides):
    det = {'cashier': [], 'customer': [], 'scanner': [], 'item': [],
           'phone': [], 'cash': [], 'counter': []}
    det.update(overrides)
    return det


class TestEvidence:

    def test_accumulates_while_active(self):
        e = _Evidence()
        assert e.update(0.0, True) == 0.0
        assert e.update(0.3, True) == pytest.approx(0.3)

    def test_short_dropout_keeps_evidence(self):
        e = _Evidence()
        e.update(0.0, True)
        e.update(0.3, True)
        # inactive observation within gap tolerance does not reset
        assert e.update(0.3 + ACTION_GAP_TOLERANCE / 2, False) == pytest.approx(0.3)
        # resuming within tolerance keeps the original start
        t = 0.3 + ACTION_GAP_TOLERANCE * 0.9
        assert e.update(t, True) == pytest.approx(t)

    def test_long_gap_resets(self):
        e = _Evidence()
        e.update(0.0, True)
        e.update(0.3, True)
        t = 0.3 + ACTION_GAP_TOLERANCE * 2
        assert e.update(t, True) == 0.0  # new evidence window

    def test_cooldown(self):
        e = _Evidence()
        e.confirm(10.0)
        assert e.in_cooldown(10.0 + 0.5, cooldown=1.5)
        assert not e.in_cooldown(10.0 + 2.0, cooldown=1.5)


class TestItemScan:

    def _scan_scene(self):
        det = empty_detections(
            item=[make_det(1, 125, 125)],
            scanner=[make_det(2, 135, 135)],
        )
        status = {2: {'moving': True}}
        return det, status

    def test_sustained_evidence_confirms_scan(self):
        r = ActionRecognizer()
        det, status = self._scan_scene()
        actions = []
        t, step = 0.0, 0.1
        while t <= SCAN_MIN_DURATION + step:
            actions += r.update(det, status, [], t)
            t += step
        assert len(actions) == 1
        assert actions[0]['action'] == 'item_scan'
        assert actions[0]['duration'] >= SCAN_MIN_DURATION

    def test_single_frame_blip_does_not_fire(self):
        r = ActionRecognizer()
        det, status = self._scan_scene()
        assert r.update(det, status, [], 0.0) == []

    def test_stationary_scanner_never_fires(self):
        r = ActionRecognizer()
        det, _ = self._scan_scene()
        status = {2: {'moving': False}}
        actions = []
        for i in range(20):
            actions += r.update(det, status, [], i * 0.1)
        assert actions == []

    def test_wrist_on_item_alone_confirms_scan(self):
        r = ActionRecognizer()
        det = empty_detections(item=[make_det(1, 125, 125)])
        wrists = [(128, 128)]
        actions = []
        t, step = 0.0, 0.2
        while t <= SCAN_MIN_DURATION + step:
            actions += r.update(det, {}, wrists, t)
            t += step
        assert len(actions) == 1
        assert actions[0]['wrist_evidence'] is True

    def test_cooldown_blocks_double_count(self):
        r = ActionRecognizer()
        det, status = self._scan_scene()
        actions = []
        t, step = 0.0, 0.1
        # keep scanning continuously for less than SCAN_MIN + COOLDOWN
        while t <= SCAN_MIN_DURATION + SCAN_COOLDOWN * 0.5:
            actions += r.update(det, status, [], t)
            t += step
        assert len(actions) == 1  # second scan of the same item suppressed


class TestPhonePayment:

    def _payment_scene(self):
        return empty_detections(
            phone=[make_det(7, 200, 200)],
            scanner=[make_det(2, 210, 210)],
        )

    def test_sustained_proximity_completes_payment(self):
        r = ActionRecognizer()
        det = self._payment_scene()
        actions = []
        t, step = 0.0, 0.2
        while t <= PAYMENT_COMPLETE_TIME + step:
            actions += r.update(det, {}, [], t)
            t += step
        assert len(actions) == 1
        assert actions[0]['action'] == 'phone_payment'

    def test_detection_dropout_mid_payment_still_completes(self):
        """The old code reset instantly on one missed frame — regression guard."""
        r = ActionRecognizer()
        det = self._payment_scene()
        gone = empty_detections(scanner=det['scanner'])
        actions = []
        t, step = 0.0, 0.2
        dropout_at = PAYMENT_COMPLETE_TIME / 2
        while t <= PAYMENT_COMPLETE_TIME + 2 * step:
            frame = gone if abs(t - dropout_at) < step / 2 else det
            actions += r.update(frame, {}, [], t)
            t += step
        assert len(actions) == 1

    def test_payment_cooldown_prevents_duplicates(self):
        r = ActionRecognizer()
        det = self._payment_scene()
        actions = []
        t, step = 0.0, 0.2
        while t <= PAYMENT_COMPLETE_TIME + PAYMENT_COOLDOWN * 0.5:
            actions += r.update(det, {}, [], t)
            t += step
        assert len(actions) == 1


class TestCashHandover:

    def test_cash_on_customer_confirms(self):
        r = ActionRecognizer()
        det = empty_detections(
            cash=[make_det(5, 300, 300, size=30)],
            customer=[make_det(4, 305, 305, size=120)],
        )
        actions = []
        for i in range(6):
            actions += r.update(det, {}, [], i * 0.15)
        assert len(actions) == 1
        assert actions[0]['action'] == 'cash_handover'
        assert actions[0]['customer_id'] == 4

    def test_wrist_on_cash_confirms(self):
        r = ActionRecognizer()
        det = empty_detections(cash=[make_det(5, 300, 300, size=30)])
        actions = []
        for i in range(6):
            actions += r.update(det, {}, [(302, 302)], i * 0.15)
        assert len(actions) == 1
        assert actions[0]['wrist_evidence'] is True

    def test_isolated_cash_never_fires(self):
        r = ActionRecognizer()
        det = empty_detections(cash=[make_det(5, 300, 300, size=30)])
        actions = []
        for i in range(20):
            actions += r.update(det, {}, [], i * 0.15)
        assert actions == []
