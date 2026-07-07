"""Tests for RetailAnalytics ledgers, counter dwell and staff classification."""
import numpy as np

from app.helper_functions import boxes_overlap, get_box_iou, get_distance
from app.retail_analytics import RetailAnalytics
from app.variables import CUSTOMER_DWELL_TIME, SCAN_MIN_DURATION


def make_det(track_id, cx, cy, size=50):
    half = size / 2
    return {
        'track_id': track_id,
        'box': np.array([cx - half, cy - half, cx + half, cy + half]),
        'center': (cx, cy),
        'conf': 0.9,
    }


class TestGeometry:

    def test_iou_identical_boxes(self):
        box = [0, 0, 10, 10]
        assert get_box_iou(box, box) == 1.0

    def test_iou_disjoint_boxes(self):
        assert get_box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_overlap(self):
        assert boxes_overlap([0, 0, 10, 10], [5, 5, 15, 15])
        assert not boxes_overlap([0, 0, 10, 10], [11, 11, 20, 20])

    def test_distance(self):
        assert get_distance((0, 0), (3, 4)) == 5.0


class TestScannerMovement:

    def test_scanner_ever_moved_flag(self):
        a = RetailAnalytics()
        assert a.scanner_ever_moved is False
        s1 = make_det(2, 100, 100)
        s2 = make_det(2, 150, 150)  # big jump = moving
        a.update_scanner_movement([s1], 0.0)
        a.update_scanner_movement([s2], 0.1)
        assert a.scanner_ever_moved is True

    def test_stationary_scanner_does_not_set_flag(self):
        a = RetailAnalytics()
        s = make_det(2, 100, 100)
        for t in (0.0, 0.1, 0.2):
            a.update_scanner_movement([s], t)
        assert a.scanner_ever_moved is False


class TestActionsLedger:

    def test_confirmed_scan_lands_in_ledger(self):
        a = RetailAnalytics()
        det = {'cashier': [], 'customer': [], 'counter': [], 'phone': [], 'cash': [],
               'item': [make_det(1, 125, 125)],
               'scanner': [make_det(2, 135, 135)]}
        status = {2: {'moving': True}}
        events = []
        t, step = 0.0, 0.1
        while t <= SCAN_MIN_DURATION + step:
            events += a.update_actions(det, status, [], t)
            t += step
        assert len(a.scanned_items) == 1
        assert any('SCANNED' in e for e in events)


class TestCustomerAtCounter:

    def test_dwell_below_threshold_not_counted(self):
        a = RetailAnalytics()
        cust = [make_det(4, 100, 100, size=120)]
        counter = [make_det(9, 110, 110, size=200)]
        a.update_customer_at_counter(cust, counter, 0.0)
        a.update_customer_at_counter(cust, counter, CUSTOMER_DWELL_TIME * 0.5)
        assert a.customer_visits == []

    def test_dwell_above_threshold_counted_once(self):
        a = RetailAnalytics()
        cust = [make_det(4, 100, 100, size=120)]
        counter = [make_det(9, 110, 110, size=200)]
        for t in (0.0, CUSTOMER_DWELL_TIME + 0.1, CUSTOMER_DWELL_TIME + 1.0):
            a.update_customer_at_counter(cust, counter, t)
        assert len(a.customer_visits) == 1

    def test_leaving_records_service_time(self):
        a = RetailAnalytics()
        cust = [make_det(4, 100, 100, size=120)]
        counter = [make_det(9, 110, 110, size=200)]
        a.update_customer_at_counter(cust, counter, 0.0)
        a.update_customer_at_counter([], counter, 5.0)  # customer gone
        assert len(a.service_times) == 1
        assert a.service_times[0] == 5.0


class TestStaffClassification:

    def test_detected_cashier_is_primary_staff(self):
        a = RetailAnalytics()
        cashier = [make_det(3, 400, 400, size=120)]
        a.update_person_behavior(customers=[], current_time=0.0, cashiers=cashier)
        rec = a.person_records[3]
        assert rec['classification'] == 'staff'
        assert rec['staff_source'] == 'primary'
        assert a.cashier_seen() is True

    def test_plain_customer_is_not_staff(self):
        a = RetailAnalytics()
        cust = [make_det(4, 100, 100, size=120)]
        a.update_person_behavior(customers=cust, current_time=0.0, cashiers=[])
        assert a.person_records[4]['classification'] == 'customer'
        assert a.cashier_seen() is False

    def test_person_label(self):
        a = RetailAnalytics()
        cashier = [make_det(3, 400, 400, size=120)]
        a.update_person_behavior(customers=[], current_time=0.0, cashiers=cashier)
        assert a.get_person_label(3) == 'staff-primary'
        assert a.get_person_label(999) == 'unknown'
