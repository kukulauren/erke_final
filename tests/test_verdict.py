"""Tests for the suspicious-activity decision matrix in Prediction.print_output."""
import threading

import pytest

from app.retail_analytics import RetailAnalytics
from pipeline import Prediction


def make_prediction(analytics=None):
    """Build a Prediction without loading YOLO or opening a video source."""
    p = Prediction.__new__(Prediction)
    p._lock = threading.Lock()
    p.analytics = analytics or RetailAnalytics()
    p.suspicious = False
    return p


def full_evidence_analytics():
    a = RetailAnalytics()
    a.customer_visits.append({'customer_id': 4})
    a.cash_detected.append({'action': 'cash_handover', 'time': 1.0})
    a.completed_payments.append({'action': 'phone_payment', 'time': 2.0})
    return a


class TestWalletPayment:

    def test_wallet_never_suspicious_even_without_cv_evidence(self):
        p = make_prediction()  # empty analytics
        output, dev = p.print_output(pos_wallet=True, pos_member=True)
        assert output['suspicious_activity'] is False
        assert output['purchasing_customer'] is True
        assert output['member_use'] is True
        assert p.suspicious is False
        assert dev == {}

    def test_wallet_non_member(self):
        p = make_prediction()
        output, _ = p.print_output(pos_wallet=True, pos_member=False)
        assert output['member_use'] is False
        assert output['customer_paid_wallet'] is True
        assert output['customer_paid_cash'] is False


class TestMemberCash:

    def test_all_evidence_present_is_clean(self):
        p = make_prediction(full_evidence_analytics())
        output, dev = p.print_output(pos_wallet=False, pos_member=True)
        assert output['suspicious_activity'] is False
        assert output['member_use'] is True
        assert p.suspicious is False
        assert dev == {}

    @pytest.mark.parametrize("missing,code_key,code", [
        ("customer", "customer_detection", "POSM1-MODELC0"),
        ("cash", "cash_detection", "POSM1-MODELB0"),
        ("member_scan", "member_detection", "POSM1-MODELM0"),
    ])
    def test_each_missing_signal_flags_suspicious(self, missing, code_key, code):
        a = full_evidence_analytics()
        if missing == "customer":
            a.customer_visits.clear()
        elif missing == "cash":
            a.cash_detected.clear()
        else:
            a.completed_payments.clear()

        p = make_prediction(a)
        output, dev = p.print_output(pos_wallet=False, pos_member=True)
        assert output['suspicious_activity'] is True
        assert p.suspicious is True
        assert dev == {code_key: code}

    def test_all_signals_missing_reports_all_codes(self):
        p = make_prediction()  # empty analytics
        output, dev = p.print_output(pos_wallet=False, pos_member=True)
        assert output['suspicious_activity'] is True
        assert set(dev) == {"customer_detection", "cash_detection", "member_detection"}


class TestNonMemberCash:

    def test_cv_not_enforced(self):
        p = make_prediction()  # no CV evidence at all
        output, dev = p.print_output(pos_wallet=False, pos_member=False)
        assert output['suspicious_activity'] is False
        assert output['purchasing_customer'] is True
        assert output['member_use'] is False
        assert dev == {}


class TestObservedSignals:

    def test_signals_reflect_analytics_not_hardcoded(self):
        p = make_prediction()  # empty analytics
        output, _ = p.print_output(pos_wallet=False, pos_member=False)
        assert output['items_scanned'] is False
        assert output['cashier'] is False
        assert output['scanner_moving'] is False

        a = full_evidence_analytics()
        a.scanned_items.append({'action': 'item_scan', 'time': 1.0})
        a.scanner_ever_moved = True
        a.person_records[3] = {'classification': 'staff', 'staff_source': 'primary'}
        p2 = make_prediction(a)
        output2, _ = p2.print_output(pos_wallet=False, pos_member=False)
        assert output2['items_scanned'] is True
        assert output2['cashier'] is True
        assert output2['scanner_moving'] is True
