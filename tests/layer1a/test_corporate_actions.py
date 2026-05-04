"""Tests for layer1a.corporate_actions — CorporateActionsManager."""

from __future__ import annotations

import sqlite3
import pytest

from layer1a.security_master import SecurityMaster
from layer1a.corporate_actions import (
    CorporateActionsManager,
    DuplicateCorporateActionError,
    InvalidActionTypeError,
    InvalidAdjustmentFactorError,
    CorporateActionNotFoundError,
)


@pytest.fixture()
def sm(tmp_path) -> SecurityMaster:
    sm = SecurityMaster.__new__(SecurityMaster)
    sm._db_path = tmp_path / "sm.db"
    sm._alert = None
    sm.conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
    sm.conn.row_factory = sqlite3.Row
    sm._apply_schema()
    return sm


@pytest.fixture()
def ca(sm) -> CorporateActionsManager:
    return CorporateActionsManager(sm)


@pytest.fixture()
def nid(sm) -> str:
    return sm.add_instrument("EQUITY", "XNAS", "USD", "Corp")


class TestCorporateActions:

    def test_valid_split(self, ca, nid):
        aid = ca.add_action(nid, "SPLIT", "2024-06-01", 0.5, "test")
        assert isinstance(aid, int)

    def test_valid_reverse_split(self, ca, nid):
        aid = ca.add_action(nid, "REVERSE_SPLIT", "2024-06-01", 4.0, "test")
        assert isinstance(aid, int)

    def test_split_with_factor_gte_1_raises(self, ca, nid):
        with pytest.raises(InvalidAdjustmentFactorError):
            ca.add_action(nid, "SPLIT", "2024-06-01", 2.0, "test")

    def test_reverse_split_with_factor_lte_1_raises(self, ca, nid):
        with pytest.raises(InvalidAdjustmentFactorError):
            ca.add_action(nid, "REVERSE_SPLIT", "2024-06-01", 0.25, "test")

    def test_duplicate_raises(self, ca, nid):
        ca.add_action(nid, "SPLIT", "2024-06-01", 0.5, "test")
        with pytest.raises(DuplicateCorporateActionError):
            ca.add_action(nid, "SPLIT", "2024-06-01", 0.25, "test")

    def test_unapplied_filter_by_date(self, ca, nid):
        ca.add_action(nid, "SPLIT", "2023-01-01", 0.5, "test")
        ca.add_action(nid, "DIVIDEND_CASH", "2025-12-31", 0.95, "test")
        unapplied = ca.get_unapplied_actions("2024-01-01")
        assert len(unapplied) == 1
        assert unapplied[0]["ex_date"] == "2023-01-01"

    def test_mark_applied(self, ca, nid):
        aid = ca.add_action(nid, "SPLIT", "2024-01-01", 0.25, "test")
        ca.mark_action_applied(aid)
        remaining = ca.get_unapplied_actions("2025-01-01")
        assert all(a["action_id"] != aid for a in remaining)

    def test_mark_applied_not_found_raises(self, ca, nid):
        with pytest.raises(CorporateActionNotFoundError):
            ca.mark_action_applied(99999)

    def test_cumulative_factor_multiplication(self, ca, nid):
        ca.add_action(nid, "SPLIT", "2023-01-01", 0.5, "test")
        ca.add_action(nid, "REVERSE_SPLIT", "2024-01-01", 2.0, "test")
        factor = ca.get_cumulative_adjustment_factor(nid, "2023-01-01", "2024-06-01")
        assert abs(factor - 1.0) < 1e-9  # 0.5 × 2.0 = 1.0

    def test_missed_action_heuristic_detection(self, ca, nid):
        price_series = [
            ("2024-01-01", 100.0),
            ("2024-01-02", 160.0),  # +60% — suspicious
            ("2024-01-03", 162.0),  # normal
        ]
        suspects = ca.check_for_missed_actions(nid, price_series)
        assert "2024-01-02" in suspects

    def test_no_missed_action_when_known_action_exists(self, ca, nid):
        ca.add_action(nid, "SPLIT", "2024-01-02", 0.5, "test")
        price_series = [
            ("2024-01-01", 100.0),
            ("2024-01-02", 50.0),  # halved due to split — not suspicious
        ]
        suspects = ca.check_for_missed_actions(nid, price_series)
        assert len(suspects) == 0
