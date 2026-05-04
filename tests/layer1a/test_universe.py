"""Tests for layer1a.universe — UniverseManager."""

from __future__ import annotations

import sqlite3
import pytest

from layer1a.security_master import SecurityMaster
from layer1a.universe import UniverseManager, AlreadyInUniverseError, NotInUniverseError


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
def um(sm) -> UniverseManager:
    return UniverseManager(sm)


@pytest.fixture()
def nid(sm) -> str:
    return sm.add_instrument("EQUITY", "XNAS", "USD", "Test Corp")


class TestUniverseManager:

    def test_add_succeeds(self, um, nid):
        um.add_to_universe("TEST_U", nid, "2024-01-01")
        current = um.get_current_universe("TEST_U")
        assert nid in current

    def test_duplicate_raises(self, um, nid):
        um.add_to_universe("TEST_U", nid, "2024-01-01")
        with pytest.raises(AlreadyInUniverseError):
            um.add_to_universe("TEST_U", nid, "2024-01-01")

    def test_remove_sets_date_and_reason(self, um, nid):
        um.add_to_universe("TEST_U", nid, "2024-01-01")
        um.remove_from_universe("TEST_U", nid, "2025-01-01", "DELISTED")
        history = um.get_universe_history("TEST_U")
        assert history[-1]["removed_date"] == "2025-01-01"
        assert history[-1]["removal_reason"] == "DELISTED"

    def test_remove_non_member_raises(self, um, nid):
        with pytest.raises(NotInUniverseError):
            um.remove_from_universe("TEST_U", nid, "2025-01-01", "MANUAL_REMOVAL")

    def test_point_in_time_correctness(self, um, sm):
        a = sm.add_instrument("EQUITY", "XNAS", "USD", "Company A")
        b = sm.add_instrument("EQUITY", "XNAS", "USD", "Company B")
        um.add_to_universe("U", a, "2023-01-01")
        um.add_to_universe("U", b, "2023-01-01")
        um.remove_from_universe("U", b, "2024-01-01", "DELISTED")

        # Before removal: both
        at_2023 = um.get_universe_at_date("U", "2023-06-01")
        assert a in at_2023 and b in at_2023

        # After removal: only a
        at_2025 = um.get_universe_at_date("U", "2025-01-01")
        assert a in at_2025 and b not in at_2025

    def test_current_universe_null_filter(self, um, sm):
        a = sm.add_instrument("EQUITY", "XNAS", "USD", "A")
        b = sm.add_instrument("EQUITY", "XNAS", "USD", "B")
        um.add_to_universe("U", a, "2023-01-01")
        um.add_to_universe("U", b, "2023-01-01")
        um.remove_from_universe("U", b, "2024-01-01", "MANUAL_REMOVAL")
        current = um.get_current_universe("U")
        assert a in current and b not in current

    def test_size_over_time_counts(self, um, sm):
        a = sm.add_instrument("EQUITY", "XNAS", "USD", "A")
        b = sm.add_instrument("EQUITY", "XNAS", "USD", "B")
        um.add_to_universe("U", a, "2024-01-01")
        um.add_to_universe("U", b, "2024-01-01")
        um.remove_from_universe("U", b, "2024-06-01", "MANUAL_REMOVAL")

        sizes = um.get_universe_size_over_time("U", "2024-01-01", "2024-07-01", frequency="monthly")
        # January: 2, July: 1
        jan = next(s for s in sizes if s["date"] == "2024-01-01")
        jul = next(s for s in sizes if s["date"] == "2024-07-01")
        assert jan["size"] == 2
        assert jul["size"] == 1
