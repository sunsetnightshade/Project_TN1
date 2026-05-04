"""Tests for layer1a.security_master."""

from __future__ import annotations

import pytest

from layer1a.security_master import (
    SecurityMaster,
    InstrumentNotFoundError,
    InvalidInstrumentTypeError,
    InvalidExchangeCodeError,
    InvalidCurrencyCodeError,
    InvalidFieldError,
    SymbolNotFoundError,
    AmbiguousSymbolError,
)


@pytest.fixture()
def sm(tmp_path) -> SecurityMaster:
    """In-memory-like SecurityMaster backed by a temp SQLite file."""
    sm = SecurityMaster.__new__(SecurityMaster)
    import sqlite3
    sm._db_path = tmp_path / "sm.db"
    sm._alert = None
    sm.conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
    sm.conn.row_factory = sqlite3.Row
    sm._apply_schema()
    return sm


class TestAddInstrument:
    def test_valid_add_returns_uuid(self, sm):
        nid = sm.add_instrument("EQUITY", "XNAS", "USD", "Test Corp")
        assert len(nid) == 36  # UUID4

    def test_invalid_type_raises(self, sm):
        with pytest.raises(InvalidInstrumentTypeError):
            sm.add_instrument("BOND", "XNAS", "USD", "Test")

    def test_invalid_exchange_raises(self, sm):
        with pytest.raises(InvalidExchangeCodeError):
            sm.add_instrument("EQUITY", "X", "USD", "Test")  # 1 letter

    def test_invalid_currency_raises(self, sm):
        with pytest.raises(InvalidCurrencyCodeError):
            sm.add_instrument("EQUITY", "XNAS", "US", "Test")


class TestResolve:
    def test_resolve_with_effective_dates(self, sm):
        nid = sm.add_instrument("EQUITY", "XNAS", "USD", "Old Corp")
        sm.add_symbol_mapping(nid, "polygon", "OLDTICK", "2020-01-01T00:00:00+00:00")

        resolved = sm.resolve("polygon", "OLDTICK", "2021-06-01T00:00:00+00:00")
        assert resolved == nid

    def test_symbol_not_found_raises(self, sm):
        with pytest.raises(SymbolNotFoundError):
            sm.resolve("polygon", "NOTEXIST")

    def test_recycled_ticker_closes_old_mapping(self, sm):
        old_id = sm.add_instrument("EQUITY", "XNAS", "USD", "Old Co")
        new_id = sm.add_instrument("EQUITY", "XNAS", "USD", "New Co")
        sm.add_symbol_mapping(old_id, "polygon", "RECYCLE", "2020-01-01T00:00:00+00:00")
        sm.add_symbol_mapping(new_id, "polygon", "RECYCLE", "2023-01-01T00:00:00+00:00")

        # Old mapping should be closed
        resolved = sm.resolve("polygon", "RECYCLE", "2024-01-01T00:00:00+00:00")
        assert resolved == new_id

    def test_time_based_resolve_past(self, sm):
        old_id = sm.add_instrument("EQUITY", "XNAS", "USD", "Old Co")
        new_id = sm.add_instrument("EQUITY", "XNAS", "USD", "New Co")
        sm.add_symbol_mapping(old_id, "polygon", "TICK", "2020-01-01T00:00:00+00:00")
        sm.add_symbol_mapping(new_id, "polygon", "TICK", "2023-01-01T00:00:00+00:00")

        resolved_past = sm.resolve("polygon", "TICK", "2021-06-01T00:00:00+00:00")
        assert resolved_past == old_id


class TestDeactivate:
    def test_deactivation_closes_mappings(self, sm):
        nid = sm.add_instrument("EQUITY", "XNAS", "USD", "Going Dark")
        sm.add_symbol_mapping(nid, "polygon", "DARK")
        sm.deactivate_instrument(nid, "2024-01-01")
        instr = sm.get_instrument(nid)
        assert instr["is_active"] == 0
        # Mapping should be closed
        with pytest.raises(SymbolNotFoundError):
            sm.resolve("polygon", "DARK")


class TestSearch:
    def test_name_search(self, sm):
        sm.add_instrument("EQUITY", "XNAS", "USD", "Apple Inc.")
        sm.add_instrument("EQUITY", "XNAS", "USD", "Applebees Corp")
        results = sm.search_instruments("Apple")
        assert len(results) == 2

    def test_statistics_accuracy(self, sm):
        sm.add_instrument("EQUITY", "XNAS", "USD", "Corp A")
        sm.add_instrument("ETF", "XNYS", "USD", "ETF B")
        stats = sm.get_statistics()
        assert stats["total_instruments"] == 2
        assert stats["by_type"]["EQUITY"] == 1
        assert stats["by_exchange"]["XNAS"] == 1


class TestUpdateInstrument:
    def test_update_allowed_field(self, sm):
        nid = sm.add_instrument("EQUITY", "XNAS", "USD", "Corp")
        sm.update_instrument(nid, name="Updated Corp")
        assert sm.get_instrument(nid)["name"] == "Updated Corp"

    def test_update_protected_field_raises(self, sm):
        nid = sm.add_instrument("EQUITY", "XNAS", "USD", "Corp")
        with pytest.raises(InvalidFieldError):
            sm.update_instrument(nid, created_at="hacked")
