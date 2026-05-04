"""Tests for layer1c.market_hours — MarketHoursManager."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from layer1c.market_hours import MarketHoursManager


@pytest.fixture()
def mhm() -> MarketHoursManager:
    """MarketHoursManager using default fallback schedule (no YAML required)."""
    m = MarketHoursManager.__new__(MarketHoursManager)
    import threading
    m._config_path = "nonexistent.yaml"
    m._lock = threading.Lock()
    m._schedule = MarketHoursManager._default_schedule()
    return m


def _ns(date_str: str, hour: int, minute: int, tz_name: str = "US/Eastern") -> int:
    import pytz
    tz = pytz.timezone(tz_name)
    dt = datetime(
        int(date_str[:4]), int(date_str[5:7]), int(date_str[8:]),
        hour, minute, tzinfo=tz
    ).astimezone(timezone.utc)
    return int(dt.timestamp() * 1e9)


class TestMarketHours:

    def test_regular_session_xnas(self, mhm):
        ts = _ns("2024-01-03", 10, 30)  # 10:30 ET Wednesday
        state = mhm.get_session_state("XNAS", ts)
        assert state == "REGULAR"

    def test_pre_market_xnas(self, mhm):
        ts = _ns("2024-01-03", 7, 0)  # 7:00 AM ET
        state = mhm.get_session_state("XNAS", ts)
        assert state == "PRE_MARKET"

    def test_post_market_xnas(self, mhm):
        ts = _ns("2024-01-03", 17, 0)  # 5:00 PM ET
        state = mhm.get_session_state("XNAS", ts)
        assert state == "POST_MARKET"

    def test_weekend_closed(self, mhm):
        # 2024-01-06 is a Saturday
        ts = _ns("2024-01-06", 12, 0)
        state = mhm.get_session_state("XNAS", ts)
        assert state == "CLOSED"

    def test_holiday_classification(self, mhm):
        # Add a holiday manually
        mhm._schedule["exchanges"]["XNAS"]["holidays_2024"].append("2024-01-03")
        ts = _ns("2024-01-03", 10, 30)
        state = mhm.get_session_state("XNAS", ts)
        assert state == "HOLIDAY"

    def test_nse_regular_session(self, mhm):
        ts = _ns("2024-01-03", 11, 0, "Asia/Kolkata")  # 11 AM IST
        state = mhm.get_session_state("XNSE", ts)
        assert state == "REGULAR"

    def test_nse_before_open_closed(self, mhm):
        ts = _ns("2024-01-03", 8, 0, "Asia/Kolkata")  # 8 AM IST
        state = mhm.get_session_state("XNSE", ts)
        assert state == "CLOSED"

    def test_is_regular_session_returns_bool(self, mhm):
        ts = _ns("2024-01-03", 10, 30)
        assert mhm.is_regular_session("XNAS", ts) is True

    def test_get_session_boundaries_returns_dict(self, mhm):
        bounds = mhm.get_session_boundaries("XNAS", "2024-01-03")
        assert isinstance(bounds, dict)
        assert "regular_open_utc_ns" in bounds
        assert bounds["regular_open_utc_ns"] is not None

    def test_get_next_market_open_skips_weekend(self, mhm):
        # Friday 4 PM ET → next open is Monday
        ts = _ns("2024-01-05", 16, 0)  # Friday
        next_open = mhm.get_next_market_open("XNAS", ts)
        assert next_open > 0
        next_dt = datetime.fromtimestamp(next_open / 1e9, tz=timezone.utc)
        assert next_dt.weekday() < 5  # Weekday
