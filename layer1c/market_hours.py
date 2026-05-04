"""Layer 1C — Market Hours Manager.

Reads exchange_hours.yaml and provides session classification
(PRE_MARKET, REGULAR, POST_MARKET, CLOSED, HOLIDAY).
"""

from __future__ import annotations

import threading
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Optional

import yaml

from layer0.logging_config import get_logger

logger = get_logger(__name__)

_SESSION_PRE_MARKET = "PRE_MARKET"
_SESSION_REGULAR = "REGULAR"
_SESSION_POST_MARKET = "POST_MARKET"
_SESSION_CLOSED = "CLOSED"
_SESSION_HOLIDAY = "HOLIDAY"


class MarketHoursManager:
    """Exchange-aware session classifier.

    Reloads exchange_hours.yaml at midnight UTC daily.
    """

    def __init__(self, config_path: str | Path = "data/exchange_hours.yaml") -> None:
        self._config_path = Path(config_path)
        self._schedule: dict = {}
        self._lock = threading.Lock()
        self._reload()
        self._start_midnight_reloader()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_session_state(self, exchange_mic: str, ts_utc_ns: int) -> str:
        """Return session state for the given exchange at the given time."""
        with self._lock:
            return self._classify(exchange_mic, ts_utc_ns)

    def is_regular_session(self, exchange_mic: str, ts_utc_ns: int) -> bool:
        return self.get_session_state(exchange_mic, ts_utc_ns) == _SESSION_REGULAR

    def get_next_market_open(self, exchange_mic: str, ts_utc_ns: int) -> int:
        """Return the next regular open timestamp (ns UTC), skipping weekends + holidays."""
        from datetime import date, timedelta
        dt = datetime.fromtimestamp(ts_utc_ns / 1e9, tz=timezone.utc)
        check_date = dt.date()
        for _ in range(30):  # Max 30 calendar days
            check_date += timedelta(days=1)
            if check_date.weekday() >= 5:  # Weekend
                continue
            if self._is_holiday(exchange_mic, check_date.isoformat()):
                continue
            boundaries = self.get_session_boundaries(exchange_mic, check_date.isoformat())
            if boundaries and boundaries.get("regular_open_utc_ns"):
                return boundaries["regular_open_utc_ns"]
        return 0

    def get_session_boundaries(self, exchange_mic: str, date_str: str) -> Optional[dict]:
        """Return session boundary timestamps (ns UTC) for the given date."""
        with self._lock:
            return self._compute_boundaries(exchange_mic, date_str)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reload(self) -> None:
        try:
            with open(self._config_path, encoding="utf-8") as f:
                with self._lock:
                    self._schedule = yaml.safe_load(f) or {}
            logger.debug("MarketHoursManager: reloaded %s", self._config_path)
        except Exception as exc:
            logger.warning("MarketHoursManager: could not load %s: %s", self._config_path, exc)
            with self._lock:
                self._schedule = self._default_schedule()

    def _start_midnight_reloader(self) -> None:
        def loop() -> None:
            import time as _time
            while True:
                now = datetime.now(timezone.utc)
                # Seconds until next midnight UTC
                seconds_until_midnight = (
                    (23 - now.hour) * 3600
                    + (59 - now.minute) * 60
                    + (60 - now.second)
                )
                _time.sleep(seconds_until_midnight + 1)
                self._reload()

        t = threading.Thread(target=loop, daemon=True, name="market-hours-reloader")
        t.start()

    def _classify(self, exchange_mic: str, ts_utc_ns: int) -> str:
        dt = datetime.fromtimestamp(ts_utc_ns / 1e9, tz=timezone.utc)
        date_str = dt.date().isoformat()

        # Weekends
        if dt.weekday() >= 5:
            return _SESSION_CLOSED

        # Holidays
        if self._is_holiday(exchange_mic, date_str):
            return _SESSION_HOLIDAY

        boundaries = self._compute_boundaries(exchange_mic, date_str)
        if not boundaries:
            return _SESSION_CLOSED

        ts_s = ts_utc_ns / 1e9

        def ns_to_s(ns_val):
            return ns_val / 1e9 if ns_val else None

        pre_open = ns_to_s(boundaries.get("pre_market_open_utc_ns"))
        reg_open = ns_to_s(boundaries.get("regular_open_utc_ns"))
        reg_close = ns_to_s(boundaries.get("regular_close_utc_ns"))
        post_close = ns_to_s(boundaries.get("post_market_close_utc_ns"))

        if reg_open and reg_close and reg_open <= ts_s < reg_close:
            return _SESSION_REGULAR
        if pre_open and reg_open and pre_open <= ts_s < reg_open:
            return _SESSION_PRE_MARKET
        if reg_close and post_close and reg_close <= ts_s < post_close:
            return _SESSION_POST_MARKET

        return _SESSION_CLOSED

    def _is_holiday(self, exchange_mic: str, date_str: str) -> bool:
        exchanges = self._schedule.get("exchanges", {})
        exch = exchanges.get(exchange_mic, {})
        holidays = exch.get("holidays_2024", []) + exch.get("holidays_2025", [])
        return date_str in holidays

    def _compute_boundaries(self, exchange_mic: str, date_str: str) -> Optional[dict]:
        exchanges = self._schedule.get("exchanges", {})
        exch = exchanges.get(exchange_mic)
        if not exch:
            return None

        import pytz
        tz = pytz.timezone(exch["timezone"])
        date = datetime.strptime(date_str, "%Y-%m-%d").date()

        def to_utc_ns(local_time_str: Optional[str]) -> Optional[int]:
            if not local_time_str:
                return None
            try:
                h, m = map(int, local_time_str.split(":"))
                local_dt = datetime(date.year, date.month, date.day, h, m, tzinfo=tz)
                utc_dt = local_dt.astimezone(timezone.utc)
                return int(utc_dt.timestamp() * 1e9)
            except Exception:
                return None

        # Check for early close
        early_closes = exch.get("early_close_days", [])
        early_close_time = exch.get("early_close_time")
        reg_close = early_close_time if date_str in early_closes and early_close_time else exch.get("regular_close")

        return {
            "pre_market_open_utc_ns": to_utc_ns(exch.get("pre_market_open")),
            "regular_open_utc_ns": to_utc_ns(exch.get("regular_open")),
            "regular_close_utc_ns": to_utc_ns(reg_close),
            "post_market_close_utc_ns": to_utc_ns(exch.get("post_market_close")),
        }

    @staticmethod
    def _default_schedule() -> dict:
        """Hardcoded fallback schedule when YAML is unavailable."""
        return {
            "exchanges": {
                "XNAS": {
                    "timezone": "US/Eastern",
                    "regular_open": "09:30",
                    "regular_close": "16:00",
                    "pre_market_open": "04:00",
                    "post_market_close": "20:00",
                    "early_close_time": "13:00",
                    "early_close_days": [],
                    "holidays_2024": [],
                    "holidays_2025": [],
                },
                "XNYS": {
                    "timezone": "US/Eastern",
                    "regular_open": "09:30",
                    "regular_close": "16:00",
                    "pre_market_open": "04:00",
                    "post_market_close": "20:00",
                    "early_close_time": "13:00",
                    "early_close_days": [],
                    "holidays_2024": [],
                    "holidays_2025": [],
                },
                "XNSE": {
                    "timezone": "Asia/Kolkata",
                    "regular_open": "09:15",
                    "regular_close": "15:30",
                    "pre_market_open": None,
                    "post_market_close": None,
                    "holidays_2024": [],
                    "holidays_2025": [],
                },
            }
        }
