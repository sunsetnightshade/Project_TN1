"""Layer 1B — Data Quality Scorer.

Normalizes raw messages from multiple sources and assigns a quality score 0–4.
Maintains internal price history for deviation detection.
"""

from __future__ import annotations

import collections
import math
import time
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer1a.security_master import SecurityMaster

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DataQualityError(Exception):
    """Base exception for data quality errors."""

class InvalidPriceError(DataQualityError):
    """Raised for NaN or infinite prices."""

class InvalidTimestampError(DataQualityError):
    """Raised for clearly invalid timestamps."""

class UnresolvableSymbolError(DataQualityError):
    """Raised when symbol cannot be resolved to a nightshade_id."""


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def convert_float_to_fixed(price_float: float) -> int:
    """Convert a float price to fixed-point integer (multiply by 10000)."""
    if math.isnan(price_float) or math.isinf(price_float):
        raise InvalidPriceError(f"Invalid price: {price_float}")
    return round(price_float * 10_000)


def convert_fixed_to_float(price_fixed: int) -> float:
    """Convert fixed-point integer back to float — for display only."""
    return price_fixed / 10_000.0


# ---------------------------------------------------------------------------
# Condition codes that affect scoring
# ---------------------------------------------------------------------------
_SUSPICIOUS_CONDITIONS = {12, 41, 52}  # odd lot, extended hours, out-of-sequence


# ---------------------------------------------------------------------------
# DataQualityScorer
# ---------------------------------------------------------------------------

class DataQualityScorer:
    """Normalizes messages from multiple sources and scores quality 0–4.

    Maintains per-instrument price history for deviation detection.
    """

    _PRICE_HISTORY_LEN = 10

    def __init__(self, security_master: Optional["SecurityMaster"] = None) -> None:
        self._sm = security_master
        # last N prices per nightshade_id
        self._price_history: dict[str, collections.deque] = {}
        # last ts_event_ns per nightshade_id
        self._last_ts: dict[str, int] = {}

    def score(self, raw_message: dict, source: str) -> tuple[dict, int]:
        """Normalize *raw_message* from *source* and return (tick_dict, score).

        Score 0 means rejected — callers must not store the tick.
        """
        try:
            tick = self._normalize(raw_message, source)
        except Exception as exc:
            logger.debug("Normalization failed for %s: %s", source, exc)
            return ({}, 0)

        if tick is None:
            return ({}, 0)

        q = self._compute_score(tick, source, raw_message)
        if q > 0:
            self._update_history(tick)
        return (tick, q)

    # ------------------------------------------------------------------
    # Normalization by source
    # ------------------------------------------------------------------

    def _normalize(self, raw: dict, source: str) -> Optional[dict]:
        if source == "polygon_ws":
            return self._normalize_polygon_ws(raw)
        elif source == "databento_hist":
            return self._normalize_databento_hist(raw)
        elif source == "yfinance_hist":
            return self._normalize_yfinance_hist(raw)
        else:
            logger.debug("Unknown source: %s", source)
            return None

    def _normalize_polygon_ws(self, raw: dict) -> Optional[dict]:
        if raw.get("ev") != "T":
            return None  # Not a trade event
        sym = raw.get("sym", "")
        nightshade_id = self._resolve(sym, "polygon_ws")
        price_fixed = convert_float_to_fixed(float(raw["p"]))
        ts_event_ns = int(raw["t"]) * 1_000_000  # ms → ns
        # Exchange ID → MIC
        exchange = str(raw.get("x", ""))
        # Conditions list → bitmask
        conds = raw.get("c", []) or []
        conditions_bitmask = 0
        for c in conds:
            conditions_bitmask |= (1 << int(c))
        return {
            "nightshade_id": nightshade_id,
            "source": "polygon_ws",
            "ts_event_ns": ts_event_ns,
            "ts_recv_ns": raw.get("_ts_recv_ns", int(time.time_ns())),
            "price_fixed": price_fixed,
            "size": int(raw.get("s", 0)),
            "exchange": exchange,
            "conditions_bitmask": conditions_bitmask,
            "instrument_type": "EQUITY",
        }

    def _normalize_databento_hist(self, raw: dict) -> Optional[dict]:
        if raw.get("action") != "T":
            return None
        instrument_id = str(raw.get("instrument_id", ""))
        nightshade_id = self._resolve(instrument_id, "databento")
        # Databento price is 1e-9 scale; Nightshade is 1e-4
        db_price = int(raw["price"])
        price_fixed = round(db_price / 100_000)  # 1e-9 → 1e-4
        return {
            "nightshade_id": nightshade_id,
            "source": "databento_hist",
            "ts_event_ns": int(raw["ts_event"]),
            "ts_recv_ns": int(raw.get("ts_recv", raw["ts_event"])),
            "price_fixed": price_fixed,
            "size": int(raw.get("size", 0)),
            "exchange": "XNAS",
            "conditions_bitmask": int(raw.get("flags", 0)),
            "instrument_type": "EQUITY",
        }

    def _normalize_yfinance_hist(self, raw: dict) -> Optional[dict]:
        ticker = raw.get("ticker", "")
        nightshade_id = self._resolve(ticker, "yfinance")
        close_price = float(raw.get("Close", raw.get("close", 0)))
        price_fixed = convert_float_to_fixed(close_price)
        # Market close 16:00 ET = 21:00 UTC
        from datetime import datetime, timezone
        date_str = str(raw.get("Date", raw.get("date", "")))[:10]
        try:
            dt = datetime.strptime(date_str + " 21:00:00", "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
            ts_event_ns = int(dt.timestamp() * 1e9)
        except Exception:
            ts_event_ns = int(time.time_ns())

        exchange = "XNSE" if ticker.endswith(".NS") else "XNAS"
        return {
            "nightshade_id": nightshade_id,
            "source": "yfinance_hist",
            "ts_event_ns": ts_event_ns,
            "ts_recv_ns": ts_event_ns,
            "price_fixed": price_fixed,
            "size": int(raw.get("Volume", 0)),
            "exchange": exchange,
            "conditions_bitmask": 0,
            "instrument_type": "EQUITY",
        }

    # ------------------------------------------------------------------
    # Quality scoring
    # ------------------------------------------------------------------

    def _compute_score(self, tick: dict, source: str, raw: dict) -> int:
        ts_event = tick["ts_event_ns"]
        ts_recv = tick["ts_recv_ns"]
        price = tick["price_fixed"]
        size = tick["size"]
        now_ns = int(time.time_ns())

        # Score 0 — Reject
        if not tick.get("nightshade_id"):
            return 0
        if price <= 0:
            return 0
        if size <= 0:
            return 0
        if ts_event > now_ns + 60 * 1_000_000_000:  # >60s in future
            return 0
        if ts_event < now_ns - 86400 * 1_000_000_000:  # >24h in past
            return 0

        lag_ns = ts_recv - ts_event
        conditions = tick.get("conditions_bitmask", 0)

        # Score 1 — Suspicious
        if self._price_deviation(tick) > 0.15:
            return 1
        if lag_ns > 10 * 1_000_000_000:  # >10s
            return 1
        if any(conditions & (1 << c) for c in _SUSPICIOUS_CONDITIONS):
            return 1

        # Score 2 — Marginal
        if lag_ns > 2 * 1_000_000_000:  # 2–10s
            return 2
        if size < 10:
            return 2
        if 0.05 < self._price_deviation(tick) <= 0.15:
            return 2

        # Score 3 — Good
        if lag_ns > 500 * 1_000_000:  # 500ms–2s
            return 3

        # Score 4 — Clean
        return 4

    def _price_deviation(self, tick: dict) -> float:
        nid = tick.get("nightshade_id", "")
        hist = self._price_history.get(nid)
        if not hist:
            return 0.0
        last_price = hist[-1]
        if last_price == 0:
            return 0.0
        return abs(tick["price_fixed"] - last_price) / last_price

    def _update_history(self, tick: dict) -> None:
        nid = tick["nightshade_id"]
        if nid not in self._price_history:
            self._price_history[nid] = collections.deque(maxlen=self._PRICE_HISTORY_LEN)
        self._price_history[nid].append(tick["price_fixed"])
        self._last_ts[nid] = tick["ts_event_ns"]

    def _resolve(self, external_id: str, source: str) -> str:
        if self._sm is None:
            return external_id  # Test mode: pass through
        try:
            return self._sm.resolve(source, external_id)
        except Exception as exc:
            raise UnresolvableSymbolError(f"Cannot resolve {source}/{external_id}: {exc}") from exc
