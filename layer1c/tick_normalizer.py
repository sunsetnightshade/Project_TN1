"""Layer 1C — Tick Normalizer.

Registry of normalization functions keyed by source name.
Explicit, testable normalization step separate from quality scoring.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, TYPE_CHECKING

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer1a.security_master import SecurityMaster

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Polygon exchange ID → MIC code map
# ---------------------------------------------------------------------------

_POLYGON_EXCHANGE_MAP = {
    1: "XNYS",   # NYSE
    2: "XNAS",   # NASDAQ
    3: "XASE",   # NYSE American (AMEX)
    4: "XNAS",   # NASDAQ (alternate)
    5: "XNAS",
    6: "XNYS",
    7: "XNAS",
    8: "XNAS",
    9: "XBOS",   # NASDAQ BX
    10: "XCIS",  # NYSE National
    11: "XISX",  # ISE
    12: "XEDG",  # EDGA
    13: "XEDX",  # EDGX
    14: "XCBO",  # CBOE
    15: "XNYS",
    16: "XNYS",
    17: "XNYS",
    18: "XNYS",
    19: "XNYS",
    20: "XNYS",
    21: "XNYS",
}


def get_polygon_exchange_map() -> dict[int, str]:
    """Return complete Polygon exchange ID → MIC code mapping."""
    return dict(_POLYGON_EXCHANGE_MAP)


def conditions_list_to_bitmask(conditions: Optional[list]) -> int:
    """Convert a list of condition codes to a bitmask.  Handles None/empty → 0."""
    if not conditions:
        return 0
    mask = 0
    for c in conditions:
        try:
            mask |= (1 << int(c))
        except (ValueError, TypeError):
            pass
    return mask


# ---------------------------------------------------------------------------
# TickNormalizer
# ---------------------------------------------------------------------------

class TickNormalizer:
    """Registry of normalization functions keyed by source name.

    Extension point: call register_normalizer() to add new sources.
    """

    def __init__(self, security_master: Optional["SecurityMaster"] = None) -> None:
        self._sm = security_master
        self._normalizers: dict[str, Callable[[dict], Optional[dict]]] = {}
        # Register built-in normalizers
        self.register_normalizer("polygon_ws", self._normalize_polygon_ws)
        self.register_normalizer("databento_ws", self._normalize_databento_ws)
        self.register_normalizer("yfinance_hist", self._normalize_yfinance_hist)

    def register_normalizer(self, source_name: str, normalizer_fn: Callable) -> None:
        self._normalizers[source_name] = normalizer_fn

    def normalize(self, raw_message: "RawMessageProtocol") -> Optional[dict]:
        """Normalize a raw message. Returns canonical tick dict or None.  Never raises."""
        from layer1c.source_protocol import RawMessageProtocol as _RM
        source = raw_message.source if isinstance(raw_message, _RM) else str(raw_message.get("source", ""))
        payload = raw_message.raw_payload if isinstance(raw_message, _RM) else raw_message

        fn = self._normalizers.get(source)
        if fn is None:
            logger.debug("No normalizer for source: %s", source)
            return None
        try:
            return fn(payload)
        except Exception as exc:
            logger.debug("Normalization failed for %s: %s", source, exc)
            return None

    # ------------------------------------------------------------------
    # Built-in normalizers
    # ------------------------------------------------------------------

    def _normalize_polygon_ws(self, raw: dict) -> Optional[dict]:
        if raw.get("ev") != "T":
            return None
        from layer1b.data_quality import convert_float_to_fixed
        sym = raw.get("sym", "")
        nightshade_id = self._resolve(sym, "polygon_ws")
        price_fixed = convert_float_to_fixed(float(raw["p"]))
        ts_event_ns = int(raw["t"]) * 1_000_000  # ms → ns
        exchange_id = int(raw.get("x", 0))
        exchange = _POLYGON_EXCHANGE_MAP.get(exchange_id, str(exchange_id))
        conditions_bitmask = conditions_list_to_bitmask(raw.get("c"))
        return {
            "nightshade_id": nightshade_id,
            "ts_event_ns": ts_event_ns,
            "ts_recv_ns": raw.get("_ts_recv_ns", int(time.time_ns())),
            "source": "polygon_ws",
            "price_fixed": price_fixed,
            "size": int(raw.get("s", 0)),
            "exchange": exchange,
            "conditions_bitmask": conditions_bitmask,
            "raw_sequence_number": raw.get("_seq"),
            "instrument_type": "EQUITY",
        }

    def _normalize_databento_ws(self, raw: dict) -> Optional[dict]:
        if raw.get("action") != "T":
            return None
        instrument_id = str(raw.get("instrument_id", ""))
        nightshade_id = self._resolve(instrument_id, "databento")
        # Databento 1e-9 scale → Nightshade 1e-4 (factor = 1e5)
        db_price = int(raw["price"])
        price_fixed = round(db_price / 100_000)
        return {
            "nightshade_id": nightshade_id,
            "ts_event_ns": int(raw["ts_event"]),
            "ts_recv_ns": int(raw.get("ts_recv", raw["ts_event"])),
            "source": "databento_ws",
            "price_fixed": price_fixed,
            "size": int(raw.get("size", 0)),
            "exchange": "XNAS",
            "conditions_bitmask": int(raw.get("flags", 0)),
            "raw_sequence_number": raw.get("sequence"),
            "instrument_type": "EQUITY",
        }

    def _normalize_yfinance_hist(self, raw: dict) -> Optional[dict]:
        from layer1b.data_quality import convert_float_to_fixed
        ticker = raw.get("ticker", "")
        nightshade_id = self._resolve(ticker, "yfinance")
        close_price = float(raw.get("Close", raw.get("close", 0)))
        price_fixed = convert_float_to_fixed(close_price)
        from datetime import datetime, timezone
        date_str = str(raw.get("Date", raw.get("date", "")))[:10]
        try:
            dt = datetime.strptime(date_str + " 21:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            ts_event_ns = int(dt.timestamp() * 1e9)
        except Exception:
            ts_event_ns = int(time.time_ns())
        exchange = "XNSE" if ticker.endswith(".NS") else "XNAS"
        return {
            "nightshade_id": nightshade_id,
            "ts_event_ns": ts_event_ns,
            "ts_recv_ns": ts_event_ns,
            "source": "yfinance_hist",
            "price_fixed": price_fixed,
            "size": int(raw.get("Volume", 0)),
            "exchange": exchange,
            "conditions_bitmask": 0,
            "raw_sequence_number": None,
            "instrument_type": "EQUITY",
        }

    def _resolve(self, external_id: str, source: str) -> str:
        if self._sm is None:
            return external_id  # Test mode: pass through
        try:
            return self._sm.resolve(source, external_id)
        except Exception:
            return external_id  # Best-effort: return raw id
