"""Layer 1B — Bronze → Silver Aggregator.

Aggregates raw ticks into OHLCV bars at configurable intervals.
Uses a checkpoint SQLite DB to avoid re-processing.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer1b.questdb_client import QuestDBClient

logger = get_logger(__name__)

_CHECKPOINT_DDL = """
CREATE TABLE IF NOT EXISTS aggregation_checkpoints (
    nightshade_id   TEXT NOT NULL,
    bar_interval    TEXT NOT NULL,
    last_aggregated_ts_ns INTEGER DEFAULT 0,
    last_run_at     TEXT,
    PRIMARY KEY (nightshade_id, bar_interval)
);
"""

_BAR_INTERVALS_NS = {
    "1m": 60 * 1_000_000_000,
    "5m": 5 * 60 * 1_000_000_000,
    "1d": 24 * 60 * 60 * 1_000_000_000,
}


class BronzeToSilverAggregator:
    """Idempotent Bronze → Silver OHLCV bar aggregator with checkpointing."""

    _DEFAULT_INTERVALS = ["1m", "5m", "1d"]

    def __init__(
        self,
        questdb_client: "QuestDBClient",
        checkpoint_db_path: Optional[str] = None,
    ) -> None:
        self._qdb = questdb_client
        path = Path(checkpoint_db_path or "~/.nightshade/aggregation_state.db").expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._cp_conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._cp_conn.execute(_CHECKPOINT_DDL)
        logger.debug("BronzeToSilverAggregator ready")

    def run_aggregation(
        self,
        nightshade_ids: list[str],
        bar_intervals: Optional[list[str]] = None,
    ) -> dict:
        intervals = bar_intervals or self._DEFAULT_INTERVALS
        results: dict[str, int] = {}
        for nid in nightshade_ids:
            for interval in intervals:
                count = self._aggregate_instrument_interval(nid, interval)
                results[f"{nid}:{interval}"] = count
        return results

    def _aggregate_instrument_interval(
        self, nightshade_id: str, bar_interval: str
    ) -> int:
        interval_ns = _BAR_INTERVALS_NS.get(bar_interval, 60 * 1_000_000_000)
        checkpoint = self._get_checkpoint(nightshade_id, bar_interval)

        try:
            ticks = self._qdb.query_ticks(
                nightshade_id=nightshade_id,
                start_ts_ns=checkpoint,
                end_ts_ns=int(datetime.now(timezone.utc).timestamp() * 1e9),
                min_quality_score=2,
            )
        except Exception as exc:
            logger.error("Failed to query ticks for %s: %s", nightshade_id, exc)
            return 0

        if not ticks:
            return 0

        bars = self._group_into_bars(ticks, interval_ns)
        written = 0
        for bar in bars:
            if bar.get("is_complete"):
                try:
                    self._write_bar(bar)
                    written += 1
                except Exception as exc:
                    logger.error("Failed to write bar: %s", exc)

        if ticks:
            last_ts = max(t["ts_event"] for t in ticks)
            self._update_checkpoint(nightshade_id, bar_interval, last_ts)
        return written

    def _group_into_bars(self, ticks: list[dict], interval_ns: int) -> list[dict]:
        if not ticks:
            return []

        bars: dict[int, dict] = {}
        for tick in ticks:
            ts = int(tick.get("ts_event", 0))
            bar_start = (ts // interval_ns) * interval_ns
            if bar_start not in bars:
                bars[bar_start] = {
                    "ts_bar_open": bar_start,
                    "ts_bar_close": bar_start + interval_ns,
                    "nightshade_id": tick["nightshade_id"],
                    "open_fixed": tick["price_fixed"],
                    "high_fixed": tick["price_fixed"],
                    "low_fixed": tick["price_fixed"],
                    "close_fixed": tick["price_fixed"],
                    "volume": 0,
                    "vwap_numerator": 0,  # sum(price * size)
                    "trade_count": 0,
                    "quality_scores": [],
                    "is_complete": True,
                }
            b = bars[bar_start]
            pf = int(tick["price_fixed"])
            sz = int(tick.get("size", 0))
            b["high_fixed"] = max(b["high_fixed"], pf)
            b["low_fixed"] = min(b["low_fixed"], pf)
            b["close_fixed"] = pf
            b["volume"] += sz
            b["vwap_numerator"] += pf * sz
            b["trade_count"] += 1
            b["quality_scores"].append(int(tick.get("data_quality_score", 2)))

        result = []
        for bar in bars.values():
            vol = bar["volume"]
            vwap = round(bar["vwap_numerator"] / vol) if vol > 0 else bar["close_fixed"]
            bar["vwap_fixed"] = vwap
            bar["data_quality_score"] = min(bar["quality_scores"]) if bar["quality_scores"] else 0
            bar["source_row_count"] = bar["trade_count"]
            del bar["vwap_numerator"]
            del bar["quality_scores"]
            result.append(bar)

        return sorted(result, key=lambda b: b["ts_bar_open"])

    def _write_bar(self, bar: dict) -> None:
        nid = bar["nightshade_id"]
        ts = bar["ts_bar_open"]
        tags = f"nightshade_id={nid},bar_interval=1m"
        fields = (
            f"ts_bar_close={bar['ts_bar_close']}i,"
            f"open_fixed={bar['open_fixed']}i,"
            f"high_fixed={bar['high_fixed']}i,"
            f"low_fixed={bar['low_fixed']}i,"
            f"close_fixed={bar['close_fixed']}i,"
            f"volume={bar['volume']}i,"
            f"vwap_fixed={bar['vwap_fixed']}i,"
            f"trade_count={bar['trade_count']}i,"
            f"data_quality_score={bar['data_quality_score']}i,"
            f"is_complete=t,"
            f"source_row_count={bar['source_row_count']}i"
        )
        ilp = f"bars_silver,{tags} {fields} {ts}"
        self._qdb._ilp_write(ilp + "\n")

    def run_integrity_check(
        self,
        nightshade_id: str,
        bar_interval: str,
        start_date: str,
        end_date: str,
    ) -> dict:
        try:
            bars = self._qdb.query_bars(nightshade_id, bar_interval, start_date, end_date)
        except Exception:
            return {}
        return {
            "expected_bar_count": None,  # Requires market-hours logic
            "actual_bar_count": len(bars),
            "missing_bars": [],
            "duplicate_bars": [],
        }

    def _get_checkpoint(self, nightshade_id: str, bar_interval: str) -> int:
        row = self._cp_conn.execute(
            "SELECT last_aggregated_ts_ns FROM aggregation_checkpoints WHERE nightshade_id=? AND bar_interval=?",
            (nightshade_id, bar_interval),
        ).fetchone()
        return row[0] if row else 0

    def _update_checkpoint(self, nightshade_id: str, bar_interval: str, ts_ns: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._cp_conn.execute(
            """INSERT INTO aggregation_checkpoints (nightshade_id, bar_interval, last_aggregated_ts_ns, last_run_at)
               VALUES (?,?,?,?)
               ON CONFLICT(nightshade_id, bar_interval) DO UPDATE SET last_aggregated_ts_ns=excluded.last_aggregated_ts_ns, last_run_at=excluded.last_run_at""",
            (nightshade_id, bar_interval, ts_ns, now),
        )
