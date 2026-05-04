"""Layer 1B — Silver → Gold Feature Computer.

Computes log returns, rolling statistics, and VPIN from daily bars.
Runs after market close on completed bars only. Zero look-ahead bias.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer1b.questdb_client import QuestDBClient

logger = get_logger(__name__)

_STATE_DDL = """
CREATE TABLE IF NOT EXISTS feature_checkpoints (
    nightshade_id   TEXT NOT NULL,
    feature_name    TEXT NOT NULL,
    last_feature_ts TEXT,
    last_run_at     TEXT,
    PRIMARY KEY (nightshade_id, feature_name)
);
"""


class SilverToGoldFeatureComputer:
    """Computes Gold-tier features from completed Silver bars."""

    def __init__(
        self,
        questdb_client: "QuestDBClient",
        checkpoint_db_path: Optional[str] = None,
        min_history_days: int = 252,
    ) -> None:
        self._qdb = questdb_client
        self._min_history = min_history_days
        path = Path(checkpoint_db_path or "~/.nightshade/feature_state.db").expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._cp = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._cp.execute(_STATE_DDL)

    def run_features(self, nightshade_ids: list[str], as_of_date: str) -> int:
        """Compute all features for all instruments as of *as_of_date*. Returns total written."""
        total = 0
        for nid in nightshade_ids:
            total += self._compute_for_instrument(nid, as_of_date)
        return total

    def get_feature_coverage(self, nightshade_id: str, as_of_date: str) -> dict:
        """Return map of feature_name → available+valid."""
        try:
            rows = self._qdb.query(
                "SELECT feature_name, is_valid FROM features_gold WHERE nightshade_id=? AND ts_feature<=? ORDER BY ts_feature DESC",
                (nightshade_id, as_of_date),
            )
            seen: dict[str, bool] = {}
            for r in rows:
                if r["feature_name"] not in seen:
                    seen[r["feature_name"]] = bool(r["is_valid"])
            return seen
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Internal feature computation
    # ------------------------------------------------------------------

    def _compute_for_instrument(self, nightshade_id: str, as_of_date: str) -> int:
        try:
            bars = self._qdb.query_bars(nightshade_id, "1d", "2020-01-01", as_of_date)
        except Exception as exc:
            logger.error("Feature compute: failed to query bars for %s: %s", nightshade_id, exc)
            return 0

        if not bars:
            return 0

        closes = [b["close_fixed"] for b in bars]
        volumes = [b.get("volume", 0) for b in bars]
        ts_ns = int(datetime.strptime(as_of_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1e9)
        has_enough = len(closes) >= self._min_history

        written = 0

        def write_feature(name: str, value: float, lookback: int, is_valid: bool) -> None:
            nonlocal written
            safe_value = 0.0
            safe_valid = is_valid
            if math.isnan(value) or math.isinf(value):
                logger.warning("Feature %s/%s is NaN/inf — storing 0.0 invalid", nightshade_id, name)
                safe_value = 0.0
                safe_valid = False
            else:
                safe_value = value
            try:
                tags = f"nightshade_id={nightshade_id},feature_name={name}"
                fields = f"feature_value={safe_value},lookback_days={lookback}i,is_valid={'t' if safe_valid else 'f'}"
                ilp = f"features_gold,{tags} {fields} {ts_ns}\n"
                self._qdb._ilp_write(ilp)
                written += 1
            except Exception as exc:
                logger.error("Feature write failed for %s/%s: %s", nightshade_id, name, exc)

        # Log returns
        for lag, name in [(1, "log_return_1d"), (5, "log_return_5d"), (21, "log_return_21d")]:
            if len(closes) > lag and closes[-lag - 1] > 0 and closes[-1] > 0:
                val = math.log(closes[-1] / closes[-lag - 1])
                write_feature(name, val, lag, True)
            else:
                write_feature(name, 0.0, lag, False)

        # Rolling stats (20d and 252d)
        for window, suffix in [(20, "20d"), (252, "252d")]:
            window_closes = closes[-window:] if len(closes) >= window else closes
            valid = len(window_closes) == window
            if window_closes:
                mean = sum(window_closes) / len(window_closes)
                std = math.sqrt(sum((c - mean) ** 2 for c in window_closes) / len(window_closes))
                write_feature(f"rolling_mean_close_{suffix}", mean, window, valid)
                write_feature(f"rolling_std_close_{suffix}", std, window, valid)
                zscore = (closes[-1] - mean) / std if std > 0 else 0.0
                write_feature(f"rolling_zscore_close_{suffix}", zscore, window, valid)
                # Realized volatility = annualized stddev of log returns
                if len(window_closes) >= 2:
                    log_returns = [math.log(window_closes[i] / window_closes[i - 1])
                                   for i in range(1, len(window_closes))
                                   if window_closes[i - 1] > 0]
                    if log_returns:
                        r_mean = sum(log_returns) / len(log_returns)
                        r_std = math.sqrt(sum((r - r_mean) ** 2 for r in log_returns) / len(log_returns))
                        realized_vol = r_std * math.sqrt(252)
                        write_feature(f"realized_volatility_{suffix}", realized_vol, window, valid)
            else:
                for feat in ["rolling_mean_close", "rolling_std_close", "rolling_zscore_close", "realized_volatility"]:
                    write_feature(f"{feat}_{suffix}", 0.0, window, False)

        # VPIN (simplified bucket approximation)
        for bucket_size, name in [(25, "vpin_bucket_25"), (50, "vpin_bucket_50")]:
            vpin = self._compute_vpin(volumes, bucket_size)
            write_feature(name, vpin, bucket_size, len(volumes) >= bucket_size * 2)

        return written

    @staticmethod
    def _compute_vpin(volumes: list, bucket_size: int) -> float:
        """Simplified VPIN: |buy_vol - sell_vol| / total_vol per bucket, averaged."""
        if not volumes or bucket_size <= 0:
            return 0.0
        # Approximate: odd periods = buy, even = sell (placeholder for real VPIN with flow decomposition)
        buckets = [volumes[i:i + bucket_size] for i in range(0, len(volumes), bucket_size)]
        if not buckets:
            return 0.0
        vpin_vals = []
        for bucket in buckets:
            if not bucket:
                continue
            total = sum(bucket)
            if total == 0:
                continue
            half = len(bucket) // 2
            buy_vol = sum(bucket[:half])
            sell_vol = sum(bucket[half:])
            vpin_vals.append(abs(buy_vol - sell_vol) / total)
        return sum(vpin_vals) / len(vpin_vals) if vpin_vals else 0.0
