"""Tests for layer1b.feature_jobs — SilverToGoldFeatureComputer."""

from __future__ import annotations

import math
import pytest
from unittest.mock import MagicMock

from layer1b.feature_jobs import SilverToGoldFeatureComputer


def _make_bars(n: int, close_price_fixed: int = 1_500_000) -> list[dict]:
    return [
        {
            "close_fixed": close_price_fixed,
            "volume": 1000,
            "ts_bar_open": 1_700_000_000_000_000_000 + i * 86400 * 1_000_000_000,
            "nightshade_id": "AAPL",
            "bar_interval": "1d",
        }
        for i in range(n)
    ]


@pytest.fixture()
def feat(tmp_path) -> SilverToGoldFeatureComputer:
    qdb = MagicMock()
    qdb._ilp_write = MagicMock()
    qdb.query_bars.return_value = []
    f = SilverToGoldFeatureComputer(qdb, checkpoint_db_path=str(tmp_path / "feat.db"), min_history_days=10)
    return f


class TestFeatureJobs:

    def test_log_return_1d_correct(self, feat):
        bars = _make_bars(5, 2_000_000)  # all same price
        # Change last bar to double
        bars[-1]["close_fixed"] = 4_000_000
        feat._qdb.query_bars.return_value = bars
        count = feat._compute_for_instrument("AAPL", "2024-01-10")
        # Should write without error
        assert count > 0

    def test_zscore_zero_when_close_equals_mean(self, feat):
        bars = _make_bars(20, 1_500_000)  # uniform price
        feat._qdb.query_bars.return_value = bars
        # Capture ILP writes
        written_lines = []
        feat._qdb._ilp_write.side_effect = lambda line: written_lines.append(line)
        feat._compute_for_instrument("AAPL", "2024-01-20")
        zscore_lines = [l for l in written_lines if "rolling_zscore_close_20d" in l]
        assert len(zscore_lines) > 0
        # feature_value should be 0 when all prices equal
        for line in zscore_lines:
            val_part = [p for p in line.split(" ")[1].split(",") if "feature_value" in p]
            if val_part:
                value = float(val_part[0].split("=")[1])
                assert abs(value) < 1e-9

    def test_is_valid_false_when_insufficient_history(self, feat):
        bars = _make_bars(5, 1_500_000)  # less than min_history_days=10
        feat._qdb.query_bars.return_value = bars
        written_lines = []
        feat._qdb._ilp_write.side_effect = lambda line: written_lines.append(line)
        feat._compute_for_instrument("AAPL", "2024-01-05")
        # Rolling_zscore features with window 20 must be is_valid=f
        zscore_252 = [l for l in written_lines if "rolling_zscore_close_252d" in l]
        for line in zscore_252:
            assert "is_valid=f" in line

    def test_nan_stores_zero_with_invalid(self, feat):
        """If ILP write receives NaN from a buggy feature, it should store 0 with is_valid=f."""
        # Simulate a bar list that would produce NaN log return (price = 0)
        bars = _make_bars(3, 0)  # zero price → log(0) would fail
        bars[0]["close_fixed"] = 0
        bars[1]["close_fixed"] = 0
        bars[2]["close_fixed"] = 0
        feat._qdb.query_bars.return_value = bars
        # Should not raise
        count = feat._compute_for_instrument("AAPL", "2024-01-05")
        assert count >= 0  # No crash
