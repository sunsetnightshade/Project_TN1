"""tests/test_cleaner.py — unit tests for data_cleaner.clean_and_replace_zombies"""
from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from data_cleaner import clean_and_replace_zombies


def _make_prices(
    n_days: int = 100,
    tickers: list[str] | None = None,
    reserve: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    tickers = tickers or [f"T{i}" for i in range(30)]
    reserve = reserve or [f"R{i}" for i in range(5)]
    all_tickers = tickers + reserve
    rng = pd.date_range(end=date.today(), periods=n_days, freq="B", tz="UTC")
    data = np.exp(np.random.randn(n_days, len(all_tickers)) * 0.02 + 5.0)
    return pd.DataFrame(data, index=rng, columns=all_tickers), tickers, reserve


class TestCleanAndReplaceZombies(unittest.TestCase):

    def test_clean_data_no_changes(self):
        """All tickers healthy — no zombies, no replacements."""
        df, primaries, reserves = _make_prices()
        result = clean_and_replace_zombies(
            df, primary_tickers=primaries, reserve_tickers=reserves
        )
        self.assertEqual(len(result.dropped_primaries), 0)
        self.assertEqual(len(result.replacements), 0)
        self.assertEqual(result.prices.shape[1], 30)

    def test_zombie_exactly_at_threshold_is_NOT_dropped(self):
        """Ticker with exactly 5% missing should NOT be flagged (> not >=)."""
        df, primaries, reserves = _make_prices(n_days=100)
        # Set exactly 5 out of 100 rows (5.0%) to NaN
        df.iloc[:5, 0] = np.nan
        result = clean_and_replace_zombies(
            df, primary_tickers=primaries, reserve_tickers=reserves
        )
        self.assertNotIn(primaries[0], result.dropped_primaries)

    def test_zombie_above_threshold_is_dropped_and_replaced(self):
        """Ticker with >5% missing should be dropped and replaced with reserve."""
        df, primaries, reserves = _make_prices(n_days=100)
        # Set 6 out of 100 rows (6.0%) to NaN
        df.iloc[:6, 0] = np.nan
        zombie = primaries[0]
        result = clean_and_replace_zombies(
            df, primary_tickers=primaries, reserve_tickers=reserves
        )
        self.assertIn(zombie, result.dropped_primaries)
        self.assertEqual(len(result.replacements), 1)
        # Matrix still has 30 columns
        self.assertEqual(result.prices.shape[1], 30)
        # The zombie column is gone, replaced by the reserve
        self.assertNotIn(zombie, result.prices.columns)

    def test_output_always_30_columns(self):
        """Even with multiple zombies, output must be exactly 30 columns."""
        df, primaries, reserves = _make_prices(n_days=100)
        # Make 3 tickers into zombies
        for i in range(3):
            df.iloc[:10, i] = np.nan  # 10% missing
        result = clean_and_replace_zombies(
            df, primary_tickers=primaries, reserve_tickers=reserves
        )
        self.assertEqual(result.prices.shape[1], 30)

    def test_no_nas_interpolated(self):
        """Short 1–2 day gaps should be interpolated away on US open days."""
        df, primaries, reserves = _make_prices(n_days=50)
        # Introduce a 2-day gap in one ticker
        df.iloc[20:22, 1] = np.nan
        result = clean_and_replace_zombies(
            df, primary_tickers=primaries, reserve_tickers=reserves,
            missing_frac_threshold=0.05,
        )
        # After interpolation, the 2-day gap should be filled
        col = result.prices.columns[1]
        self.assertFalse(result.prices[col].iloc[20:22].isna().any())

    def test_raises_on_reserve_exhaustion(self):
        """If there are more zombies than reserves, raise RuntimeError."""
        df, primaries, reserves = _make_prices(n_days=100)
        # Make 6 zombies but only 5 reserves
        for i in range(6):
            df.iloc[:20, i] = np.nan  # 20% missing
        with self.assertRaises(RuntimeError):
            clean_and_replace_zombies(
                df, primary_tickers=primaries, reserve_tickers=reserves
            )

    def test_raises_on_non_datetime_index(self):
        df = pd.DataFrame({"A": [1.0, 2.0]}, index=[0, 1])
        with self.assertRaises(TypeError):
            clean_and_replace_zombies(
                df, primary_tickers=["A"], reserve_tickers=[]
            )


if __name__ == "__main__":
    unittest.main()
