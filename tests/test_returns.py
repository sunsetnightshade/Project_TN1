"""tests/test_returns.py — unit tests for matrix_math.build_aligned_log_return_matrix"""
from __future__ import annotations

import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from matrix_math import build_aligned_log_return_matrix


class TestBuildAlignedLogReturnMatrix(unittest.TestCase):

    def _make_prices(self, n_days: int = 10, n_tickers: int = 3) -> pd.DataFrame:
        rng = pd.date_range(end=date.today(), periods=n_days, freq="B", tz="UTC")
        data = np.exp(np.random.randn(n_days, n_tickers) * 0.02 + 5.0)  # synthetic prices ~148
        return pd.DataFrame(data, index=rng, columns=[f"T{i}" for i in range(n_tickers)])

    def test_output_has_datetime_index(self):
        prices = self._make_prices()
        result = build_aligned_log_return_matrix(prices)
        self.assertIsInstance(result.index, pd.DatetimeIndex)

    def test_output_shape_one_less_row(self):
        """Log returns drop first row (NaN from shift). dropna removes it."""
        n_days = 10
        prices = self._make_prices(n_days=n_days)
        result = build_aligned_log_return_matrix(prices)
        self.assertEqual(result.shape[0], n_days - 1)

    def test_same_columns_as_input(self):
        prices = self._make_prices(n_tickers=5)
        result = build_aligned_log_return_matrix(prices)
        self.assertEqual(list(result.columns), list(prices.columns))

    def test_no_nans_in_result(self):
        """dropna should eliminate all NaNs."""
        prices = self._make_prices()
        result = build_aligned_log_return_matrix(prices)
        self.assertFalse(result.isna().any().any())

    def test_log_returns_are_approximately_correct(self):
        """Manual check: ln(P[t]/P[t-1]) for a 2-day price series."""
        idx = pd.date_range("2026-01-01", periods=2, freq="B", tz="UTC")
        prices = pd.DataFrame({"A": [100.0, 110.0]}, index=idx)
        result = build_aligned_log_return_matrix(prices)
        expected = np.log(110.0 / 100.0)
        self.assertAlmostEqual(float(result["A"].iloc[0]), expected, places=8)

    def test_raises_on_non_datetime_index(self):
        prices = pd.DataFrame({"A": [1.0, 2.0]}, index=[0, 1])
        with self.assertRaises(TypeError):
            build_aligned_log_return_matrix(prices)

    def test_values_sorted_chronologically(self):
        """Output index must be sorted ascending."""
        prices = self._make_prices(n_days=20)
        result = build_aligned_log_return_matrix(prices)
        self.assertTrue(result.index.is_monotonic_increasing)

    def test_no_us_shift_applied(self):
        """Verify .shift(1) is NOT applied — all columns should have the same
        row count with no extra NaNs introduced from a per-column shift."""
        prices = self._make_prices(n_days=5, n_tickers=2)
        result = build_aligned_log_return_matrix(prices)
        # All columns should have the same non-NaN count (4 = 5 - 1 for log return row)
        counts = result.notna().sum()
        self.assertEqual(counts.nunique(), 1, "All ticker columns should have same row count")
        self.assertEqual(int(counts.iloc[0]), 4)


if __name__ == "__main__":
    unittest.main()
