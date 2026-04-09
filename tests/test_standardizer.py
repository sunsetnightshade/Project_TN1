"""tests/test_standardizer.py — unit tests for standardizer.py"""
from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from standardizer import (
    StandardizationResult,
    compute_correlation_outliers,
    standardize_and_plot_heatmap,
)


def _make_log_returns(n_days: int = 100, n_tickers: int = 30) -> pd.DataFrame:
    rng = pd.date_range(end=date.today(), periods=n_days, freq="B", tz="UTC")
    data = np.random.randn(n_days, n_tickers) * 0.01
    return pd.DataFrame(data, index=rng, columns=[f"T{i}" for i in range(n_tickers)])


class TestStandardize(unittest.TestCase):

    def test_returns_standardization_result(self):
        r = _make_log_returns()
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            result = standardize_and_plot_heatmap(r, heatmap_path=pathlib.Path(tmp) / "h.png")
        self.assertIsInstance(result, StandardizationResult)

    def test_output_shape_matches_input(self):
        r = _make_log_returns(n_days=50, n_tickers=30)
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            result = standardize_and_plot_heatmap(r, heatmap_path=pathlib.Path(tmp) / "h.png")
        self.assertEqual(result.standardized.shape, r.shape)

    def test_column_means_near_zero(self):
        """StandardScaler should produce zero-mean per column."""
        r = _make_log_returns(n_days=200)
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            result = standardize_and_plot_heatmap(r, heatmap_path=pathlib.Path(tmp) / "h.png")
        means = result.standardized.mean(axis=0).abs()
        self.assertTrue((means < 1e-10).all(), f"Column means not near zero: {means.max():.2e}")

    def test_column_stds_near_one(self):
        """StandardScaler should produce unit standard deviation per column."""
        r = _make_log_returns(n_days=200)
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            result = standardize_and_plot_heatmap(r, heatmap_path=pathlib.Path(tmp) / "h.png")
        stds = result.standardized.std(axis=0, ddof=0)
        self.assertTrue((abs(stds - 1.0) < 1e-10).all(), f"Column stds not near 1: {stds.max():.4f}")

    def test_scaler_params_keys(self):
        """scaler_params must contain mean, scale, feature_names, correlation_outliers."""
        r = _make_log_returns()
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            result = standardize_and_plot_heatmap(r, heatmap_path=pathlib.Path(tmp) / "h.png")
        keys = result.scaler_params.keys()
        for expected in ("mean", "scale", "feature_names", "correlation_outliers"):
            self.assertIn(expected, keys)

    def test_both_heatmaps_created(self):
        """Both matrix_heatmap.png and correlation_heatmap.png should be created."""
        import tempfile, pathlib
        r = _make_log_returns(n_days=100)
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "matrix_heatmap.png"
            standardize_and_plot_heatmap(r, heatmap_path=p)
            self.assertTrue(p.exists(), "matrix_heatmap.png not created")
            corr_p = pathlib.Path(tmp) / "correlation_heatmap.png"
            self.assertTrue(corr_p.exists(), "correlation_heatmap.png not created")

    def test_raises_on_non_datetime_index(self):
        df = pd.DataFrame({"A": [1.0, 2.0]}, index=[0, 1])
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                standardize_and_plot_heatmap(df, heatmap_path=pathlib.Path(tmp) / "h.png")


class TestComputeCorrelationOutliers(unittest.TestCase):

    def test_perfectly_correlated_data_no_outliers(self):
        """All tickers perfectly correlated → no outliers."""
        rng = pd.date_range("2026-01-01", periods=50, freq="B", tz="UTC")
        base = np.random.randn(50)
        df = pd.DataFrame({f"T{i}": base for i in range(5)}, index=rng)
        corr = df.corr()
        result = compute_correlation_outliers(corr, low_threshold=0.3)
        self.assertEqual(result, [])

    def test_uncorrelated_data_has_outliers(self):
        """Independent random series should have many low-correlation pairs."""
        rng = pd.date_range("2026-01-01", periods=100, freq="B", tz="UTC")
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 5), index=rng, columns=[f"T{i}" for i in range(5)])
        corr = df.corr()
        result = compute_correlation_outliers(corr, low_threshold=0.9)
        # With random data, most pairs will be below 0.9
        self.assertGreater(len(result), 0)

    def test_outliers_sorted_ascending(self):
        """Outlier list must be sorted by correlation ascending (worst first)."""
        rng = pd.date_range("2026-01-01", periods=100, freq="B", tz="UTC")
        np.random.seed(77)
        df = pd.DataFrame(np.random.randn(100, 4), index=rng, columns=["A", "B", "C", "D"])
        corr = df.corr()
        result = compute_correlation_outliers(corr, low_threshold=1.0)  # all pairs
        if len(result) >= 2:
            for i in range(len(result) - 1):
                self.assertLessEqual(result[i]["correlation"], result[i + 1]["correlation"])


if __name__ == "__main__":
    unittest.main()
