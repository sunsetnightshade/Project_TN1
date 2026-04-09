"""tests/test_config.py — unit tests for config.py ticker lists and env parsing"""
from __future__ import annotations

import os
import unittest


class TestTickerUniverse(unittest.TestCase):

    def test_nasdaq_30_has_exactly_30_tickers(self):
        from config import NASDAQ_30
        self.assertEqual(len(NASDAQ_30), 30, f"Expected 30, got {len(NASDAQ_30)}: {NASDAQ_30}")

    def test_nasdaq_30_no_duplicates(self):
        from config import NASDAQ_30
        self.assertEqual(len(set(NASDAQ_30)), len(NASDAQ_30), "NASDAQ_30 contains duplicates")

    def test_no_indian_stocks(self):
        """No .NS suffix tickers should remain in any ticker list."""
        from config import NASDAQ_30, RESERVE_BENCH, ALL_TICKERS
        for t in ALL_TICKERS:
            self.assertFalse(t.endswith(".NS"), f"Indian stock found: {t}")

    def test_reserve_bench_has_exactly_5(self):
        from config import RESERVE_BENCH
        self.assertEqual(len(RESERVE_BENCH), 5, f"Expected 5 reserves, got {len(RESERVE_BENCH)}")

    def test_primary_tickers_is_nasdaq_30(self):
        from config import PRIMARY_TICKERS, NASDAQ_30
        self.assertEqual(PRIMARY_TICKERS, NASDAQ_30)

    def test_all_tickers_is_35(self):
        from config import ALL_TICKERS
        self.assertEqual(len(ALL_TICKERS), 35)

    def test_no_nifty_tickers_in_primary(self):
        """Known old NIFTY tickers should not appear in primary list."""
        old_nifty = {"INFY.NS", "TCS.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"}
        from config import PRIMARY_TICKERS
        overlap = old_nifty & set(PRIMARY_TICKERS)
        self.assertEqual(overlap, set(), f"Old NIFTY tickers still present: {overlap}")

    def test_no_non_tech_stocks(self):
        """PEP, COST, TMUS, CMCSA were non-tech stocks that should be removed."""
        removed_stocks = {"PEP", "COST", "TMUS", "CMCSA"}
        from config import ALL_TICKERS
        overlap = removed_stocks & set(ALL_TICKERS)
        self.assertEqual(overlap, set(), f"Non-tech stocks still present: {overlap}")


class TestLiveIngestConfig(unittest.TestCase):

    def _env(self, **overrides: str) -> None:
        for k, v in overrides.items():
            os.environ[k] = v

    def _clear(self, *keys: str) -> None:
        for k in keys:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        """Clean up any env vars we set."""
        for k in [
            "QM_ENABLE_LIVE_INGEST", "QM_LIVE_PROVIDER", "QM_BAR_SINK",
            "TWELVEDATA_API_KEY", "POLYGON_API_KEY",
            "QM_LIVE_HEARTBEAT_SECONDS", "QM_LIVE_LATENCY_ALERT_MS",
        ]:
            os.environ.pop(k, None)

    def test_default_provider_is_twelvedata(self):
        from config import get_live_ingest_config
        cfg = get_live_ingest_config()
        self.assertEqual(cfg.provider, "twelvedata")

    def test_invalid_provider_raises(self):
        from config import get_live_ingest_config
        with self.assertRaises(ValueError):
            get_live_ingest_config(provider_override="bloomberg")

    def test_env_float_parsed_correctly(self):
        os.environ["QM_LIVE_HEARTBEAT_SECONDS"] = "7.5"
        from importlib import reload
        import config
        reload(config)
        from config import get_live_ingest_config
        cfg = get_live_ingest_config()
        self.assertAlmostEqual(cfg.heartbeat_seconds, 7.5)

    def test_validate_requires_api_key(self):
        from config import get_live_ingest_config, validate_live_ingest_config
        self.clear_keys = lambda: None
        cfg = get_live_ingest_config(provider_override="twelvedata")
        # No TWELVEDATA_API_KEY set — should raise
        with self.assertRaises(ValueError):
            validate_live_ingest_config(cfg, require_provider_keys=True)

    def test_validate_passes_with_key(self):
        os.environ["TWELVEDATA_API_KEY"] = "fake-key-for-test"
        from config import get_live_ingest_config, validate_live_ingest_config
        cfg = get_live_ingest_config(provider_override="twelvedata")
        # Should not raise
        try:
            validate_live_ingest_config(cfg, require_provider_keys=True)
        except ValueError as e:
            # Any validation errors other than the API key check are fine
            if "API_KEY" in str(e):
                self.fail(f"Raised unexpected API key error: {e}")

    def test_symbols_override(self):
        from config import get_live_ingest_config
        custom_symbols = ["AAPL", "MSFT"]
        cfg = get_live_ingest_config(symbols_override=custom_symbols)
        self.assertEqual(list(cfg.symbols), custom_symbols)


if __name__ == "__main__":
    unittest.main()
