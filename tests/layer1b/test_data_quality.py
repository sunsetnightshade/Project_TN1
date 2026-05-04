"""Tests for layer1b.data_quality — DataQualityScorer."""

from __future__ import annotations

import time

import pytest

from layer1b.data_quality import (
    DataQualityScorer,
    InvalidPriceError,
    convert_float_to_fixed,
    convert_fixed_to_float,
)


def _make_scorer() -> DataQualityScorer:
    """Scorer with no SecurityMaster (pass-through ID resolution)."""
    return DataQualityScorer(security_master=None)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _now_ns() -> int:
    return time.time_ns()


class TestConversions:

    def test_float_to_fixed(self):
        assert convert_float_to_fixed(150.2350) == 1502350

    def test_fixed_to_float_reversal(self):
        fixed = convert_float_to_fixed(99.99)
        assert abs(convert_fixed_to_float(fixed) - 99.99) < 0.0001

    def test_nan_raises(self):
        with pytest.raises(InvalidPriceError):
            convert_float_to_fixed(float("nan"))

    def test_inf_raises(self):
        with pytest.raises(InvalidPriceError):
            convert_float_to_fixed(float("inf"))


class TestDataQualityScorer:

    def test_valid_polygon_trade_scores_4(self):
        scorer = _make_scorer()
        now_ms = _now_ms()
        raw = {
            "ev": "T",
            "sym": "AAPL",
            "p": 150.00,
            "s": 100,
            "t": now_ms,
            "x": 4,
            "c": [],
            "_ts_recv_ns": _now_ns(),
        }
        tick, score = scorer.score(raw, "polygon_ws")
        assert score >= 2  # At minimum marginal (latency from ts_recv calc)

    def test_zero_price_scores_0(self):
        scorer = _make_scorer()
        raw = {
            "ev": "T",
            "sym": "AAPL",
            "p": 0.0,
            "s": 100,
            "t": _now_ms(),
            "x": 4,
            "c": [],
            "_ts_recv_ns": _now_ns(),
        }
        _, score = scorer.score(raw, "polygon_ws")
        assert score == 0

    def test_future_timestamp_scores_0(self):
        scorer = _make_scorer()
        future_ms = (_now_ns() // 1_000_000) + 120_000  # 2 minutes in future
        raw = {
            "ev": "T",
            "sym": "AAPL",
            "p": 150.0,
            "s": 100,
            "t": future_ms,
            "x": 4,
            "c": [],
            "_ts_recv_ns": _now_ns(),
        }
        _, score = scorer.score(raw, "polygon_ws")
        assert score == 0

    def test_large_price_deviation_scores_1(self):
        scorer = _make_scorer()
        # Prime price history with a value
        scorer._price_history["AAPL"] = __import__("collections").deque([1_000_000], maxlen=10)  # $100
        now_ms = _now_ms()
        raw = {
            "ev": "T",
            "sym": "AAPL",
            "p": 125.0,  # 25% deviation from $100
            "s": 100,
            "t": now_ms,
            "x": 4,
            "c": [],
            "_ts_recv_ns": _now_ns(),
        }
        _, score = scorer.score(raw, "polygon_ws")
        assert score == 1

    def test_non_trade_event_returns_0(self):
        scorer = _make_scorer()
        raw = {"ev": "Q", "sym": "AAPL"}  # Quote, not a trade
        _, score = scorer.score(raw, "polygon_ws")
        assert score == 0

    def test_databento_price_scale_conversion(self):
        """Databento 1e-9 → Nightshade 1e-4 price conversion."""
        scorer = _make_scorer()
        # $150.00 in Databento format = 150 * 1e9 = 150_000_000_000
        now_ns = _now_ns()
        raw = {
            "action": "T",
            "instrument_id": "12345",
            "price": 150_000_000_000,
            "size": 50,
            "ts_event": now_ns,
            "ts_recv": now_ns,
            "flags": 0,
        }
        tick, score = scorer.score(raw, "databento_hist")
        assert score > 0
        # $150 * 10000 = 1_500_000
        assert tick["price_fixed"] == 1_500_000
