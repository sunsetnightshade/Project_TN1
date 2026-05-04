"""Tests for layer1c.tick_normalizer — TickNormalizer."""

from __future__ import annotations

import time
import pytest

from layer1c.tick_normalizer import TickNormalizer, conditions_list_to_bitmask, get_polygon_exchange_map
from layer1c.source_protocol import RawMessageProtocol


def _make_msg(source: str, payload: dict, ts_recv_ns: int) -> RawMessageProtocol:
    return RawMessageProtocol(source=source, raw_payload=payload, ts_recv_ns=ts_recv_ns)


class TestTickNormalizer:

    def test_polygon_trade_normalizes_correctly(self):
        norm = TickNormalizer(security_master=None)
        now_ns = time.time_ns()
        payload = {
            "ev": "T",
            "sym": "AAPL",
            "p": 150.00,
            "s": 100,
            "t": now_ns // 1_000_000,  # ms
            "x": 4,
            "c": [],
            "_ts_recv_ns": now_ns,
        }
        tick = norm.normalize(_make_msg("polygon_ws", payload, now_ns))
        assert tick is not None
        assert tick["nightshade_id"] == "AAPL"  # pass-through
        assert tick["price_fixed"] == 1_500_000
        assert tick["size"] == 100

    def test_polygon_non_trade_returns_none(self):
        norm = TickNormalizer()
        payload = {"ev": "Q", "sym": "AAPL"}
        tick = norm.normalize(_make_msg("polygon_ws", payload, time.time_ns()))
        assert tick is None

    def test_unknown_source_returns_none(self):
        norm = TickNormalizer()
        payload = {"something": "else"}
        tick = norm.normalize(_make_msg("unknown_source", payload, time.time_ns()))
        assert tick is None

    def test_custom_normalizer_registered(self):
        norm = TickNormalizer()
        custom_called = [False]

        def my_norm(raw: dict):
            custom_called[0] = True
            return {"nightshade_id": "CUSTOM", "price_fixed": 99}

        norm.register_normalizer("custom_source", my_norm)
        result = norm.normalize(_make_msg("custom_source", {}, time.time_ns()))
        assert result is not None
        assert result["nightshade_id"] == "CUSTOM"
        assert custom_called[0]

    def test_databento_price_scale(self):
        norm = TickNormalizer()
        now_ns = time.time_ns()
        payload = {
            "action": "T",
            "instrument_id": "99999",
            "price": 150_000_000_000,  # $150 in 1e-9
            "size": 25,
            "ts_event": now_ns,
            "ts_recv": now_ns,
            "flags": 0,
        }
        tick = norm.normalize(_make_msg("databento_ws", payload, now_ns))
        assert tick is not None
        assert tick["price_fixed"] == 1_500_000  # $150 * 10000

    def test_conditions_bitmask_conversion(self):
        assert conditions_list_to_bitmask([12]) == (1 << 12)
        assert conditions_list_to_bitmask([]) == 0
        assert conditions_list_to_bitmask(None) == 0
        assert conditions_list_to_bitmask([0, 1, 2]) == 7  # 1 + 2 + 4

    def test_polygon_exchange_map_completeness(self):
        mp = get_polygon_exchange_map()
        assert 1 in mp   # NYSE
        assert 2 in mp   # NASDAQ
        assert all(v.isupper() and len(v) == 4 for v in mp.values())
