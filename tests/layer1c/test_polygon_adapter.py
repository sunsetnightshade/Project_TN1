"""Tests for the Polygon WebSocket Adapter."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from layer1c.polygon_adapter import PolygonWebSocketAdapter
from layer1c.source_protocol import AuthenticationError

@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.get.side_effect = lambda key, default=None: {
        "live_data.polygon.ws_url": "wss://socket.polygon.io/stocks",
        "live_data.polygon.write_buffer_size": "2",
        "live_data.polygon.write_buffer_timeout_ms": "500",
        "live_data.polygon.ws_reconnect_critical_after_attempts": "3",
        "live_data.polygon.ws_reconnect_initial_delay_seconds": "1",
        "live_data.polygon.ws_reconnect_max_delay_seconds": "2"
    }.get(key, default)
    return cfg

@pytest.fixture
def mock_secrets():
    secrets = MagicMock()
    secrets.get.return_value = "TEST_POLYGON_KEY"
    return secrets

@pytest.fixture
def mock_qdb():
    qdb = MagicMock()
    qdb.write_ticks_batch.return_value = 1
    return qdb

@pytest.fixture
def mock_redis():
    redis = MagicMock()
    return redis

@pytest.fixture
def mock_dq():
    dq = MagicMock()
    # Always return a valid score > 0 for tests unless specified
    dq.score.return_value = (None, 5.0)
    return dq

@pytest.fixture
def mock_gaps():
    return MagicMock()

@pytest.fixture
def mock_normalizer():
    normalizer = MagicMock()
    normalizer.normalize.return_value = {
        "nightshade_id": "TEST_ID",
        "price": 100.0,
        "size": 100,
        "conditions": 0,
        "exchange_id": 1,
    }
    return normalizer

@pytest.fixture
def mock_seq():
    return MagicMock()

@pytest.fixture
def mock_alert():
    return MagicMock()

@pytest.fixture
def adapter(mock_config, mock_secrets, mock_qdb, mock_redis, mock_dq, mock_gaps, mock_normalizer, mock_seq, mock_alert):
    return PolygonWebSocketAdapter(
        config=mock_config,
        secrets_manager=mock_secrets,
        questdb_client=mock_qdb,
        redis_client=mock_redis,
        data_quality_scorer=mock_dq,
        gap_tracker=mock_gaps,
        tick_normalizer=mock_normalizer,
        sequence_tracker=mock_seq,
        alert_manager=mock_alert,
    )

@pytest.mark.asyncio
async def test_polygon_adapter_auth_failure(adapter, mock_secrets):
    mock_secrets.get.side_effect = Exception("Not found")
    with pytest.raises(AuthenticationError):
        await adapter._connect()

@pytest.mark.asyncio
async def test_polygon_adapter_connect_success(adapter):
    mock_ws = AsyncMock()
    mock_ws.__aiter__.return_value = []

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_ws
    mock_ctx.__aexit__.return_value = None

    with patch("websockets.connect", return_value=mock_ctx):
        await adapter._connect()
        mock_ws.send.assert_called_once_with('{"action": "auth", "params": "TEST_POLYGON_KEY"}')
        assert adapter.get_connection_state() == "CONNECTED"

@pytest.mark.asyncio
async def test_polygon_handle_event_auth_success(adapter):
    adapter.subscribe(["AAPL"])
    with patch("websockets.connect") as mock_ws_mod:
        await adapter._handle_event({"ev": "auth_success"}, time.time_ns())
    # State doesn't change here, but no exception should be raised.

@pytest.mark.asyncio
async def test_polygon_handle_event_auth_failed(adapter):
    with pytest.raises(AuthenticationError, match="Polygon authentication failed"):
        await adapter._handle_event({"ev": "auth_failed"}, time.time_ns())

@pytest.mark.asyncio
async def test_polygon_handle_event_tick(adapter, mock_normalizer, mock_redis, mock_qdb, mock_dq, mock_seq):
    event = {"ev": "T", "sym": "AAPL", "p": 150.0, "s": 100, "seq": 100}
    ts_recv = time.time_ns()
    
    # Send first tick
    await adapter._handle_event(event, ts_recv)
    
    mock_normalizer.normalize.assert_called_once()
    mock_redis.write_tick_to_stream.assert_called_once()
    assert adapter._metrics["messages_received_total"] == 1
    mock_seq.record_message.assert_called_once_with("polygon_ws", adapter._session_id, 100, ts_recv)
    
    # Buffer size is 2, so QDB write should not be called yet
    mock_qdb.write_ticks_batch.assert_not_called()
    
    # Send second tick to trigger flush
    await adapter._handle_event(event, ts_recv)
    mock_qdb.write_ticks_batch.assert_called_once()

@patch("layer1c.polygon_adapter.asyncio.sleep", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_polygon_connect_with_retry(mock_sleep, adapter, mock_alert):
    # Mock _connect to raise an exception 3 times, which matches ws_reconnect_critical_after_attempts = 3
    adapter._connect = AsyncMock()
    adapter._connect.side_effect = [Exception("Drop 1"), Exception("Drop 2"), Exception("Drop 3"), asyncio.CancelledError()]
    adapter._running = True
    
    try:
        await adapter._connect_with_retry()
    except asyncio.CancelledError:
        pass
        
    assert adapter._metrics["reconnections"] == 3
    mock_alert.send_critical.assert_called_once()
