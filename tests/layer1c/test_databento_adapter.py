"""Tests for the Databento WebSocket Adapter."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from layer1c.databento_adapter import DatabentоWebSocketAdapter
from layer1c.source_protocol import AuthenticationError, SourceUnavailableError

@pytest.fixture
def mock_config():
    cfg = MagicMock()
    return cfg

@pytest.fixture
def mock_secrets():
    secrets = MagicMock()
    secrets.get.return_value = "TEST_DB_KEY"
    return secrets

@pytest.fixture
def mock_qdb():
    return MagicMock()

@pytest.fixture
def mock_redis():
    return MagicMock()

@pytest.fixture
def mock_normalizer():
    return MagicMock()

@pytest.fixture
def mock_seq():
    return MagicMock()

@pytest.fixture
def mock_alert():
    return MagicMock()

@pytest.fixture
def adapter(mock_config, mock_secrets, mock_qdb, mock_redis, mock_normalizer, mock_seq, mock_alert):
    return DatabentоWebSocketAdapter(
        config=mock_config,
        secrets_manager=mock_secrets,
        questdb_client=mock_qdb,
        redis_client=mock_redis,
        tick_normalizer=mock_normalizer,
        sequence_tracker=mock_seq,
        alert_manager=mock_alert,
    )

def test_databento_connect_unavailable(adapter, monkeypatch):
    # Hide the databento module to trigger SourceUnavailableError
    import sys
    monkeypatch.setitem(sys.modules, "databento", None)
    
    with pytest.raises(SourceUnavailableError, match="databento library not installed"):
        adapter.connect()

def test_databento_connect_auth_failure(adapter, mock_secrets, monkeypatch):
    # Mock databento module presence
    import sys
    monkeypatch.setitem(sys.modules, "databento", MagicMock())
    
    mock_secrets.get.side_effect = Exception("Not found")
    
    with pytest.raises(AuthenticationError, match="databento.api_key not found"):
        adapter.connect()

def test_databento_connect_success(adapter, mock_secrets, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "databento", MagicMock())
    
    adapter.connect()
    
    assert adapter.get_connection_state() == "CONNECTED"
    metrics = adapter.get_health_metrics()
    assert metrics["connection_state"] == "CONNECTED"
    assert metrics["source_name"] == "databento_ws"
    
    adapter.disconnect()
    assert adapter.get_connection_state() == "DISCONNECTED"
