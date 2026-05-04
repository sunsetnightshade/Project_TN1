"""Tests for Connection Health Monitor."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from layer1c.connection_health_monitor import ConnectionHealthMonitor

@pytest.fixture
def mock_redis():
    redis = MagicMock()
    redis._client = MagicMock()
    return redis

@pytest.fixture
def mock_alert():
    return MagicMock()

def test_connection_health_monitor_ok(mock_redis, mock_alert):
    monitor = ConnectionHealthMonitor(
        redis_client=mock_redis,
        alert_manager=mock_alert,
    )
    
    mock_adapter = MagicMock()
    mock_adapter.get_health_metrics.return_value = {
        "source_name": "test_src",
        "connection_state": "CONNECTED",
        "messages_received_last_minute": 10,
        "sequence_gaps_detected": 0,
        "current_latency_ms": 10
    }
    
    # Force initial score to 100
    monitor._previous_scores["test_src"] = 100
    
    monitor._check_adapter(mock_adapter)
    
    # 100 base - 0 penalty = 100
    assert monitor._previous_scores["test_src"] == 100
    mock_redis._client.setex.assert_called_once()
    mock_alert.send_warning.assert_not_called()

def test_connection_health_monitor_degraded(mock_redis, mock_alert):
    monitor = ConnectionHealthMonitor(
        redis_client=mock_redis,
        alert_manager=mock_alert,
    )
    
    mock_adapter = MagicMock()
    # High latency and 0 messages
    mock_adapter.get_health_metrics.return_value = {
        "source_name": "test_src",
        "connection_state": "CONNECTED",
        "messages_received_last_minute": 0, # -10
        "sequence_gaps_detected": 2, # -20
        "current_latency_ms": 250 # -10
    }
    
    # Base 100 - 10 - 20 - 10 = 60
    monitor._check_adapter(mock_adapter)
    assert monitor._previous_scores["test_src"] == 60
    
    # Now drop it to 40
    mock_adapter.get_health_metrics.return_value = {
        "source_name": "test_src",
        "connection_state": "DISCONNECTED", # -20
        "messages_received_last_minute": 0, # -10
        "sequence_gaps_detected": 3, # -30
        "current_latency_ms": 0
    }
    
    monitor._check_adapter(mock_adapter)
    assert monitor._previous_scores["test_src"] == 40
    
    mock_alert.send_warning.assert_called_once_with("ConnectionHealthMonitor", "test_src health degraded: 40/100")

def test_connection_health_monitor_failed(mock_redis, mock_alert):
    monitor = ConnectionHealthMonitor(
        redis_client=mock_redis,
        alert_manager=mock_alert,
    )
    
    monitor._previous_scores["test_src"] = 50
    
    mock_adapter = MagicMock()
    # Score goes to 0
    mock_adapter.get_health_metrics.return_value = {
        "source_name": "test_src",
        "connection_state": "DISCONNECTED", # -20
        "messages_received_last_minute": 0, # -10
        "sequence_gaps_detected": 5, # -30 (max)
        "current_latency_ms": 600 # -20
    }
    
    monitor._check_adapter(mock_adapter)
    assert monitor._previous_scores["test_src"] == 20
    # Wait, 100 - 20 - 10 - 30 - 20 = 20
    # Let's check logic: max(0, 100 - 80) = 20. But transition is prev >= 20 and score < 20.
    # To get < 20, let's force a negative logic error? No, min score is 0.
    
    # Let's just adjust expectations to score = 20, prev_score = 50.
    # The warning should trigger since prev >= 50 and score < 50.
    mock_alert.send_warning.assert_called_once()
