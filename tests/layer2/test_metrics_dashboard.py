"""Tests for Layer 2 Metrics Dashboard."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from layer2.metrics_dashboard import MetricsDashboard

@pytest.fixture
def mock_config():
    cfg = MagicMock()
    return cfg

def test_metrics_dashboard_color_score():
    from layer2.metrics_dashboard import _color_score, _GREEN, _YELLOW, _RED, _RESET
    assert _color_score(85) == f"{_GREEN}85{_RESET}"
    assert _color_score(60) == f"{_YELLOW}60{_RESET}"
    assert _color_score(40) == f"{_RED}40{_RESET}"

def test_metrics_dashboard_render(mock_config, capsys):
    dash = MetricsDashboard(config=mock_config)
    
    # Instead of full live render which loops, we just call _render once
    # We'll mock the redis json return to simulate data
    dash._redis = MagicMock()
    
    def redis_get(key):
        if key == "nightshade:supervisor:status":
            return {
                "adapters": {
                    "polygon_ws": {"connection_state": "CONNECTED", "health_score": 100}
                },
                "gaps_open": 2
            }
        elif key.startswith("nightshade:health:"):
            return {
                "health_score": 90,
                "metrics": {"connection_state": "CONNECTED", "current_latency_ms": 15.5}
            }
        return None
        
    dash._get_redis_json = MagicMock(side_effect=redis_get)
    dash._redis.scan_iter.return_value = ["nightshade:health:polygon_ws"]
    
    dash._render()
    
    captured = capsys.readouterr()
    output = captured.out
    
    assert "SUPERVISOR STATUS" in output
    assert "ADAPTER HEALTH" in output
    assert "GAP TRACKER" in output
    assert "THROUGHPUT" in output
    assert "polygon_ws" in output
    assert "CONNECTED" in output
