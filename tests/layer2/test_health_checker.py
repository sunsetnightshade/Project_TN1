"""Tests for Layer 2 Health Checker."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from layer2.health_checker import HealthChecker, HealthState

@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.get.return_value = 30 # interval seconds
    return cfg

def test_health_checker_initialization(mock_config):
    checker = HealthChecker(config=mock_config)
    assert checker.get_system_health_score() == 100

def test_health_checker_check_components(mock_config):
    checker = HealthChecker(config=mock_config)
    
    checker.register_check("component1", lambda: {"healthy": True, "score": 100})
    checker.register_check("component2", lambda: {"healthy": False, "score": 60})
    
    results = checker.run_all_checks_once()
    
    assert checker.get_component_state("component1") == HealthState.HEALTHY
    assert checker.get_component_state("component2") == HealthState.DEGRADED
    
    # 100 and 50 average = 75
    assert checker.get_system_health_score() == 75

def test_health_checker_stale_component(mock_config):
    checker = HealthChecker(config=mock_config)
    
    # Score 0 triggers FAILED
    checker.register_check("component1", lambda: {"healthy": False, "score": 0})
    
    checker.run_all_checks_once()
    
    assert checker.get_component_state("component1") == HealthState.FAILED
    
    # 0 system health
    assert checker.get_system_health_score() == 0

def test_health_checker_missing_component(mock_config):
    checker = HealthChecker(config=mock_config)
    
    # Function throws error
    def bad_check():
        raise Exception("Failed")
        
    checker.register_check("core", bad_check)
    
    checker.run_all_checks_once()
    
    assert checker.get_component_state("core") == HealthState.FAILED
    assert checker.get_system_health_score() == 0
