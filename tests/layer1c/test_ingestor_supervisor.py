"""Tests for Ingestor Supervisor."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, AsyncMock

import pytest
from layer1c.ingestor_supervisor import IngestorSupervisor
from layer1c.source_protocol import LiveDataSourceProtocol

class MockSource(LiveDataSourceProtocol):
    def __init__(self, name="mock_src"):
        self.name = name
        self.connected = False
        self.subscribed = False
        
    def connect(self) -> None:
        self.connected = True
        
    def disconnect(self) -> None:
        self.connected = False
        
    def subscribe(self, nightshade_ids: list[str]) -> None:
        self.subscribed = True
        
    def get_source_name(self) -> str:
        return self.name
        
    def get_connection_state(self) -> str:
        return "CONNECTED" if self.connected else "DISCONNECTED"
        
    def get_health_metrics(self) -> dict:
        return {"state": self.get_connection_state()}

@pytest.fixture
def supervisor():
    # Provide mocks for config, secrets, alert
    config = MagicMock()
    secrets = MagicMock()
    alert = MagicMock()
    
    # We bypass _init_components to avoid spinning up heavy classes
    with pytest.MonkeyPatch.context() as m:
        m.setattr(IngestorSupervisor, "_init_components", lambda self: None)
        sup = IngestorSupervisor(config, secrets, alert)
        
        # Manually attach mock attributes normally created in _init_components
        sup.polygon = MockSource("polygon")
        sup.gap_orchestrator = MagicMock()
        sup.health_monitor = MagicMock()
        sup.qdb = MagicMock()
        sup.redis = MagicMock()
        
        return sup

def test_supervisor_start(supervisor):
    # Mock universe manager to avoid DB lookup
    import sys
    sys.modules["layer1a.universe"] = MagicMock()
    
    supervisor.start = MagicMock()
    supervisor.start()
    supervisor.start.assert_called_once()

def test_supervisor_stop(supervisor):
    supervisor.stop()
    assert supervisor._running is False
    assert supervisor.polygon.connected is False
    supervisor.gap_orchestrator.stop.assert_called_once()
    supervisor.health_monitor.stop.assert_called_once()
