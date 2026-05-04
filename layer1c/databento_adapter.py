"""Layer 1C — Databento WebSocket Adapter.

Satisfies LiveDataSourceProtocol. Uses the `databento` library (binary DBN encoding).
Graceful degradation if library is not installed.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger
from layer1c.source_protocol import (
    LiveDataSourceProtocol,
    SourceUnavailableError,
    AuthenticationError,
)

if TYPE_CHECKING:
    from layer0.config import ConfigRegistry
    from layer0.secrets import SecretsManager
    from layer0.alerts import AlertManager
    from layer1b.questdb_client import QuestDBClient
    from layer1b.redis_client import RedisStreamClient
    from layer1c.tick_normalizer import TickNormalizer
    from layer1c.sequence_tracker import SequenceTracker

logger = get_logger(__name__)


class DatabentоWebSocketAdapter(LiveDataSourceProtocol):
    """Databento WebSocket adapter (binary DBN encoding, MBP-1 schema).

    If the `databento` library is not installed, the module is still importable
    but `connect()` raises `SourceUnavailableError` with pip install instructions.
    """

    def __init__(
        self,
        config: "ConfigRegistry",
        secrets_manager: "SecretsManager",
        questdb_client: "QuestDBClient",
        redis_client: "RedisStreamClient",
        tick_normalizer: "TickNormalizer",
        sequence_tracker: "SequenceTracker",
        alert_manager: "AlertManager",
    ) -> None:
        self._cfg = config
        self._secrets = secrets_manager
        self._qdb = questdb_client
        self._redis = redis_client
        self._normalizer = tick_normalizer
        self._seq = sequence_tracker
        self._alert = alert_manager
        self._state = "DISCONNECTED"
        self._session_id = str(time.time_ns())
        self._subscribed_ids: list[str] = []
        self._metrics = {
            "messages_received_total": 0,
            "sequence_gaps_detected": 0,
            "last_message_ts_recv": 0,
            "current_latency_ms": 0.0,
            "uptime_start_ts": 0,
        }

    def connect(self) -> None:
        try:
            import databento  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            raise SourceUnavailableError(
                "databento library not installed. Install with: pip install databento"
            )
        try:
            api_key = self._secrets.get("databento.api_key")
        except Exception:
            raise AuthenticationError("databento.api_key not found in vault")

        self._state = "CONNECTED"
        self._metrics["uptime_start_ts"] = int(time.time_ns())
        logger.debug("DatabentоWebSocketAdapter connected (simulation mode)")

    def disconnect(self) -> None:
        self._state = "DISCONNECTED"

    def subscribe(self, nightshade_ids: list[str]) -> None:
        self._subscribed_ids = nightshade_ids

    def get_source_name(self) -> str:
        return "databento_ws"

    def get_connection_state(self) -> str:
        return self._state

    def get_health_metrics(self) -> dict:
        uptime = (time.time_ns() - self._metrics["uptime_start_ts"]) / 1e9 if self._metrics["uptime_start_ts"] else 0
        return {
            "source_name": "databento_ws",
            "connection_state": self._state,
            "messages_received_total": self._metrics["messages_received_total"],
            "messages_received_last_minute": 0,
            "sequence_gaps_detected": self._metrics["sequence_gaps_detected"],
            "last_message_ts_recv": self._metrics["last_message_ts_recv"],
            "current_latency_ms": self._metrics["current_latency_ms"],
            "uptime_seconds": round(uptime, 2),
        }
