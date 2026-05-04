"""Layer 1C — Ingestor Supervisor.

Top-level orchestrator for the live data layer. Initializes all components
in strict dependency order and manages lifecycle.
"""

from __future__ import annotations

import json
import signal
import threading
import time
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from layer0.config import ConfigRegistry
    from layer0.secrets import SecretsManager
    from layer0.alerts import AlertManager


class IngestorSupervisor:
    """Top-level orchestrator for Layers 1B + 1C.

    Strict initialization order per spec:
    SecurityMaster → QuestDBClient → RedisStreamClient → SchemaManager →
    DataQualityScorer → GapTracker → TickNormalizer → SequenceTracker →
    MarketHoursManager → PolygonWebSocketAdapter → DatabentоWebSocketAdapter (optional) →
    GapFillOrchestrator → ConnectionHealthMonitor
    """

    def __init__(
        self,
        config: "ConfigRegistry",
        secrets_manager: "SecretsManager",
        alert_manager: "AlertManager",
    ) -> None:
        self._cfg = config
        self._secrets = secrets_manager
        self._alert = alert_manager
        self._running = False
        self._status_thread: Optional[threading.Thread] = None

        logger.debug("IngestorSupervisor initializing...")
        self._init_components()

    def _init_components(self) -> None:
        from layer1a.security_master import SecurityMaster
        from layer1b.questdb_client import QuestDBClient
        from layer1b.redis_client import RedisStreamClient
        from layer1b.schema import SchemaManager
        from layer1b.data_quality import DataQualityScorer
        from layer1b.gap_tracker import GapTracker
        from layer1c.tick_normalizer import TickNormalizer
        from layer1c.sequence_tracker import SequenceTracker
        from layer1c.market_hours import MarketHoursManager
        from layer1c.polygon_adapter import PolygonWebSocketAdapter
        from layer1c.databento_adapter import DatabentоWebSocketAdapter
        from layer1c.gap_fill_orchestrator import GapFillOrchestrator
        from layer1c.connection_health_monitor import ConnectionHealthMonitor

        self.sm = SecurityMaster(self._cfg, self._alert)
        self.qdb = QuestDBClient(self._cfg, self._alert)
        self.redis = RedisStreamClient(self._cfg, self._alert)
        schema = SchemaManager(self.qdb)
        schema.create_all_tables()

        self.dq = DataQualityScorer(self.sm)
        self.gap_tracker = GapTracker(alert_manager=self._alert)
        self.normalizer = TickNormalizer(self.sm)
        self.seq_tracker = SequenceTracker()
        self.market_hours = MarketHoursManager()

        self.polygon = PolygonWebSocketAdapter(
            self._cfg, self._secrets, self.qdb, self.redis,
            self.dq, self.gap_tracker, self.normalizer, self.seq_tracker, self._alert,
        )
        # Databento optional
        self.databento = None
        try:
            self.databento = DatabentоWebSocketAdapter(
                self._cfg, self._secrets, self.qdb, self.redis,
                self.normalizer, self.seq_tracker, self._alert,
            )
        except Exception as exc:
            logger.info("Databento adapter skipped: %s", exc)

        self.gap_orchestrator = GapFillOrchestrator(self.gap_tracker, alert_manager=self._alert)
        self.health_monitor = ConnectionHealthMonitor(self.redis, self.market_hours, self._alert)
        self.health_monitor.register_adapter(self.polygon)

    def start(self) -> None:
        self._alert.send_info("IngestorSupervisor", "System startup")
        self._running = True

        # Subscribe adapters to current universe
        from layer1a.universe import UniverseManager
        um = UniverseManager(self.sm)
        current = um.get_current_universe("NIGHTSHADE_US_TECH") + um.get_current_universe("NIGHTSHADE_NIFTY_IT")
        self.polygon.subscribe(current)
        self.polygon.connect()

        # Start gap orchestrator
        try:
            key = self._secrets.get("polygon.api_key")
            self.gap_orchestrator.start(key)
        except Exception:
            pass

        # Start health monitor
        self.health_monitor.start()

        # Status writer background thread
        self._status_thread = threading.Thread(
            target=self._status_writer, daemon=True, name="supervisor-status"
        )
        self._status_thread.start()

        # Block on SIGINT/SIGTERM
        def _signal_handler(sig, frame):
            logger.info("IngestorSupervisor: shutdown signal received")
            self.stop()

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        logger.info("IngestorSupervisor: running. Press Ctrl+C to stop.")
        while self._running:
            time.sleep(1)

    def stop(self) -> None:
        self._running = False
        self.polygon.disconnect()
        self.gap_orchestrator.stop()
        self.health_monitor.stop()
        self.qdb.stop()
        self.redis.stop()
        self._alert.send_info("IngestorSupervisor", "System shutdown")

    def handle_universe_change(self, added_ids: list[str], removed_ids: list[str]) -> None:
        """Handle dynamic universe changes — subscribe added, don't unsubscribe removed."""
        if added_ids:
            self.polygon.subscribe(added_ids)

    def _status_writer(self) -> None:
        interval = int(self._cfg.get("supervisor.status_write_interval_seconds", 10))
        while self._running:
            try:
                status = {
                    "ts_utc": time.time(),
                    "adapters": {
                        "polygon_ws": {
                            "health_score": 100,
                            "connection_state": self.polygon.get_connection_state(),
                        }
                    },
                    "gaps_open": len(self.gap_tracker.get_open_gaps()),
                }
                import redis as redis_lib  # type: ignore[import-untyped]
                r = redis_lib.Redis()
                r.setex("nightshade:supervisor:status", 60, json.dumps(status))
            except Exception as exc:
                logger.debug("Status write failed: %s", exc)
            time.sleep(interval)
