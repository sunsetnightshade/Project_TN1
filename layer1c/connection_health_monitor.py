"""Layer 1C — Connection Health Monitor.

Polls all registered adapters every 10 seconds and writes health scores
to Redis. Sends alert transitions on score changes.
"""

from __future__ import annotations

import json
import threading
import time
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer0.alerts import AlertManager
    from layer1b.redis_client import RedisStreamClient
    from layer1c.market_hours import MarketHoursManager

logger = get_logger(__name__)


class ConnectionHealthMonitor:
    """Polls adapters and computes 0–100 health scores, written to Redis."""

    _CHECK_INTERVAL_SEC = 10
    _REDIS_KEY_PREFIX = "nightshade:health:"
    _REDIS_KEY_TTL = 60

    def __init__(
        self,
        redis_client: Optional["RedisStreamClient"] = None,
        market_hours_manager: Optional["MarketHoursManager"] = None,
        alert_manager: Optional["AlertManager"] = None,
    ) -> None:
        self._redis = redis_client
        self._market_hours = market_hours_manager
        self._alert = alert_manager
        self._adapters: list = []
        self._previous_scores: dict[str, int] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register_adapter(self, adapter) -> None:
        self._adapters.append(adapter)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="conn-health-monitor"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        while self._running:
            for adapter in self._adapters:
                self._check_adapter(adapter)
            time.sleep(self._CHECK_INTERVAL_SEC)

    def _check_adapter(self, adapter) -> None:
        try:
            metrics = adapter.get_health_metrics()
            source = metrics.get("source_name", "unknown")
            score = self._compute_score(metrics)
            prev_score = self._previous_scores.get(source, 100)

            # Alert transitions
            if prev_score >= 50 and score < 50 and self._alert:
                self._alert.send_warning("ConnectionHealthMonitor", f"{source} health degraded: {score}/100")
            elif prev_score >= 20 and score < 20 and self._alert:
                self._alert.send_critical("ConnectionHealthMonitor", f"{source} health failed: {score}/100")
            elif prev_score < 50 and score >= 80 and self._alert:
                self._alert.send_info("ConnectionHealthMonitor", f"{source} health recovered: {score}/100")

            self._previous_scores[source] = score

            # Write to Redis
            if self._redis:
                payload = json.dumps({
                    "health_score": score,
                    "metrics": metrics,
                    "ts_written_utc": time.time(),
                })
                try:
                    self._redis._client.setex(
                        f"{self._REDIS_KEY_PREFIX}{source}",
                        self._REDIS_KEY_TTL,
                        payload,
                    )
                except Exception as exc:
                    logger.debug("Redis health write failed: %s", exc)

        except Exception as exc:
            logger.error("Health check error: %s", exc)

    @staticmethod
    def _compute_score(metrics: dict) -> int:
        score = 100
        if metrics.get("connection_state") != "CONNECTED":
            score -= 20
        msgs_last_min = metrics.get("messages_received_last_minute", 1)
        if msgs_last_min == 0:
            score -= 10
        gaps = metrics.get("sequence_gaps_detected", 0)
        score -= min(30, gaps * 10)
        latency = metrics.get("current_latency_ms", 0)
        if latency > 500:
            score -= 20
        elif latency > 200:
            score -= 10
        return max(0, score)
