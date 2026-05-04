"""Layer 2 — Health Checker.

Periodic health checks with HTTP/PG ping, score computation, QuestDB write.
Reports HEALTHY/DEGRADED/FAILED states and manages alert transitions.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer0.alerts import AlertManager
    from layer2.metrics_emitter import MetricsEmitter

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class HealthState(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HealthRecord:
    component: str
    state: HealthState
    health_score: int
    consecutive_failures: int
    message: str
    ts_ns: int = field(default_factory=time.time_ns)
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HealthChecker
# ---------------------------------------------------------------------------

class HealthChecker:
    """Orchestrates periodic health checks, state machine, and alerting."""

    _CHECK_INTERVAL_SEC = 30

    def __init__(
        self,
        config=None,
        metrics_emitter: Optional["MetricsEmitter"] = None,
        alert_manager: Optional["AlertManager"] = None,
    ) -> None:
        self._metrics = metrics_emitter
        self._alert = alert_manager
        self._interval = int(
            config.get("layer2.health_check_interval_seconds", 30) if config else 30
        )
        self._checks: dict[str, Callable[[], dict]] = {}
        self._state: dict[str, HealthState] = {}
        self._consecutive_failures: dict[str, int] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register_check(self, component_name: str, check_fn: Callable[[], dict]) -> None:
        """Register a health check function.

        check_fn must return:
            {"healthy": bool, "score": int, "message": str, "extra": dict}
        """
        self._checks[component_name] = check_fn
        self._state[component_name] = HealthState.HEALTHY
        self._consecutive_failures[component_name] = 0

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._check_loop, daemon=True, name="health-checker"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def run_all_checks_once(self) -> dict[str, HealthRecord]:
        results = {}
        for component, check_fn in self._checks.items():
            record = self._run_single_check(component, check_fn)
            results[component] = record
        return results

    def get_component_state(self, component: str) -> Optional[HealthState]:
        return self._state.get(component)

    def get_system_health_score(self) -> int:
        """Return weighted average of all component scores (equal weights)."""
        if not self._checks:
            return 100
        scores = [
            100 if self._state.get(c) == HealthState.HEALTHY else
            (50 if self._state.get(c) == HealthState.DEGRADED else 0)
            for c in self._checks
        ]
        return round(sum(scores) / len(scores))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_loop(self) -> None:
        while self._running:
            for component, check_fn in self._checks.items():
                self._run_single_check(component, check_fn)
            time.sleep(self._interval)

    def _run_single_check(self, component: str, check_fn: Callable) -> HealthRecord:
        try:
            result = check_fn()
            is_healthy = bool(result.get("healthy", False))
            score = int(result.get("score", 100 if is_healthy else 0))
            message = str(result.get("message", ""))
            extra = result.get("extra", {})
        except Exception as exc:
            is_healthy = False
            score = 0
            message = f"check_fn raised: {exc}"
            extra = {}

        prev_state = self._state.get(component, HealthState.HEALTHY)
        failures = self._consecutive_failures.get(component, 0)

        if is_healthy:
            new_state = HealthState.HEALTHY
            failures = 0
        elif score >= 50:
            new_state = HealthState.DEGRADED
            failures += 1
        else:
            new_state = HealthState.FAILED
            failures += 1

        self._state[component] = new_state
        self._consecutive_failures[component] = failures

        # Alert transitions
        if self._alert:
            if prev_state == HealthState.HEALTHY and new_state == HealthState.DEGRADED:
                self._alert.send_warning("HealthChecker", f"{component} degraded: {message}")
            elif prev_state != HealthState.FAILED and new_state == HealthState.FAILED:
                self._alert.send_critical("HealthChecker", f"{component} FAILED: {message}", extra)
            elif prev_state != HealthState.HEALTHY and new_state == HealthState.HEALTHY:
                self._alert.send_info("HealthChecker", f"{component} recovered: {message}")

        record = HealthRecord(
            component=component,
            state=new_state,
            health_score=score,
            consecutive_failures=failures,
            message=message,
            extra=extra,
        )

        # Write to metrics
        if self._metrics:
            self._metrics.emit("HealthChecker", f"{component}.health_score", score)
            self._metrics.emit("HealthChecker", f"{component}.consecutive_failures", failures)

        logger.debug("Health check %s: %s score=%d", component, new_state.value, score)
        return record
