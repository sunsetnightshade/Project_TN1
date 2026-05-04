"""Layer 2 — Metrics Emitter.

Thread-safe, non-blocking, fire-and-forget metrics emission.
Writes to QuestDB metrics_layer2 table via ILP.
Never raises — errors are logged, never propagated.
"""

from __future__ import annotations

import json
import platform
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from layer0.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metric dataclass
# ---------------------------------------------------------------------------

@dataclass
class MetricPoint:
    component: str
    metric_name: str
    metric_value: float
    ts_ns: int = field(default_factory=time.time_ns)
    host: str = field(default_factory=lambda: platform.node() or "unknown")
    environment: str = "paper"
    tags: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MetricsEmitterError(Exception):
    """Base exception for MetricsEmitter errors."""

class MetricsQueueFullError(MetricsEmitterError):
    """Raised when the internal queue is full (non-blocking)."""


# ---------------------------------------------------------------------------
# MetricsEmitter
# ---------------------------------------------------------------------------

class MetricsEmitter:
    """Non-blocking, thread-safe metrics emitter for all Nightshade layers.

    Uses an internal queue and a background thread.
    set_metrics_emitter() pattern: inject one instance per component.
    """

    _QUEUE_MAX = 10_000
    _FLUSH_INTERVAL_SEC = 5
    _FLUSH_BATCH_SIZE = 500

    def __init__(
        self,
        config=None,
        alert_manager=None,
        environment: str = "paper",
    ) -> None:
        if config:
            self._ilp_host = config.get("database.questdb.host", "localhost")
            self._ilp_port = int(config.get("database.questdb.ilp_port", 9009))
            environment = config.get("system.environment", "paper")
        else:
            self._ilp_host = "localhost"
            self._ilp_port = 9009

        self._environment = environment
        self._alert = alert_manager
        self._host = platform.node() or "unknown"
        self._queue: queue.Queue[MetricPoint] = queue.Queue(maxsize=self._QUEUE_MAX)
        self._ilp_sock: Optional[socket.socket] = None
        self._running = True
        self._worker = threading.Thread(
            target=self._flush_loop, daemon=True, name="metrics-emitter"
        )
        self._worker.start()

        # Stats
        self._total_emitted = 0
        self._total_dropped = 0
        self._total_write_errors = 0

    def emit(
        self,
        component: str,
        metric_name: str,
        metric_value: float,
        tags: Optional[dict] = None,
        ts_ns: Optional[int] = None,
    ) -> None:
        """Emit a metric. Fire-and-forget — never raises."""
        if not isinstance(metric_value, (int, float)):
            logger.debug("Ignoring non-numeric metric %s.%s: %s", component, metric_name, metric_value)
            return
        try:
            point = MetricPoint(
                component=component,
                metric_name=metric_name,
                metric_value=float(metric_value),
                ts_ns=ts_ns or time.time_ns(),
                host=self._host,
                environment=self._environment,
                tags=tags or {},
            )
            self._queue.put_nowait(point)
        except queue.Full:
            self._total_dropped += 1
            # Only log every 1000 drops to avoid flooding
            if self._total_dropped % 1000 == 1:
                logger.warning("MetricsEmitter queue full — %d dropped", self._total_dropped)

    def emit_timer(self, component: str, metric_name: str) -> "TimerContext":
        """Return a context manager that emits elapsed_ms on exit."""
        return TimerContext(self, component, metric_name)

    def get_statistics(self) -> dict:
        return {
            "total_emitted": self._total_emitted,
            "total_dropped": self._total_dropped,
            "total_write_errors": self._total_write_errors,
            "queue_size": self._queue.qsize(),
        }

    def stop(self) -> None:
        self._running = False
        self._flush_now()
        if self._ilp_sock:
            try:
                self._ilp_sock.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_loop(self) -> None:
        while self._running:
            time.sleep(self._FLUSH_INTERVAL_SEC)
            self._flush_now()

    def _flush_now(self) -> None:
        batch = []
        try:
            while len(batch) < self._FLUSH_BATCH_SIZE:
                batch.append(self._queue.get_nowait())
        except queue.Empty:
            pass

        if not batch:
            return

        lines = "".join(self._point_to_ilp(p) for p in batch)
        try:
            self._ilp_send(lines)
            self._total_emitted += len(batch)
        except Exception as exc:
            self._total_write_errors += 1
            logger.debug("MetricsEmitter write failed: %s", exc)

    def _ilp_send(self, payload: str) -> None:
        if self._ilp_sock is None:
            self._ilp_sock = socket.create_connection((self._ilp_host, self._ilp_port), timeout=5)
        try:
            self._ilp_sock.sendall(payload.encode("utf-8"))
        except Exception:
            self._ilp_sock = None
            self._ilp_sock = socket.create_connection((self._ilp_host, self._ilp_port), timeout=5)
            self._ilp_sock.sendall(payload.encode("utf-8"))

    def _point_to_ilp(self, p: MetricPoint) -> str:
        tags_part = f",tags={json.dumps(p.tags)}" if p.tags else ""
        return (
            f"metrics_layer2,"
            f"component={p.component},"
            f"metric_name={p.metric_name},"
            f"host={p.host},"
            f"environment={p.environment}"
            f"{tags_part} "
            f"metric_value={p.metric_value} "
            f"{p.ts_ns}\n"
        )


# ---------------------------------------------------------------------------
# TimerContext
# ---------------------------------------------------------------------------

class TimerContext:
    """Context manager for timing operations and emitting elapsed_ms metric."""

    def __init__(self, emitter: MetricsEmitter, component: str, metric_name: str) -> None:
        self._emitter = emitter
        self._component = component
        self._name = metric_name
        self._start = 0.0

    def __enter__(self) -> "TimerContext":
        self._start = time.monotonic()
        return self

    def __exit__(self, *args) -> None:
        elapsed_ms = (time.monotonic() - self._start) * 1000
        self._emitter.emit(self._component, self._name, elapsed_ms)
