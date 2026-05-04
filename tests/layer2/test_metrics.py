"""Tests for layer2.metrics_emitter — MetricsEmitter."""

from __future__ import annotations

import queue
import time
import pytest
from unittest.mock import MagicMock, patch

from layer2.metrics_emitter import MetricsEmitter, MetricPoint, TimerContext


class TestMetricPoint:

    def test_default_fields(self):
        p = MetricPoint(component="test", metric_name="latency", metric_value=1.5)
        assert p.environment == "paper"
        assert isinstance(p.ts_ns, int)
        assert p.tags == {}


class TestMetricsEmitter:

    def _make_emitter(self) -> MetricsEmitter:
        e = MetricsEmitter.__new__(MetricsEmitter)
        e._ilp_host = "localhost"
        e._ilp_port = 9009
        e._environment = "test"
        e._host = "testhost"
        e._queue = queue.Queue(maxsize=MetricsEmitter._QUEUE_MAX)
        e._ilp_sock = None
        e._running = True
        e._total_emitted = 0
        e._total_dropped = 0
        e._total_write_errors = 0
        e._alert = None
        return e

    def test_emit_adds_to_queue(self):
        emitter = self._make_emitter()
        emitter.emit("ingestor", "tick_latency_ms", 5.2)
        assert emitter._queue.qsize() == 1

    def test_non_numeric_metric_ignored(self):
        emitter = self._make_emitter()
        emitter.emit("ingestor", "bad_metric", "not a number")
        assert emitter._queue.qsize() == 0

    def test_queue_full_increments_dropped(self):
        emitter = self._make_emitter()
        # Fill the queue beyond capacity
        for _ in range(MetricsEmitter._QUEUE_MAX + 5):
            emitter.emit("comp", "m", 1.0)
        assert emitter._total_dropped >= 1

    def test_point_to_ilp_format(self):
        emitter = self._make_emitter()
        p = MetricPoint(
            component="ingestor", metric_name="tick_latency", metric_value=5.2,
            ts_ns=1_700_000_000_000_000_000, host="host1", environment="paper",
        )
        ilp = emitter._point_to_ilp(p)
        assert "metrics_layer2," in ilp
        assert "component=ingestor" in ilp
        assert "metric_value=5.2" in ilp
        assert "1700000000000000000" in ilp
        assert ilp.endswith("\n")

    def test_timer_context_emits_elapsed(self):
        emitter = self._make_emitter()
        with emitter.emit_timer("comp", "op_ms") as ctx:
            time.sleep(0.01)
        # Should have emitted 1 metric
        assert emitter._queue.qsize() == 1
        point = emitter._queue.get()
        assert point.metric_name == "op_ms"
        assert point.metric_value >= 10.0  # ≥10ms

    def test_get_statistics_keys(self):
        emitter = self._make_emitter()
        stats = emitter.get_statistics()
        for key in ("total_emitted", "total_dropped", "total_write_errors", "queue_size"):
            assert key in stats


class TestHealthCheckerIntegration:

    def test_healthy_to_degraded_alert(self):
        from layer2.health_checker import HealthChecker, HealthState

        alert = MagicMock()
        checker = HealthChecker(alert_manager=alert)

        call_count = [0]

        def check_fn():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"healthy": True, "score": 100, "message": "OK"}
            return {"healthy": False, "score": 60, "message": "slow"}

        checker.register_check("questdb", check_fn)

        # First run: healthy
        checker.run_all_checks_once()
        assert checker.get_component_state("questdb") == HealthState.HEALTHY

        # Second run: degraded
        checker.run_all_checks_once()
        assert checker.get_component_state("questdb") == HealthState.DEGRADED
        alert.send_warning.assert_called_once()

    def test_failed_state_sends_critical(self):
        from layer2.health_checker import HealthChecker, HealthState

        alert = MagicMock()
        checker = HealthChecker(alert_manager=alert)

        checker.register_check("redis", lambda: {"healthy": False, "score": 0, "message": "down"})
        checker.run_all_checks_once()
        assert checker.get_component_state("redis") == HealthState.FAILED
        alert.send_critical.assert_called_once()

    def test_recovered_state_sends_info(self):
        from layer2.health_checker import HealthChecker, HealthState

        alert = MagicMock()
        checker = HealthChecker(alert_manager=alert)

        call_count = [0]

        def check_fn():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"healthy": False, "score": 0, "message": "down"}
            return {"healthy": True, "score": 100, "message": "OK"}

        checker.register_check("comp", check_fn)
        checker.run_all_checks_once()  # → FAILED
        checker.run_all_checks_once()  # → HEALTHY (recovered)
        alert.send_info.assert_called_once()

    def test_system_health_score_average(self):
        from layer2.health_checker import HealthChecker

        checker = HealthChecker()
        checker.register_check("a", lambda: {"healthy": True, "score": 100, "message": ""})
        checker.register_check("b", lambda: {"healthy": False, "score": 0, "message": ""})
        checker.run_all_checks_once()
        score = checker.get_system_health_score()
        assert score == 50  # (100 + 0) / 2
