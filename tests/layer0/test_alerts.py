"""Tests for layer0.alerts — AlertManager."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from layer0.alerts import Alert, AlertManager, AlertSeverity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> MagicMock:
    log_path = tmp_path / "alerts.log"
    mock = MagicMock()
    mock.get.side_effect = lambda key, default=None: {
        "alerting.method": "file",
        "alerting.alert_log_path": str(log_path),
        "alerting.email.smtp_host": "smtp.example.com",
        "alerting.email.smtp_port": 587,
        "alerting.email.sender_address": "sender@example.com",
        "alerting.email.recipient_address": "recv@example.com",
    }.get(key, default)
    return mock


@pytest.fixture()
def mgr(tmp_path: Path) -> AlertManager:
    return AlertManager(_make_config(tmp_path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAlertManager:

    def test_info_written_to_file(self, mgr: AlertManager, tmp_path: Path):
        mgr.send_info("test_component", "info message")
        log = tmp_path / "alerts.log"
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["severity"] == "INFO"
        assert d["component"] == "test_component"

    def test_critical_written_to_file(self, mgr: AlertManager, tmp_path: Path):
        mgr.send_critical("test_component", "critical message", {"key": "value"})
        log = tmp_path / "alerts.log"
        d = json.loads(log.read_text().strip().splitlines()[-1])
        assert d["severity"] == "CRITICAL"
        assert d["data"]["key"] == "value"

    def test_email_fallback_on_smtp_failure(self, tmp_path: Path):
        """Email method with SMTP failure → file still written, no exception."""
        cfg = _make_config(tmp_path)
        cfg.get.side_effect = lambda key, default=None: {
            "alerting.method": "email",
            "alerting.alert_log_path": str(tmp_path / "alerts.log"),
            "alerting.email.smtp_host": "bad-host",
            "alerting.email.smtp_port": 587,
            "alerting.email.sender_address": "",
            "alerting.email.recipient_address": "",
        }.get(key, default)

        secrets = MagicMock()
        secrets.get.side_effect = Exception("no secret")

        mgr = AlertManager(cfg, secrets)
        # Should not raise even when email fails
        mgr.send_critical("component", "critical but no email")
        log = tmp_path / "alerts.log"
        assert log.exists()

    def test_get_recent_alerts_count_and_order(self, mgr: AlertManager):
        for i in range(5):
            mgr.send_info("comp", f"message {i}")
        recent = mgr.get_recent_alerts(3)
        assert len(recent) == 3
        # Sorted descending — last sent should be first
        assert recent[0].message == "message 4"

    def test_get_recent_alerts_severity_filter(self, mgr: AlertManager):
        mgr.send_info("comp", "info msg")
        mgr.send_warning("comp", "warn msg")
        warnings = mgr.get_recent_alerts(10, severity_filter=AlertSeverity.WARNING)
        assert all(a.severity == AlertSeverity.WARNING for a in warnings)

    def test_alert_json_validity(self, mgr: AlertManager, tmp_path: Path):
        mgr.send_warning("comp", "test", {"extra": 42})
        log = tmp_path / "alerts.log"
        d = json.loads(log.read_text().strip())
        assert "alert_id" in d
        assert "ts_utc" in d

    def test_concurrent_write_safety(self, mgr: AlertManager, tmp_path: Path):
        """Multiple threads writing concurrently must not corrupt the log."""
        def write() -> None:
            for _ in range(20):
                mgr.send_info("thread", "concurrent msg")

        threads = [threading.Thread(target=write) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        log = tmp_path / "alerts.log"
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 100
        for line in lines:
            json.loads(line)  # must be valid JSON
