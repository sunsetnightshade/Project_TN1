"""Layer 0 — Alerting System.

Provides AlertManager for sending WARNING/CRITICAL/INFO alerts via file
log or email.  Alert log entries are newline-delimited JSON.
"""

from __future__ import annotations

import json
import smtplib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer0.config import ConfigRegistry
    from layer0.secrets import SecretsManager

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    severity: AlertSeverity
    component: str
    message: str
    data: dict = field(default_factory=dict)
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "ts_utc": self.ts_utc,
            "severity": self.severity.value,
            "component": self.component,
            "message": self.message,
            "data": self.data,
        }


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """Sends alerts via file log and optionally email.

    Never raises — all failures are logged at ERROR level internally.
    """

    def __init__(
        self,
        config: "ConfigRegistry",
        secrets_manager: Optional["SecretsManager"] = None,
    ) -> None:
        self._method: str = config.get("alerting.method", "file")
        alert_log_path_raw = config.get("alerting.alert_log_path", "~/.nightshade/alerts.log")
        self._alert_log_path = Path(str(alert_log_path_raw)).expanduser()
        self._alert_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Email settings
        self._smtp_host: str = config.get("alerting.email.smtp_host", "smtp.gmail.com")
        self._smtp_port: int = int(config.get("alerting.email.smtp_port", 587))
        self._sender: str = config.get("alerting.email.sender_address", "")
        self._recipient: str = config.get("alerting.email.recipient_address", "")

        # Load sender from secrets if email method configured
        if self._method == "email" and secrets_manager is not None:
            try:
                self._sender = secrets_manager.get("alerting.sender_email")
            except Exception:
                logger.warning(
                    "alerting.sender_email secret not found; falling back to file alerting"
                )
                self._method = "file"

        self._secrets_manager = secrets_manager
        logger.debug("AlertManager initialised: method=%s log=%s", self._method, self._alert_log_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(
        self,
        severity: AlertSeverity,
        component: str,
        message: str,
        data: Optional[dict] = None,
    ) -> Alert:
        """Create, log, and (optionally) email an alert.  Never raises."""
        alert = Alert(severity=severity, component=component, message=message, data=data or {})
        self._write_to_file(alert)

        if self._method == "email" and severity in (AlertSeverity.WARNING, AlertSeverity.CRITICAL):
            self._send_email(alert)

        return alert

    def send_info(self, component: str, message: str, data: Optional[dict] = None) -> Alert:
        return self.send(AlertSeverity.INFO, component, message, data)

    def send_warning(self, component: str, message: str, data: Optional[dict] = None) -> Alert:
        return self.send(AlertSeverity.WARNING, component, message, data)

    def send_critical(self, component: str, message: str, data: Optional[dict] = None) -> Alert:
        return self.send(AlertSeverity.CRITICAL, component, message, data)

    def get_recent_alerts(
        self,
        n: int = 10,
        severity_filter: Optional[AlertSeverity] = None,
    ) -> list[Alert]:
        """Return last *n* alerts sorted by timestamp descending."""
        alerts: list[Alert] = []
        try:
            if not self._alert_log_path.exists():
                return []
            with open(self._alert_log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        alert = Alert(
                            severity=AlertSeverity(d["severity"]),
                            component=d["component"],
                            message=d["message"],
                            data=d.get("data", {}),
                            alert_id=d["alert_id"],
                            ts_utc=d["ts_utc"],
                        )
                        if severity_filter is None or alert.severity == severity_filter:
                            alerts.append(alert)
                    except Exception:
                        pass
        except Exception as exc:
            logger.error("get_recent_alerts failed: %s", exc)

        alerts.sort(key=lambda a: a.ts_utc, reverse=True)
        return alerts[:n]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_to_file(self, alert: Alert) -> None:
        try:
            line = json.dumps(alert.to_dict()) + "\n"
            with open(self._alert_log_path, "a", encoding="utf-8") as fh:
                # Portable file locking
                try:
                    import fcntl
                    fcntl.flock(fh, fcntl.LOCK_EX)
                    fh.write(line)
                    fcntl.flock(fh, fcntl.LOCK_UN)
                except ImportError:
                    # Windows — no fcntl, just write
                    fh.write(line)
        except Exception as exc:
            logger.error("Failed to write alert to log: %s", exc)

    def _send_email(self, alert: Alert) -> None:
        try:
            smtp_password: Optional[str] = None
            if self._secrets_manager is not None:
                try:
                    smtp_password = self._secrets_manager.get("alerting.smtp_password")
                except Exception:
                    pass

            subject = (
                f"[NIGHTSHADE {alert.severity.value}] "
                f"{alert.component}: {alert.message[:50]}"
            )
            body = json.dumps(alert.to_dict(), indent=2)
            msg = MIMEText(body, "plain")
            msg["Subject"] = subject
            msg["From"] = self._sender
            msg["To"] = self._recipient

            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=10) as smtp:
                smtp.ehlo()
                smtp.starttls()
                if smtp_password:
                    smtp.login(self._sender, smtp_password)
                smtp.sendmail(self._sender, [self._recipient], msg.as_string())
        except Exception as exc:
            logger.error("Email alert failed, falling back to file only: %s", exc)
            # Already written to file above — no further action needed
