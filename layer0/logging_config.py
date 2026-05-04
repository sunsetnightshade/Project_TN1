"""Layer 0 — Logging Configuration.

Provides configure_logging() and get_logger() for all Nightshade modules.
No module should ever call logging.getLogger() directly or use print().
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layer0.config import ConfigRegistry


# ---------------------------------------------------------------------------
# Structured JSON formatter
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Single-line JSON log records with all required fields."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        extra: dict = {}
        if record.exc_info:
            extra["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            extra["exc_value"] = str(record.exc_info[1])
            extra["exc_traceback"] = traceback.format_exception(*record.exc_info)

        payload = {
            "ts": ts,
            "level": record.levelname,
            "module": record.name,
            "msg": record.getMessage(),
            "pid": os.getpid(),
        }
        if extra:
            payload["extra"] = extra

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Human-readable formatter
# ---------------------------------------------------------------------------

_HUMAN_FORMAT = "{asctime} | {levelname:<8} | {name:<30} | {message}"


class _HumanFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            fmt="{asctime} | {levelname:<8} | {name:<30} | {message}",
            datefmt="%Y-%m-%dT%H:%M:%S",
            style="{",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_logging(config: "ConfigRegistry") -> logging.Logger:
    """Configure the root logger from *config* and return it.

    Sets up:
    - StreamHandler to stdout
    - TimedRotatingFileHandler (daily rotation, 90-day retention)
    """
    level_name: str = config.get("logging.level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    structured: bool = bool(config.get("logging.structured", True))
    log_dir = Path(str(config.get("logging.log_dir", "~/.nightshade/logs"))).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter: logging.Formatter = _JsonFormatter() if structured else _HumanFormatter()

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    root.handlers.clear()

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # File handler — rotates daily, keeps 90 backups
    log_file = log_dir / "nightshade.log"
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        backupCount=config.get("logging.backup_count", 90),
        utc=True,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    root.info("Logging configured: level=%s structured=%s log_dir=%s", level_name, structured, log_dir)
    return root


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper — all modules call this instead of logging.getLogger."""
    return logging.getLogger(name)
