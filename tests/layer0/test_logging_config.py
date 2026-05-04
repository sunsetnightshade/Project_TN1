"""Tests for layer0.logging_config."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from layer0.logging_config import configure_logging, get_logger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, structured: bool = True) -> MagicMock:
    mock = MagicMock()
    mock.get.side_effect = lambda key, default=None: {
        "logging.level": "DEBUG",
        "logging.log_dir": str(tmp_path / "logs"),
        "logging.backup_count": 5,
        "logging.structured": structured,
    }.get(key, default)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConfigureLogging:

    def test_log_dir_created(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        configure_logging(cfg)
        assert (tmp_path / "logs").is_dir()

    def test_structured_json_fields(self, tmp_path: Path):
        cfg = _make_config(tmp_path, structured=True)
        root = configure_logging(cfg)
        log = get_logger("test.module")

        # Capture output from stream handler
        from io import StringIO
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        from layer0.logging_config import _JsonFormatter
        handler.setFormatter(_JsonFormatter())
        log.addHandler(handler)
        log.debug("hello world")
        out = stream.getvalue().strip()

        parsed = json.loads(out)
        for key in ("ts", "level", "module", "msg", "pid"):
            assert key in parsed, f"Missing field: {key}"
        assert parsed["msg"] == "hello world"

    def test_exception_captured_in_json(self, tmp_path: Path):
        cfg = _make_config(tmp_path, structured=True)
        configure_logging(cfg)
        log = get_logger("test.exc")

        from io import StringIO
        from layer0.logging_config import _JsonFormatter
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(_JsonFormatter())
        log.addHandler(handler)
        try:
            raise ValueError("boom")
        except ValueError:
            log.exception("error occurred")

        out = stream.getvalue().strip()
        parsed = json.loads(out)
        assert "extra" in parsed
        assert "exc_type" in parsed["extra"]
        assert parsed["extra"]["exc_type"] == "ValueError"

    def test_human_readable_format(self, tmp_path: Path):
        cfg = _make_config(tmp_path, structured=False)
        configure_logging(cfg)
        log = get_logger("test.human")

        from io import StringIO
        from layer0.logging_config import _HumanFormatter
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(_HumanFormatter())
        log.addHandler(handler)
        log.info("test message")
        out = stream.getvalue().strip()
        assert "|" in out
        assert "test message" in out

    def test_get_logger_name(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        configure_logging(cfg)
        log = get_logger("my.custom.module")
        assert log.name == "my.custom.module"
