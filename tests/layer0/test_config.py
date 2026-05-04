"""Tests for layer0.config — ConfigRegistry."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from layer0.config import (
    ConfigRegistry,
    ConfigFileNotFoundError,
    ConfigKeyNotFoundError,
    ConfigValidationError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_VALID_CONFIG = {
    "secrets": {"vault_path": "~/.ns/v.enc", "salt_path": "~/.ns/v.salt"},
    "logging": {"level": "INFO", "log_dir": "/tmp", "structured": True},
    "alerting": {"method": "file", "alert_log_path": "/tmp/a.log"},
    "database": {
        "questdb": {"host": "localhost", "http_port": 9000},
        "redis": {"host": "localhost", "port": 6379},
    },
    "system": {"environment": "paper", "timezone": "UTC"},
    "research": {"data_lookback_days": 504},
}


@pytest.fixture()
def cfg_file(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(MINIMAL_VALID_CONFIG), encoding="utf-8")
    return p


@pytest.fixture()
def cfg(cfg_file: Path) -> ConfigRegistry:
    return ConfigRegistry(cfg_file)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConfigRegistry:

    def test_valid_load(self, cfg: ConfigRegistry):
        assert cfg.get("system.environment") == "paper"

    def test_dot_notation_nesting(self, cfg: ConfigRegistry):
        assert cfg.get("database.questdb.host") == "localhost"

    def test_missing_key_returns_default(self, cfg: ConfigRegistry):
        assert cfg.get("nonexistent.key", "fallback") == "fallback"

    def test_missing_key_no_default_returns_none(self, cfg: ConfigRegistry):
        assert cfg.get("nonexistent.key") is None

    def test_get_required_raises_on_missing(self, cfg: ConfigRegistry):
        with pytest.raises(ConfigKeyNotFoundError):
            cfg.get_required("nonexistent.deep.key")

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(ConfigFileNotFoundError):
            ConfigRegistry(tmp_path / "does_not_exist.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path):
        p = tmp_path / "bad.yaml"
        p.write_text("{\nnot valid yaml{{{{", encoding="utf-8")
        with pytest.raises(ConfigValidationError):
            ConfigRegistry(p)

    def test_reload(self, cfg: ConfigRegistry, cfg_file: Path):
        cfg_file.write_text(
            yaml.dump({**MINIMAL_VALID_CONFIG, "system": {"environment": "live", "timezone": "UTC"}}),
            encoding="utf-8",
        )
        cfg.reload()
        assert cfg.get("system.environment") == "live"

    def test_as_dict_is_deep_copy(self, cfg: ConfigRegistry):
        d = cfg.as_dict()
        d["system"]["environment"] = "tampered"
        # Original must be unchanged
        assert cfg.get("system.environment") == "paper"

    def test_missing_required_top_level_key(self, tmp_path: Path):
        incomplete = {"secrets": {}, "logging": {}, "alerting": {}, "database": {}}
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(incomplete), encoding="utf-8")
        with pytest.raises(ConfigValidationError):
            ConfigRegistry(p)
