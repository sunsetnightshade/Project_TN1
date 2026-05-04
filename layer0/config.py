"""Layer 0 — Configuration Registry.

Loads and validates config.yaml, provides dot-notation access to any
nested key, and supports runtime reload.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from layer0.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Base exception for all configuration errors."""


class ConfigFileNotFoundError(ConfigError):
    """Raised when config.yaml cannot be found at the given path."""


class ConfigValidationError(ConfigError):
    """Raised when the loaded YAML fails structural validation."""


class ConfigKeyNotFoundError(ConfigError):
    """Raised by get_required() when a key path is absent."""


# ---------------------------------------------------------------------------
# ConfigRegistry
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path("config.yaml")

# Minimum required top-level keys for a valid Nightshade config
_REQUIRED_TOP_LEVEL_KEYS = {"secrets", "logging", "alerting", "database", "system"}


class ConfigRegistry:
    """Dot-notation accessor for the Nightshade master config.yaml.

    Args:
        config_path: Path to config.yaml.  Defaults to ./config.yaml.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._data: dict = {}
        self._load()
        logger.debug("ConfigRegistry loaded: %s", self._config_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key_path: str, default: Any = None) -> Any:
        """Return value at dot-separated *key_path*.

        Returns *default* when the path is absent (even if default is None).
        """
        try:
            return self._traverse(key_path)
        except ConfigKeyNotFoundError:
            return default

    def get_required(self, key_path: str) -> Any:
        """Return value at *key_path*.  Raises ConfigKeyNotFoundError if absent."""
        return self._traverse(key_path)

    def reload(self) -> None:
        """Re-read config.yaml from disk."""
        self._load()
        logger.debug("ConfigRegistry reloaded: %s", self._config_path)

    def as_dict(self) -> dict:
        """Return a deep copy of the entire configuration dict."""
        return copy.deepcopy(self._data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._config_path.exists():
            raise ConfigFileNotFoundError(
                f"Config file not found: {self._config_path}"
            )
        with open(self._config_path, "r", encoding="utf-8") as fh:
            try:
                data = yaml.safe_load(fh)
            except yaml.YAMLError as exc:
                raise ConfigValidationError(f"YAML parse error: {exc}") from exc

        if not isinstance(data, dict):
            raise ConfigValidationError("Config YAML root must be a mapping.")

        missing = _REQUIRED_TOP_LEVEL_KEYS - set(data.keys())
        if missing:
            raise ConfigValidationError(
                f"Config missing required top-level keys: {sorted(missing)}"
            )
        self._data = data

    def _traverse(self, key_path: str) -> Any:
        parts = key_path.split(".")
        node = self._data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                raise ConfigKeyNotFoundError(
                    f"Config key not found: {key_path!r}"
                )
            node = node[part]
        return node
