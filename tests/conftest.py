"""Shared pytest fixtures for all Nightshade tests."""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(autouse=True)
def suppress_log_propagation(caplog):
    """Suppress verbose logging output in tests."""
    import logging
    caplog.set_level(logging.ERROR, logger="layer0")
    caplog.set_level(logging.ERROR, logger="layer1a")
    caplog.set_level(logging.ERROR, logger="layer1b")
    caplog.set_level(logging.ERROR, logger="layer1c")
    caplog.set_level(logging.ERROR, logger="layer2")
