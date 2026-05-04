"""Layer 1C — Source Protocol.

Defines the structural interface all live data sources must satisfy.
Uses Protocol (structural subtyping) — no inheritance required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SourceProtocolError(Exception):
    """Base exception for source protocol errors."""

class ConnectionError(SourceProtocolError):
    """Connection failure."""

class SubscriptionError(SourceProtocolError):
    """Subscription failure."""

class AuthenticationError(SourceProtocolError):
    """Authentication failure."""

class SourceUnavailableError(SourceProtocolError):
    """Source library not installed or source not available."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RawMessageProtocol:
    """A raw message from any live data source."""
    source: str
    raw_payload: bytes | str | dict
    ts_recv_ns: int
    sequence_number: Optional[int] = None


# ---------------------------------------------------------------------------
# Protocol definition (structural)
# ---------------------------------------------------------------------------

class LiveDataSourceProtocol:
    """Base class / interface for all live data sources.

    Concrete adapters (PolygonWebSocketAdapter, DatabentоWebSocketAdapter)
    must implement all of these methods.  Typing is structural — adapters
    do not need to inherit from this class.
    """

    def connect(self) -> None:
        """Connect to the data source. Idempotent."""
        raise NotImplementedError

    def disconnect(self) -> None:
        """Disconnect cleanly, flush pending messages."""
        raise NotImplementedError

    def subscribe(self, nightshade_ids: list[str]) -> None:
        """Subscribe to tick feed for the given instrument IDs.

        Resolves nightshade_ids → external symbols via SecurityMaster.
        Raises SubscriptionError on failure.
        """
        raise NotImplementedError

    def get_source_name(self) -> str:
        """Return canonical source name (e.g., 'polygon_ws', 'databento_ws')."""
        raise NotImplementedError

    def get_connection_state(self) -> str:
        """Return one of: DISCONNECTED, CONNECTING, CONNECTED, RECONNECTING, ERROR."""
        raise NotImplementedError

    def get_health_metrics(self) -> dict:
        """Return health metrics dict.

        Keys:
          source_name, connection_state, messages_received_total,
          messages_received_last_minute, sequence_gaps_detected,
          last_message_ts_recv, current_latency_ms, uptime_seconds
        """
        raise NotImplementedError
