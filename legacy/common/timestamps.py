from __future__ import annotations

from datetime import datetime, timezone


def parse_iso_utc(raw: str) -> datetime:
    """
    Parse an ISO 8601 timestamp string and return a UTC-aware datetime.

    Handles:
      - Strings ending in 'Z' (UTC shorthand)
      - Strings with explicit UTC offset (e.g. '+00:00', '+05:30')
      - Naive strings (assumed UTC)

    Used across: consumer, janitor, type2_fallback, provider, service.
    Single source of truth — do not duplicate this in individual modules.
    """
    if raw.endswith("Z"):
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    else:
        dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_iso_utc_optional(raw: str | None) -> datetime | None:
    """Same as parse_iso_utc but returns None when raw is None or unparseable."""
    if raw is None:
        return None
    try:
        return parse_iso_utc(raw)
    except (ValueError, AttributeError):
        return None
