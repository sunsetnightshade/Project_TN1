"""Layer 1C — Sequence Tracker.

Detects per-message gaps in ordered sequences from each data source.
Tracks state per (source_name, session_id) pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from layer0.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SequenceGap:
    source_name: str
    session_id: str
    gap_start_sequence: int
    gap_end_sequence: int
    estimated_missing_count: int
    detected_at_ts_recv_ns: int


@dataclass
class SequenceState:
    last_sequence: int
    first_sequence: int
    total_messages: int
    total_gaps: int
    session_start_ts_recv_ns: int
    last_message_ts_recv_ns: int


class SequenceTracker:
    """Per-(source_name, session_id) sequence tracking for gap detection."""

    def __init__(self) -> None:
        self._states: dict[tuple, SequenceState] = {}
        self._gaps: dict[str, list[SequenceGap]] = {}
        self._archived: dict[tuple, list[SequenceState]] = {}

    def record_message(
        self,
        source_name: str,
        session_id: str,
        sequence_number: int,
        ts_recv_ns: int,
    ) -> Optional[SequenceGap]:
        """Record a message. Returns SequenceGap if a gap was detected, else None."""
        key = (source_name, session_id)

        if key not in self._states:
            # First message for this session
            self._states[key] = SequenceState(
                last_sequence=sequence_number,
                first_sequence=sequence_number,
                total_messages=1,
                total_gaps=0,
                session_start_ts_recv_ns=ts_recv_ns,
                last_message_ts_recv_ns=ts_recv_ns,
            )
            return None

        state = self._states[key]

        if sequence_number == state.last_sequence + 1:
            # Sequential — normal
            state.last_sequence = sequence_number
            state.total_messages += 1
            state.last_message_ts_recv_ns = ts_recv_ns
            return None

        if sequence_number <= state.last_sequence:
            # Out of order
            logger.debug(
                "Out-of-order seq from %s/%s: expected >%d, got %d",
                source_name, session_id, state.last_sequence, sequence_number,
            )
            return None

        # Gap detected
        missing = sequence_number - state.last_sequence - 1
        gap = SequenceGap(
            source_name=source_name,
            session_id=session_id,
            gap_start_sequence=state.last_sequence + 1,
            gap_end_sequence=sequence_number - 1,
            estimated_missing_count=missing,
            detected_at_ts_recv_ns=ts_recv_ns,
        )
        state.total_gaps += 1
        state.last_sequence = sequence_number
        state.total_messages += 1
        state.last_message_ts_recv_ns = ts_recv_ns

        # Store gap
        if source_name not in self._gaps:
            self._gaps[source_name] = []
        self._gaps[source_name].append(gap)

        logger.warning(
            "Sequence gap in %s/%s: missing %d messages (%d-%d)",
            source_name, session_id, missing, gap.gap_start_sequence, gap.gap_end_sequence,
        )
        return gap

    def reset_session(self, source_name: str, session_id: str) -> None:
        """Archive old state and clear for a new session (reconnect)."""
        key = (source_name, session_id)
        if key in self._states:
            archived = self._archived.setdefault(key, [])
            archived.append(self._states.pop(key))
        logger.debug("Session reset: %s/%s", source_name, session_id)

    def get_session_statistics(self, source_name: str, session_id: str) -> dict:
        key = (source_name, session_id)
        state = self._states.get(key)
        if not state:
            return {}
        total = state.total_messages
        expected = state.last_sequence - state.first_sequence + 1
        coverage = (total / expected * 100) if expected > 0 else 100.0
        gap_rate = state.total_gaps / total if total > 0 else 0.0
        return {
            "last_sequence": state.last_sequence,
            "first_sequence": state.first_sequence,
            "total_messages": total,
            "total_gaps": state.total_gaps,
            "gap_rate": round(gap_rate, 6),
            "coverage_pct": round(coverage, 4),
        }

    def get_all_gaps(self, source_name: Optional[str] = None) -> list[SequenceGap]:
        if source_name:
            return list(self._gaps.get(source_name, []))
        return [gap for gaps in self._gaps.values() for gap in gaps]
