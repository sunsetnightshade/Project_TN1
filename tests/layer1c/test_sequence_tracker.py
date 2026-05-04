"""Tests for layer1c.sequence_tracker — SequenceTracker."""

from __future__ import annotations

import time
import pytest
from layer1c.sequence_tracker import SequenceTracker, SequenceGap


@pytest.fixture()
def tracker() -> SequenceTracker:
    return SequenceTracker()


class TestSequenceTracker:

    def test_sequential_messages_no_gap(self, tracker):
        ns = time.time_ns()
        for i in range(1, 11):
            gap = tracker.record_message("polygon_ws", "sess1", i, ns + i)
            assert gap is None

    def test_gap_detected(self, tracker):
        ns = time.time_ns()
        tracker.record_message("polygon_ws", "sess1", 1, ns)
        gap = tracker.record_message("polygon_ws", "sess1", 5, ns + 100)
        assert isinstance(gap, SequenceGap)
        assert gap.gap_start_sequence == 2
        assert gap.gap_end_sequence == 4
        assert gap.estimated_missing_count == 3

    def test_out_of_order_no_gap(self, tracker):
        ns = time.time_ns()
        tracker.record_message("polygon_ws", "sess1", 5, ns)
        gap = tracker.record_message("polygon_ws", "sess1", 3, ns + 1)  # OOO
        assert gap is None

    def test_statistics_accuracy(self, tracker):
        ns = time.time_ns()
        tracker.record_message("polygon_ws", "sess1", 1, ns)
        tracker.record_message("polygon_ws", "sess1", 2, ns + 1)
        tracker.record_message("polygon_ws", "sess1", 5, ns + 2)  # gap: 3,4 missing
        tracker.record_message("polygon_ws", "sess1", 6, ns + 3)
        stats = tracker.get_session_statistics("polygon_ws", "sess1")
        assert stats["total_messages"] == 4
        assert stats["total_gaps"] == 1

    def test_session_reset_clears_state(self, tracker):
        ns = time.time_ns()
        tracker.record_message("polygon_ws", "sess1", 1, ns)
        tracker.record_message("polygon_ws", "sess1", 2, ns + 1)
        tracker.reset_session("polygon_ws", "sess1")
        # New session: should start fresh
        gap = tracker.record_message("polygon_ws", "sess1", 100, ns + 2)
        assert gap is None  # First message of new session

    def test_independent_sessions_for_different_sources(self, tracker):
        ns = time.time_ns()
        tracker.record_message("polygon_ws", "sess1", 1, ns)
        tracker.record_message("databento_ws", "sess1", 1, ns)
        # Both sources track independently
        stats_poly = tracker.get_session_statistics("polygon_ws", "sess1")
        stats_db = tracker.get_session_statistics("databento_ws", "sess1")
        assert stats_poly["total_messages"] == 1
        assert stats_db["total_messages"] == 1

    def test_get_all_gaps_returns_list(self, tracker):
        ns = time.time_ns()
        tracker.record_message("polygon_ws", "s", 1, ns)
        tracker.record_message("polygon_ws", "s", 10, ns + 1)
        gaps = tracker.get_all_gaps("polygon_ws")
        assert len(gaps) == 1
        assert gaps[0].estimated_missing_count == 8
