"""Tests for layer1b.gap_tracker — GapTracker."""

from __future__ import annotations

import pytest

from layer1b.gap_tracker import GapTracker


@pytest.fixture()
def tracker(tmp_path) -> GapTracker:
    return GapTracker(db_path=str(tmp_path / "gaps.db"))


class TestGapTracker:

    def test_record_creates_open_entry(self, tracker):
        gap_id = tracker.record_gap("AAPL", "polygon_ws", 1_000_000, 2_000_000)
        gaps = tracker.get_open_gaps()
        assert any(g["gap_id"] == gap_id for g in gaps)
        assert gaps[0]["status"] == "OPEN"

    def test_get_open_gaps_filters_correctly(self, tracker):
        tracker.record_gap("AAPL", "polygon_ws", 1_000, 2_000)
        tracker.record_gap("MSFT", "polygon_ws", 3_000, 4_000)
        gaps = tracker.get_open_gaps()
        assert len(gaps) == 2

    def test_run_fill_cycle_marks_unfillable_after_3(self, tracker):
        gap_id = tracker.record_gap("AAPL", "polygon_ws", 1_000, 2_000)
        for _ in range(3):
            tracker.attempt_gap_fill(gap_id, "bad-api-key")
        row = tracker._conn.execute(
            "SELECT status FROM gaps WHERE gap_id=?", (gap_id,)
        ).fetchone()
        assert row[0] == "UNFILLABLE"

    def test_detect_historical_gaps_returns_list(self, tracker):
        result = tracker.detect_historical_gaps("AAPL", "2024-01-01", "2024-01-31")
        assert isinstance(result, list)
