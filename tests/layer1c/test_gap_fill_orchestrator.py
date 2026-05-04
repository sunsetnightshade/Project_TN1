"""Tests for Gap Fill Orchestrator and Token Bucket."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from layer1c.gap_fill_orchestrator import GapFillOrchestrator, TokenBucket

def test_token_bucket_initialization():
    bucket = TokenBucket(rate_per_minute=60) # 1 token per second
    assert bucket.available == 60

def test_token_bucket_consume():
    bucket = TokenBucket(rate_per_minute=60, burst=5)
    assert bucket.available == 5
    assert bucket.consume(3) is True
    assert bucket.available == pytest.approx(2.0, abs=0.1)
    assert bucket.consume(3) is False # Not enough tokens

def test_token_bucket_refill():
    bucket = TokenBucket(rate_per_minute=60, burst=1)
    bucket.consume(1)
    assert bucket.consume(1) is False
    
    # Simulate time passing
    with patch("time.monotonic", return_value=time.monotonic() + 1.1):
        assert bucket.consume(1) is True

@pytest.fixture
def mock_gap_tracker():
    tracker = MagicMock()
    tracker.get_open_gaps.return_value = [
        {"gap_id": "gap_1", "nightshade_id": "AAPL", "gap_start_ts_ns": time.time_ns(), "gap_end_ts_ns": time.time_ns() + 2000},
        {"gap_id": "gap_2", "nightshade_id": "TSLA", "gap_start_ts_ns": time.time_ns() - 86400 * 1e9, "gap_end_ts_ns": time.time_ns()}
    ]
    tracker.attempt_gap_fill.return_value = True
    return tracker

@pytest.fixture
def mock_um():
    um = MagicMock()
    um.list_universes.return_value = ["TEST"]
    um.get_current_universe.return_value = ["AAPL"]
    return um

def test_gap_fill_orchestrator_priority(mock_gap_tracker, mock_um):
    orch = GapFillOrchestrator(gap_tracker=mock_gap_tracker, universe_manager=mock_um)
    
    gaps = mock_gap_tracker.get_open_gaps()
    # AAPL is recent, in universe, and missing >= 1000ns. Score should be high.
    score_aapl = orch._priority_score(gaps[0])
    # TSLA is old, not in universe, missing large amount. Score should be lower.
    score_tsla = orch._priority_score(gaps[1])
    
    assert score_aapl > score_tsla

def test_gap_fill_orchestrator_run_once(mock_gap_tracker, mock_um):
    orch = GapFillOrchestrator(gap_tracker=mock_gap_tracker, universe_manager=mock_um, rate_per_minute=60)
    # Don't call start() to avoid background thread racing with explicit _run_once()
    
    orch._run_once()
    
    assert mock_gap_tracker.attempt_gap_fill.call_count == 2
    stats = orch.get_statistics()
    assert stats["total_gaps_processed"] == 2
    assert stats["total_gaps_filled"] == 2

def test_gap_fill_orchestrator_rate_limit(mock_gap_tracker, mock_um):
    # Bucket starts with 1 token, can only process 1 gap immediately
    orch = GapFillOrchestrator(gap_tracker=mock_gap_tracker, universe_manager=mock_um, rate_per_minute=1)
    # Patch bucket burst to 1
    orch._bucket = TokenBucket(rate_per_minute=1, burst=1)
    
    orch.start("FAKE_API_KEY")
    orch._run_once()
    
    # Second gap deferred
    assert mock_gap_tracker.attempt_gap_fill.call_count == 1
    stats = orch.get_statistics()
    assert stats["total_gaps_processed"] == 1
    
    orch.stop()
