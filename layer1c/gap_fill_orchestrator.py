"""Layer 1C — Gap Fill Orchestrator.

Wraps GapTracker with token-bucket rate limiting, priority scoring,
and a background fill loop.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer0.alerts import AlertManager
    from layer1b.gap_tracker import GapTracker
    from layer1a.universe import UniverseManager
    from layer1c.tick_normalizer import TickNormalizer

logger = get_logger(__name__)


class TokenBucket:
    """Thread-safe token bucket for rate limiting."""

    def __init__(self, rate_per_minute: float, burst: Optional[float] = None) -> None:
        self._rate_per_sec = rate_per_minute / 60.0
        self._tokens = burst if burst is not None else rate_per_minute
        self._max_tokens = burst if burst is not None else rate_per_minute
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def add_tokens(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate_per_sec)
            self._last_refill = now

    def consume(self, count: int = 1) -> bool:
        self.add_tokens()
        with self._lock:
            if self._tokens >= count:
                self._tokens -= count
                return True
            return False

    @property
    def available(self) -> float:
        self.add_tokens()
        return self._tokens


class GapFillOrchestrator:
    """Rate-limited, prioritized gap filler with background fill loop."""

    _FILL_LOOP_INTERVAL_SEC = 10

    def __init__(
        self,
        gap_tracker: "GapTracker",
        universe_manager: Optional["UniverseManager"] = None,
        alert_manager: Optional["AlertManager"] = None,
        rate_per_minute: float = 5.0,
    ) -> None:
        self._gaps = gap_tracker
        self._um = universe_manager
        self._alert = alert_manager
        self._bucket = TokenBucket(rate_per_minute)
        self._polygon_api_key: Optional[str] = None
        self._running = False
        self._fill_thread: Optional[threading.Thread] = None
        self._stats = {
            "total_gaps_processed": 0,
            "total_gaps_filled": 0,
            "total_gaps_failed": 0,
            "total_gaps_unfillable": 0,
            "last_fill_ts": 0,
        }

    def start(self, polygon_api_key: str) -> None:
        self._polygon_api_key = polygon_api_key
        self._running = True
        self._fill_thread = threading.Thread(
            target=self._fill_loop, daemon=True, name="gap-fill"
        )
        self._fill_thread.start()

    def stop(self) -> None:
        self._running = False

    def _fill_loop(self) -> None:
        while self._running:
            self._run_once()
            time.sleep(self._FILL_LOOP_INTERVAL_SEC)

    def _run_once(self) -> None:
        gaps = self._gaps.get_open_gaps()
        if not gaps:
            return

        # Score and sort by priority
        scored = [(self._priority_score(g), g) for g in gaps]
        scored.sort(key=lambda x: x[0], reverse=True)

        for _, gap in scored:
            if not self._bucket.consume(1):
                logger.debug("Rate limit — token bucket empty, deferring gap fill")
                break
            ok = self._gaps.attempt_gap_fill(gap["gap_id"], self._polygon_api_key or "")
            self._stats["total_gaps_processed"] += 1
            if ok:
                self._stats["total_gaps_filled"] += 1
            else:
                # Check if now unfillable
                row = self._gaps._conn.execute(
                    "SELECT status FROM gaps WHERE gap_id=?", (gap["gap_id"],)
                ).fetchone()
                status = row[0] if row else ""
                if status == "UNFILLABLE":
                    self._stats["total_gaps_unfillable"] += 1
                else:
                    self._stats["total_gaps_failed"] += 1
        self._stats["last_fill_ts"] = int(time.time_ns())

    def _priority_score(self, gap: dict) -> int:
        score = 0
        # Recency
        gap_start_ns = gap.get("gap_start_ts_ns", 0)
        age_sec = (time.time_ns() - gap_start_ns) / 1e9
        if age_sec < 3600:
            score += 100
        elif age_sec < 86400:
            score += 50
        else:
            score += 10

        # Universe membership
        if self._um:
            nid = gap.get("nightshade_id", "")
            for uname in (self._um.list_universes() if self._um else []):
                if nid in self._um.get_current_universe(uname):
                    score += 50
                    break

        # Size
        missing = gap.get("gap_end_ts_ns", 0) - gap.get("gap_start_ts_ns", 0)
        if missing < 1000:
            score += 30
        else:
            score += 10

        return score

    def get_statistics(self) -> dict:
        return {
            **self._stats,
            "tokens_available": round(self._bucket.available, 2),
            "queue_depth": len(self._gaps.get_open_gaps()),
            "fill_rate_per_hour": self._stats["total_gaps_filled"],  # simplified
        }
