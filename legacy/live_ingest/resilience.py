"""
live_ingest/resilience.py
─────────────────────────
Operational health / monitoring utilities consolidated from three micro-modules:
  - FreezeController  (was freeze.py)
  - GapLogger         (was gap_logger.py)
  - LatencyWindow     (was metrics.py)

All three are small (57 lines total) and share the same conceptual domain:
"things that track and record system health during live ingestion".
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# FreezeController — writes / clears a freeze marker file
# ---------------------------------------------------------------------------

class FreezeController:
    """
    Writes `outputs/live/freeze.flag` when the ingest service is frozen
    (e.g. heartbeat timeout) and removes it when reconnected.
    """

    def __init__(self, *, root_dir: Path) -> None:
        self.path = Path(root_dir) / "outputs" / "live" / "freeze.flag"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def freeze(self, *, reason: str) -> None:
        payload = f"frozen_at={datetime.now(timezone.utc).isoformat()} reason={reason}\n"
        self.path.write_text(payload, encoding="utf-8")

    def unfreeze(self) -> None:
        if self.path.exists():
            self.path.unlink()


# ---------------------------------------------------------------------------
# GapLogger — appends gap / latency events to gaps.jsonl
# ---------------------------------------------------------------------------

class GapLogger:
    """
    Appends JSONL records to `outputs/live/gaps.jsonl` for every notable event:
    heartbeat timeout, latency alert, type2 backfill applied, etc.
    """

    def __init__(self, *, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self.path = self.root_dir / "outputs" / "live" / "gaps.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("logged_at", datetime.now(timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True) + "\n")


# ---------------------------------------------------------------------------
# LatencyWindow — rolling p95 latency tracker
# ---------------------------------------------------------------------------

@dataclass
class LatencyWindow:
    """
    Fixed-size ring buffer of Redis write latencies (ms).
    Computes p95 on demand.
    """
    maxlen: int = 600

    def __post_init__(self) -> None:
        self._values: deque[float] = deque(maxlen=self.maxlen)

    def add(self, value: float) -> None:
        self._values.append(float(value))

    def p95(self) -> float | None:
        if not self._values:
            return None
        arr = sorted(self._values)
        idx = int(round((len(arr) - 1) * 0.95))
        return arr[idx]
