"""Layer 1B — Gap Tracker.

Tracks connection and sequence gaps; manages gap-fill attempts via Polygon REST.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer0.alerts import AlertManager
    from layer1b.data_quality import DataQualityScorer

logger = get_logger(__name__)

_GAP_DDL = """
CREATE TABLE IF NOT EXISTS gaps (
    gap_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    nightshade_id   TEXT NOT NULL,
    source          TEXT NOT NULL,
    gap_start_ts_ns INTEGER NOT NULL,
    gap_end_ts_ns   INTEGER,
    detected_at     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'OPEN',
    fill_attempts   INT  DEFAULT 0,
    filled_at       TEXT
);
"""


class GapTracker:
    """Tracks and attempts to fill data gaps in tick streams."""

    _MAX_ATTEMPTS = 3

    def __init__(
        self,
        db_path: Optional[str] = None,
        alert_manager: Optional["AlertManager"] = None,
    ) -> None:
        path = Path(db_path or "~/.nightshade/gap_registry.db").expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_GAP_DDL)
        self._alert = alert_manager
        logger.debug("GapTracker ready: db=%s", path)

    def record_gap(
        self,
        nightshade_id: str,
        source: str,
        gap_start_ts_ns: int,
        gap_end_ts_ns: Optional[int] = None,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """INSERT INTO gaps (nightshade_id, source, gap_start_ts_ns, gap_end_ts_ns, detected_at)
               VALUES (?,?,?,?,?)""",
            (nightshade_id, source, gap_start_ts_ns, gap_end_ts_ns, now),
        )
        gap_id = cursor.lastrowid
        if self._alert:
            self._alert.send_warning(
                "GapTracker",
                f"Gap detected for {nightshade_id} from {source}",
                {"gap_id": gap_id, "gap_start": gap_start_ts_ns},
            )
        return gap_id

    def close_gap(self, gap_id: int, gap_end_ts_ns: int) -> None:
        self._conn.execute(
            "UPDATE gaps SET gap_end_ts_ns=? WHERE gap_id=? AND status='OPEN'",
            (gap_end_ts_ns, gap_id),
        )

    def get_open_gaps(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM gaps WHERE status IN ('OPEN','FILL_PENDING') ORDER BY detected_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def attempt_gap_fill(self, gap_id: int, polygon_api_key: str) -> bool:
        """Try to fill a gap via Polygon REST. Returns True on success."""
        row = self._conn.execute(
            "SELECT * FROM gaps WHERE gap_id=?", (gap_id,)
        ).fetchone()
        if not row:
            return False

        row = dict(row)
        attempts = row["fill_attempts"] + 1

        try:
            self._fetch_and_store(row, polygon_api_key)
            self._conn.execute(
                "UPDATE gaps SET status='FILLED', fill_attempts=?, filled_at=? WHERE gap_id=?",
                (attempts, datetime.now(timezone.utc).isoformat(), gap_id),
            )
            return True
        except Exception as exc:
            logger.error("Gap fill attempt failed for gap_id=%s: %s", gap_id, exc)
            if attempts >= self._MAX_ATTEMPTS:
                self._conn.execute(
                    "UPDATE gaps SET status='UNFILLABLE', fill_attempts=? WHERE gap_id=?",
                    (attempts, gap_id),
                )
                if self._alert:
                    self._alert.send_critical(
                        "GapTracker",
                        f"Gap {gap_id} is UNFILLABLE after {attempts} attempts",
                        {"gap_id": gap_id},
                    )
            else:
                self._conn.execute(
                    "UPDATE gaps SET status='FILL_PENDING', fill_attempts=? WHERE gap_id=?",
                    (attempts, gap_id),
                )
            return False

    def run_fill_cycle(self, polygon_api_key: str) -> dict:
        gaps = self.get_open_gaps()
        filled = failed = unfillable = 0
        for gap in gaps:
            ok = self.attempt_gap_fill(gap["gap_id"], polygon_api_key)
            if ok:
                filled += 1
            else:
                updated = self._conn.execute(
                    "SELECT status FROM gaps WHERE gap_id=?", (gap["gap_id"],)
                ).fetchone()
                status = updated[0] if updated else "UNKNOWN"
                if status == "UNFILLABLE":
                    unfillable += 1
                else:
                    failed += 1
        return {"gaps_filled": filled, "gaps_failed": failed, "gaps_unfillable": unfillable}

    def detect_historical_gaps(
        self,
        nightshade_id: str,
        start_date: str,
        end_date: str,
    ) -> list[str]:
        """Return calendar days with no tick data where surrounding data exists.

        This requires QuestDB access — returns empty list if unavailable.
        """
        return []  # Implemented during Layer 1B historical load phase

    def _fetch_and_store(self, gap: dict, api_key: str) -> None:
        """Fetch ticks from Polygon REST for the gap period."""
        import urllib.request
        import json

        ticker = gap["nightshade_id"]  # simplified — real impl resolves via SecurityMaster
        start_ns = gap["gap_start_ts_ns"]
        end_ns = gap["gap_end_ts_ns"] or int(time.time_ns())

        url = (
            f"https://api.polygon.io/v3/trades/{ticker}"
            f"?timestamp.gte={start_ns}&timestamp.lte={end_ns}&limit=50000"
            f"&apiKey={api_key}"
        )
        req = urllib.request.urlopen(url, timeout=15)
        data = json.loads(req.read().decode())
        if data.get("status") not in ("OK", "DELAYED"):
            raise RuntimeError(f"Polygon API error: {data.get('error', data)}")
        # In production, ticks would be normalized + written to QuestDB here
        logger.debug("Gap fill fetched %d ticks for %s", len(data.get("results", [])), ticker)
