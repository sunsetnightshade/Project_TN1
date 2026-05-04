"""Layer 1A — Universe Manager.

Tracks membership of instruments in named universes with full point-in-time
history to prevent survivorship bias.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer1a.security_master import SecurityMaster

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UniverseError(Exception):
    """Base exception for Universe errors."""

class AlreadyInUniverseError(UniverseError):
    """Raised when adding an instrument already in the universe."""

class NotInUniverseError(UniverseError):
    """Raised when removing an instrument that is not a member."""

class UniverseNotFoundError(UniverseError):
    """Raised when querying a universe that has never been defined."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class UniverseManager:
    """Point-in-time universe membership manager.

    Shares the SecurityMaster's SQLite connection to avoid lock conflicts.
    """

    def __init__(self, security_master: "SecurityMaster") -> None:
        self._sm = security_master
        self._conn = security_master.conn

    def add_to_universe(
        self,
        universe_name: str,
        nightshade_id: str,
        added_date: str,
    ) -> None:
        self._sm._assert_exists(nightshade_id)
        existing = self._conn.execute(
            """SELECT membership_id FROM universe_memberships
               WHERE universe_name=? AND nightshade_id=? AND removed_date IS NULL""",
            (universe_name, nightshade_id),
        ).fetchone()
        if existing:
            raise AlreadyInUniverseError(
                f"{nightshade_id!r} is already an active member of {universe_name!r}"
            )
        self._conn.execute(
            """INSERT INTO universe_memberships
               (universe_name, nightshade_id, added_date, created_at)
               VALUES (?,?,?,?)""",
            (universe_name, nightshade_id, added_date, _utcnow()),
        )
        logger.debug("add_to_universe: %s → %s", nightshade_id, universe_name)

    def remove_from_universe(
        self,
        universe_name: str,
        nightshade_id: str,
        removed_date: str,
        reason: str = "MANUAL_REMOVAL",
    ) -> None:
        from layer0.alerts import AlertSeverity
        row = self._conn.execute(
            """SELECT membership_id FROM universe_memberships
               WHERE universe_name=? AND nightshade_id=? AND removed_date IS NULL""",
            (universe_name, nightshade_id),
        ).fetchone()
        if row is None:
            raise NotInUniverseError(
                f"{nightshade_id!r} is not an active member of {universe_name!r}"
            )
        self._conn.execute(
            """UPDATE universe_memberships
               SET removed_date=?, removal_reason=?
               WHERE membership_id=?""",
            (removed_date, reason, row[0]),
        )
        if reason == "DELISTED" and hasattr(self._sm, "_alert") and self._sm._alert:
            self._sm._alert.send_warning(
                "UniverseManager",
                f"Delisted instrument removed from universe: {nightshade_id}",
                {"universe": universe_name},
            )
        logger.debug("remove_from_universe: %s from %s (reason=%s)", nightshade_id, universe_name, reason)

    def get_universe_at_date(self, universe_name: str, as_of_date: str) -> list[str]:
        """Point-in-time query — prevents survivorship bias."""
        rows = self._conn.execute(
            """SELECT nightshade_id FROM universe_memberships
               WHERE universe_name=?
               AND added_date <= ?
               AND (removed_date > ? OR removed_date IS NULL)""",
            (universe_name, as_of_date, as_of_date),
        ).fetchall()
        return [r[0] for r in rows]

    def get_current_universe(self, universe_name: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT nightshade_id FROM universe_memberships WHERE universe_name=? AND removed_date IS NULL",
            (universe_name,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_universe_history(self, universe_name: str) -> list[dict]:
        rows = self._conn.execute(
            """SELECT * FROM universe_memberships WHERE universe_name=? ORDER BY added_date""",
            (universe_name,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_membership_record(self, universe_name: str, nightshade_id: str) -> list[dict]:
        rows = self._conn.execute(
            """SELECT * FROM universe_memberships
               WHERE universe_name=? AND nightshade_id=? ORDER BY added_date""",
            (universe_name, nightshade_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_universes(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT universe_name FROM universe_memberships ORDER BY universe_name"
        ).fetchall()
        return [r[0] for r in rows]

    def get_universe_size_over_time(
        self,
        universe_name: str,
        start_date: str,
        end_date: str,
        frequency: str = "daily",
    ) -> list[dict]:
        """Return list of {date, size} dicts for the given date range."""
        from datetime import date, timedelta

        freq_delta = {"daily": timedelta(days=1), "weekly": timedelta(weeks=1), "monthly": None}
        result = []
        current = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        while current <= end:
            ds = current.isoformat()
            count = len(self.get_universe_at_date(universe_name, ds))
            result.append({"date": ds, "size": count})
            if frequency == "monthly":
                # Advance one month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
            else:
                current += freq_delta.get(frequency, timedelta(days=1))

        return result
