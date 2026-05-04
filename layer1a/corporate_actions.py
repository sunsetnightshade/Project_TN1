"""Layer 1A — Corporate Actions Manager."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer1a.security_master import SecurityMaster

logger = get_logger(__name__)

_ALLOWED_ACTION_TYPES = {
    "SPLIT", "REVERSE_SPLIT", "DIVIDEND_CASH", "DIVIDEND_STOCK",
    "MERGER_ACQUIRED", "MERGER_ACQUIRER", "SPINOFF", "NAME_CHANGE", "TICKER_CHANGE",
}


class CorporateActionsError(Exception):
    """Base exception for corporate action errors."""

class DuplicateCorporateActionError(CorporateActionsError):
    """Raised on duplicate (nightshade_id, action_type, ex_date)."""

class CorporateActionNotFoundError(CorporateActionsError):
    """Raised when action_id does not exist."""

class InvalidActionTypeError(CorporateActionsError):
    """Raised when action_type is not in the allowed set."""

class InvalidAdjustmentFactorError(CorporateActionsError):
    """Raised when adjustment_factor violates type-specific constraints."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class CorporateActionsManager:
    """Manages corporate actions — splits, dividends, mergers, etc.

    Shares the SecurityMaster's SQLite connection.
    """

    def __init__(self, security_master: "SecurityMaster") -> None:
        self._sm = security_master
        self._conn = security_master.conn
        self._alert = security_master._alert

    def add_action(
        self,
        nightshade_id: str,
        action_type: str,
        ex_date: str,
        adjustment_factor: float,
        data_source: str,
        record_date: Optional[str] = None,
        pay_date: Optional[str] = None,
        raw_value: Optional[float] = None,
        raw_value_unit: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        self._sm._assert_exists(nightshade_id)
        if action_type not in _ALLOWED_ACTION_TYPES:
            raise InvalidActionTypeError(f"Invalid action type: {action_type!r}")
        if not isinstance(adjustment_factor, (int, float)) or adjustment_factor <= 0:
            raise InvalidAdjustmentFactorError("adjustment_factor must be a positive float")
        if action_type == "SPLIT" and adjustment_factor >= 1.0:
            raise InvalidAdjustmentFactorError("SPLIT factor must be < 1.0")
        if action_type == "REVERSE_SPLIT" and adjustment_factor <= 1.0:
            raise InvalidAdjustmentFactorError("REVERSE_SPLIT factor must be > 1.0")

        # Duplicate check
        existing = self._conn.execute(
            "SELECT action_id FROM corporate_actions WHERE nightshade_id=? AND action_type=? AND ex_date=?",
            (nightshade_id, action_type, ex_date),
        ).fetchone()
        if existing:
            raise DuplicateCorporateActionError(
                f"Duplicate action: {nightshade_id}/{action_type}/{ex_date}"
            )

        now = _utcnow()
        cursor = self._conn.execute(
            """INSERT INTO corporate_actions
               (nightshade_id, action_type, ex_date, record_date, pay_date,
                adjustment_factor, raw_value, raw_value_unit, notes, data_source,
                is_applied, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?)""",
            (nightshade_id, action_type, ex_date, record_date, pay_date,
             adjustment_factor, raw_value, raw_value_unit, notes, data_source, now, now),
        )
        action_id = cursor.lastrowid
        if self._alert:
            self._alert.send_warning(
                "CorporateActionsManager",
                f"New corporate action requires human verification: {action_type} on {nightshade_id}",
                {"action_id": action_id, "ex_date": ex_date, "factor": adjustment_factor},
            )
        logger.debug("add_action: %s %s ex=%s factor=%s", nightshade_id, action_type, ex_date, adjustment_factor)
        return action_id

    def get_unapplied_actions(self, as_of_date: str) -> list[dict]:
        rows = self._conn.execute(
            """SELECT * FROM corporate_actions
               WHERE is_applied=0 AND ex_date <= ?
               ORDER BY ex_date ASC""",
            (as_of_date,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_action_applied(self, action_id: int) -> None:
        row = self._conn.execute(
            "SELECT action_id FROM corporate_actions WHERE action_id=?", (action_id,)
        ).fetchone()
        if row is None:
            raise CorporateActionNotFoundError(f"Action not found: {action_id}")
        self._conn.execute(
            "UPDATE corporate_actions SET is_applied=1, updated_at=? WHERE action_id=?",
            (_utcnow(), action_id),
        )

    def get_cumulative_adjustment_factor(
        self,
        nightshade_id: str,
        from_date: str,
        to_date: str,
    ) -> float:
        rows = self._conn.execute(
            """SELECT adjustment_factor FROM corporate_actions
               WHERE nightshade_id=? AND ex_date >= ? AND ex_date <= ?
               ORDER BY ex_date""",
            (nightshade_id, from_date, to_date),
        ).fetchall()
        factor = 1.0
        for r in rows:
            factor *= r[0]
        return factor

    def get_actions_for_instrument(
        self,
        nightshade_id: str,
        action_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict]:
        sql = "SELECT * FROM corporate_actions WHERE nightshade_id=?"
        params: list = [nightshade_id]
        if action_type:
            sql += " AND action_type=?"
            params.append(action_type)
        if start_date:
            sql += " AND ex_date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND ex_date <= ?"
            params.append(end_date)
        sql += " ORDER BY ex_date"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def check_for_missed_actions(
        self,
        nightshade_id: str,
        price_series: list[tuple[str, float]],  # [(date_str, close_price), ...]
    ) -> list[str]:
        """Detect overnight price changes >40% not explained by a known corporate action."""
        threshold = 0.40
        suspect_dates: list[str] = []
        for i in range(1, len(price_series)):
            prev_date, prev_price = price_series[i - 1]
            curr_date, curr_price = price_series[i]
            if prev_price == 0:
                continue
            change = abs(curr_price - prev_price) / prev_price
            if change > threshold:
                # Check for known action on curr_date
                known = self._conn.execute(
                    "SELECT 1 FROM corporate_actions WHERE nightshade_id=? AND ex_date=?",
                    (nightshade_id, curr_date),
                ).fetchone()
                if known is None:
                    suspect_dates.append(curr_date)

        if suspect_dates and self._alert:
            self._alert.send_warning(
                "CorporateActionsManager",
                f"Possible missed corporate actions detected for {nightshade_id}",
                {"suspect_dates": suspect_dates},
            )
        return suspect_dates
