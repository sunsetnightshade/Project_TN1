"""Layer 1A — Security Master.

Manages instrument identity, symbol mappings across sources, and
resolves external symbols to permanent Nightshade IDs.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer0.config import ConfigRegistry
    from layer0.alerts import AlertManager

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SecurityMasterError(Exception):
    """Base exception for SecurityMaster errors."""

class SecurityMasterSchemaError(SecurityMasterError):
    """Raised when the DB schema is invalid or migration failed."""

class InstrumentNotFoundError(SecurityMasterError):
    """Raised when a nightshade_id does not exist."""

class InvalidInstrumentTypeError(SecurityMasterError):
    """Raised when instrument_type is not in the allowed set."""

class InvalidExchangeCodeError(SecurityMasterError):
    """Raised when exchange code is not a 4-letter uppercase MIC."""

class InvalidCurrencyCodeError(SecurityMasterError):
    """Raised when currency code is not a 3-letter uppercase ISO 4217 code."""

class InvalidFieldError(SecurityMasterError):
    """Raised when trying to update a protected field."""

class SymbolNotFoundError(SecurityMasterError):
    """Raised when resolve() finds no matching mapping."""

class AmbiguousSymbolError(SecurityMasterError):
    """Raised when resolve() finds multiple active mappings."""


# ---------------------------------------------------------------------------
# SecurityMaster
# ---------------------------------------------------------------------------

_ALLOWED_TYPES = {"EQUITY", "ETF", "FUTURE", "CRYPTO", "INDEX"}
_PROTECTED_FIELDS = {"nightshade_id", "instrument_type", "primary_exchange", "currency",
                     "listed_date", "created_at"}

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=FULL;

CREATE TABLE IF NOT EXISTS instruments (
    nightshade_id       TEXT PRIMARY KEY,
    instrument_type     TEXT NOT NULL,
    primary_exchange    TEXT NOT NULL,
    currency            TEXT NOT NULL,
    name                TEXT NOT NULL,
    sector              TEXT,
    industry            TEXT,
    is_active           INT  DEFAULT 1,
    listed_date         TEXT,
    delisted_date       TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS symbol_mappings (
    mapping_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    nightshade_id   TEXT NOT NULL REFERENCES instruments(nightshade_id),
    source          TEXT NOT NULL,
    external_id     TEXT NOT NULL,
    effective_from  TEXT NOT NULL,
    effective_to    TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE (source, external_id, effective_from)
);
CREATE INDEX IF NOT EXISTS idx_sm_source_ext ON symbol_mappings (source, external_id);

CREATE TABLE IF NOT EXISTS universe_memberships (
    membership_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    universe_name   TEXT NOT NULL,
    nightshade_id   TEXT NOT NULL REFERENCES instruments(nightshade_id),
    added_date      TEXT NOT NULL,
    removed_date    TEXT,
    removal_reason  TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE (universe_name, nightshade_id, added_date)
);
CREATE INDEX IF NOT EXISTS idx_um_universe ON universe_memberships (universe_name, added_date, removed_date);

CREATE TABLE IF NOT EXISTS corporate_actions (
    action_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    nightshade_id       TEXT NOT NULL REFERENCES instruments(nightshade_id),
    action_type         TEXT NOT NULL,
    ex_date             TEXT NOT NULL,
    record_date         TEXT,
    pay_date            TEXT,
    adjustment_factor   REAL NOT NULL,
    raw_value           REAL,
    raw_value_unit      TEXT,
    notes               TEXT,
    data_source         TEXT NOT NULL,
    is_applied          INT DEFAULT 0,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ca_instrument ON corporate_actions (nightshade_id, ex_date);
CREATE INDEX IF NOT EXISTS idx_ca_applied ON corporate_actions (is_applied, ex_date);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class SecurityMaster:
    """Institutional-grade instrument identity store backed by SQLite."""

    def __init__(
        self,
        config: Optional["ConfigRegistry"] = None,
        alert_manager: Optional["AlertManager"] = None,
    ) -> None:
        if config is not None:
            db_path = Path(str(config.get("security_master.db_path", "~/.nightshade/security_master.db"))).expanduser()
        else:
            db_path = Path("~/.nightshade/security_master.db").expanduser()

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._alert = alert_manager

        self.conn = sqlite3.connect(
            str(db_path), check_same_thread=False, isolation_level=None
        )
        self.conn.row_factory = sqlite3.Row
        self._apply_schema()
        logger.debug("SecurityMaster ready: db=%s", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _apply_schema(self) -> None:
        try:
            for statement in _DDL.strip().split(";"):
                stmt = statement.strip()
                if stmt:
                    self.conn.execute(stmt)
        except sqlite3.Error as exc:
            raise SecurityMasterSchemaError(f"Schema error: {exc}") from exc

    # ------------------------------------------------------------------
    # Instruments
    # ------------------------------------------------------------------

    def add_instrument(
        self,
        instrument_type: str,
        primary_exchange: str,
        currency: str,
        name: str,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        listed_date: Optional[str] = None,
    ) -> str:
        self._validate_type(instrument_type)
        self._validate_exchange(primary_exchange)
        self._validate_currency(currency)

        nightshade_id = str(uuid.uuid4())
        now = _utcnow()
        self.conn.execute(
            """INSERT INTO instruments
               (nightshade_id, instrument_type, primary_exchange, currency, name,
                sector, industry, is_active, listed_date, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,1,?,?,?)""",
            (nightshade_id, instrument_type, primary_exchange, currency, name,
             sector, industry, listed_date, now, now),
        )
        logger.debug("add_instrument: %s (%s)", name, nightshade_id)
        return nightshade_id

    def get_instrument(self, nightshade_id: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM instruments WHERE nightshade_id=?", (nightshade_id,)
        ).fetchone()
        if row is None:
            raise InstrumentNotFoundError(f"Instrument not found: {nightshade_id!r}")
        return dict(row)

    def update_instrument(self, nightshade_id: str, **kwargs) -> None:
        for field in kwargs:
            if field in _PROTECTED_FIELDS:
                raise InvalidFieldError(f"Cannot update protected field: {field!r}")
        self._assert_exists(nightshade_id)
        updates = ", ".join(f"{k}=?" for k in kwargs)
        values = list(kwargs.values()) + [_utcnow(), nightshade_id]
        self.conn.execute(
            f"UPDATE instruments SET {updates}, updated_at=? WHERE nightshade_id=?",
            values,
        )

    def deactivate_instrument(
        self,
        nightshade_id: str,
        delisted_date: str,
        reason: str = "DELISTED",
    ) -> None:
        self._assert_exists(nightshade_id)
        now = _utcnow()
        self.conn.execute(
            "UPDATE instruments SET is_active=0, delisted_date=?, updated_at=? WHERE nightshade_id=?",
            (delisted_date, now, nightshade_id),
        )
        # Close all active mappings
        self.conn.execute(
            "UPDATE symbol_mappings SET effective_to=? WHERE nightshade_id=? AND effective_to IS NULL",
            (now, nightshade_id),
        )
        if self._alert:
            self._alert.send_info("SecurityMaster", f"Instrument deactivated: {nightshade_id}", {"reason": reason})

    def search_instruments(
        self,
        query: str,
        instrument_type: Optional[str] = None,
        exchange: Optional[str] = None,
        active_only: bool = True,
    ) -> list[dict]:
        sql = "SELECT * FROM instruments WHERE name LIKE ?"
        params: list = [f"%{query}%"]
        if instrument_type:
            sql += " AND instrument_type=?"
            params.append(instrument_type)
        if exchange:
            sql += " AND primary_exchange=?"
            params.append(exchange)
        if active_only:
            sql += " AND is_active=1"
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_statistics(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) FROM instruments").fetchone()[0]
        active = self.conn.execute("SELECT COUNT(*) FROM instruments WHERE is_active=1").fetchone()[0]
        total_mappings = self.conn.execute("SELECT COUNT(*) FROM symbol_mappings").fetchone()[0]
        active_mappings = self.conn.execute("SELECT COUNT(*) FROM symbol_mappings WHERE effective_to IS NULL").fetchone()[0]
        by_type = {r[0]: r[1] for r in self.conn.execute("SELECT instrument_type, COUNT(*) FROM instruments GROUP BY instrument_type").fetchall()}
        by_exchange = {r[0]: r[1] for r in self.conn.execute("SELECT primary_exchange, COUNT(*) FROM instruments GROUP BY primary_exchange").fetchall()}
        return {
            "total_instruments": total,
            "active_instruments": active,
            "inactive_instruments": total - active,
            "total_mappings": total_mappings,
            "active_mappings": active_mappings,
            "by_type": by_type,
            "by_exchange": by_exchange,
        }

    # ------------------------------------------------------------------
    # Symbol mappings
    # ------------------------------------------------------------------

    def add_symbol_mapping(
        self,
        nightshade_id: str,
        source: str,
        external_id: str,
        effective_from: Optional[str] = None,
    ) -> None:
        self._assert_exists(nightshade_id)
        if effective_from is None:
            effective_from = _utcnow()

        # Close any existing active mapping for same (source, external_id)
        self.conn.execute(
            """UPDATE symbol_mappings SET effective_to=?
               WHERE source=? AND external_id=? AND effective_to IS NULL
               AND nightshade_id != ?""",
            (effective_from, source, external_id, nightshade_id),
        )
        try:
            self.conn.execute(
                """INSERT INTO symbol_mappings
                   (nightshade_id, source, external_id, effective_from, created_at)
                   VALUES (?,?,?,?,?)""",
                (nightshade_id, source, external_id, effective_from, _utcnow()),
            )
        except sqlite3.IntegrityError:
            pass  # Duplicate (idempotent)

    def resolve(
        self,
        source: str,
        external_id: str,
        at_time: Optional[str] = None,
    ) -> str:
        if at_time is None:
            at_time = _utcnow()
        rows = self.conn.execute(
            """SELECT nightshade_id FROM symbol_mappings
               WHERE source=? AND external_id=?
               AND effective_from <= ?
               AND (effective_to IS NULL OR effective_to > ?)""",
            (source, external_id, at_time, at_time),
        ).fetchall()
        if not rows:
            raise SymbolNotFoundError(f"Symbol not found: {source}/{external_id} at {at_time}")
        if len(rows) > 1:
            raise AmbiguousSymbolError(f"Ambiguous symbol: {source}/{external_id} at {at_time}")
        return rows[0][0]

    def get_all_mappings(self, nightshade_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM symbol_mappings WHERE nightshade_id=? ORDER BY effective_from",
            (nightshade_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _assert_exists(self, nightshade_id: str) -> None:
        row = self.conn.execute(
            "SELECT 1 FROM instruments WHERE nightshade_id=?", (nightshade_id,)
        ).fetchone()
        if row is None:
            raise InstrumentNotFoundError(f"Instrument not found: {nightshade_id!r}")

    @staticmethod
    def _validate_type(instrument_type: str) -> None:
        if instrument_type not in _ALLOWED_TYPES:
            raise InvalidInstrumentTypeError(
                f"Invalid instrument type: {instrument_type!r}. Allowed: {_ALLOWED_TYPES}"
            )

    @staticmethod
    def _validate_exchange(code: str) -> None:
        if not (len(code) == 4 and code.isupper() and code.isalpha()):
            raise InvalidExchangeCodeError(
                f"Exchange code must be 4 uppercase letters: {code!r}"
            )

    @staticmethod
    def _validate_currency(code: str) -> None:
        if not (len(code) == 3 and code.isupper() and code.isalpha()):
            raise InvalidCurrencyCodeError(
                f"Currency code must be 3 uppercase letters: {code!r}"
            )
