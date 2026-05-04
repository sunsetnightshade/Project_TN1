"""Layer 1B — QuestDB Client.

Wraps ILP writes (high throughput) and psycopg2 queries (analytical reads).
All methods are connection-resilient: one reconnect attempt on failure.
Never raises from write methods — catches all exceptions and alerts.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer0.config import ConfigRegistry
    from layer0.alerts import AlertManager

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class QuestDBError(Exception):
    """Base exception for QuestDB client errors."""

class QuestDBConnectionError(QuestDBError):
    """Cannot connect to QuestDB."""

class QuestDBWriteError(QuestDBError):
    """ILP write failure."""

class QuestDBQueryError(QuestDBError):
    """SQL query failure."""


# ---------------------------------------------------------------------------
# QuestDBClient
# ---------------------------------------------------------------------------

class QuestDBClient:
    """Dual-protocol QuestDB client: ILP for writes, psycopg2 for queries."""

    def __init__(
        self,
        config: Optional["ConfigRegistry"] = None,
        alert_manager: Optional["AlertManager"] = None,
    ) -> None:
        if config:
            self._host = config.get("database.questdb.host", "localhost")
            self._ilp_port = int(config.get("database.questdb.ilp_port", 9009))
            self._pg_port = int(config.get("database.questdb.pg_port", 8812))
            self._http_port = int(config.get("database.questdb.http_port", 9000))
        else:
            self._host = "localhost"
            self._ilp_port = 9009
            self._pg_port = 8812
            self._http_port = 9000

        self._alert = alert_manager
        self._ilp_sock: Optional[socket.socket] = None
        self._pg_conn = None
        self._lock = threading.Lock()
        self._last_health_check_ok = False

        # Background health-check thread
        self._running = True
        self._health_thread = threading.Thread(
            target=self._health_loop, daemon=True, name="questdb-health"
        )
        self._health_thread.start()

    # ------------------------------------------------------------------
    # ILP Writes
    # ------------------------------------------------------------------

    def write_tick(self, tick: dict) -> None:
        """Write a single tick via ILP. Never raises."""
        try:
            line = self._tick_to_ilp(tick)
            self._ilp_write(line)
        except Exception as exc:
            logger.error("write_tick failed: %s", exc)
            if self._alert:
                self._alert.send_critical("QuestDBClient", f"Tick write failed: {exc}")

    def write_ticks_batch(self, ticks: list[dict]) -> int:
        """Write ticks as a batch. Returns count written."""
        if not ticks:
            return 0
        try:
            lines = "\n".join(self._tick_to_ilp(t) for t in ticks) + "\n"
            self._ilp_write(lines)
            return len(ticks)
        except Exception as exc:
            logger.error("write_ticks_batch failed: %s", exc)
            if self._alert:
                self._alert.send_critical("QuestDBClient", f"Batch write failed: {exc}")
            return 0

    # ------------------------------------------------------------------
    # SQL Queries (psycopg2)
    # ------------------------------------------------------------------

    def query(self, sql: str, params: Optional[tuple] = None) -> list[dict]:
        """Execute SQL via PostgreSQL wire protocol. One reconnect on failure."""
        try:
            return self._execute(sql, params)
        except Exception:
            self._reconnect_pg()
            return self._execute(sql, params)

    def query_ticks(
        self,
        nightshade_id: str,
        start_ts_ns: int,
        end_ts_ns: int,
        min_quality_score: int = 0,
    ) -> list[dict]:
        sql = (
            "SELECT * FROM ticks_bronze "
            "WHERE nightshade_id=? AND ts_event BETWEEN ? AND ? "
            "AND data_quality_score >= ? ORDER BY ts_event ASC"
        )
        return self.query(sql, (nightshade_id, start_ts_ns, end_ts_ns, min_quality_score))

    def query_bars(
        self,
        nightshade_id: str,
        bar_interval: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        sql = (
            "SELECT * FROM bars_silver "
            "WHERE nightshade_id=? AND bar_interval=? "
            "AND ts_bar_open BETWEEN ? AND ? ORDER BY ts_bar_open ASC"
        )
        return self.query(sql, (nightshade_id, bar_interval, start_date, end_date))

    def query_features(
        self,
        nightshade_id: str,
        feature_names: list[str],
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        placeholders = ",".join("?" * len(feature_names))
        sql = (
            f"SELECT * FROM features_gold "
            f"WHERE nightshade_id=? AND feature_name IN ({placeholders}) "
            f"AND ts_feature BETWEEN ? AND ? ORDER BY ts_feature ASC"
        )
        return self.query(sql, (nightshade_id, *feature_names, start_date, end_date))

    def get_latest_tick_timestamp(self, nightshade_id: str) -> Optional[int]:
        try:
            rows = self.query(
                "SELECT max(ts_event) as latest FROM ticks_bronze WHERE nightshade_id=?",
                (nightshade_id,),
            )
            if rows and rows[0]["latest"]:
                return rows[0]["latest"]
            return None
        except Exception:
            return None

    def get_table_row_count(self, table_name: str) -> int:
        try:
            rows = self.query(f"SELECT count() as cnt FROM {table_name}")
            return rows[0]["cnt"] if rows else 0
        except Exception:
            return -1

    def health_check(self) -> dict:
        result = {
            "ilp_healthy": False,
            "pg_healthy": False,
            "ilp_latency_ms": -1.0,
            "pg_latency_ms": -1.0,
        }
        # ILP health: TCP connect
        t0 = time.monotonic()
        try:
            s = socket.create_connection((self._host, self._ilp_port), timeout=2)
            s.close()
            result["ilp_healthy"] = True
            result["ilp_latency_ms"] = (time.monotonic() - t0) * 1000
        except Exception:
            pass

        # PG health: simple query
        t0 = time.monotonic()
        try:
            self.query("SELECT 1")
            result["pg_healthy"] = True
            result["pg_latency_ms"] = (time.monotonic() - t0) * 1000
        except Exception:
            pass

        return result

    def stop(self) -> None:
        self._running = False
        if self._ilp_sock:
            try:
                self._ilp_sock.close()
            except Exception:
                pass
        if self._pg_conn:
            try:
                self._pg_conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ilp_write(self, payload: str) -> None:
        with self._lock:
            if self._ilp_sock is None:
                self._connect_ilp()
            try:
                self._ilp_sock.sendall(payload.encode("utf-8"))
            except Exception:
                self._ilp_sock = None
                self._connect_ilp()
                self._ilp_sock.sendall(payload.encode("utf-8"))

    def _connect_ilp(self) -> None:
        s = socket.create_connection((self._host, self._ilp_port), timeout=5)
        self._ilp_sock = s

    def _execute(self, sql: str, params: Optional[tuple]) -> list[dict]:
        if self._pg_conn is None:
            self._reconnect_pg()
        cursor = self._pg_conn.cursor()
        cursor.execute(sql, params or ())
        if cursor.description:
            cols = [desc[0] for desc in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        return []

    def _reconnect_pg(self) -> None:
        try:
            import psycopg2
            self._pg_conn = psycopg2.connect(
                host=self._host,
                port=self._pg_port,
                database="qdb",
                user="admin",
                password="quest",
                connect_timeout=5,
            )
            self._pg_conn.autocommit = True
        except Exception as exc:
            raise QuestDBConnectionError(f"Cannot connect to QuestDB PG wire: {exc}") from exc

    def _health_loop(self) -> None:
        while self._running:
            time.sleep(60)
            try:
                h = self.health_check()
                self._last_health_check_ok = h["pg_healthy"]
            except Exception:
                pass

    @staticmethod
    def _tick_to_ilp(tick: dict) -> str:
        """Convert tick dict to ILP line protocol string."""
        nightshade_id = tick.get("nightshade_id", "")
        source = tick.get("source", "")
        exchange = tick.get("exchange", "")
        ts_event_ns = int(tick.get("ts_event_ns", tick.get("ts_event", 0)))

        fields = [
            f"ts_recv={int(tick.get('ts_recv_ns', tick.get('ts_recv', 0)))}i",
            f"ts_db_write={int(tick.get('ts_db_write', 0))}i",
            f"price_fixed={int(tick.get('price_fixed', 0))}i",
            f"size={int(tick.get('size', 0))}i",
            f"conditions={int(tick.get('conditions_bitmask', tick.get('conditions', 0)))}i",
            f"data_quality_score={int(tick.get('data_quality_score', 0))}i",
        ]
        # ILP: measurement,tag=val field=val timestamp(ns)
        tags = f"nightshade_id={nightshade_id},source={source},exchange={exchange}"
        return f"ticks_bronze,{tags} {','.join(fields)} {ts_event_ns}"
