"""Layer 1B — Redis Stream Client.

Uses connection pooling and XADD/XREADGROUP for durable tick streaming.
Never raises from write methods.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer0.config import ConfigRegistry
    from layer0.alerts import AlertManager

logger = get_logger(__name__)


class RedisClientError(Exception):
    """Base exception for Redis client errors."""

class RedisConnectionError(RedisClientError):
    """Cannot connect to Redis."""

class RedisStreamError(RedisClientError):
    """Stream operation failure."""


class RedisStreamClient:
    """Redis stream client for real-time tick distribution to consumer groups."""

    def __init__(
        self,
        config: Optional["ConfigRegistry"] = None,
        alert_manager: Optional["AlertManager"] = None,
    ) -> None:
        import redis as redis_lib

        if config:
            host = config.get("database.redis.host", "localhost")
            port = int(config.get("database.redis.port", 6379))
            db = int(config.get("database.redis.db", 0))
            self._max_stream_len = int(config.get("database.redis.max_stream_len", 100000))
        else:
            host, port, db = "localhost", 6379, 0
            self._max_stream_len = 100000

        self._alert = alert_manager
        self._pool = redis_lib.ConnectionPool(
            host=host, port=port, db=db,
            max_connections=20,
            socket_connect_timeout=5,
            decode_responses=True,
        )
        self._client = redis_lib.Redis(connection_pool=self._pool)

        # Background health check
        self._running = True
        self._health_ok = False
        self._health_thread = threading.Thread(
            target=self._health_loop, daemon=True, name="redis-health"
        )
        self._health_thread.start()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_tick_to_stream(self, tick: dict) -> str:
        """XADD tick to stream. Never raises. Returns stream entry ID."""
        try:
            nid = tick.get("nightshade_id", "UNKNOWN")
            key = f"ticks:{nid}"
            entry_id = self._client.xadd(
                key,
                tick,
                maxlen=self._max_stream_len,
                approximate=True,
            )
            return entry_id or ""
        except Exception as exc:
            logger.error("write_tick_to_stream failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def create_consumer_group(
        self,
        nightshade_id: str,
        consumer_group: str,
        start_from: str = "$",
    ) -> None:
        key = f"ticks:{nightshade_id}"
        try:
            self._client.xgroup_create(key, consumer_group, id=start_from, mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" in str(exc):
                pass  # Already exists
            else:
                logger.error("create_consumer_group failed: %s", exc)

    def read_ticks_from_stream(
        self,
        nightshade_id: str,
        consumer_group: str,
        consumer_name: str,
        count: int = 100,
        block_ms: int = 1000,
    ) -> list[dict]:
        key = f"ticks:{nightshade_id}"
        try:
            result = self._client.xreadgroup(
                consumer_group, consumer_name, {key: ">"}, count=count, block=block_ms
            )
            ticks = []
            if result:
                for _key, entries in result:
                    for stream_id, fields in entries:
                        fields["stream_id"] = stream_id
                        ticks.append(fields)
            return ticks
        except Exception as exc:
            logger.error("read_ticks_from_stream failed: %s", exc)
            return []

    def acknowledge_ticks(
        self,
        nightshade_id: str,
        consumer_group: str,
        stream_ids: list[str],
    ) -> int:
        key = f"ticks:{nightshade_id}"
        try:
            return self._client.xack(key, consumer_group, *stream_ids)
        except Exception as exc:
            logger.error("acknowledge_ticks failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_stream_info(self, nightshade_id: str) -> dict:
        key = f"ticks:{nightshade_id}"
        try:
            info = self._client.xinfo_stream(key)
            groups = self._client.xinfo_groups(key)
            return {
                "length": info.get("length", 0),
                "first_entry_id": info.get("first-entry", [None])[0] if info.get("first-entry") else None,
                "last_entry_id": info.get("last-entry", [None])[0] if info.get("last-entry") else None,
                "consumer_groups": [g.get("name") for g in groups],
            }
        except Exception as exc:
            logger.error("get_stream_info failed: %s", exc)
            return {}

    def get_pending_count(self, nightshade_id: str, consumer_group: str) -> int:
        key = f"ticks:{nightshade_id}"
        try:
            pending = self._client.xpending(key, consumer_group)
            return pending.get("pending", 0)
        except Exception:
            return 0

    def get_all_stream_keys(self) -> list[str]:
        try:
            return [k for k in self._client.scan_iter("ticks:*")]
        except Exception:
            return []

    def flush_stream(self, nightshade_id: str) -> None:
        """Delete all entries. Sends WARNING in non-test environments."""
        key = f"ticks:{nightshade_id}"
        try:
            if self._alert:
                self._alert.send_warning("RedisStreamClient", f"Flushing stream: {key}")
            self._client.delete(key)
        except Exception as exc:
            logger.error("flush_stream failed: %s", exc)

    def health_check(self) -> dict:
        t0 = time.monotonic()
        try:
            self._client.ping()
            latency_ms = (time.monotonic() - t0) * 1000
            info = self._client.info("memory")
            keys = self._client.scan_iter("ticks:*")
            key_list = list(keys)
            return {
                "connected": True,
                "latency_ms": round(latency_ms, 2),
                "memory_used_mb": round(info.get("used_memory", 0) / 1_048_576, 2),
                "total_streams": len(key_list),
                "total_stream_entries": 0,  # too expensive to count all
            }
        except Exception as exc:
            return {"connected": False, "error": str(exc)}

    def stop(self) -> None:
        self._running = False

    def _health_loop(self) -> None:
        while self._running:
            time.sleep(30)
            try:
                self._client.ping()
                self._health_ok = True
            except Exception:
                self._health_ok = False
