"""Layer 1B — WebSocket Ingestor (Polygon).

Connects to Polygon.io WebSocket, normalizes trades, scores quality,
buffers, and batch-writes to QuestDB + Redis.
"""

from __future__ import annotations

import asyncio
import json
import time
import threading
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer0.config import ConfigRegistry
    from layer0.secrets import SecretsManager
    from layer0.alerts import AlertManager
    from layer1b.questdb_client import QuestDBClient
    from layer1b.redis_client import RedisStreamClient
    from layer1b.data_quality import DataQualityScorer
    from layer1b.gap_tracker import GapTracker

logger = get_logger(__name__)


class IngestorError(Exception):
    """Base exception for ingestor errors."""

class AuthenticationError(IngestorError):
    """Polygon authentication failed."""

class ConnectionError(IngestorError):
    """WebSocket connection failed."""

class SubscriptionError(IngestorError):
    """WebSocket subscription failed."""


class PolygonWebSocketIngestor:
    """Hardened Polygon.io WebSocket ingestor with exponential backoff reconnect."""

    def __init__(
        self,
        config: "ConfigRegistry",
        secrets_manager: "SecretsManager",
        questdb_client: "QuestDBClient",
        redis_client: "RedisStreamClient",
        data_quality_scorer: "DataQualityScorer",
        gap_tracker: "GapTracker",
        alert_manager: "AlertManager",
    ) -> None:
        self._cfg = config
        self._secrets = secrets_manager
        self._qdb = questdb_client
        self._redis = redis_client
        self._dq = data_quality_scorer
        self._gaps = gap_tracker
        self._alert = alert_manager

        self._ws_url = config.get("websocket.polygon_url", "wss://socket.polygon.io/stocks")
        self._batch_size = int(config.get("data_lake.bronze.write_batch_size", 500))
        self._batch_timeout_ms = int(config.get("data_lake.bronze.write_batch_timeout_ms", 100))
        self._reconnect_max = int(config.get("websocket.reconnect_max_attempts_before_critical", 10))

        self._write_buffer: list[dict] = []
        self._last_flush_ts: float = time.monotonic()
        self._stats = {
            "connection_status": "DISCONNECTED",
            "ticks_received_today": 0,
            "ticks_written_today": 0,
            "ticks_rejected_today": 0,
            "buffer_current_size": 0,
            "last_tick_ts_recv": 0,
            "reconnection_count": 0,
            "current_backoff_seconds": 1,
        }
        self._running = False
        self._ws = None

    def start(self) -> None:
        """Start the ingestor event loop (blocking)."""
        self._running = True
        asyncio.run(self._connect_with_retry())

    def stop(self) -> None:
        self._running = False
        self._flush_buffer()
        self._qdb.stop()
        logger.debug("PolygonWebSocketIngestor stopped")

    async def _connect_with_retry(self) -> None:
        import websockets
        backoff = 1
        failures = 0
        while self._running:
            try:
                self._stats["current_backoff_seconds"] = backoff
                await self._connect()
                backoff = 1
                failures = 0
            except Exception as exc:
                failures += 1
                self._stats["reconnection_count"] += 1
                logger.warning("Polygon WS disconnected (%s). Retrying in %ss", exc, backoff)
                if failures >= self._reconnect_max:
                    self._alert.send_critical(
                        "PolygonWebSocketIngestor",
                        f"Failed to reconnect after {failures} attempts",
                    )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _connect(self) -> None:
        import websockets

        # Get API key
        try:
            api_key = self._secrets.get("polygon.api_key")
        except Exception:
            logger.error("polygon.api_key not in vault. Add it: python -m layer0.secrets set polygon.api_key YOUR_KEY_HERE")
            raise AuthenticationError("polygon.api_key not found in vault")

        self._stats["connection_status"] = "CONNECTING"
        async with websockets.connect(self._ws_url) as ws:
            self._ws = ws
            self._stats["connection_status"] = "CONNECTED"
            # Auth handshake
            auth_msg = json.dumps({"action": "auth", "params": api_key})
            await ws.send(auth_msg)
            await self._message_loop(ws)

    async def _message_loop(self, ws) -> None:
        import websockets
        async for message in ws:
            ts_recv_ns = time.time_ns()
            try:
                events = json.loads(message)
                if isinstance(events, list):
                    for event in events:
                        ev = event.get("ev")
                        if ev == "connected":
                            logger.debug("Polygon WS: connected")
                        elif ev == "auth_success":
                            await ws.send(json.dumps({"action": "subscribe", "params": "T.*"}))
                        elif ev == "auth_failed":
                            raise AuthenticationError("Polygon authentication failed")
                        elif ev == "T":
                            event["_ts_recv_ns"] = ts_recv_ns
                            self._stats["ticks_received_today"] += 1
                            self._stats["last_tick_ts_recv"] = ts_recv_ns
                            await self._process_event(event)
            except AuthenticationError:
                raise
            except Exception as exc:
                logger.error("Message loop error: %s", exc)

    async def _process_event(self, event: dict) -> None:
        tick, score = self._dq.score(event, "polygon_ws")
        if score == 0:
            self._stats["ticks_rejected_today"] += 1
            return
        tick["data_quality_score"] = score
        self._write_buffer.append(tick)
        self._redis.write_tick_to_stream(tick)
        self._stats["buffer_current_size"] = len(self._write_buffer)

        should_flush = (
            len(self._write_buffer) >= self._batch_size
            or (time.monotonic() - self._last_flush_ts) * 1000 >= self._batch_timeout_ms
        )
        if should_flush:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._write_buffer:
            return
        now_us = time.time_ns() // 1000
        for t in self._write_buffer:
            t["ts_db_write"] = now_us
        written = self._qdb.write_ticks_batch(self._write_buffer)
        if written > 0:
            self._stats["ticks_written_today"] += written
            self._write_buffer.clear()
        # On failure: retain buffer (will retry next flush)
        self._last_flush_ts = time.monotonic()
        self._stats["buffer_current_size"] = len(self._write_buffer)

    def get_statistics(self) -> dict:
        return dict(self._stats)
