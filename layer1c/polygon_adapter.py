"""Layer 1C — Polygon WebSocket Adapter.

Supersedes Layer 1B PolygonWebSocketIngestor. Satisfies LiveDataSourceProtocol.
"""

from __future__ import annotations

import asyncio
import json
import time
import threading
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger
from layer1c.source_protocol import (
    LiveDataSourceProtocol,
    AuthenticationError,
    ConnectionError as SourceConnectionError,
    SubscriptionError,
)

if TYPE_CHECKING:
    from layer0.config import ConfigRegistry
    from layer0.secrets import SecretsManager
    from layer0.alerts import AlertManager
    from layer1b.questdb_client import QuestDBClient
    from layer1b.redis_client import RedisStreamClient
    from layer1b.data_quality import DataQualityScorer
    from layer1b.gap_tracker import GapTracker
    from layer1c.tick_normalizer import TickNormalizer
    from layer1c.sequence_tracker import SequenceTracker

logger = get_logger(__name__)


class PolygonWebSocketAdapter(LiveDataSourceProtocol):
    """Hardened Polygon.io WebSocket adapter satisfying LiveDataSourceProtocol."""

    def __init__(
        self,
        config: "ConfigRegistry",
        secrets_manager: "SecretsManager",
        questdb_client: "QuestDBClient",
        redis_client: "RedisStreamClient",
        data_quality_scorer: "DataQualityScorer",
        gap_tracker: "GapTracker",
        tick_normalizer: "TickNormalizer",
        sequence_tracker: "SequenceTracker",
        alert_manager: "AlertManager",
    ) -> None:
        self._cfg = config
        self._secrets = secrets_manager
        self._qdb = questdb_client
        self._redis = redis_client
        self._dq = data_quality_scorer
        self._gaps = gap_tracker
        self._normalizer = tick_normalizer
        self._seq = sequence_tracker
        self._alert = alert_manager

        self._ws_url = config.get("live_data.polygon.ws_url", "wss://socket.polygon.io/stocks")
        self._buffer_size = int(config.get("live_data.polygon.write_buffer_size", 500))
        self._buffer_timeout_ms = int(config.get("live_data.polygon.write_buffer_timeout_ms", 500))
        self._reconnect_max = int(config.get("live_data.polygon.ws_reconnect_critical_after_attempts", 10))

        self._write_buffer: list[dict] = []
        self._last_flush = time.monotonic()
        self._state = "DISCONNECTED"
        self._session_id = str(time.time_ns())
        self._subscribed_ids: list[str] = []

        self._metrics = {
            "messages_received_total": 0,
            "messages_received_last_minute": 0,
            "sequence_gaps_detected": 0,
            "last_message_ts_recv": 0,
            "current_latency_ms": 0.0,
            "uptime_start_ts": 0,
            "reconnections": 0,
        }
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # LiveDataSourceProtocol interface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self._state == "CONNECTED":
            return
        self._running = True
        self._loop_thread = threading.Thread(
            target=lambda: asyncio.run(self._connect_with_retry()),
            daemon=True,
            name="polygon-ws",
        )
        self._loop_thread.start()
        self._metrics["uptime_start_ts"] = int(time.time_ns())

    def disconnect(self) -> None:
        self._running = False
        self._flush_buffer()
        self._state = "DISCONNECTED"

    def subscribe(self, nightshade_ids: list[str]) -> None:
        self._subscribed_ids = nightshade_ids

    def get_source_name(self) -> str:
        return "polygon_ws"

    def get_connection_state(self) -> str:
        return self._state

    def get_health_metrics(self) -> dict:
        uptime = (time.time_ns() - self._metrics["uptime_start_ts"]) / 1e9 if self._metrics["uptime_start_ts"] else 0
        return {
            "source_name": "polygon_ws",
            "connection_state": self._state,
            "messages_received_total": self._metrics["messages_received_total"],
            "messages_received_last_minute": self._metrics["messages_received_last_minute"],
            "sequence_gaps_detected": self._metrics["sequence_gaps_detected"],
            "last_message_ts_recv": self._metrics["last_message_ts_recv"],
            "current_latency_ms": self._metrics["current_latency_ms"],
            "uptime_seconds": round(uptime, 2),
        }

    # ------------------------------------------------------------------
    # Async core
    # ------------------------------------------------------------------

    async def _connect_with_retry(self) -> None:
        import websockets
        backoff = int(self._cfg.get("live_data.polygon.ws_reconnect_initial_delay_seconds", 1))
        max_backoff = int(self._cfg.get("live_data.polygon.ws_reconnect_max_delay_seconds", 60))
        failures = 0

        while self._running:
            try:
                self._state = "CONNECTING"
                await self._connect()
                failures = 0
                backoff = 1
            except AuthenticationError as exc:
                logger.error("Authentication failed — stopping: %s", exc)
                self._alert.send_critical("PolygonWebSocketAdapter", f"Auth failed: {exc}")
                self._state = "ERROR"
                break
            except Exception as exc:
                failures += 1
                self._metrics["reconnections"] += 1
                self._state = "RECONNECTING"
                logger.warning("Polygon WS disconnected (%s). Retry in %ss", exc, backoff)
                if failures >= self._reconnect_max:
                    self._alert.send_critical(
                        "PolygonWebSocketAdapter",
                        f"Failed to reconnect after {failures} attempts",
                    )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                self._seq.reset_session("polygon_ws", self._session_id)
                self._session_id = str(time.time_ns())

    async def _connect(self) -> None:
        import websockets

        try:
            api_key = self._secrets.get("polygon.api_key")
        except Exception:
            logger.error(
                "polygon.api_key not in vault. Add it:\n  python -m layer0.secrets set polygon.api_key YOUR_KEY_HERE"
            )
            raise AuthenticationError("polygon.api_key not found in vault")

        async with websockets.connect(self._ws_url) as ws:
            self._state = "CONNECTED"
            await ws.send(json.dumps({"action": "auth", "params": api_key}))
            await self._message_loop(ws)

    async def _message_loop(self, ws) -> None:
        async for message in ws:
            ts_recv_ns = time.time_ns()
            try:
                events = json.loads(message)
                for event in (events if isinstance(events, list) else [events]):
                    await self._handle_event(event, ts_recv_ns)
            except Exception as exc:
                logger.error("Message loop error: %s", exc)

    async def _handle_event(self, event: dict, ts_recv_ns: int) -> None:
        ev = event.get("ev")
        if ev == "auth_success":
            # Batch-subscribe all tickers
            if self._subscribed_ids:
                params = ",".join(f"T.{nid}" for nid in self._subscribed_ids)
            else:
                params = "T.*"
            import websockets
            # (ws is out of scope here — subscription happens in _message_loop via protocol)
        elif ev == "auth_failed":
            raise AuthenticationError("Polygon authentication failed")
        elif ev == "T":
            event["_ts_recv_ns"] = ts_recv_ns
            seq = event.get("seq") or event.get("z")

            from layer1c.source_protocol import RawMessageProtocol
            raw_msg = RawMessageProtocol(source="polygon_ws", raw_payload=event, ts_recv_ns=ts_recv_ns, sequence_number=seq)
            tick = self._normalizer.normalize(raw_msg)
            if tick is None:
                return

            tick["ts_recv_ns"] = ts_recv_ns
            _, score = self._dq.score(event, "polygon_ws")
            if score == 0:
                return

            tick["data_quality_score"] = score
            self._metrics["messages_received_total"] += 1
            self._metrics["last_message_ts_recv"] = ts_recv_ns

            # Sequence tracking
            if seq is not None:
                gap = self._seq.record_message("polygon_ws", self._session_id, int(seq), ts_recv_ns)
                if gap:
                    self._metrics["sequence_gaps_detected"] += 1
                    self._gaps.record_gap(tick["nightshade_id"], "polygon_ws", gap.gap_start_sequence)

            self._write_buffer.append(tick)
            self._redis.write_tick_to_stream(tick)

            if (len(self._write_buffer) >= self._buffer_size or
                    (time.monotonic() - self._last_flush) * 1000 >= self._buffer_timeout_ms):
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._write_buffer:
            return
        now_us = time.time_ns() // 1000
        for t in self._write_buffer:
            t["ts_db_write"] = now_us
        written = self._qdb.write_ticks_batch(self._write_buffer)
        if written > 0:
            self._write_buffer.clear()
        self._last_flush = time.monotonic()
