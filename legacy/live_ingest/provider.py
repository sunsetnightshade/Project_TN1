from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterable

from .heartbeat import HeartbeatWatchdog
from .models import TickEvent


class ProviderError(RuntimeError):
    pass


class HeartbeatTimeoutError(ProviderError):
    pass


class ProviderProtocolError(ProviderError):
    pass


def _parse_timestamp(raw: Any) -> datetime:
    if raw is None:
        return datetime.now(timezone.utc)

    if isinstance(raw, (int, float)):
        value = float(raw)
        if value > 1_000_000_000_000_000:  # ns
            return datetime.fromtimestamp(value / 1_000_000_000.0, tz=timezone.utc)
        if value > 1_000_000_000_000:  # us
            return datetime.fromtimestamp(value / 1_000_000.0, tz=timezone.utc)
        if value > 10_000_000_000:  # ms
            return datetime.fromtimestamp(value / 1_000.0, tz=timezone.utc)
        return datetime.fromtimestamp(value, tz=timezone.utc)

    if isinstance(raw, str):
        try:
            if raw.endswith("Z"):
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return datetime.fromisoformat(raw)
        except ValueError:
            return datetime.now(timezone.utc)

    return datetime.now(timezone.utc)


class RealtimeProvider(ABC):
    def __init__(
        self,
        *,
        heartbeat_seconds: float,
        pong_timeout_seconds: float,
    ) -> None:
        self.heartbeat_seconds = heartbeat_seconds
        self.pong_timeout_seconds = pong_timeout_seconds

    @abstractmethod
    async def stream_ticks(self, symbols: Iterable[str]) -> AsyncIterator[TickEvent | None]:
        raise NotImplementedError


class TwelveDataProvider(RealtimeProvider):
    def __init__(
        self,
        *,
        api_key: str,
        heartbeat_seconds: float,
        pong_timeout_seconds: float,
    ) -> None:
        super().__init__(
            heartbeat_seconds=heartbeat_seconds,
            pong_timeout_seconds=pong_timeout_seconds,
        )
        self.api_key = api_key
        self.url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={api_key}"

    async def stream_ticks(self, symbols: Iterable[str]) -> AsyncIterator[TickEvent | None]:
        import websockets  # type: ignore

        wanted = set(symbols)
        subscribe_msg = {
            "action": "subscribe",
            "params": {"symbols": ",".join(sorted(wanted))},
        }

        async with websockets.connect(
            self.url,
            ping_interval=None,
            ping_timeout=None,
            max_queue=10000,
        ) as ws:
            await ws.send(json.dumps(subscribe_msg))
            watchdog = HeartbeatWatchdog.create(
                interval_seconds=self.heartbeat_seconds,
                pong_timeout_seconds=self.pong_timeout_seconds,
                now=time.monotonic(),
            )

            while True:
                now = time.monotonic()
                if watchdog.should_send_ping(now):
                    pong_waiter = await ws.ping()
                    watchdog.mark_ping_sent(now)
                    try:
                        await asyncio.wait_for(pong_waiter, timeout=self.pong_timeout_seconds)
                    except asyncio.TimeoutError as exc:
                        raise HeartbeatTimeoutError("TwelveData pong timeout") from exc
                    watchdog.mark_pong_seen(time.monotonic())

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    if watchdog.has_timed_out(time.monotonic()):
                        raise HeartbeatTimeoutError("TwelveData heartbeat timeout")
                    yield None
                    continue

                watchdog.mark_pong_seen(time.monotonic())
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    continue

                status = str(payload.get("status") or "").lower()
                if status in {"error", "failed", "fail"}:
                    raise ProviderProtocolError(f"TwelveData protocol error: {payload}")

                symbol = str(payload.get("symbol") or payload.get("s") or "").strip()
                if not symbol or symbol not in wanted:
                    continue

                raw_price = payload.get("price")
                if raw_price is None:
                    raw_price = payload.get("p")
                if raw_price is None:
                    continue

                try:
                    price = float(raw_price)
                except (TypeError, ValueError):
                    continue

                raw_size = payload.get("size", 0.0)
                try:
                    size = float(raw_size)
                except (TypeError, ValueError):
                    size = 0.0

                ts = _parse_timestamp(payload.get("timestamp") or payload.get("t"))
                yield TickEvent(
                    symbol=symbol,
                    price=price,
                    size=size,
                    timestamp=ts,
                    source="twelvedata",
                )


class PolygonProvider(RealtimeProvider):
    def __init__(
        self,
        *,
        api_key: str,
        heartbeat_seconds: float,
        pong_timeout_seconds: float,
    ) -> None:
        super().__init__(
            heartbeat_seconds=heartbeat_seconds,
            pong_timeout_seconds=pong_timeout_seconds,
        )
        self.api_key = api_key
        self.url = "wss://socket.polygon.io/stocks"

    async def stream_ticks(self, symbols: Iterable[str]) -> AsyncIterator[TickEvent | None]:
        import websockets  # type: ignore

        wanted = set(symbols)
        trades = ",".join([f"T.{s}" for s in sorted(wanted)])

        async with websockets.connect(
            self.url,
            ping_interval=None,
            ping_timeout=None,
            max_queue=10000,
        ) as ws:
            await ws.send(json.dumps({"action": "auth", "params": self.api_key}))
            await ws.send(json.dumps({"action": "subscribe", "params": trades}))

            watchdog = HeartbeatWatchdog.create(
                interval_seconds=self.heartbeat_seconds,
                pong_timeout_seconds=self.pong_timeout_seconds,
                now=time.monotonic(),
            )

            while True:
                now = time.monotonic()
                if watchdog.should_send_ping(now):
                    pong_waiter = await ws.ping()
                    watchdog.mark_ping_sent(now)
                    try:
                        await asyncio.wait_for(pong_waiter, timeout=self.pong_timeout_seconds)
                    except asyncio.TimeoutError as exc:
                        raise HeartbeatTimeoutError("Polygon pong timeout") from exc
                    watchdog.mark_pong_seen(time.monotonic())

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    if watchdog.has_timed_out(time.monotonic()):
                        raise HeartbeatTimeoutError("Polygon heartbeat timeout")
                    yield None
                    continue

                watchdog.mark_pong_seen(time.monotonic())
                payload = json.loads(raw)
                events: list[dict[str, Any]] = []
                if isinstance(payload, list):
                    events = [x for x in payload if isinstance(x, dict)]
                elif isinstance(payload, dict):
                    events = [payload]

                for event in events:
                    if event.get("ev") == "status":
                        status = str(event.get("status") or "").lower()
                        message = str(event.get("message") or "")
                        if status in {"auth_failed", "error", "failed"}:
                            raise ProviderProtocolError(
                                f"Polygon protocol error: status={status} message={message}"
                            )
                        continue

                    if event.get("ev") != "T":
                        continue

                    symbol = str(event.get("sym") or "").strip()
                    if not symbol or symbol not in wanted:
                        continue

                    raw_price = event.get("p")
                    if raw_price is None:
                        continue

                    try:
                        price = float(raw_price)
                    except (TypeError, ValueError):
                        continue

                    raw_size = event.get("s", 0.0)
                    try:
                        size = float(raw_size)
                    except (TypeError, ValueError):
                        size = 0.0

                    ts = _parse_timestamp(event.get("t"))
                    yield TickEvent(
                        symbol=symbol,
                        price=price,
                        size=size,
                        timestamp=ts,
                        source="polygon",
                    )


# ---------------------------------------------------------------------------
# Public factory — shared by runner.py and bakeoff.py (single source of truth)
# ---------------------------------------------------------------------------

def build_provider(
    cfg: "LiveIngestConfig",
    *,
    provider_name: str | None = None,
) -> RealtimeProvider:
    """
    Construct the correct RealtimeProvider from a LiveIngestConfig.

    Args:
        cfg: live ingest configuration
        provider_name: override provider name (default: cfg.provider)

    Raises:
        ValueError: if provider is not supported or API key is missing
    """
    from config import LiveIngestConfig  # local import to avoid circular

    name = (provider_name or cfg.provider).lower()
    if name == "twelvedata":
        return TwelveDataProvider(
            api_key=str(cfg.twelvedata_api_key),
            heartbeat_seconds=cfg.heartbeat_seconds,
            pong_timeout_seconds=cfg.pong_timeout_seconds,
        )
    if name == "polygon":
        return PolygonProvider(
            api_key=str(cfg.polygon_api_key),
            heartbeat_seconds=cfg.heartbeat_seconds,
            pong_timeout_seconds=cfg.pong_timeout_seconds,
        )
    raise ValueError(f"Unsupported provider: '{name}'. Allowed: twelvedata, polygon")
