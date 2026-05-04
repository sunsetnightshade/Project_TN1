from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.timestamps import parse_iso_utc

from .aggregator import MinuteBarAggregator
from .resilience import FreezeController, GapLogger, LatencyWindow
from .provider import HeartbeatTimeoutError, RealtimeProvider
from .sinks import BarSink
from .type2_fallback import Type2FallbackVerifier


class LiveIngestService:
    def __init__(
        self,
        *,
        provider: RealtimeProvider,
        producer: BarSink,
        aggregator: MinuteBarAggregator,
        symbols: tuple[str, ...],
        reconnect_backoff_seconds: float,
        root_dir: Path,
        latency_alert_ms: float,
        type2_parquet_root: str,
        price_scale: int,
    ) -> None:
        self.provider = provider
        self.producer = producer
        self.aggregator = aggregator
        self.symbols = symbols
        self.reconnect_backoff_seconds = reconnect_backoff_seconds
        self.latency_alert_ms = latency_alert_ms
        self.price_scale = price_scale

        self.state: str = "connecting"
        self.last_error: str | None = None
        self.ticks_received: int = 0
        self.bars_written: int = 0
        self.freeze_events: int = 0
        self.reconnect_attempts: int = 0
        self.last_bar_end: str | None = None
        self.last_redis_write_ms: float | None = None
        self.max_redis_write_ms: float = 0.0
        self.latency_p95_ms: float | None = None

        self._freeze = FreezeController(root_dir=root_dir)
        self._gaps = GapLogger(root_dir=root_dir)
        self._type2 = Type2FallbackVerifier(
            root_dir=root_dir,
            parquet_root=type2_parquet_root,
        )
        self._latency = LatencyWindow(maxlen=600)

    async def _write_bars(self, bars: list[Any]) -> None:
        for bar in bars:
            write_ms = await self.producer.write_bar(bar)
            self._latency.add(write_ms)
            self.last_redis_write_ms = write_ms
            self.max_redis_write_ms = max(self.max_redis_write_ms, write_ms)
            self.latency_p95_ms = self._latency.p95()
            self.bars_written += 1
            self.last_bar_end = bar.bar_end.isoformat()

            if write_ms > self.latency_alert_ms:
                self._gaps.write(
                    {
                        "event": "latency_alert",
                        "symbol": bar.symbol,
                        "bar_end": bar.bar_end.isoformat(),
                        "write_ms": write_ms,
                        "threshold_ms": self.latency_alert_ms,
                    }
                )

    def snapshot(self) -> dict[str, Any]:
        ingest_lag_ms: float | None = None
        if self.last_bar_end is not None:
            try:
                bar_end = parse_iso_utc(self.last_bar_end)
                now = datetime.now(timezone.utc)
                ingest_lag_ms = max(0.0, (now - bar_end).total_seconds() * 1000.0)
            except ValueError:
                ingest_lag_ms = None

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": self.state,
            "last_error": self.last_error,
            "ticks_received": self.ticks_received,
            "bars_written": self.bars_written,
            "freeze_events": self.freeze_events,
            "reconnect_attempts": self.reconnect_attempts,
            "last_bar_end": self.last_bar_end,
            "ingest_lag_ms": ingest_lag_ms,
            "last_redis_write_ms": self.last_redis_write_ms,
            "max_redis_write_ms": self.max_redis_write_ms,
            "p95_redis_write_ms": self.latency_p95_ms,
            "late_ticks_dropped": self.aggregator.late_ticks_dropped,
        }

    async def run_forever(self) -> None:
        while True:
            try:
                self.state = "connecting"
                self._freeze.unfreeze()
                async for tick in self.provider.stream_ticks(self.symbols):
                    self.state = "connected"
                    if tick is None:
                        due = self.aggregator.finalize_due(now=datetime.now(timezone.utc))
                        await self._write_bars(due)
                        continue

                    self.ticks_received += 1
                    bars = self.aggregator.ingest_tick(tick)
                    bars.extend(self.aggregator.finalize_due(now=datetime.now(timezone.utc)))
                    await self._write_bars(bars)
            except HeartbeatTimeoutError as exc:
                self.state = "frozen"
                self.freeze_events += 1
                self.last_error = repr(exc)
                self._freeze.freeze(reason="heartbeat_timeout")
                result = self._type2.verify_market_state(
                    reason="heartbeat_timeout",
                    last_bar_end=self.last_bar_end,
                )
                self._gaps.write(
                    {
                        "event": "heartbeat_timeout",
                        "error": repr(exc),
                        "state": self.state,
                        "last_bar_end": self.last_bar_end,
                        "type2_ok": result.ok,
                        "type2_note": result.note,
                        "type2_checked_at": result.checked_at.isoformat(),
                    }
                )

                backfill = self._type2.backfill_bars(
                    symbols=self.symbols,
                    after_bar_end=self.last_bar_end,
                    until=datetime.now(timezone.utc),
                    price_scale=self.price_scale,
                )
                if backfill:
                    await self._write_bars(backfill)
                    self._gaps.write(
                        {
                            "event": "type2_backfill_applied",
                            "count": len(backfill),
                            "after_bar_end": self.last_bar_end,
                        }
                    )

                await asyncio.sleep(self.reconnect_backoff_seconds)
                self.state = "reconnecting"
                self.reconnect_attempts += 1
            except Exception as exc:  # noqa: BLE001
                self.state = "frozen"
                self.freeze_events += 1
                self.last_error = repr(exc)
                self._freeze.freeze(reason="producer_or_provider_error")
                self._gaps.write(
                    {
                        "event": "runtime_exception",
                        "error": repr(exc),
                        "state": self.state,
                        "last_bar_end": self.last_bar_end,
                    }
                )
                await asyncio.sleep(self.reconnect_backoff_seconds)
                self.state = "reconnecting"
                self.reconnect_attempts += 1
