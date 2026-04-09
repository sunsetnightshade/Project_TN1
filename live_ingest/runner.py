from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from pathlib import Path

from config import LiveIngestConfig
from live_persistence import HourlyParquetJanitor

from .aggregator import MinuteBarAggregator
from .provider import PolygonProvider, RealtimeProvider, TwelveDataProvider, build_provider
from .redis_streams import RedisBarProducer
from .service import LiveIngestService
from .sinks import BarSink
from .zmq_fallback import ZeroMQBarProducer



def _build_sink(cfg: LiveIngestConfig) -> BarSink:
    if cfg.bar_sink == "redis":
        return RedisBarProducer(
            host=cfg.redis_host,
            port=cfg.redis_port,
            db=cfg.redis_db,
            stream_60=cfg.redis_stream_60,
            stream_500=cfg.redis_stream_500,
            maxlen_60=cfg.redis_maxlen_60,
            maxlen_500=cfg.redis_maxlen_500,
        )
    if cfg.bar_sink == "zeromq":
        return ZeroMQBarProducer(endpoint=cfg.zmq_endpoint)
    raise ValueError(f"Unsupported bar sink: {cfg.bar_sink}")


async def _telemetry_printer(service: LiveIngestService, interval_seconds: float = 5.0) -> None:
    while True:
        print(json.dumps(service.snapshot(), indent=2))
        await asyncio.sleep(interval_seconds)


async def _janitor_loop(cfg: LiveIngestConfig, root_dir: Path) -> None:
    janitor = HourlyParquetJanitor(cfg=cfg, root_dir=root_dir)
    while True:
        result = await asyncio.to_thread(janitor.persist_recent, hours=cfg.janitor_lookback_hours)
        payload = {
            "janitor": {
                "rows_scanned": result.total_rows_scanned,
                "rows_written": result.rows_written,
                "rows_deduped": result.deduped_rows,
                "files_written": [str(p) for p in result.files_written],
            }
        }
        print(json.dumps(payload, indent=2))
        await asyncio.sleep(cfg.janitor_interval_seconds)


async def run_live_ingest_forever(cfg: LiveIngestConfig, *, root_dir: Path | None = None) -> None:
    root = Path(".") if root_dir is None else Path(root_dir)
    provider = build_provider(cfg)
    producer = _build_sink(cfg)
    aggregator = MinuteBarAggregator(price_scale=cfg.price_scale, source=cfg.provider)
    service = LiveIngestService(
        provider=provider,
        producer=producer,
        aggregator=aggregator,
        symbols=cfg.symbols,
        reconnect_backoff_seconds=cfg.reconnect_backoff_seconds,
        root_dir=root,
        latency_alert_ms=cfg.latency_alert_ms,
        type2_parquet_root=cfg.type2_parquet_root,
        price_scale=cfg.price_scale,
    )

    await producer.connect()
    telemetry_task = asyncio.create_task(_telemetry_printer(service))
    janitor_task = None
    if cfg.enable_hourly_janitor and cfg.bar_sink == "redis":
        janitor_task = asyncio.create_task(_janitor_loop(cfg, root))
    try:
        await service.run_forever()
    finally:
        telemetry_task.cancel()
        with suppress(asyncio.CancelledError):
            await telemetry_task
        if janitor_task is not None:
            janitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await janitor_task
        await producer.close()
