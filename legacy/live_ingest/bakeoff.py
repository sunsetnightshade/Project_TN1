from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import LiveIngestConfig

from .aggregator import MinuteBarAggregator
from .provider import (
    HeartbeatTimeoutError,
    PolygonProvider,
    ProviderError,
    ProviderProtocolError,
    RealtimeProvider,
    TwelveDataProvider,
    build_provider,
)


@dataclass(frozen=True)
class ProviderScore:
    provider: str
    seconds_requested: int
    seconds_observed: float
    status: str
    ticks_received: int
    bars_emitted: int
    symbols_seen: int
    symbol_coverage_ratio: float
    late_ticks_dropped: int
    errors: int
    heartbeat_timeouts: int
    protocol_errors: int
    unexpected_errors: int



async def _run_one(cfg: LiveIngestConfig, *, provider_name: str, seconds: int) -> ProviderScore:
    # Skip providers with no key in environment.
    if provider_name == "twelvedata" and not cfg.twelvedata_api_key:
        return ProviderScore(
            provider=provider_name,
            seconds_requested=seconds,
            seconds_observed=0.0,
            status="skipped_missing_key",
            ticks_received=0,
            bars_emitted=0,
            symbols_seen=0,
            symbol_coverage_ratio=0.0,
            late_ticks_dropped=0,
            errors=0,
            heartbeat_timeouts=0,
            protocol_errors=0,
            unexpected_errors=0,
        )
    if provider_name == "polygon" and not cfg.polygon_api_key:
        return ProviderScore(
            provider=provider_name,
            seconds_requested=seconds,
            seconds_observed=0.0,
            status="skipped_missing_key",
            ticks_received=0,
            bars_emitted=0,
            symbols_seen=0,
            symbol_coverage_ratio=0.0,
            late_ticks_dropped=0,
            errors=0,
            heartbeat_timeouts=0,
            protocol_errors=0,
            unexpected_errors=0,
        )

    provider = build_provider(cfg, provider_name=provider_name)
    agg = MinuteBarAggregator(price_scale=cfg.price_scale, source=provider_name)

    symbols_seen: set[str] = set()
    ticks_received = 0
    bars_emitted = 0
    heartbeat_timeouts = 0
    protocol_errors = 0
    unexpected_errors = 0
    status = "ok"

    start = time.monotonic()
    deadline = start + max(1, seconds)

    try:
        async for tick in provider.stream_ticks(cfg.symbols):
            now = time.monotonic()
            if now >= deadline:
                break

            if tick is None:
                bars = agg.finalize_due(now=datetime.now(timezone.utc))
                bars_emitted += len(bars)
                continue

            ticks_received += 1
            symbols_seen.add(tick.symbol)
            bars = agg.ingest_tick(tick)
            bars.extend(agg.finalize_due(now=datetime.now(timezone.utc)))
            bars_emitted += len(bars)
    except HeartbeatTimeoutError:
        heartbeat_timeouts += 1
        status = "heartbeat_timeout"
    except ProviderProtocolError:
        protocol_errors += 1
        status = "protocol_error"
    except ProviderError:
        unexpected_errors += 1
        status = "provider_error"
    except Exception:
        unexpected_errors += 1
        status = "unexpected_error"

    # Flush any remaining open minute buckets.
    bars_emitted += len(agg.flush())

    observed = max(0.0, time.monotonic() - start)
    errors = heartbeat_timeouts + protocol_errors + unexpected_errors
    coverage = len(symbols_seen) / float(max(1, len(cfg.symbols)))

    return ProviderScore(
        provider=provider_name,
        seconds_requested=seconds,
        seconds_observed=observed,
        status=status,
        ticks_received=ticks_received,
        bars_emitted=bars_emitted,
        symbols_seen=len(symbols_seen),
        symbol_coverage_ratio=coverage,
        late_ticks_dropped=agg.late_ticks_dropped,
        errors=errors,
        heartbeat_timeouts=heartbeat_timeouts,
        protocol_errors=protocol_errors,
        unexpected_errors=unexpected_errors,
    )


def _winner(scores: list[ProviderScore]) -> str | None:
    eligible = [s for s in scores if s.status != "skipped_missing_key"]
    if not eligible:
        return None

    def key(s: ProviderScore) -> tuple[int, float, int, int, int]:
        # Lower errors and late ticks are better; higher coverage/ticks are better.
        return (
            -s.errors,
            s.symbol_coverage_ratio,
            s.ticks_received,
            s.bars_emitted,
            -s.late_ticks_dropped,
        )

    best = sorted(eligible, key=key, reverse=True)[0]
    return best.provider


async def run_bakeoff(
    *,
    cfg: LiveIngestConfig,
    root_dir: Path,
    seconds_per_provider: int,
) -> tuple[list[ProviderScore], Path]:
    providers = ["twelvedata", "polygon"]
    scores: list[ProviderScore] = []

    for name in providers:
        score = await _run_one(cfg, provider_name=name, seconds=seconds_per_provider)
        scores.append(score)
        await asyncio.sleep(1.0)

    winner = _winner(scores)

    root = Path(root_dir)
    out_dir = root / "outputs" / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"provider_bakeoff_{stamp}.json"

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seconds_per_provider": int(seconds_per_provider),
        "winner": winner,
        "scores": [asdict(s) for s in scores],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return scores, out_path
