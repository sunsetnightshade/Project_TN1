from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .models import MinuteBar, TickEvent
from .precision import to_fixed_price


@dataclass
class _Bucket:
    minute_start: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float


class MinuteBarAggregator:
    def __init__(self, *, price_scale: int, source: str) -> None:
        self._price_scale = price_scale
        self._source = source
        self._buckets: dict[str, _Bucket] = {}
        self.late_ticks_dropped: int = 0

    @staticmethod
    def _minute_floor(ts: datetime) -> datetime:
        utc_ts = ts.astimezone(timezone.utc)
        return utc_ts.replace(second=0, microsecond=0)

    def _finalize(self, symbol: str, bucket: _Bucket) -> MinuteBar:
        bar_start = bucket.minute_start
        bar_end = bar_start + timedelta(minutes=1)
        return MinuteBar(
            symbol=symbol,
            bar_start=bar_start,
            bar_end=bar_end,
            open_fixed=to_fixed_price(bucket.open_price, scale=self._price_scale),
            high_fixed=to_fixed_price(bucket.high_price, scale=self._price_scale),
            low_fixed=to_fixed_price(bucket.low_price, scale=self._price_scale),
            close_fixed=to_fixed_price(bucket.close_price, scale=self._price_scale),
            volume=max(0, int(round(bucket.volume))),
            source=self._source,
        )

    def ingest_tick(self, tick: TickEvent) -> list[MinuteBar]:
        emitted: list[MinuteBar] = []
        minute_start = self._minute_floor(tick.timestamp)
        bucket = self._buckets.get(tick.symbol)

        if bucket is None:
            self._buckets[tick.symbol] = _Bucket(
                minute_start=minute_start,
                open_price=tick.price,
                high_price=tick.price,
                low_price=tick.price,
                close_price=tick.price,
                volume=tick.size,
            )
            return emitted

        if minute_start < bucket.minute_start:
            self.late_ticks_dropped += 1
            return emitted

        if minute_start > bucket.minute_start:
            emitted.append(self._finalize(tick.symbol, bucket))
            bucket = _Bucket(
                minute_start=minute_start,
                open_price=tick.price,
                high_price=tick.price,
                low_price=tick.price,
                close_price=tick.price,
                volume=tick.size,
            )
            self._buckets[tick.symbol] = bucket
            return emitted

        bucket.high_price = max(bucket.high_price, tick.price)
        bucket.low_price = min(bucket.low_price, tick.price)
        bucket.close_price = tick.price
        bucket.volume += tick.size
        return emitted

    def finalize_due(self, *, now: datetime) -> list[MinuteBar]:
        now_minute = self._minute_floor(now)
        due_symbols = [
            symbol
            for symbol, bucket in self._buckets.items()
            if bucket.minute_start < now_minute
        ]

        emitted: list[MinuteBar] = []
        for symbol in due_symbols:
            bucket = self._buckets.pop(symbol, None)
            if bucket is None:
                continue
            emitted.append(self._finalize(symbol, bucket))
        return emitted

    def flush(self) -> list[MinuteBar]:
        bars = [self._finalize(symbol, bucket) for symbol, bucket in self._buckets.items()]
        self._buckets.clear()
        return bars
