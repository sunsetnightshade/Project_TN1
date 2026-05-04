from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class TickEvent:
    symbol: str
    price: float
    size: float
    timestamp: datetime
    source: str

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            object.__setattr__(self, "timestamp", self.timestamp.replace(tzinfo=timezone.utc))


@dataclass(frozen=True)
class MinuteBar:
    symbol: str
    bar_start: datetime
    bar_end: datetime
    open_fixed: int
    high_fixed: int
    low_fixed: int
    close_fixed: int
    volume: int
    source: str

    def to_redis_fields(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "bar_start": self.bar_start.isoformat(),
            "bar_end": self.bar_end.isoformat(),
            "open": str(self.open_fixed),
            "high": str(self.high_fixed),
            "low": str(self.low_fixed),
            "close": str(self.close_fixed),
            "volume": str(self.volume),
            "source": self.source,
        }
