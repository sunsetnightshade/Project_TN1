from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd

from common.timestamps import parse_iso_utc
from .precision import from_fixed_price


class RedisLiveConsumer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        db: int,
        stream_key: str,
        symbols: Iterable[str],
        price_scale: int,
    ) -> None:
        self.host = host
        self.port = port
        self.db = db
        self.stream_key = stream_key
        self.symbols = tuple(symbols)
        self.price_scale = price_scale

        import redis  # type: ignore

        self._client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=True,
        )

    def ping(self) -> bool:
        return bool(self._client.ping())


    def build_close_matrix(self, *, limit_minutes: int) -> pd.DataFrame:
        count = max(1, limit_minutes * max(1, len(self.symbols)) * 3)
        entries = self._client.xrevrange(self.stream_key, max="+", min="-", count=count)

        # xrevrange returns newest first; reverse to get chronological order.
        rows: list[tuple[datetime, str, float]] = []
        for _, fields in reversed(entries):
            if not isinstance(fields, dict):
                continue
            symbol = str(fields.get("symbol") or "").strip()
            if symbol not in self.symbols:
                continue

            ts_raw = fields.get("bar_end")
            close_raw = fields.get("close")
            if ts_raw is None or close_raw is None:
                continue

            try:
                ts = parse_iso_utc(str(ts_raw))
                close_fixed = int(close_raw)
                close = from_fixed_price(close_fixed, scale=self.price_scale)
            except (ValueError, TypeError):
                continue

            rows.append((ts, symbol, close))

        if not rows:
            return pd.DataFrame(columns=list(self.symbols), dtype=float)

        frame = pd.DataFrame(rows, columns=["bar_end", "symbol", "close"])
        matrix = frame.pivot_table(index="bar_end", columns="symbol", values="close", aggfunc="last")
        matrix = matrix.sort_index()
        matrix = matrix.reindex(columns=list(self.symbols))
        matrix = matrix.tail(limit_minutes)
        matrix.index = pd.to_datetime(matrix.index, utc=True)
        return matrix
