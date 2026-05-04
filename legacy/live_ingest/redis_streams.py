from __future__ import annotations

import time

from .models import MinuteBar


class RedisBarProducer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        db: int,
        stream_60: str,
        stream_500: str,
        maxlen_60: int,
        maxlen_500: int,
    ) -> None:
        self.host = host
        self.port = port
        self.db = db
        self.stream_60 = stream_60
        self.stream_500 = stream_500
        self.maxlen_60 = maxlen_60
        self.maxlen_500 = maxlen_500
        self._client = None

    async def connect(self) -> None:
        import redis.asyncio as redis  # type: ignore

        self._client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=True,
        )
        await self._client.ping()

    async def write_bar(self, bar: MinuteBar) -> float:
        if self._client is None:
            raise RuntimeError("RedisBarProducer.connect() must be called first")

        t0 = time.perf_counter()
        fields = bar.to_redis_fields()
        # Pipeline both XADD commands into a single round-trip.
        # approximate=True uses O(1) trimming instead of O(N).
        async with self._client.pipeline(transaction=False) as pipe:
            pipe.xadd(
                self.stream_60,
                fields,
                maxlen=self.maxlen_60,
                approximate=True,
            )
            pipe.xadd(
                self.stream_500,
                fields,
                maxlen=self.maxlen_500,
                approximate=True,
            )
            await pipe.execute()
        return (time.perf_counter() - t0) * 1000.0

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
