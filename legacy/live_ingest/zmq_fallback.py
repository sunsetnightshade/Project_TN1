from __future__ import annotations

import asyncio
import json
import time

from .models import MinuteBar


class ZeroMQBarProducer:
    def __init__(self, *, endpoint: str) -> None:
        self.endpoint = endpoint
        self._ctx = None
        self._socket = None

    async def connect(self) -> None:
        import zmq  # type: ignore

        self._ctx = zmq.asyncio.Context.instance()
        self._socket = self._ctx.socket(zmq.PUSH)
        self._socket.connect(self.endpoint)

    async def write_bar(self, bar: MinuteBar) -> float:
        if self._socket is None:
            raise RuntimeError("ZeroMQBarProducer.connect() must be called first")

        t0 = time.perf_counter()
        payload = {
            "kind": "minute_bar",
            "symbol": bar.symbol,
            "fields": bar.to_redis_fields(),
        }
        await self._socket.send_string(json.dumps(payload))
        return (time.perf_counter() - t0) * 1000.0

    async def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._ctx is not None:
            # Context is process-wide singleton; do not terminate here.
            await asyncio.sleep(0)
