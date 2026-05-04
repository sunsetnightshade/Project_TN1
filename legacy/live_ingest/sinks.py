from __future__ import annotations

from typing import Protocol

from .models import MinuteBar


class BarSink(Protocol):
    async def connect(self) -> None: ...

    async def write_bar(self, bar: MinuteBar) -> float: ...

    async def close(self) -> None: ...
