from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pipeline import build_and_serialize


async def _safe_build(
    *,
    root_dir: Path,
    start_date: date,
    end_date: date,
    missing_threshold: float,
) -> dict[str, Any]:
    try:
        artifacts = build_and_serialize(
            start_date=start_date,
            end_date=end_date,
            missing_threshold=missing_threshold,
            root_dir=root_dir,
        )
        c = artifacts.cleaning
        return {
            "ok": True,
            "paths": {k: str(v) for k, v in artifacts.paths.items()},
            "dropped": list(c.dropped_primaries),
            "replacements": list([{"from": a, "to": b} for a, b in c.replacements]),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


async def run_websocket_live_server(
    *,
    host: str,
    port: int,
    root_dir: Path,
    interval_seconds: float,
    start_date: date,
    end_date: date,
    missing_threshold: float,
) -> None:
    """
    Local websocket server that triggers a rebuild every `interval_seconds`
    and broadcasts a status event to all connected clients.

    Note: yfinance is not designed for high-frequency refresh; this is intended
    for a "live dashboard" feel, not tick-level trading data.
    """
    import websockets  # type: ignore

    clients: set[Any] = set()

    async def handler(ws: Any) -> None:
        clients.add(ws)
        try:
            await ws.send(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "status": "connected",
                    }
                )
            )
            async for _ in ws:
                # This server is broadcast-only; ignore client messages.
                pass
        finally:
            clients.discard(ws)

    async def broadcaster() -> None:
        """
        Broadcast a message every `interval_seconds`.

        If a rebuild takes longer than the interval (common with yfinance),
        we still push a heartbeat every interval with status='building' until
        the build completes, then we publish the completion payload.
        """
        build_task: asyncio.Task[dict[str, Any]] | None = None
        last_result: dict[str, Any] | None = None

        while True:
            ts = datetime.now(timezone.utc).isoformat()

            if build_task is None:
                build_task = asyncio.create_task(
                    _safe_build(
                        root_dir=root_dir,
                        start_date=start_date,
                        end_date=end_date,
                        missing_threshold=missing_threshold,
                    )
                )

            status: str
            error: str | None = None
            payload: dict[str, Any]

            if build_task.done():
                last_result = build_task.result()
                build_task = None  # schedule next one on next tick
                ok = bool(last_result.get("ok"))
                status = "ok" if ok else "error"
                error = last_result.get("error")
                payload = {
                    "timestamp": ts,
                    "status": status,
                    "error": error,
                    "paths": last_result.get("paths"),
                    "dropped": last_result.get("dropped"),
                    "replacements": last_result.get("replacements"),
                }
            else:
                payload = {
                    "timestamp": ts,
                    "status": "building",
                    "error": None,
                    "paths": last_result.get("paths") if last_result else None,
                    "dropped": last_result.get("dropped") if last_result else None,
                    "replacements": last_result.get("replacements")
                    if last_result
                    else None,
                }

            if clients:
                msg = json.dumps(payload)
                await asyncio.gather(*(c.send(msg) for c in list(clients)))

            await asyncio.sleep(interval_seconds)

    server = await websockets.serve(handler, host, port)
    try:
        await broadcaster()
    finally:
        server.close()
        await server.wait_closed()


def default_start_end() -> tuple[date, date]:
    today = date.today()
    return today - timedelta(days=730), today

