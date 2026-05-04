"""Layer 2 — Metrics Dashboard.

Terminal text-based dashboard showing system health in real time.
Reads metrics from Redis and QuestDB. Rendered without curses for portability.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer0.config import ConfigRegistry

logger = get_logger(__name__)


# ANSI escape codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_CYAN = "\033[96m"
_DIM = "\033[2m"


def _color_score(score: int) -> str:
    if score >= 80:
        return f"{_GREEN}{score}{_RESET}"
    elif score >= 50:
        return f"{_YELLOW}{score}{_RESET}"
    else:
        return f"{_RED}{score}{_RESET}"


class MetricsDashboard:
    """Terminal metrics dashboard using Redis + QuestDB as data sources."""

    _REFRESH_INTERVAL_SEC = 5

    def __init__(self, config: Optional["ConfigRegistry"] = None) -> None:
        self._cfg = config
        self._redis = None
        self._qdb = None
        self._last_refresh = 0.0

    def _init_connections(self) -> None:
        try:
            import redis
            self._redis = redis.Redis(
                host=self._cfg.get("database.redis.host", "localhost") if self._cfg else "localhost",
                port=int(self._cfg.get("database.redis.port", 6379) if self._cfg else 6379),
                decode_responses=True,
            )
        except Exception as exc:
            logger.debug("Redis connection failed for dashboard: %s", exc)

    def run(self, refresh_sec: int = 5) -> None:
        """Run the live dashboard. Blocking. Press Ctrl+C to exit."""
        self._init_connections()
        try:
            while True:
                self._render()
                time.sleep(refresh_sec)
        except KeyboardInterrupt:
            print(f"\n{_RESET}Dashboard stopped.")

    def _render(self) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\033[2J\033[H", end="")  # Clear screen

        print(f"{_BOLD}{_CYAN}═══════════════════════════════════════════════════════════════{_RESET}")
        print(f"{_BOLD}{_CYAN}          NIGHTSHADE QUANTITATIVE SYSTEM — OBSERVABILITY         {_RESET}")
        print(f"{_BOLD}{_CYAN}═══════════════════════════════════════════════════════════════{_RESET}")
        print(f"  {_DIM}Refreshed at {now}{_RESET}")
        print()

        # Supervisor status
        self._render_supervisor_status()
        print()

        # Connection health per adapter
        self._render_adapter_health()
        print()

        # Gap tracker summary
        self._render_gap_summary()
        print()

        # Ingestor throughput
        self._render_throughput()
        print()

    def _render_supervisor_status(self) -> None:
        print(f"  {_BOLD}SUPERVISOR STATUS{_RESET}")
        print(f"  {'─' * 60}")
        status = self._get_redis_json("nightshade:supervisor:status")
        if not status:
            print(f"  {_RED}No supervisor data (not running?){_RESET}")
            return
        for adapter, info in status.get("adapters", {}).items():
            state = info.get("connection_state", "UNKNOWN")
            score = info.get("health_score", 0)
            state_color = _GREEN if state == "CONNECTED" else (_YELLOW if state == "RECONNECTING" else _RED)
            print(f"  {adapter:20s}  state={state_color}{state}{_RESET}  score={_color_score(score)}")
        print(f"  Open gaps: {status.get('gaps_open', 'N/A')}")

    def _render_adapter_health(self) -> None:
        print(f"  {_BOLD}ADAPTER HEALTH{_RESET}")
        print(f"  {'─' * 60}")
        if not self._redis:
            print(f"  {_DIM}Redis unavailable{_RESET}")
            return
        keys = list(self._redis.scan_iter("nightshade:health:*"))
        if not keys:
            print(f"  {_DIM}No health data{_RESET}")
            return
        for key in sorted(keys):
            data = self._get_redis_json(key)
            if not data:
                continue
            source = key.split(":")[-1]
            score = data.get("health_score", 0)
            metrics = data.get("metrics", {})
            state = metrics.get("connection_state", "UNKNOWN")
            lag = metrics.get("current_latency_ms", 0)
            msgs = metrics.get("messages_received_total", 0)
            print(f"  {source:20s}  score={_color_score(score)}  state={state}  lag={lag:.1f}ms  msgs={msgs:,}")

    def _render_gap_summary(self) -> None:
        print(f"  {_BOLD}GAP TRACKER{_RESET}")
        print(f"  {'─' * 60}")
        status = self._get_redis_json("nightshade:supervisor:status")
        if status:
            open_gaps = status.get("gaps_open", "N/A")
            color = _GREEN if open_gaps == 0 else (_YELLOW if isinstance(open_gaps, int) and open_gaps < 5 else _RED)
            print(f"  Open gaps: {color}{open_gaps}{_RESET}")
        else:
            print(f"  {_DIM}Unavailable{_RESET}")

    def _render_throughput(self) -> None:
        print(f"  {_BOLD}THROUGHPUT (today){_RESET}")
        print(f"  {'─' * 60}")
        status = self._get_redis_json("nightshade:supervisor:status")
        if not status:
            print(f"  {_DIM}Unavailable{_RESET}")
            return
        for adapter, info in status.get("adapters", {}).items():
            recv = info.get("ticks_received_today", "N/A")
            written = info.get("ticks_written_today", "N/A")
            rejected = info.get("ticks_rejected_today", "N/A")
            print(f"  {adapter:20s}  recv={recv}  written={written}  rejected={rejected}")

    def _get_redis_json(self, key: str) -> Optional[dict]:
        if not self._redis:
            return None
        try:
            data = self._redis.get(key)
            return json.loads(data) if data else None
        except Exception:
            return None
