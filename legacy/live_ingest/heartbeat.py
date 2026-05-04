from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HeartbeatWatchdog:
    interval_seconds: float
    pong_timeout_seconds: float
    last_ping_sent: float
    last_pong_seen: float

    @classmethod
    def create(cls, *, interval_seconds: float, pong_timeout_seconds: float, now: float) -> "HeartbeatWatchdog":
        return cls(
            interval_seconds=interval_seconds,
            pong_timeout_seconds=pong_timeout_seconds,
            last_ping_sent=now,
            last_pong_seen=now,
        )

    def should_send_ping(self, now: float) -> bool:
        return (now - self.last_ping_sent) >= self.interval_seconds

    def mark_ping_sent(self, now: float) -> None:
        self.last_ping_sent = now

    def mark_pong_seen(self, now: float) -> None:
        self.last_pong_seen = now

    def has_timed_out(self, now: float) -> bool:
        waiting_for_pong = self.last_ping_sent > self.last_pong_seen
        return waiting_for_pong and (now - self.last_ping_sent) > self.pong_timeout_seconds
