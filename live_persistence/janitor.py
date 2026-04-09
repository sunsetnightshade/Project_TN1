from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config import LiveIngestConfig
from common.timestamps import parse_iso_utc
from live_ingest.precision import from_fixed_price


@dataclass(frozen=True)
class HourlyPersistenceResult:
    total_rows_scanned: int
    rows_written: int
    deduped_rows: int
    files_written: tuple[Path, ...]


class HourlyParquetJanitor:
    def __init__(self, *, cfg: LiveIngestConfig, root_dir: Path) -> None:
        self.cfg = cfg
        self.root_dir = Path(root_dir)

        import redis  # type: ignore

        self._client = redis.Redis(
            host=cfg.redis_host,
            port=cfg.redis_port,
            db=cfg.redis_db,
            decode_responses=True,
        )

    def _partition_path(self, hour_start: datetime) -> Path:
        day = hour_start.strftime("%Y-%m-%d")
        hour = hour_start.strftime("%H")
        return self.root_dir / "outputs" / "live" / "parquet" / f"date={day}" / f"hour={hour}" / "bars.parquet"

    def fetch_recent_bars(self, *, hours: int) -> pd.DataFrame:
        if hours < 1:
            raise ValueError("hours must be >= 1")

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        count = max(1, hours * 60 * max(1, len(self.cfg.symbols)) * 5)
        entries = self._client.xrevrange(self.cfg.redis_stream_500, max="+", min="-", count=count)

        rows: list[dict[str, object]] = []
        symbols = set(self.cfg.symbols)

        for stream_id, fields in reversed(entries):
            if not isinstance(fields, dict):
                continue

            symbol = str(fields.get("symbol") or "").strip()
            if symbol not in symbols:
                continue

            bar_start_raw = fields.get("bar_start")
            bar_end_raw = fields.get("bar_end")
            if bar_start_raw is None or bar_end_raw is None:
                continue

            try:
                bar_start_dt = parse_iso_utc(str(bar_start_raw))
                bar_end_dt = parse_iso_utc(str(bar_end_raw))
            except ValueError:
                continue

            if bar_end_dt < cutoff:
                continue

            try:
                open_fixed = int(fields.get("open", "0"))
                high_fixed = int(fields.get("high", "0"))
                low_fixed = int(fields.get("low", "0"))
                close_fixed = int(fields.get("close", "0"))
                volume = int(float(fields.get("volume", "0")))
            except (TypeError, ValueError):
                continue

            rows.append(
                {
                    "stream_id": str(stream_id),
                    "symbol": symbol,
                    "source": str(fields.get("source") or ""),
                    "bar_start_dt": bar_start_dt,
                    "bar_end_dt": bar_end_dt,
                    "bar_start": bar_start_dt.isoformat(),
                    "bar_end": bar_end_dt.isoformat(),
                    "open_fixed": open_fixed,
                    "high_fixed": high_fixed,
                    "low_fixed": low_fixed,
                    "close_fixed": close_fixed,
                    "open": from_fixed_price(open_fixed, scale=self.cfg.price_scale),
                    "high": from_fixed_price(high_fixed, scale=self.cfg.price_scale),
                    "low": from_fixed_price(low_fixed, scale=self.cfg.price_scale),
                    "close": from_fixed_price(close_fixed, scale=self.cfg.price_scale),
                    "volume": volume,
                }
            )

        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(rows)
        frame = frame.sort_values(["bar_end_dt", "symbol"], kind="stable").reset_index(drop=True)
        return frame

    def persist_recent(self, *, hours: int = 1) -> HourlyPersistenceResult:
        frame = self.fetch_recent_bars(hours=hours)
        if frame.empty:
            return HourlyPersistenceResult(
                total_rows_scanned=0,
                rows_written=0,
                deduped_rows=0,
                files_written=(),
            )

        frame["hour_bucket"] = frame["bar_end_dt"].dt.floor("h")
        files_written: list[Path] = []
        rows_written = 0
        deduped_rows = 0

        for hour_start, group in frame.groupby("hour_bucket", sort=True):
            part = group.drop(columns=["hour_bucket"]).copy()
            part = part.drop(columns=["bar_start_dt", "bar_end_dt"])

            path = self._partition_path(hour_start)
            path.parent.mkdir(parents=True, exist_ok=True)

            previous_unique = 0
            if path.exists():
                existing = pd.read_parquet(path)
                previous_unique = len(
                    existing.drop_duplicates(subset=["symbol", "bar_end"], keep="last")
                )
                combined = pd.concat([existing, part], ignore_index=True)
            else:
                combined = part

            before = len(combined)
            combined = combined.drop_duplicates(subset=["symbol", "bar_end"], keep="last")
            combined = combined.sort_values(["bar_end", "symbol"], kind="stable").reset_index(drop=True)
            after = len(combined)
            deduped_rows += max(0, before - after)

            rows_written += max(0, after - previous_unique)

            combined.to_parquet(path, index=False)
            files_written.append(path)

        return HourlyPersistenceResult(
            total_rows_scanned=len(frame),
            rows_written=rows_written,
            deduped_rows=deduped_rows,
            files_written=tuple(files_written),
        )
