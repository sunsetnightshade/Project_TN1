from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common.timestamps import parse_iso_utc_optional
from .models import MinuteBar
from .precision import to_fixed_price


@dataclass(frozen=True)
class Type2VerificationResult:
    ok: bool
    checked_at: datetime
    note: str


class Type2FallbackVerifier:
    """
    Phase-scope scaffold for TYPE2 verification path.
    Bloomberg integration is deferred; this verifier records checkpoints so
    gap reconciliation can be automated in a later phase.
    """

    def __init__(self, *, root_dir: Path, parquet_root: str) -> None:
        base = Path(root_dir)
        raw = Path(parquet_root)
        self.parquet_root = raw if raw.is_absolute() else (base / raw)

    def verify_market_state(
        self,
        *,
        reason: str,
        last_bar_end: str | None = None,
    ) -> Type2VerificationResult:
        if not self.parquet_root.exists():
            return Type2VerificationResult(
                ok=False,
                checked_at=datetime.now(timezone.utc),
                note=f"TYPE2 root missing: {self.parquet_root} reason={reason}",
            )

        files = sorted(self.parquet_root.glob("**/bars.parquet"))
        if not files:
            return Type2VerificationResult(
                ok=False,
                checked_at=datetime.now(timezone.utc),
                note=f"TYPE2 parquet files not found under {self.parquet_root} reason={reason}",
            )

        latest_file = files[-1]
        try:
            frame = pd.read_parquet(latest_file, columns=["bar_end"])
        except Exception as exc:  # noqa: BLE001
            return Type2VerificationResult(
                ok=False,
                checked_at=datetime.now(timezone.utc),
                note=f"TYPE2 read failed: {exc!r} file={latest_file}",
            )

        if frame.empty:
            return Type2VerificationResult(
                ok=False,
                checked_at=datetime.now(timezone.utc),
                note=f"TYPE2 file empty: {latest_file}",
            )

        latest_ts = pd.to_datetime(frame["bar_end"], utc=True, errors="coerce").max()
        if pd.isna(latest_ts):
            return Type2VerificationResult(
                ok=False,
                checked_at=datetime.now(timezone.utc),
                note=f"TYPE2 timestamps invalid in {latest_file}",
            )

        reference = parse_iso_utc_optional(last_bar_end)
        if reference is None:
            return Type2VerificationResult(
                ok=True,
                checked_at=datetime.now(timezone.utc),
                note=f"TYPE2 available latest={latest_ts.isoformat()} reason={reason}",
            )

        gap_seconds = (reference - latest_ts.to_pydatetime()).total_seconds()
        ok = gap_seconds <= 120.0
        return Type2VerificationResult(
            ok=ok,
            checked_at=datetime.now(timezone.utc),
            note=(
                f"TYPE2 latest={latest_ts.isoformat()} reference={reference.isoformat()} "
                f"gap_seconds={gap_seconds:.1f} reason={reason}"
            ),
        )

    def backfill_bars(
        self,
        *,
        symbols: tuple[str, ...],
        after_bar_end: str | None,
        until: datetime,
        price_scale: int,
    ) -> list[MinuteBar]:
        start = parse_iso_utc_optional(after_bar_end)
        if start is None:
            return []

        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        else:
            until = until.astimezone(timezone.utc)

        files = sorted(self.parquet_root.glob("**/bars.parquet"))
        if not files:
            return []

        parts: list[pd.DataFrame] = []
        cols = [
            "symbol",
            "source",
            "bar_start",
            "bar_end",
            "open_fixed",
            "high_fixed",
            "low_fixed",
            "close_fixed",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for path in files:
            try:
                frame = pd.read_parquet(path)
            except Exception:  # noqa: BLE001
                continue
            if frame.empty:
                continue
            available = [c for c in cols if c in frame.columns]
            frame = frame[available].copy()
            parts.append(frame)

        if not parts:
            return []

        frame = pd.concat(parts, ignore_index=True)
        if "symbol" not in frame.columns or "bar_end" not in frame.columns:
            return []

        frame["bar_end_dt"] = pd.to_datetime(frame["bar_end"], utc=True, errors="coerce")
        frame["bar_start_dt"] = pd.to_datetime(
            frame.get("bar_start", frame["bar_end"]),
            utc=True,
            errors="coerce",
        )
        frame = frame.dropna(subset=["bar_end_dt", "bar_start_dt"])
        frame = frame[frame["symbol"].isin(symbols)]
        frame = frame[(frame["bar_end_dt"] > start) & (frame["bar_end_dt"] <= until)]
        if frame.empty:
            return []

        frame = frame.sort_values(["bar_end_dt", "symbol"], kind="stable")
        bars: list[MinuteBar] = []

        for _, row in frame.iterrows():
            source = str(row.get("source") or "type2")
            symbol = str(row["symbol"])

            def _fixed(col_fixed: str, col_float: str) -> int:
                raw_fixed = row.get(col_fixed)
                if pd.notna(raw_fixed):
                    try:
                        return int(raw_fixed)
                    except (TypeError, ValueError):
                        pass
                raw_float = row.get(col_float)
                if pd.notna(raw_float):
                    return to_fixed_price(float(raw_float), scale=price_scale)
                return 0

            volume_raw = row.get("volume", 0)
            try:
                volume = int(float(volume_raw))
            except (TypeError, ValueError):
                volume = 0

            bars.append(
                MinuteBar(
                    symbol=symbol,
                    bar_start=row["bar_start_dt"].to_pydatetime(),
                    bar_end=row["bar_end_dt"].to_pydatetime(),
                    open_fixed=_fixed("open_fixed", "open"),
                    high_fixed=_fixed("high_fixed", "high"),
                    low_fixed=_fixed("low_fixed", "low"),
                    close_fixed=_fixed("close_fixed", "close"),
                    volume=max(0, volume),
                    source=f"{source}:backfill",
                )
            )

        return bars
