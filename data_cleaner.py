from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleaningResult:
    prices: pd.DataFrame
    dropped_primaries: tuple[str, ...]
    replacements: tuple[tuple[str, str], ...]  # (dropped_primary, reserve_used)


def clean_and_replace_zombies(
    prices: pd.DataFrame,
    *,
    primary_tickers: list[str],
    reserve_tickers: list[str],
    missing_frac_threshold: float = 0.05,
) -> CleaningResult:
    """
    Zombie ticker rule:
    - If any primary ticker has >5% NaNs over the window, drop it entirely.
    - Replace with the next available reserve ticker to keep strict 30 columns.
    Interpolation rule (guarded):
    - Only interpolate tiny gaps *within* each market's open days.
    - Do NOT interpolate across days where a market is closed (e.g., US holidays),
      so downstream return/shift/dropna alignment remains calendar-faithful.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices index must be a DatetimeIndex")

    df = prices.copy()
    df = df.sort_index()

    primaries = list(primary_tickers)
    reserves = list(reserve_tickers)

    # Ensure we have at least the columns needed available in fetched data
    for t in primaries + reserves:
        if t not in df.columns:
            df[t] = np.nan

    nan_frac = df[primaries].isna().mean(axis=0)
    zombies = [t for t, frac in nan_frac.items() if float(frac) > missing_frac_threshold]

    dropped: list[str] = []
    replacements: list[tuple[str, str]] = []

    # Replace in-place while preserving the primary column order
    for z in zombies:
        dropped.append(z)

        next_reserve: str | None = None
        while reserves:
            candidate = reserves.pop(0)
            if candidate in primaries:
                continue
            if candidate in df.columns:
                next_reserve = candidate
                break
        if next_reserve is None:
            raise RuntimeError(
                f"Reserve bench exhausted; cannot replace zombie ticker: {z}"
            )

        replacements.append((z, next_reserve))

        # Drop zombie column and substitute reserve at the same position in the primary list
        if z in df.columns:
            df = df.drop(columns=[z])
        idx = primaries.index(z)
        primaries[idx] = next_reserve

    # Enforce strict width & order: final primary list must be length 30
    if len(primaries) != 30:
        raise RuntimeError(f"Expected 30 primary tickers, got {len(primaries)}")

    df = df.reindex(columns=primaries)
    df = df.astype(float)

    def _interpolate_on_open_days(frame: pd.DataFrame, open_mask: pd.Series) -> pd.DataFrame:
        if frame.empty:
            return frame
        open_mask = open_mask.reindex(frame.index).fillna(False)
        if not bool(open_mask.any()):
            return frame
        subset = frame.loc[open_mask]
        subset = subset.interpolate(method="linear", limit=2, limit_direction="both")
        frame.loc[open_mask] = subset
        return frame

    india_cols = [c for c in df.columns if str(c).endswith(".NS")]
    us_cols = [c for c in df.columns if c not in india_cols]

    if india_cols:
        india_open = df[india_cols].notna().any(axis=1)
        df[india_cols] = _interpolate_on_open_days(df[india_cols], india_open)
    if us_cols:
        us_open = df[us_cols].notna().any(axis=1)
        df[us_cols] = _interpolate_on_open_days(df[us_cols], us_open)

    return CleaningResult(
        prices=df,
        dropped_primaries=tuple(dropped),
        replacements=tuple(replacements),
    )

