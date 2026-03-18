from __future__ import annotations

from dataclasses import dataclass

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
    Interpolation rule:
    - After replacement, interpolate remaining tiny gaps linearly (both directions).
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
            df[t] = pd.NA

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

    # Fill remaining tiny gaps (1-2 days) after replacements
    df = df.interpolate(method="linear", limit_direction="both")

    return CleaningResult(
        prices=df,
        dropped_primaries=tuple(dropped),
        replacements=tuple(replacements),
    )

