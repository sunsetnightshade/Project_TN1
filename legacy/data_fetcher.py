from __future__ import annotations

import os
import time
from datetime import date
from typing import Iterable

import pandas as pd


def _legacy_yfinance_enabled() -> bool:
    return os.getenv("QM_ENABLE_LEGACY_YFINANCE", "0").strip() == "1"


def fetch_adj_close_prices(
    tickers: Iterable[str],
    start_date: date,
    end_date: date,
    *,
    max_retries: int = 3,
    sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    """
    LEGACY PATH ONLY.
    Historical yfinance fetch is disabled by default because this project now
    uses websocket -> Redis Streams for TYPE1 live data.

    To temporarily re-enable this legacy path, set:
      QM_ENABLE_LEGACY_YFINANCE=1
    """
    if not _legacy_yfinance_enabled():
        raise RuntimeError(
            "Legacy yfinance path is disabled. Use websocket live ingestion instead: "
            "'py main.py --ingest-live'. To force-enable old behavior, set "
            "QM_ENABLE_LEGACY_YFINANCE=1."
        )

    try:
        import yfinance as yf  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "yfinance is not installed. This path is legacy-only; prefer websocket "
            "live ingestion commands."
        ) from exc

    ticker_list = list(dict.fromkeys(tickers))  # preserve order, drop duplicates
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                tickers=ticker_list,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                auto_adjust=False,
                group_by="column",
                progress=False,
                threads=True,
            )

            if df is None or df.empty:
                raise RuntimeError("yfinance returned empty dataframe")

            # yfinance returns:
            # - MultiIndex columns for multi-ticker
            # - single-level columns for single ticker
            if isinstance(df.columns, pd.MultiIndex):
                if "Adj Close" not in df.columns.get_level_values(0):
                    raise RuntimeError("yfinance response missing 'Adj Close'")
                adj = df["Adj Close"].copy()
            else:
                if "Adj Close" not in df.columns:
                    raise RuntimeError("yfinance response missing 'Adj Close'")
                adj = df[["Adj Close"]].copy()
                adj.columns = ticker_list[:1]

            adj.index = pd.to_datetime(adj.index)
            adj = adj.sort_index()
            adj = adj.reindex(columns=ticker_list)
            return adj
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries:
                time.sleep(sleep_seconds)
                continue
            raise

    # unreachable, but keeps type-checkers happy
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("fetch failed unexpectedly")

