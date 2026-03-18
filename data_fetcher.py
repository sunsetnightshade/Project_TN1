from __future__ import annotations

import time
from datetime import date
from typing import Iterable

import pandas as pd
import yfinance as yf


def fetch_adj_close_prices(
    tickers: Iterable[str],
    start_date: date,
    end_date: date,
    *,
    max_retries: int = 3,
    sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    """
    Fetch Adj Close prices for the given tickers over [start_date, end_date).
    Implements a strict retry loop to handle yfinance throttling.
    """
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

