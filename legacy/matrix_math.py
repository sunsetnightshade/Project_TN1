from __future__ import annotations

import numpy as np
import pandas as pd


def build_aligned_log_return_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """
    STRICT ORDER OF OPERATIONS (do not change):
    1) Compute log returns for the entire dataframe.
    2) Drop all rows with NaNs (removes the first row from the log-return calc,
       plus any rows with missing data across any ticker).

    NOTE: The old Step 2 (shift US columns by 1 to align with India open) has
    been removed. The universe is now 30 US-only Nasdaq-100 stocks — no cross-
    market alignment is needed. All tickers trade on the same US calendar.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices index must be a DatetimeIndex")

    df = prices.copy().sort_index()

    # Step 1 — log returns
    log_ret = np.log(df / df.shift(1))

    # Step 2 — drop NaN rows (removes t=0 and any data gaps)
    log_ret = log_ret.dropna(axis=0, how="any")

    return log_ret
