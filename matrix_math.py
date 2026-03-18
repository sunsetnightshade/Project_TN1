from __future__ import annotations

import numpy as np
import pandas as pd


def build_aligned_log_return_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """
    STRICT ORDER OF OPERATIONS (do not change):
    1) Compute log returns for the entire dataframe.
    2) Shift US columns (no '.NS' suffix) by 1 to align last night's NY with morning India.
    3) Drop all rows with NaNs (inner join across markets/holidays).
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices index must be a DatetimeIndex")

    df = prices.copy().sort_index()

    # Step 1
    log_ret = np.log(df / df.shift(1))

    # Step 2
    us_cols = [c for c in log_ret.columns if not str(c).endswith(".NS")]
    log_ret[us_cols] = log_ret[us_cols].shift(1)

    # Step 3
    log_ret = log_ret.dropna(axis=0, how="any")

    return log_ret

