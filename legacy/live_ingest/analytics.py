from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA


def compute_log_returns_vectorized(close_matrix: pd.DataFrame) -> pd.DataFrame:
    if close_matrix.empty or len(close_matrix.index) < 2:
        raise ValueError("Need at least 2 bars to compute log returns")

    values = close_matrix.values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.log(values[1:] / values[:-1])

    out = pd.DataFrame(log_ret, index=close_matrix.index[1:], columns=close_matrix.columns)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna(axis=0, how="any")


def rolling_zscore_latest(
    matrix: pd.DataFrame,
    *,
    window: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if window < 2:
        raise ValueError("window must be >= 2")
    if len(matrix.index) < window:
        raise ValueError(f"Need at least {window} rows for rolling z-score")

    values = matrix.values.astype(float)
    # Build windows as (n_features, n_windows, window), then transpose to
    # (n_windows, window, n_features).
    windows = sliding_window_view(values.T, window_shape=window, axis=1)
    windows = np.transpose(windows, (1, 2, 0))

    means = windows.mean(axis=1)
    stds = windows.std(axis=1)
    stds = np.where(stds == 0.0, 1e-12, stds)

    current = windows[:, -1, :]
    zscores = (current - means) / stds

    zindex = matrix.index[window - 1 :]
    zdf = pd.DataFrame(zscores, index=zindex, columns=matrix.columns)
    latest = zdf.iloc[-1].copy()
    latest.name = zindex[-1]
    return zdf, latest


def residual_zscore_latest(
    log_returns: pd.DataFrame,
    *,
    window: int = 500,
    factors: int = 1,
) -> pd.Series:
    """
    Residual-first signal:
    1) Fit PCA factors on prior window-1 rows.
    2) Compute residuals for prior rows and current row.
    3) Z-score current residual vs prior residual distribution.
    """
    if window < 3:
        raise ValueError("window must be >= 3")
    if len(log_returns.index) < window:
        raise ValueError(f"Need at least {window} rows for residual z-score")

    segment = log_returns.tail(window)
    train = segment.iloc[:-1]
    current = segment.iloc[-1:]

    k = max(1, min(factors, train.shape[1]))
    pca = PCA(n_components=k)
    train_scores = pca.fit_transform(train.values)
    train_recon = pca.inverse_transform(train_scores)
    train_res = train.values - train_recon

    current_scores = pca.transform(current.values)
    current_recon = pca.inverse_transform(current_scores)
    current_res = (current.values - current_recon)[0]

    mu = train_res.mean(axis=0)
    sigma = train_res.std(axis=0)
    sigma = np.where(sigma == 0.0, 1e-12, sigma)

    z = (current_res - mu) / sigma
    out = pd.Series(z, index=log_returns.columns, name=segment.index[-1])
    return out
