from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class StandardizationResult:
    standardized: pd.DataFrame
    scaler_params: dict[str, object]


# ---------------------------------------------------------------------------
# 30×T matrix heatmap (tickers as rows, days as columns)
# ---------------------------------------------------------------------------

def render_aligned_matrix_heatmap(
    standardized: pd.DataFrame,
    *,
    heatmap_path: Path | None,
    title: str = "Aligned Standardized Matrix Heatmap (30×T)",
) -> "plt.Figure":
    """
    Render a 30×T heatmap (tickers as rows, days as columns).
    Uses imshow with interpolation='nearest' to avoid cell-edge split artifacts.
    """
    if not isinstance(standardized.index, pd.DatetimeIndex):
        raise TypeError("standardized index must be a DatetimeIndex")

    data = standardized.T.values  # (tickers, days)
    dates = standardized.index
    tickers = list(standardized.columns)

    fig, ax = plt.subplots(figsize=(18, 10))
    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-3.0,
        vmax=3.0,
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Ticker")
    ax.set_xlabel("Date")

    ax.set_yticks(np.arange(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=9)

    n_days = len(dates)
    if n_days <= 15:
        xticks = np.arange(n_days)
    else:
        target = 12
        step = max(1, int(round(n_days / target)))
        xticks = np.arange(0, n_days, step)
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [dates[i].strftime("%Y-%m-%d") for i in xticks],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Z-score")
    fig.tight_layout()

    if heatmap_path is not None:
        heatmap_path = Path(heatmap_path)
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(heatmap_path, dpi=220)

    return fig


# ---------------------------------------------------------------------------
# 30×30 correlation heatmap — the "one unified square"
# ---------------------------------------------------------------------------

def compute_correlation_outliers(
    corr: pd.DataFrame,
    *,
    low_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """
    Identify ticker pairs with suspiciously low pairwise correlation.
    Returns a list of {ticker_a, ticker_b, correlation} dicts sorted by correlation.

    For a 30-stock US tech universe, any correlation below 0.3 is a red flag
    (possible zombie ticker, data gap, or stale price).
    """
    outliers: list[dict[str, Any]] = []
    tickers = list(corr.columns)
    for i, ta in enumerate(tickers):
        for tb in tickers[i + 1:]:
            c = float(corr.loc[ta, tb])
            if c < low_threshold:
                outliers.append({"ticker_a": ta, "ticker_b": tb, "correlation": round(c, 4)})
    return sorted(outliers, key=lambda x: x["correlation"])


def render_correlation_heatmap(
    standardized: pd.DataFrame,
    *,
    heatmap_path: Path | None,
    title: str = "30×30 Pairwise Correlation Matrix (US Nasdaq-100 Tech)",
    low_threshold: float = 0.3,
) -> tuple["plt.Figure", list[dict[str, Any]]]:
    """
    Render a 30×30 symmetric correlation heatmap using seaborn.
    Returns (figure, outliers) where outliers is a list of low-correlation pairs.

    Expected range for healthy US tech stocks: 0.5 – 0.9.
    Any pair below `low_threshold` (default 0.3) is flagged as suspicious.
    """
    if not isinstance(standardized.index, pd.DatetimeIndex):
        raise TypeError("standardized index must be a DatetimeIndex")
    if standardized.shape[1] < 2:
        raise ValueError("Need at least 2 tickers to compute correlations")

    corr = standardized.corr()
    outliers = compute_correlation_outliers(corr, low_threshold=low_threshold)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.3,
        linecolor="white",
        annot=False,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Highlight suspicious pairs — add a text box if any exist
    if outliers:
        worst = outliers[0]
        note = (
            f"⚠ {len(outliers)} low-corr pair(s) found\n"
            f"Worst: {worst['ticker_a']} ↔ {worst['ticker_b']} = {worst['correlation']:.3f}"
        )
        ax.text(
            0.01, 0.99, note,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffe0e0", alpha=0.85),
        )

    fig.tight_layout()

    if heatmap_path is not None:
        heatmap_path = Path(heatmap_path)
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(heatmap_path, dpi=220)

    return fig, outliers


# ---------------------------------------------------------------------------
# Main standardization entry point
# ---------------------------------------------------------------------------

def standardize_and_plot_heatmap(
    matrix: pd.DataFrame,
    *,
    heatmap_path: Path,
) -> StandardizationResult:
    """
    Z-score standardize the T×30 log-return matrix using StandardScaler,
    then render BOTH heatmaps:
      1. matrix_heatmap.png   — 30×T values (tickers × days)
      2. correlation_heatmap.png — 30×30 pairwise correlations

    Returns StandardizationResult with standardized matrix + scaler params.
    """
    if not isinstance(matrix.index, pd.DatetimeIndex):
        raise TypeError("matrix index must be a DatetimeIndex")

    # Z-score standardize (column-wise, i.e. per ticker)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix.values)
    standardized = pd.DataFrame(scaled, index=matrix.index, columns=matrix.columns)

    heatmap_path = Path(heatmap_path)
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)

    # Heatmap 1: 30×T matrix (tickers × days)
    fig_matrix = render_aligned_matrix_heatmap(standardized, heatmap_path=heatmap_path)
    plt.close(fig_matrix)

    # Heatmap 2: 30×30 correlation (always generated alongside)
    corr_path = heatmap_path.parent / "correlation_heatmap.png"
    fig_corr, outliers = render_correlation_heatmap(standardized, heatmap_path=corr_path)
    plt.close(fig_corr)

    params: dict[str, object] = {
        "mean": scaler.mean_.copy(),
        "scale": scaler.scale_.copy(),
        "feature_names": list(matrix.columns),
        "correlation_outliers": outliers,   # list of low-corr pairs — useful for debugging
    }

    return StandardizationResult(standardized=standardized, scaler_params=params)
