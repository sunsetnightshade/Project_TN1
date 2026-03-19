from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class StandardizationResult:
    standardized: pd.DataFrame
    scaler_params: dict[str, object]


def render_aligned_matrix_heatmap(
    standardized: pd.DataFrame,
    *,
    heatmap_path: Path,
    title: str = "Aligned Standardized Matrix Heatmap",
) -> None:
    """
    Render a 30×T heatmap (tickers as rows, days as columns) without cell-edge
    artifacts that can make a single cell look "split" in the saved PNG.
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
        cmap="vlag",
        vmin=-3.0,
        vmax=3.0,
    )

    ax.set_title(title)
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

    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.tight_layout()

    heatmap_path = Path(heatmap_path)
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(heatmap_path, dpi=220)
    plt.close(fig)


def standardize_and_plot_heatmap(
    matrix: pd.DataFrame,
    *,
    heatmap_path: Path,
) -> StandardizationResult:
    if not isinstance(matrix.index, pd.DatetimeIndex):
        raise TypeError("matrix index must be a DatetimeIndex")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix.values)
    standardized = pd.DataFrame(scaled, index=matrix.index, columns=matrix.columns)

    # Keep the legacy correlation plot available for debugging, but default the
    # "main" heatmap to the aligned 30×T matrix (tickers × days).
    heatmap_path = Path(heatmap_path)
    if heatmap_path.name.lower().startswith("corr"):
        import seaborn as sns

        corr = standardized.corr()
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr, cmap="vlag", center=0, square=True, cbar_kws={"shrink": 0.75})
        plt.tight_layout()
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(heatmap_path, dpi=220)
        plt.close()
    else:
        render_aligned_matrix_heatmap(standardized, heatmap_path=heatmap_path)

    params = {
        "mean": scaler.mean_.copy(),
        "scale": scaler.scale_.copy(),
        "feature_names": list(matrix.columns),
    }

    return StandardizationResult(standardized=standardized, scaler_params=params)

