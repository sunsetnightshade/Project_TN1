from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class StandardizationResult:
    standardized: pd.DataFrame
    scaler_params: dict[str, object]


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

    corr = standardized.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="vlag", center=0, square=True, cbar_kws={"shrink": 0.75})
    plt.tight_layout()
    heatmap_path = Path(heatmap_path)
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    params = {
        "mean": scaler.mean_.copy(),
        "scale": scaler.scale_.copy(),
        "feature_names": list(matrix.columns),
    }

    return StandardizationResult(standardized=standardized, scaler_params=params)

