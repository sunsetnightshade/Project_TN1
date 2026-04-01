from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class PCASummary:
    explained_variance_ratio: pd.Series
    cumulative_explained_variance: pd.Series


def pca_fit_summary(matrix: pd.DataFrame, *, n_components: int) -> dict[str, pd.DataFrame]:
    """
    Fit PCA on the (T×30) standardized matrix and return explained variance tables.
    """
    if n_components < 1:
        raise ValueError("n_components must be >= 1")

    pca = PCA(n_components=min(n_components, matrix.shape[1]))
    pca.fit(matrix.values)

    evr = pd.Series(
        pca.explained_variance_ratio_,
        index=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        name="explained_variance_ratio",
    )
    cev = evr.cumsum().rename("cumulative_explained_variance")

    return {
        "explained_variance_ratio": evr.to_frame(),
        "cumulative_explained_variance": cev.to_frame(),
    }


def pca_beta_alpha(
    standardized: pd.DataFrame,
    *,
    k: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """
    Decompose standardized returns into:
    - beta_component: reconstruction from first k principal components
    - alpha_residual: standardized - beta_component
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    n_features = standardized.shape[1]
    k = min(k, n_features)

    pca = PCA(n_components=k)
    scores = pca.fit_transform(standardized.values)  # (T, k)
    recon = pca.inverse_transform(scores)  # (T, 30)

    beta = pd.DataFrame(recon, index=standardized.index, columns=standardized.columns)
    alpha = standardized - beta

    return beta, alpha, pca

