from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def pca_fit_summary(matrix: pd.DataFrame, *, n_components: int) -> dict[str, pd.DataFrame]:
    """
    Fit PCA on the (T×30) standardised matrix and return variance tables.

    Safely clamps n_components to min(T, n_features) — scikit-learn raises
    if n_components > min(n_samples, n_features).
    """
    if n_components < 1:
        raise ValueError("n_components must be >= 1")

    # sklearn constraint: n_components <= min(n_samples, n_features)
    safe_n = min(n_components, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=safe_n)
    pca.fit(matrix.values)

    evr = pd.Series(
        pca.explained_variance_ratio_,
        index=[f"PC{i+1}" for i in range(safe_n)],
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
    Decompose standardised returns into:
    - beta_component : reconstruction from first k principal components
                       (systematic / market factor)
    - alpha_residual : standardized - beta_component
                       (idiosyncratic returns)

    Safely clamps k to min(T, n_features) to prevent sklearn from crashing.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    n_samples, n_features = standardized.shape
    k = min(k, n_samples, n_features)

    pca = PCA(n_components=k)
    scores = pca.fit_transform(standardized.values)   # (T, k)
    recon = pca.inverse_transform(scores)              # (T, n_features)

    beta = pd.DataFrame(recon, index=standardized.index, columns=standardized.columns)
    alpha = standardized - beta

    return beta, alpha, pca
