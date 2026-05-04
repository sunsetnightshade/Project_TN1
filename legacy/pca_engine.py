import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

def rolling_pca_alpha_beta(
    standardized: pd.DataFrame,
    *,
    window: int = 60,
    k: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Silver Layer: Project standardized returns onto the rolling Principal Components.
    
    Decompose standardised returns into:
    - beta_component : reconstruction from first k principal components (systematic)
    - alpha_residual : standardized - beta_component (idiosyncratic returns)
    
    Calculated over a strict rolling window to preserve bitemporal causality 
    and eliminate look-ahead bias. Highly vectorized via numpy eigh.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if window < 2:
        raise ValueError("window must be >= 2 for covariance computation")

    data = standardized.values
    n_samples, n_features = data.shape
    
    if n_samples < window:
        # Not enough data for rolling window; return NaNs
        padding = np.full((n_samples, n_features), np.nan)
        beta_df = pd.DataFrame(padding, index=standardized.index, columns=standardized.columns)
        return beta_df, beta_df.copy()

    safe_k = min(k, window, n_features)

    # 1) Sliding Window
    # view shape: (T - window + 1, n_features, window)
    view = sliding_window_view(data, window_shape=window, axis=0)

    # We WANT (T - window + 1, window, n_features) for natural matrix math
    view = np.transpose(view, (0, 2, 1))

    # 2) Rolling Covariance Matrix
    # We are using pre-standardized data, so mean is approx 0.
    # To be mathematically exact for the rolling window, we center it.
    view_mean = np.mean(view, axis=1, keepdims=True)
    centered = view - view_mean
    
    # covariance = (X^T X) / (window - 1)
    # centered: (N, W, F), centered.transpose: (N, F, W)
    cov_matrices = np.matmul(centered.transpose(0, 2, 1), centered) / (window - 1)

    # 3) Rolling Eigenvalue Decomposition (EVD)
    # eigh returns eigenvalues in ascending order, eigenvectors in corresponding columns
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrices)

    # Extract top safe_k eigenvectors (last safe_k columns)
    # top_eigenvectors shape: (N, F, safe_k)
    top_eigenvectors = eigenvectors[:, :, -safe_k:]

    # 4) Market Factor Projection
    # Project the last observation of each window onto the principal components
    # current_obs shape: (N, F) -> expand to (N, 1, F)
    current_obs = view[:, -1, :]
    centered_obs = current_obs - view_mean[:, 0, :]
    
    # scores = centered_obs @ top_eigenvectors
    # (N, 1, F) @ (N, F, k) -> (N, 1, k)
    scores = np.matmul(np.expand_dims(centered_obs, axis=1), top_eigenvectors)

    # 5) Reconstruct Beta and Extract Alpha
    # beta = scores @ top_eigenvectors^T + mean
    # (N, 1, k) @ (N, k, F) -> (N, 1, F)
    beta_recon = np.matmul(scores, top_eigenvectors.transpose(0, 2, 1)).squeeze(1)
    beta_reconstructed = beta_recon + view_mean[:, 0, :]
    
    alpha_residual = current_obs - beta_reconstructed

    # Pad the beginning to maintain original (T, F) shape
    padding = np.full((window - 1, n_features), np.nan)
    beta_full = np.vstack((padding, beta_reconstructed))
    alpha_full = np.vstack((padding, alpha_residual))

    beta_df = pd.DataFrame(beta_full, index=standardized.index, columns=standardized.columns)
    alpha_df = pd.DataFrame(alpha_full, index=standardized.index, columns=standardized.columns)

    return beta_df, alpha_df


def rolling_pca_summary(
    standardized: pd.DataFrame,
    *,
    window: int = 60,
    n_components: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    Returns the explained variance ratio of the *last* rolling window.
    Useful for visualizing the current principal component strengths.
    """
    data = standardized.values
    if len(data) < window:
        evr = pd.Series([np.nan]*n_components, index=[f"PC{i+1}" for i in range(n_components)])
        return {
            "explained_variance_ratio": evr.to_frame(name="explained_variance_ratio"),
            "cumulative_explained_variance": evr.to_frame(name="cumulative_explained_variance"),
        }
        
    last_window = data[-window:]
    centered = last_window - np.mean(last_window, axis=0)
    cov = np.dot(centered.T, centered) / (window - 1)
    
    # eigh is faster for symmetric matrices
    eigenvalues, _ = np.linalg.eigh(cov)
    
    # eigenvalues are ascending, so reverse them
    eigenvalues = eigenvalues[::-1]
    
    total_var = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)
    
    safe_n = min(n_components, window, standardized.shape[1])
    evr = pd.Series(
        explained_variance_ratio[:safe_n],
        index=[f"PC{i+1}" for i in range(safe_n)],
        name="explained_variance_ratio",
    )
    cev = evr.cumsum().rename("cumulative_explained_variance")

    return {
        "explained_variance_ratio": evr.to_frame(),
        "cumulative_explained_variance": cev.to_frame(),
    }
