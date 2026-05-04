from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import LiveIngestConfig
from pca_engine import rolling_pca_summary

from .analytics import (
    compute_log_returns_vectorized,
    residual_zscore_latest,
    rolling_zscore_latest,
)
from .consumer import RedisLiveConsumer


@dataclass(frozen=True)
class LiveSnapshotArtifacts:
    close_matrix: pd.DataFrame
    log_returns: pd.DataFrame
    rolling_zscores: pd.DataFrame
    latest_zscore: pd.Series
    latest_residual_zscore: pd.Series
    pca_explained: pd.DataFrame
    paths: dict[str, Path]


def _to_accessible_matrix_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.T.copy()
    if isinstance(out.columns, pd.DatetimeIndex):
        idx = out.columns
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        out.columns = idx.strftime("%Y-%m-%d %H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=True)


def build_live_snapshot_from_redis(
    *,
    cfg: LiveIngestConfig,
    root_dir: Path,
    lookback_minutes: int = 500,
    z_window: int = 60,
    pca_components: int = 3,
) -> LiveSnapshotArtifacts:
    if lookback_minutes < z_window:
        raise ValueError("lookback_minutes must be >= z_window")

    consumer = RedisLiveConsumer(
        host=cfg.redis_host,
        port=cfg.redis_port,
        db=cfg.redis_db,
        stream_key=cfg.redis_stream_500,
        symbols=cfg.symbols,
        price_scale=cfg.price_scale,
    )
    consumer.ping()
    close_matrix = consumer.build_close_matrix(limit_minutes=lookback_minutes)
    if close_matrix.empty:
        raise ValueError("No live bars found in Redis stream")

    log_returns = compute_log_returns_vectorized(close_matrix)
    rolling_zscores, latest_zscore = rolling_zscore_latest(log_returns, window=z_window)
    latest_residual = residual_zscore_latest(
        log_returns,
        window=lookback_minutes,
        factors=1,
    )

    pca_input = rolling_zscores.tail(max(z_window, pca_components + 2))
    pca_summary = rolling_pca_summary(
        pca_input,
        n_components=min(pca_components, pca_input.shape[1]),
    )
    pca_explained = pca_summary["explained_variance_ratio"].copy()

    root_dir = Path(root_dir)
    latest_dir = root_dir / "outputs" / "latest"
    live_dir = latest_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)

    close_path = live_dir / "live_close_1m_30xT.csv"
    returns_path = live_dir / "live_log_returns_1m_30xT.csv"
    rolling_path = live_dir / "live_rolling_zscores_1m_30xT.csv"
    latest_z_path = live_dir / "live_latest_zscore.csv"
    latest_residual_path = live_dir / "live_latest_residual_zscore.csv"
    pca_path = live_dir / "live_pca_explained.csv"

    _to_accessible_matrix_csv(close_matrix, close_path)
    _to_accessible_matrix_csv(log_returns, returns_path)
    _to_accessible_matrix_csv(rolling_zscores, rolling_path)
    latest_zscore.to_frame(name="zscore").to_csv(latest_z_path, index=True)
    latest_residual.to_frame(name="residual_zscore").to_csv(latest_residual_path, index=True)
    pca_explained.to_csv(pca_path, index=True)

    return LiveSnapshotArtifacts(
        close_matrix=close_matrix,
        log_returns=log_returns,
        rolling_zscores=rolling_zscores,
        latest_zscore=latest_zscore,
        latest_residual_zscore=latest_residual,
        pca_explained=pca_explained,
        paths={
            "live_close_csv": close_path,
            "live_log_returns_csv": returns_path,
            "live_rolling_zscores_csv": rolling_path,
            "live_latest_zscore_csv": latest_z_path,
            "live_latest_residual_zscore_csv": latest_residual_path,
            "live_pca_explained_csv": pca_path,
        },
    )
