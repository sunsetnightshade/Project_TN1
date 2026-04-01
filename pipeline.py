from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from config import ALL_TICKERS, PRIMARY_TICKERS, RESERVE_BENCH
from data_cleaner import CleaningResult, clean_and_replace_zombies
from data_fetcher import fetch_adj_close_prices
from matrix_math import build_aligned_log_return_matrix
from standardizer import StandardizationResult, standardize_and_plot_heatmap


@dataclass(frozen=True)
class PipelineArtifacts:
    prices: pd.DataFrame
    cleaning: CleaningResult
    aligned_log_returns: pd.DataFrame
    standardization: StandardizationResult
    paths: dict[str, Path]


def _stamp(d: date) -> str:
    return f"{d.year:04d}_{d.month:02d}_{d.day:02d}"


def to_accessible_30xT_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Export as 30×T with tickers as rows and dates as columns.
    """
    out = df.T.copy()
    if isinstance(out.columns, pd.DatetimeIndex):
        out.columns = out.columns.strftime("%Y-%m-%d")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=True)


def build_and_serialize(
    *,
    start_date: date,
    end_date: date,
    missing_threshold: float,
    root_dir: Path,
) -> PipelineArtifacts:
    """
    Full pipeline + serialization protocol.
    Writes pickles to storage/ and heatmap + accessible CSVs to outputs/latest + outputs/archive.
    """
    root_dir = Path(root_dir)
    storage_dir = root_dir / "storage"
    outputs_dir = root_dir / "outputs"
    latest_dir = outputs_dir / "latest"
    archive_dir = outputs_dir / "archive"

    storage_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    prices = fetch_adj_close_prices(
        tickers=ALL_TICKERS,
        start_date=start_date,
        end_date=end_date,
    )

    cleaning = clean_and_replace_zombies(
        prices,
        primary_tickers=list(PRIMARY_TICKERS),
        reserve_tickers=list(RESERVE_BENCH),
        missing_frac_threshold=missing_threshold,
    )

    aligned_log_returns = build_aligned_log_return_matrix(cleaning.prices)

    heatmap_path = latest_dir / "matrix_heatmap.png"
    standardization = standardize_and_plot_heatmap(
        aligned_log_returns, heatmap_path=heatmap_path
    )

    current_matrix_path = storage_dir / "current_matrix.pkl"
    backup_matrix_path = storage_dir / f"matrix_{_stamp(date.today())}.pkl"
    scaler_params_path = storage_dir / "scaler_params.pkl"

    standardization.standardized.to_pickle(current_matrix_path)
    standardization.standardized.to_pickle(backup_matrix_path)
    pd.to_pickle(standardization.scaler_params, scaler_params_path)

    stamp = _stamp(date.today())
    returns_csv_latest = latest_dir / "aligned_log_returns_30xT.csv"
    returns_csv_archive = archive_dir / f"aligned_log_returns_{stamp}_30xT.csv"
    standardized_csv_latest = latest_dir / "standardized_matrix_30xT.csv"
    standardized_csv_archive = archive_dir / f"standardized_matrix_{stamp}_30xT.csv"

    to_accessible_30xT_csv(aligned_log_returns, returns_csv_latest)
    to_accessible_30xT_csv(aligned_log_returns, returns_csv_archive)
    to_accessible_30xT_csv(standardization.standardized, standardized_csv_latest)
    to_accessible_30xT_csv(standardization.standardized, standardized_csv_archive)

    return PipelineArtifacts(
        prices=prices,
        cleaning=cleaning,
        aligned_log_returns=aligned_log_returns,
        standardization=standardization,
        paths={
            "current_matrix": current_matrix_path,
            "backup_matrix": backup_matrix_path,
            "scaler_params": scaler_params_path,
            "matrix_heatmap": heatmap_path,
            "returns_csv_latest": returns_csv_latest,
            "standardized_csv_latest": standardized_csv_latest,
            "returns_csv_archive": returns_csv_archive,
            "standardized_csv_archive": standardized_csv_archive,
        },
    )

