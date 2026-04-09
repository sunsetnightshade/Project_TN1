from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from config import ALL_TICKERS, PRIMARY_TICKERS, RESERVE_BENCH, LiveIngestConfig
from data_cleaner import CleaningResult, clean_and_replace_zombies
from data_fetcher import fetch_adj_close_prices
from live_ingest.snapshot import LiveSnapshotArtifacts, build_live_snapshot_from_redis
from live_persistence import HourlyPersistenceResult, HourlyParquetJanitor
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


def _ts_stamp() -> str:
    """Full ISO-like timestamp stamp for archiving (safe for filenames)."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d_%H%M%S")


def to_accessible_30xT_csv(df: pd.DataFrame, path: Path) -> None:
    """Export as 30×T with tickers as rows and dates as columns."""
    out = df.T.copy()
    if isinstance(out.columns, pd.DatetimeIndex):
        out.columns = out.columns.strftime("%Y-%m-%d")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=True)


# ---------------------------------------------------------------------------
# Archive previous latest/ before writing new outputs
# ---------------------------------------------------------------------------

def _archive_latest(latest_dir: Path, archive_dir: Path) -> Path | None:
    """
    Move ALL contents of outputs/latest/ (files AND subdirectories) into
    outputs/archive/<timestamp>/ so the next run starts with a clean latest/.

    Returns the archive subdirectory path, or None if latest was empty.
    """
    latest_dir = Path(latest_dir)
    archive_dir = Path(archive_dir)

    if not latest_dir.exists():
        return None

    # Include both files and subdirectories (e.g. live/ from snapshot builds)
    items = list(latest_dir.iterdir())
    if not items:
        return None

    ts = _ts_stamp()
    dest = archive_dir / ts
    dest.mkdir(parents=True, exist_ok=True)

    for item in items:
        shutil.move(str(item), str(dest / item.name))

    return dest


# ---------------------------------------------------------------------------
# Auto-generated GUIDE.md for the outputs folder
# ---------------------------------------------------------------------------

_GUIDE_TEMPLATE = """\
# Quant Matrix — Output Guide

> **Generated automatically by the pipeline on {timestamp}**
> Run: `{date_range}`

---

## What's Inside

| File | What It Is | How to Use It |
|------|-----------|---------------|
| `matrix_heatmap.png` | 30×T heatmap — each row is a ticker, each column is a trading day. Colour = Z-score (red = overperforming, blue = underperforming). | Open the PNG to visually scan for regime changes or outlier days. |
| `correlation_heatmap.png` | 30×30 pairwise Pearson correlation between all US tech stocks. Expected range: 0.5–0.9 (high because they're all tech). | If any cell is blue/near-zero, that ticker may have bad data or be a non-tech intruder. |
| `correlation_outliers.json` | Machine-readable list of ticker pairs with suspiciously low correlation (below 0.3). | Parse this in Python/Excel. If non-empty, investigate the flagged tickers. |
| `standardized_matrix_30xT.csv` | The final Z-score standardized matrix. Rows = tickers, columns = dates. This is the core analytical output. | Import into Excel, Python, or R. Each cell is a Z-score (mean 0, std 1 per ticker). |
| `aligned_log_returns_30xT.csv` | Raw log returns before standardization. Same 30×T orientation. | Use for your own analytics (e.g. covariance estimation, factor models). |
| `build_metadata.json` | Machine-readable build metadata: date range, ticker list, zombie replacements, timing. | Audit trail — check which tickers were replaced and when the build ran. |
| `GUIDE.md` | This file. | You're reading it! |

## How to Read the Heatmaps

- **Matrix heatmap**: If a ticker row is persistently red (high Z-score), it's outperforming the cross-section. Persistently blue = underperforming. A sudden colour change marks a regime shift.
- **Correlation heatmap**: All 30 US Nasdaq-100 tech stocks should show 0.5–0.9 correlation. If you see a blue square (< 0.3), that ticker is decoupled — possibly a data issue or a genuinely uncorrelated asset.

## Where Are Older Runs?

Previous runs are automatically moved to `../archive/<timestamp>/` before new outputs are written. Each archive folder has the exact same files.

## Storage (Internal Pipeline State)

| File | Location |
|------|----------|
| `current_matrix.pkl` | `storage/` — latest matrix as a Python pickle. Overwritten every run. |
| `current_matrix.parquet` | `storage/` — same matrix as Apache Parquet for long-term storage. |
| `matrix_YYYY_MM_DD.pkl` | `storage/` — timestamped backup (never overwritten). |
| `scaler_params.pkl` | `storage/` — StandardScaler mean_ and scale_ arrays for inverse transforms. |

## Quick Commands

```powershell
# Build the matrix (default action — just run main.py)
py main.py

# Open the Streamlit dashboard
py -m streamlit run app.py

# Verify outputs exist
py main.py --verify

# Run PCA on the current matrix
py main.py --interactive   # then choose option 4
```
"""


def _generate_guide(
    latest_dir: Path,
    *,
    start_date: date,
    end_date: date,
    timestamp: str,
) -> Path:
    """Write the GUIDE.md into latest_dir."""
    content = _GUIDE_TEMPLATE.format(
        timestamp=timestamp,
        date_range=f"{start_date.isoformat()} → {end_date.isoformat()}",
    )
    path = latest_dir / "GUIDE.md"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Build metadata JSON
# ---------------------------------------------------------------------------

def _save_build_metadata(
    latest_dir: Path,
    *,
    start_date: date,
    end_date: date,
    cleaning: CleaningResult,
    standardized_shape: tuple[int, int],
    tickers: list[str],
    outlier_count: int,
    build_ts: str,
) -> Path:
    meta = {
        "build_timestamp_utc": build_ts,
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "tickers": tickers,
        "matrix_shape": {"rows_T": standardized_shape[0], "cols_tickers": standardized_shape[1]},
        "zombies_dropped": list(cleaning.dropped_primaries),
        "replacements": [{"dropped": a, "replaced_with": b} for a, b in cleaning.replacements],
        "correlation_outlier_count": outlier_count,
    }
    path = latest_dir / "build_metadata.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------------

def build_and_serialize(
    *,
    start_date: date,
    end_date: date,
    missing_threshold: float,
    root_dir: Path,
) -> PipelineArtifacts:
    """
    Full pipeline + serialization protocol.

    Steps:
      1. Archive ALL existing outputs/latest/ files → outputs/archive/<timestamp>/
      2. Fetch raw Adj Close prices
      3. Zombie ticker detection + replacement (>5% missing → swap with reserve)
      4. Linear interpolation for 1–2 day gaps (calendar-faithful)
      5. Log returns: r_t = ln(P_t / P_{t-1})
      6. dropna() — removes first row and any residual gaps
      7. Z-score standardization via StandardScaler (column-wise)
      8. Save ALL artifacts to outputs/latest/ (clean, no stale files)
      9. Generate GUIDE.md + build_metadata.json
    """
    root_dir = Path(root_dir)
    storage_dir = root_dir / "storage"
    outputs_dir = root_dir / "outputs"
    latest_dir = outputs_dir / "latest"
    archive_dir = outputs_dir / "archive"

    storage_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    build_ts = datetime.now(timezone.utc).isoformat()

    # ---- Step 1: Archive old latest/ ----
    archived_to = _archive_latest(latest_dir, archive_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 2: Fetch ----
    prices = fetch_adj_close_prices(
        tickers=ALL_TICKERS,
        start_date=start_date,
        end_date=end_date,
    )

    # ---- Steps 3–4: Clean + zombie replacement + interpolation ----
    cleaning = clean_and_replace_zombies(
        prices,
        primary_tickers=list(PRIMARY_TICKERS),
        reserve_tickers=list(RESERVE_BENCH),
        missing_frac_threshold=missing_threshold,
    )

    # ---- Steps 5–6: Log returns + dropna ----
    aligned_log_returns = build_aligned_log_return_matrix(cleaning.prices)

    # ---- Steps 7+8: Standardize + both heatmaps ----
    heatmap_path = latest_dir / "matrix_heatmap.png"
    standardization = standardize_and_plot_heatmap(
        aligned_log_returns, heatmap_path=heatmap_path
    )

    # ---- Pickle + Parquet saves ----
    stamp = _stamp(date.today())

    current_matrix_path = storage_dir / "current_matrix.pkl"
    backup_matrix_path = storage_dir / f"matrix_{stamp}.pkl"
    current_parquet_path = storage_dir / "current_matrix.parquet"
    backup_parquet_path = storage_dir / f"matrix_{stamp}.parquet"
    scaler_params_path = storage_dir / "scaler_params.pkl"

    standardization.standardized.to_pickle(current_matrix_path)
    standardization.standardized.to_pickle(backup_matrix_path)
    standardization.standardized.to_parquet(current_parquet_path, engine="pyarrow")
    standardization.standardized.to_parquet(backup_parquet_path, engine="pyarrow")
    pd.to_pickle(standardization.scaler_params, scaler_params_path)

    # ---- CSV exports (all to latest/) ----
    returns_csv = latest_dir / "aligned_log_returns_30xT.csv"
    standardized_csv = latest_dir / "standardized_matrix_30xT.csv"

    to_accessible_30xT_csv(aligned_log_returns, returns_csv)
    to_accessible_30xT_csv(standardization.standardized, standardized_csv)

    # ---- Correlation outlier report ----
    outliers = standardization.scaler_params.get("correlation_outliers", [])
    corr_outliers_path = latest_dir / "correlation_outliers.json"
    corr_outliers_path.write_text(
        json.dumps({"outliers": outliers, "count": len(outliers)}, indent=2),
        encoding="utf-8",
    )

    corr_heatmap_path = latest_dir / "correlation_heatmap.png"

    # ---- Step 9: Guide + metadata ----
    guide_path = _generate_guide(
        latest_dir,
        start_date=start_date,
        end_date=end_date,
        timestamp=build_ts,
    )

    metadata_path = _save_build_metadata(
        latest_dir,
        start_date=start_date,
        end_date=end_date,
        cleaning=cleaning,
        standardized_shape=standardization.standardized.shape,
        tickers=list(standardization.standardized.columns),
        outlier_count=len(outliers),
        build_ts=build_ts,
    )

    return PipelineArtifacts(
        prices=prices,
        cleaning=cleaning,
        aligned_log_returns=aligned_log_returns,
        standardization=standardization,
        paths={
            "current_matrix": current_matrix_path,
            "backup_matrix": backup_matrix_path,
            "current_matrix_parquet": current_parquet_path,
            "backup_matrix_parquet": backup_parquet_path,
            "scaler_params": scaler_params_path,
            "matrix_heatmap": heatmap_path,
            "correlation_heatmap": corr_heatmap_path,
            "correlation_outliers": corr_outliers_path,
            "returns_csv": returns_csv,
            "standardized_csv": standardized_csv,
            "guide": guide_path,
            "build_metadata": metadata_path,
            "archived_to": Path(str(archived_to)) if archived_to else Path("(none)"),
        },
    )


def build_live_snapshot(
    *,
    live_cfg: LiveIngestConfig,
    root_dir: Path,
    lookback_minutes: int = 500,
    z_window: int = 60,
    pca_components: int = 3,
) -> LiveSnapshotArtifacts:
    return build_live_snapshot_from_redis(
        cfg=live_cfg,
        root_dir=root_dir,
        lookback_minutes=lookback_minutes,
        z_window=z_window,
        pca_components=pca_components,
    )


def persist_live_hourly(
    *,
    live_cfg: LiveIngestConfig,
    root_dir: Path,
    hours: int = 1,
) -> HourlyPersistenceResult:
    janitor = HourlyParquetJanitor(cfg=live_cfg, root_dir=root_dir)
    return janitor.persist_recent(hours=hours)
