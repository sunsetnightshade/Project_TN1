# Quant Matrix — Output Guide

> **Generated automatically by the pipeline on 2026-04-10T06:54:58.232579+00:00**
> Run: `2024-04-10 → 2026-04-10`

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
