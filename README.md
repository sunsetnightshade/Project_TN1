# Quick Start: How to Run

> **Important**: To run the live application, you **must use two separate terminal windows**.

### 1) Start the Live Websocket Server (Terminal 1)
Leave this terminal open and running. It fetches data in the background and pushes updates every 5 seconds.
```powershell
cd "C:\Users\Naman Sinha\Desktop\quant_matrix_cli"
py main.py --serve-live --interval 5 --host 127.0.0.1 --port 8765
```
> **Note:** The current implementation guarantees a websocket update every 5 seconds and fetches as fast as yfinance allows.

### 2) Start the Live CLI Dashboard (Terminal 2)
Open a new terminal to connect to the server and view the live streaming dashboard.
```powershell
cd "C:\Users\Naman Sinha\Desktop\quant_matrix_cli"
py main.py --live --ws-url ws://127.0.0.1:8765
```

### Understanding the Dashboard Data (Z-Scores)
The dashboard displays the tail of the **Aligned, Standardized Log-Return Matrix**. 
- The data shown are **not prices**, but rather **Standardized Z-Scores** of the daily log returns.
- A value of `0.0` means the stock performed exactly on its average.
- Positive values (e.g., `1.82`) mean the stock had an unusually good day compared to its average daily return.
- Negative values (e.g., `-2.54`) mean the stock got crushed compared to its average.
This standardization scales every stock to the exact same "baseline" (zero mean, unit variance), allowing models to compare volatility fairly across all 30 tickers.

---

### One-time Build (Alternative to live mode)
If you just want to run the pipeline once and generate the CSV/PNG artifacts:
```powershell
cd "C:\Users\Naman Sinha\Desktop\quant_matrix_cli"
py main.py --build
```

---

# Quant Matrix CLI


Quant Matrix CLI builds a synchronized log-return matrix for NIFTY IT and NASDAQ mega-cap tickers, applies a strict zombie-ticker replacement policy, standardizes the result, and preserves the latest scaler & matrix for downstream analysis. It also renders a correlation heatmap so you can visualize the normalized relationships across markets.

## What the pipeline does
1. `data_fetcher.py` downloads adjusted close prices for the configured tickers using `yfinance`.
2. `data_cleaner.py` drops primary tickers with >5% missing data, swaps in reserve tickers to keep a 30-column matrix, and linearly interpolates small gaps.
3. `matrix_math.py` converts prices to log returns, shifts U.S. columns by one day to align with Indian trading, and drops any remaining NaNs.
4. `standardizer.py` z-scores every column, saves the scaler parameters, and exports a 30×T aligned matrix heatmap image (tickers × days).
5. `main.py` orchestrates the process, saves the standardized matrix/history snapshots to `storage/`, also writes Excel-friendly `30×T` CSVs to `outputs/`, and prints summary paths plus any zombie replacements.

## Repository layout
- `main.py` – command-line entry point with `--build` and `--verify` flags.
- `config.py` – ticker lists and default [start_date, end_date) window (last two years by default).
- `data_fetcher.py` – resilient `yfinance` download with retries and column alignment.
- `data_cleaner.py` – enforces the 30-ticker rule, drops zombies, and applies replacements.
- `matrix_math.py` – log-return builder with NYC-to-India alignment.
- `standardizer.py` – scales, stores scaler metadata, and exports `correlation_heatmap.png`.
- `storage/` – runtime output (current matrix, backups, scaler params).
- `outputs/latest/` – latest heatmap + accessible `30×T` CSVs.
- `outputs/archive/` – dated snapshots of the accessible `30×T` CSVs.

## Prerequisites (Windows)
1. Install Python 3.10+ from [python.org][1] and ensure `python` is on your PATH.
2. (Optional but recommended) Install Git for Windows to manage source control and push changes.
3. Ensure Internet access so `yfinance` can reach Yahoo Finance.

## Local setup and execution
```powershell
cd path\\to\\quant_matrix_cli
python -m venv .venv
.\\.venv\\Scripts\\Activate
python -m pip install --upgrade pip
pip install pandas numpy scipy scikit-learn seaborn matplotlib yfinance
```

> Alternatively, once you add a `requirements.txt`, replace the last command with `pip install -r requirements.txt`.

### Run the pipeline
```powershell
python main.py --build
```
- Downloads prices for the configured tickers (primary + reserve bench).
- Cleans zombies, builds returns, standardizes, and saves:
  - `storage/current_matrix.pkl`
  - `storage/matrix_YYYY_MM_DD.pkl` (daily backup)
  - `storage/scaler_params.pkl`
  - `outputs/latest/matrix_heatmap.png`
  - `outputs/latest/aligned_log_returns_30xT.csv`
  - `outputs/latest/standardized_matrix_30xT.csv`

### Verify outputs without rebuilding
```powershell
python main.py --verify
```
Checks that `storage/current_matrix.pkl` and `storage/scaler_params.pkl` exist.

## Configuration
- Edit `config.py` to customize `PRIMARY_TICKERS`, `RESERVE_BENCH`, `START_DATE`, or `END_DATE`. The pipeline uses the frozen `PipelineConfig` dataclass when instantiated.

## Zombie replacement logic
- Any primary ticker with >5% missing data is removed and replaced with the next available reserve that is not already a primary.
- The replacement keeps the original primary order and maintains 30 columns total.
- After replacements, the dataframe interpolates any remaining short gaps before computing returns.

## Data science artifacts
- The standardized matrix stores z-scores (zero mean, unit variance) for each ticker over the synced window.
- Scaler parameters are persisted so you can standardize new data with the same transformation outside this CLI.
- Correlation heatmap visualizes how the standardized returns co-move after alignment.

## GitHub workflow hints
1. `git init` (if not already a repo) and commit the source + README.
2. `git remote add origin https://github.com/sunsetnightshade/stat_cli.git`.
3. Push your branch and open a pull request from that branch to the target repository.
4. Share the PR URL back here when ready.

Let me know if you want me to draft the PR description or add CI/formatting guidance.
[1]: https://www.python.org
