# Quant Matrix CLI

Quant Matrix CLI builds a synchronized log-return matrix for NIFTY IT and NASDAQ mega-cap tickers, applies a strict zombie-ticker replacement policy, standardizes the result, and preserves the latest scaler & matrix for downstream analysis. It also renders a correlation heatmap so you can visualize the normalized relationships across markets.

## What the pipeline does
1. `data_fetcher.py` downloads adjusted close prices for the configured tickers using `yfinance`.
2. `data_cleaner.py` drops primary tickers with >5% missing data, swaps in reserve tickers to keep a 30-column matrix, and linearly interpolates small gaps.
3. `matrix_math.py` converts prices to log returns, shifts U.S. columns by one day to align with Indian trading, and drops any remaining NaNs.
4. `standardizer.py` z-scores every column, saves the scaler parameters, and exports a correlation heatmap image.
5. `main.py` orchestrates the process, saves the standardized matrix/history snapshots to `storage/`, and prints summary paths plus any zombie replacements.

## Repository layout
- `main.py` – command-line entry point with `--build` and `--verify` flags.
- `config.py` – ticker lists and default [start_date, end_date) window (last two years by default).
- `data_fetcher.py` – resilient `yfinance` download with retries and column alignment.
- `data_cleaner.py` – enforces the 30-ticker rule, drops zombies, and applies replacements.
- `matrix_math.py` – log-return builder with NYC-to-India alignment.
- `standardizer.py` – scales, stores scaler metadata, and exports `correlation_heatmap.png`.
- `storage/` – runtime output (current matrix, backups, scaler params).

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
  - `correlation_heatmap.png` in the repo root.

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
