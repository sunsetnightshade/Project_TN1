from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import pandas as pd

from config import ALL_TICKERS, END_DATE, PRIMARY_TICKERS, RESERVE_BENCH, START_DATE
from data_cleaner import clean_and_replace_zombies
from data_fetcher import fetch_adj_close_prices
from matrix_math import build_aligned_log_return_matrix
from standardizer import standardize_and_plot_heatmap


ROOT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = ROOT_DIR / "storage"


def _backup_name(d: date) -> str:
    return f"matrix_{d.year:04d}_{d.month:02d}_{d.day:02d}.pkl"


def build_pipeline() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    prices = fetch_adj_close_prices(
        tickers=ALL_TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    cleaning = clean_and_replace_zombies(
        prices,
        primary_tickers=list(PRIMARY_TICKERS),
        reserve_tickers=list(RESERVE_BENCH),
        missing_frac_threshold=0.05,
    )

    returns = build_aligned_log_return_matrix(cleaning.prices)

    heatmap_path = ROOT_DIR / "correlation_heatmap.png"
    std = standardize_and_plot_heatmap(returns, heatmap_path=heatmap_path)

    current_path = STORAGE_DIR / "current_matrix.pkl"
    backup_path = STORAGE_DIR / _backup_name(date.today())
    scaler_path = STORAGE_DIR / "scaler_params.pkl"

    std.standardized.to_pickle(current_path)
    std.standardized.to_pickle(backup_path)
    pd.to_pickle(std.scaler_params, scaler_path)

    if cleaning.dropped_primaries:
        dropped = ", ".join(cleaning.dropped_primaries)
        repl = ", ".join([f"{a}->{b}" for a, b in cleaning.replacements])
        print(f"Zombie tickers dropped: {dropped}")
        print(f"Replacements applied: {repl}")

    print(f"Saved current matrix to: {current_path}")
    print(f"Saved backup matrix to: {backup_path}")
    print(f"Saved scaler params to: {scaler_path}")
    print(f"Saved correlation heatmap to: {heatmap_path}")


def verify_storage() -> int:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    expected = [
        STORAGE_DIR / "current_matrix.pkl",
        STORAGE_DIR / "scaler_params.pkl",
    ]

    missing = [p for p in expected if not p.exists()]
    if missing:
        for p in missing:
            print(f"Missing: {p}")
        return 1

    print("storage/ looks OK.")
    for p in expected:
        print(f"Found: {p}")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a synchronized T×30 log-return matrix for NIFTY IT + NASDAQ 100."
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run full pipeline and serialize outputs to storage/.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify storage/ contains expected output files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if not args.build and not args.verify:
        print("Nothing to do. Use --build and/or --verify.")
        return 2

    if args.build:
        build_pipeline()

    if args.verify:
        return verify_storage()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

