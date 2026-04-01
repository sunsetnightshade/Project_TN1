from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline import PipelineArtifacts, build_and_serialize
from pca_tools import pca_beta_alpha, pca_fit_summary


def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _press_enter() -> None:
    input("\nPress Enter to continue...")


def _fmt_path(p: Path) -> str:
    return str(p.resolve())


def _load_current_matrix(root_dir: Path) -> pd.DataFrame | None:
    p = Path(root_dir) / "storage" / "current_matrix.pkl"
    if not p.exists():
        return None
    return pd.read_pickle(p)


def _print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def _print_artifacts(artifacts: PipelineArtifacts) -> None:
    c = artifacts.cleaning
    if c.dropped_primaries:
        print("Zombie tickers dropped:", ", ".join(c.dropped_primaries))
        print("Replacements:", ", ".join([f"{a}->{b}" for a, b in c.replacements]))
    else:
        print("Zombie tickers dropped: none")

    print("\nSaved outputs:")
    for k, v in artifacts.paths.items():
        print(f"- {k}: {_fmt_path(v)}")


def _matrix_quick_stats(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "shape": tuple(df.shape),
        "start": str(df.index.min().date()) if len(df.index) else None,
        "end": str(df.index.max().date()) if len(df.index) else None,
        "columns": len(df.columns),
        "nan_frac": float(df.isna().mean().mean()),
    }


def run_interactive_cli(*, root_dir: Path) -> int:
    root_dir = Path(root_dir)
    last_artifacts: PipelineArtifacts | None = None

    start_default = date.today() - timedelta(days=730)
    end_default = date.today()
    missing_threshold_default = 0.05

    while True:
        _clear_screen()
        _print_header("Quant Matrix CLI (Interactive)")
        print("1) Build pipeline (fetch -> clean -> align -> standardize -> save)")
        print("2) Verify storage outputs")
        print("3) Show current matrix quick stats")
        print("4) Run PCA (beta/alpha) on current matrix")
        print("5) Show latest saved artifact paths")
        print("0) Exit")

        choice = input("\nSelect an option: ").strip()
        if choice == "0":
            return 0

        if choice == "1":
            _clear_screen()
            _print_header("Build pipeline")
            print(f"Default start: {start_default.isoformat()}")
            print(f"Default end  : {end_default.isoformat()}")
            print(f"Default zombie missing threshold: {missing_threshold_default}")

            s = input("Start date (YYYY-MM-DD) [Enter=default]: ").strip()
            e = input("End date   (YYYY-MM-DD) [Enter=default]: ").strip()
            t = input("Zombie threshold (e.g. 0.05) [Enter=default]: ").strip()

            start = start_default if not s else date.fromisoformat(s)
            end = end_default if not e else date.fromisoformat(e)
            thr = missing_threshold_default if not t else float(t)

            print("\nBuilding... (yfinance can be slow; retries on throttling)")
            last_artifacts = build_and_serialize(
                start_date=start,
                end_date=end,
                missing_threshold=thr,
                root_dir=root_dir,
            )
            print("\nBuild complete.\n")
            _print_artifacts(last_artifacts)
            _press_enter()
            continue

        if choice == "2":
            _clear_screen()
            _print_header("Verify storage outputs")
            expected = [
                root_dir / "storage" / "current_matrix.pkl",
                root_dir / "storage" / "scaler_params.pkl",
                root_dir / "outputs" / "latest" / "matrix_heatmap.png",
                root_dir / "outputs" / "latest" / "aligned_log_returns_30xT.csv",
                root_dir / "outputs" / "latest" / "standardized_matrix_30xT.csv",
            ]
            missing = [p for p in expected if not p.exists()]
            if missing:
                print("Missing outputs:")
                for p in missing:
                    print("-", _fmt_path(p))
            else:
                print("All expected outputs are present.")
            _press_enter()
            continue

        if choice == "3":
            _clear_screen()
            _print_header("Current matrix quick stats")
            df = _load_current_matrix(root_dir)
            if df is None:
                print("No current matrix found. Run option 1 first.")
            else:
                stats = _matrix_quick_stats(df)
                print(json.dumps(stats, indent=2))
                print("\nTail (last 5 rows):")
                print(df.tail(5).to_string())
            _press_enter()
            continue

        if choice == "4":
            _clear_screen()
            _print_header("PCA (beta/alpha)")
            df = _load_current_matrix(root_dir)
            if df is None:
                print("No current matrix found. Run option 1 first.")
                _press_enter()
                continue

            k_str = input("How many PCs to treat as Beta? [default=1]: ").strip()
            k = 1 if not k_str else int(k_str)
            summary = pca_fit_summary(df, n_components=min(max(k, 1), df.shape[1]))
            print("\nExplained variance ratio:")
            print(summary["explained_variance_ratio"].to_string())
            print("\nCumulative explained variance:")
            print(summary["cumulative_explained_variance"].to_string())

            beta, alpha, _ = pca_beta_alpha(df, k=k)
            out_dir = root_dir / "outputs" / "latest"
            out_dir.mkdir(parents=True, exist_ok=True)
            beta_path = out_dir / "beta_component_30xT.csv"
            alpha_path = out_dir / "alpha_residual_30xT.csv"
            beta.T.to_csv(beta_path, index=True)
            alpha.T.to_csv(alpha_path, index=True)
            print(f"\nSaved beta component CSV: {_fmt_path(beta_path)}")
            print(f"Saved alpha residual CSV: {_fmt_path(alpha_path)}")
            _press_enter()
            continue

        if choice == "5":
            _clear_screen()
            _print_header("Latest artifact paths")
            if last_artifacts is None:
                print("No build run yet in this session.")
                print("Tip: outputs are stored under:")
                print("-", _fmt_path(root_dir / "storage"))
                print("-", _fmt_path(root_dir / "outputs"))
            else:
                _print_artifacts(last_artifacts)
            _press_enter()
            continue

        print("Invalid option.")
        _press_enter()


async def run_live_cli_via_websocket(
    *,
    ws_url: str,
    root_dir: Path,
    show_tail_rows: int = 5,
) -> int:
    """
    Connects to a local websocket stream and prints a simple live dashboard.
    """
    import websockets  # type: ignore

    root_dir = Path(root_dir)
    _clear_screen()
    _print_header("Quant Matrix CLI (Live)")
    print(f"Connecting to {ws_url} ...\n")

    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
        while True:
            msg = await ws.recv()
            payload = json.loads(msg)

            _clear_screen()
            _print_header("Quant Matrix CLI (Live)")
            print(f"Last update: {payload.get('timestamp')}")
            print(f"Status     : {payload.get('status')}")
            if payload.get("error"):
                print("\nError:")
                print(payload["error"])

            # Show quick stats + tail from current matrix on disk (most reliable for size)
            df = _load_current_matrix(root_dir)
            if df is not None:
                print("\nCurrent matrix stats:")
                print(json.dumps(_matrix_quick_stats(df), indent=2))
                print(f"\nTail (last {show_tail_rows} rows):")
                print(df.tail(show_tail_rows).to_string())

            print("\nCtrl+C to exit.")

