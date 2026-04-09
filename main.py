#!/usr/bin/env python3
"""
Quant Matrix CLI - NASDAQ-100 Tech, 30 Stocks

Default action (no flags):  build the full pipeline
                            -> outputs land in outputs/latest/
                            -> previous run archived to outputs/archive/<timestamp>/

Quick start:
    py main.py                     # build everything
    py main.py --verify            # check outputs exist
    py main.py --interactive       # menu-driven CLI
    py -m streamlit run app.py     # launch dashboard
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import (
    END_DATE,
    START_DATE,
    NASDAQ_30,
    RESERVE_BENCH,
    get_live_ingest_config,
    validate_live_ingest_config,
)

ROOT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = ROOT_DIR / "storage"


# ---------------------------------------------------------------------------
# Banner + pretty output helpers
# ---------------------------------------------------------------------------

_BANNER = r"""
=================================================================
|        QUANT MATRIX  --  NASDAQ-100 Tech (30 Stocks)          |
=================================================================
"""

def _banner() -> None:
    print(_BANNER)

def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")

def _info(msg: str) -> None:
    print(f"  [INFO] {msg}")

def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")

def _err(msg: str) -> None:
    print(f"  [ERR]  {msg}")

def _section(title: str) -> None:
    print(f"\n{'-'*65}")
    print(f"  {title}")
    print(f"{'-'*65}")


# ---------------------------------------------------------------------------
# Build (default action)
# ---------------------------------------------------------------------------

def build_pipeline() -> int:
    import os
    os.environ["QM_ENABLE_LEGACY_YFINANCE"] = "1"
    from pipeline import build_and_serialize

    _banner()
    _section("Building Quant Matrix")
    _info(f"Date range : {START_DATE} -> {END_DATE}")
    _info(f"Tickers    : {len(NASDAQ_30)} primary + {len(RESERVE_BENCH)} reserve")
    _info(f"Threshold  : >5% NaN -> zombie replacement")
    print()

    t0 = time.perf_counter()
    try:
        artifacts = build_and_serialize(
            start_date=START_DATE,
            end_date=END_DATE,
            missing_threshold=0.05,
            root_dir=ROOT_DIR,
        )
    except RuntimeError as exc:
        _err(f"Build failed: {exc}")
        return 1

    elapsed = time.perf_counter() - t0

    # Cleaning report
    cleaning = artifacts.cleaning
    if cleaning.dropped_primaries:
        _warn(f"Zombie tickers dropped: {', '.join(cleaning.dropped_primaries)}")
        for old, new in cleaning.replacements:
            _info(f"  {old} -> {new}")
    else:
        _ok("All 30 tickers healthy - no zombies detected")

    # Correlation outlier report
    outliers = artifacts.standardization.scaler_params.get("correlation_outliers", [])
    if outliers:
        _warn(f"{len(outliers)} low-correlation pair(s) flagged (see correlation_outliers.json)")
    else:
        _ok("Correlation structure healthy - all pairs above 0.3")

    # Output summary
    _section("Outputs Saved")
    p = artifacts.paths
    _ok(f"Matrix heatmap        : outputs/latest/matrix_heatmap.png")
    _ok(f"Correlation heatmap   : outputs/latest/correlation_heatmap.png")
    _ok(f"Standardized CSV      : outputs/latest/standardized_matrix_30xT.csv")
    _ok(f"Log returns CSV       : outputs/latest/aligned_log_returns_30xT.csv")
    _ok(f"Outlier report        : outputs/latest/correlation_outliers.json")
    _ok(f"Usage guide           : outputs/latest/GUIDE.md")
    _ok(f"Build metadata        : outputs/latest/build_metadata.json")
    print()
    _ok(f"Current matrix pickle : storage/current_matrix.pkl")
    _ok(f"Current matrix parquet: storage/current_matrix.parquet")
    _ok(f"Scaler params         : storage/scaler_params.pkl")

    archived = p.get("archived_to")
    if archived and str(archived) != "(none)":
        _info(f"Previous run archived to: {archived}")

    shape = artifacts.standardization.standardized.shape
    _section("Summary")
    _ok(f"Matrix shape: {shape[0]} trading days x {shape[1]} tickers")
    _ok(f"Build time  : {elapsed:.1f}s")
    print()
    _info("Open outputs/latest/GUIDE.md for help reading the output files.")
    _info("Run 'py -m streamlit run app.py' for the interactive dashboard.")
    print()
    return 0


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify_storage() -> int:
    _banner()
    _section("Verifying Outputs")

    checks = [
        (STORAGE_DIR / "current_matrix.pkl", "Current matrix (pickle)"),
        (STORAGE_DIR / "current_matrix.parquet", "Current matrix (parquet)"),
        (STORAGE_DIR / "scaler_params.pkl", "Scaler parameters"),
        (ROOT_DIR / "outputs" / "latest" / "matrix_heatmap.png", "Matrix heatmap"),
        (ROOT_DIR / "outputs" / "latest" / "correlation_heatmap.png", "Correlation heatmap"),
        (ROOT_DIR / "outputs" / "latest" / "standardized_matrix_30xT.csv", "Standardized CSV"),
        (ROOT_DIR / "outputs" / "latest" / "aligned_log_returns_30xT.csv", "Log returns CSV"),
        (ROOT_DIR / "outputs" / "latest" / "GUIDE.md", "Output guide"),
        (ROOT_DIR / "outputs" / "latest" / "build_metadata.json", "Build metadata"),
    ]

    missing = []
    for path, label in checks:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            _ok(f"{label:30s}  ({size_kb:.1f} KB)")
        else:
            _err(f"{label:30s}  MISSING")
            missing.append(label)

    print()
    if missing:
        _warn(f"{len(missing)} file(s) missing. Run 'py main.py' to build.")
        return 1
    _ok("All outputs present and valid.")
    return 0


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="py main.py",
        description=(
            "Quant Matrix CLI - NASDAQ-100 Tech (30 stocks).\n"
            "Default (no flags): builds the full pipeline.\n"
            "Outputs -> outputs/latest/   |   Archive -> outputs/archive/<timestamp>/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Quick start:\n"
            "  py main.py                        # build everything\n"
            "  py main.py --verify               # check outputs\n"
            "  py main.py --interactive           # menu-driven CLI\n"
            "  py -m streamlit run app.py         # launch dashboard\n"
        ),
    )

    # Primary actions
    group = parser.add_argument_group("primary actions")
    group.add_argument("--build", action="store_true", help="Build full pipeline (default if no flags).")
    group.add_argument("--verify", action="store_true", help="Verify all output files exist.")
    group.add_argument("--interactive", action="store_true", help="Launch menu-driven interactive CLI.")

    # Live ingestion
    live = parser.add_argument_group("live ingestion (requires Redis + API keys)")
    live.add_argument("--ingest-live", action="store_true", help="Start WebSocket -> Redis ingestion loop.")
    live.add_argument("--snapshot-live", action="store_true", help="Build one-shot analytics snapshot from Redis.")
    live.add_argument("--persist-live-hourly", action="store_true", help="Persist Redis bars to Parquet.")
    live.add_argument("--bakeoff-live", action="store_true", help="Score providers (Twelve Data vs Polygon).")
    live.add_argument("--serve-live", action="store_true", help="Run local WebSocket live server.")
    live.add_argument("--live", action="store_true", help="Launch live CLI (connects to WebSocket server).")

    # Options
    opts = parser.add_argument_group("options")
    opts.add_argument("--provider", choices=["twelvedata", "polygon"], default=None)
    opts.add_argument("--lookback-minutes", type=int, default=500)
    opts.add_argument("--z-window", type=int, default=60)
    opts.add_argument("--pca-components", type=int, default=3)
    opts.add_argument("--persist-hours", type=int, default=1)
    opts.add_argument("--bakeoff-seconds", type=int, default=300)
    opts.add_argument("--ws-url", default="ws://127.0.0.1:8765")
    opts.add_argument("--host", default="127.0.0.1")
    opts.add_argument("--port", type=int, default=8765)
    opts.add_argument("--interval", type=float, default=5.0)

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # ── Default: build ──────────────────────────────────────────
    has_action = any([
        args.build, args.verify, args.interactive, args.live,
        args.serve_live, args.ingest_live, args.snapshot_live,
        args.persist_live_hourly, args.bakeoff_live,
    ])
    if not has_action or args.build:
        return build_pipeline()

    # ── verify ──────────────────────────────────────────────────
    if args.verify:
        return verify_storage()

    # ── interactive ─────────────────────────────────────────────
    if args.interactive:
        from cli_app import run_interactive_cli
        return run_interactive_cli(root_dir=ROOT_DIR)

    # ── snapshot-live ───────────────────────────────────────────
    if args.snapshot_live:
        from pipeline import build_live_snapshot
        try:
            cfg = get_live_ingest_config(provider_override=args.provider)
            validate_live_ingest_config(cfg, require_provider_keys=False)
            artifacts = build_live_snapshot(
                live_cfg=cfg, root_dir=ROOT_DIR,
                lookback_minutes=args.lookback_minutes,
                z_window=args.z_window, pca_components=args.pca_components,
            )
            _banner()
            _section("Live Snapshot Built")
            _ok(f"Close matrix shape: {artifacts.close_matrix.shape}")
            _ok(f"Rolling z-scores  : {artifacts.rolling_zscores.shape}")
            for key, path in artifacts.paths.items():
                _ok(f"{key}: {path}")
        except ModuleNotFoundError as exc:
            _err(f"Missing dependency: {exc}\n  Install: py -m pip install redis")
            return 3
        except ValueError as exc:
            _err(f"Config error: {exc}")
            return 2
        return 0

    # ── persist-live-hourly ─────────────────────────────────────
    if args.persist_live_hourly:
        from pipeline import persist_live_hourly
        try:
            cfg = get_live_ingest_config(provider_override=args.provider)
            validate_live_ingest_config(cfg, require_provider_keys=False)
            result = persist_live_hourly(live_cfg=cfg, root_dir=ROOT_DIR, hours=args.persist_hours)
            _banner()
            _section("Parquet Persistence Complete")
            _ok(f"Rows scanned : {result.total_rows_scanned}")
            _ok(f"Rows written : {result.rows_written}")
            _ok(f"Rows deduped : {result.deduped_rows}")
        except ModuleNotFoundError as exc:
            _err(f"Missing dependency: {exc}\n  Install: py -m pip install redis pyarrow")
            return 3
        except ValueError as exc:
            _err(f"Config error: {exc}")
            return 2
        return 0

    # ── bakeoff-live ────────────────────────────────────────────
    if args.bakeoff_live:
        import asyncio
        from live_ingest.bakeoff import run_bakeoff
        try:
            cfg = get_live_ingest_config(provider_override=None)
            validate_live_ingest_config(cfg, require_provider_keys=False)
            scores, report = asyncio.run(run_bakeoff(
                cfg=cfg, root_dir=ROOT_DIR, seconds_per_provider=max(30, args.bakeoff_seconds),
            ))
            _banner()
            _section("Provider Bake-off Results")
            for s in scores:
                _ok(f"{s.provider}: status={s.status} ticks={s.ticks_received} "
                    f"coverage={s.symbol_coverage_ratio:.2f}")
            _ok(f"Report: {report}")
        except ModuleNotFoundError as exc:
            _err(f"Missing dependency: {exc}\n  Install: py -m pip install websockets")
            return 3
        except (ValueError, KeyboardInterrupt):
            return 0
        return 0

    # ── ingest-live ─────────────────────────────────────────────
    if args.ingest_live:
        import asyncio
        from live_ingest.runner import run_live_ingest_forever
        try:
            cfg = get_live_ingest_config(provider_override=args.provider)
            if not cfg.enabled:
                _err("Live ingest is disabled. Enable:\n  set QM_ENABLE_LIVE_INGEST=1")
                return 2
            validate_live_ingest_config(cfg)
            _banner()
            _info(f"Starting live ingestion via {cfg.provider}...")
            asyncio.run(run_live_ingest_forever(cfg, root_dir=ROOT_DIR))
        except ModuleNotFoundError as exc:
            _err(f"Missing dependency: {exc}\n  Install: py -m pip install websockets redis")
            return 3
        except (ValueError, KeyboardInterrupt):
            return 0
        return 0

    # ── serve-live ──────────────────────────────────────────────
    if args.serve_live:
        import asyncio
        from live_ws import default_start_end, run_websocket_live_server
        try:
            s, e = default_start_end()
            asyncio.run(run_websocket_live_server(
                host=args.host, port=args.port, root_dir=ROOT_DIR,
                interval_seconds=args.interval, start_date=s, end_date=e,
                missing_threshold=0.05,
            ))
        except ModuleNotFoundError as exc:
            _err(f"Missing dependency: {exc}\n  Install: py -m pip install websockets")
            return 3
        except KeyboardInterrupt:
            return 0
        return 0

    # ── live ────────────────────────────────────────────────────
    if args.live:
        import asyncio
        from cli_app import run_live_cli_via_websocket
        try:
            asyncio.run(run_live_cli_via_websocket(ws_url=args.ws_url, root_dir=ROOT_DIR))
        except ModuleNotFoundError as exc:
            _err(f"Missing dependency: {exc}\n  Install: py -m pip install websockets")
            return 3
        except KeyboardInterrupt:
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
