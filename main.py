from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from config import END_DATE, START_DATE
from cli_app import run_interactive_cli, run_live_cli_via_websocket
from live_ws import default_start_end, run_websocket_live_server
from pipeline import build_and_serialize

ROOT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = ROOT_DIR / "storage"


def build_pipeline() -> None:
    artifacts = build_and_serialize(
        start_date=START_DATE,
        end_date=END_DATE,
        missing_threshold=0.05,
        root_dir=ROOT_DIR,
    )

    cleaning = artifacts.cleaning
    if cleaning.dropped_primaries:
        dropped = ", ".join(cleaning.dropped_primaries)
        repl = ", ".join([f"{a}->{b}" for a, b in cleaning.replacements])
        print(f"Zombie tickers dropped: {dropped}")
        print(f"Replacements applied: {repl}")

    p = artifacts.paths
    print(f"Saved current matrix to: {p['current_matrix']}")
    print(f"Saved backup matrix to: {p['backup_matrix']}")
    print(f"Saved scaler params to: {p['scaler_params']}")
    print(f"Saved matrix heatmap to: {p['matrix_heatmap']}")
    print(f"Saved accessible aligned returns (30xT) to: {p['returns_csv_latest']}")
    print(f"Saved accessible standardized matrix (30xT) to: {p['standardized_csv_latest']}")


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
        description="Interactive CLI to build and explore aligned NIFTY+NASDAQ matrices."
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run full pipeline and serialize outputs to storage/ and outputs/.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify storage/ contains expected output files.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch menu-driven interactive CLI (default if no flags).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Launch live CLI fed by a local websocket server.",
    )
    parser.add_argument(
        "--serve-live",
        action="store_true",
        help="Run the local websocket live server (pushes updates every --interval seconds).",
    )
    parser.add_argument(
        "--ws-url",
        default="ws://127.0.0.1:8765",
        help="Websocket URL for --live mode.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for --serve-live websocket server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for --serve-live websocket server.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Live rebuild interval in seconds (default: 5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # Default UX: interactive menu
    if not any([args.build, args.verify, args.interactive, args.live, args.serve_live]):
        return run_interactive_cli(root_dir=ROOT_DIR)

    if args.build:
        build_pipeline()

    if args.verify:
        return verify_storage()

    if args.interactive:
        return run_interactive_cli(root_dir=ROOT_DIR)

    if args.serve_live:
        try:
            import asyncio

            s, e = default_start_end()
            asyncio.run(
                run_websocket_live_server(
                    host=args.host,
                    port=args.port,
                    root_dir=ROOT_DIR,
                    interval_seconds=args.interval,
                    start_date=s,
                    end_date=e,
                    missing_threshold=0.05,
                )
            )
        except ModuleNotFoundError as exc:
            print(
                "Missing dependency for live server. Install with:\n"
                "  py -m pip install websockets\n"
                f"Details: {exc}"
            )
            return 3
        except KeyboardInterrupt:
            return 0
        return 0

    if args.live:
        try:
            import asyncio

            asyncio.run(run_live_cli_via_websocket(ws_url=args.ws_url, root_dir=ROOT_DIR))
        except ModuleNotFoundError as exc:
            print(
                "Missing dependency for live mode. Install with:\n"
                "  py -m pip install websockets\n"
                f"Details: {exc}"
            )
            return 3
        except KeyboardInterrupt:
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

