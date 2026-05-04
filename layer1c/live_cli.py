"""Layer 1C — Live CLI."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="layer1c.live_cli", description="Nightshade Live Data Layer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("start", help="Start IngestorSupervisor (blocks)")
    sub.add_parser("status", help="Read supervisor status from Redis")
    sub.add_parser("health", help="Read health dashboard from Redis")

    gaps_p = sub.add_parser("gaps", help="Manage data gaps")
    gaps_p.add_argument("--list", action="store_true")
    gaps_p.add_argument("--fill", action="store_true")

    sub_p = sub.add_parser("subscribe", help="Send subscribe command")
    sub_p.add_argument("--source", required=True)
    sub_p.add_argument("--symbols", required=True)

    mh_p = sub.add_parser("market-hours", help="Print session boundaries")
    mh_p.add_argument("--exchange", required=True)
    mh_p.add_argument("--date", required=True)

    args = parser.parse_args()

    from layer0.config import ConfigRegistry
    from layer0.logging_config import configure_logging
    cfg = ConfigRegistry()
    configure_logging(cfg)

    if args.command == "start":
        import getpass
        from layer0.secrets import SecretsManager
        from layer0.alerts import AlertManager
        from layer1c.ingestor_supervisor import IngestorSupervisor
        password = getpass.getpass("Master password: ")
        sm = SecretsManager(password)
        alert = AlertManager(cfg, sm)
        supervisor = IngestorSupervisor(cfg, sm, alert)
        supervisor.start()

    elif args.command == "status":
        import redis  # type: ignore[import-untyped]
        r = redis.Redis()
        data = r.get("nightshade:supervisor:status")
        print(data.decode() if data else "No status available")

    elif args.command == "health":
        import redis  # type: ignore[import-untyped]
        r = redis.Redis()
        keys = list(r.scan_iter("nightshade:health:*"))
        for key in keys:
            data = r.get(key)
            if data:
                d = json.loads(data.decode())
                source = key.decode().split(":")[-1]
                print(f"[{source}] score={d.get('health_score', 'N/A')}")

    elif args.command == "gaps":
        from layer1b.gap_tracker import GapTracker
        gaps = GapTracker()
        if args.list:
            open_gaps = gaps.get_open_gaps()
            print(json.dumps(open_gaps, indent=2))
        elif args.fill:
            import getpass
            from layer0.secrets import SecretsManager
            password = getpass.getpass("Master password: ")
            sm = SecretsManager(password)
            key = sm.get("polygon.api_key")
            result = gaps.run_fill_cycle(key)
            print(json.dumps(result, indent=2))

    elif args.command == "subscribe":
        import redis  # type: ignore[import-untyped]
        r = redis.Redis()
        cmd = json.dumps({"action": "subscribe", "source": args.source, "symbols": args.symbols.split(",")})
        r.publish("nightshade:supervisor:commands", cmd)
        print(f"Subscribe command sent: {cmd}")

    elif args.command == "market-hours":
        from layer1c.market_hours import MarketHoursManager
        mhm = MarketHoursManager()
        boundaries = mhm.get_session_boundaries(args.exchange, args.date)
        print(json.dumps(boundaries, indent=2))


if __name__ == "__main__":
    main()
