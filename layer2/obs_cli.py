"""Layer 2 — Observability CLI."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(prog="layer2.obs_cli", description="Nightshade Observability CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("dashboard", help="Launch live metrics dashboard")
    sub.add_parser("health", help="Run all health checks and print results")
    sub.add_parser("metrics", help="Print metric statistics")

    args = parser.parse_args()

    from layer0.config import ConfigRegistry
    from layer0.logging_config import configure_logging
    cfg = ConfigRegistry()
    configure_logging(cfg)

    if args.command == "dashboard":
        from layer2.metrics_dashboard import MetricsDashboard
        MetricsDashboard(cfg).run()

    elif args.command == "health":
        from layer2.health_checker import HealthChecker, HealthState
        from layer1b.questdb_client import QuestDBClient
        from layer1b.redis_client import RedisStreamClient

        checker = HealthChecker(cfg)

        def _qdb_check():
            try:
                qdb = QuestDBClient(cfg)
                h = qdb.health_check()
                score = (50 if h["pg_healthy"] else 0) + (50 if h["ilp_healthy"] else 0)
                return {"healthy": score == 100, "score": score, "message": str(h)}
            except Exception as exc:
                return {"healthy": False, "score": 0, "message": str(exc)}

        def _redis_check():
            try:
                redis = RedisStreamClient(cfg)
                h = redis.health_check()
                healthy = h.get("connected", False)
                return {"healthy": healthy, "score": 100 if healthy else 0, "message": str(h)}
            except Exception as exc:
                return {"healthy": False, "score": 0, "message": str(exc)}

        checker.register_check("questdb", _qdb_check)
        checker.register_check("redis", _redis_check)
        results = checker.run_all_checks_once()
        for component, record in results.items():
            print(f"[{record.state.value:8s}] {component}: score={record.health_score}  {record.message}")
        system_score = checker.get_system_health_score()
        print(f"\nSystem health score: {system_score}/100")

    elif args.command == "metrics":
        from layer2.metrics_emitter import MetricsEmitter
        emitter = MetricsEmitter(cfg)
        print(json.dumps(emitter.get_statistics(), indent=2))
        emitter.stop()


if __name__ == "__main__":
    main()
