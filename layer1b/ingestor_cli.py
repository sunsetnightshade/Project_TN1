"""Layer 1B — Ingestor CLI."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="layer1b.ingestor_cli", description="Nightshade Data Lake CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("start", help="Start WebSocket ingestor (blocks)")
    sub.add_parser("status", help="Read ingestor status from Redis")
    schema_p = sub.add_parser("schema", help="Manage QuestDB schema")
    schema_p.add_argument("--create", action="store_true")
    schema_p.add_argument("--verify", action="store_true")

    agg_p = sub.add_parser("aggregate", help="Run Bronze→Silver aggregation")
    agg_p.add_argument("--date", required=True)

    feat_p = sub.add_parser("features", help="Run Silver→Gold feature computation")
    feat_p.add_argument("--date", required=True)

    gaps_p = sub.add_parser("gaps", help="Manage data gaps")
    gaps_p.add_argument("--list", action="store_true")
    gaps_p.add_argument("--fill", action="store_true")

    sub.add_parser("health", help="Check QuestDB/Redis/WebSocket health")

    args = parser.parse_args()

    from layer0.config import ConfigRegistry
    from layer0.logging_config import configure_logging

    cfg = ConfigRegistry()
    configure_logging(cfg)

    if args.command == "start":
        import getpass
        from layer0.secrets import SecretsManager
        from layer0.alerts import AlertManager
        from layer1b.questdb_client import QuestDBClient
        from layer1b.redis_client import RedisStreamClient
        from layer1b.data_quality import DataQualityScorer
        from layer1b.gap_tracker import GapTracker
        from layer1b.websocket_ingestor import PolygonWebSocketIngestor

        password = getpass.getpass("Master password: ")
        sm = SecretsManager(password)
        alert = AlertManager(cfg, sm)
        qdb = QuestDBClient(cfg, alert)
        redis = RedisStreamClient(cfg, alert)
        dq = DataQualityScorer()
        gaps = GapTracker(alert_manager=alert)
        ingestor = PolygonWebSocketIngestor(cfg, sm, qdb, redis, dq, gaps, alert)
        ingestor.start()

    elif args.command == "status":
        from layer1b.redis_client import RedisStreamClient
        redis = RedisStreamClient(cfg)
        import redis as redis_lib
        r = redis_lib.Redis()
        data = r.get("nightshade:ingestor:status")
        print(data.decode() if data else "No status available")

    elif args.command == "schema":
        from layer1b.questdb_client import QuestDBClient
        from layer1b.schema import SchemaManager
        qdb = QuestDBClient(cfg)
        sm = SchemaManager(qdb)
        if args.create:
            sm.create_all_tables()
            print("Schema created.")
        elif args.verify:
            result = sm.verify_tables()
            print(json.dumps(result, indent=2))

    elif args.command == "aggregate":
        from layer1b.questdb_client import QuestDBClient
        from layer1b.aggregation_jobs import BronzeToSilverAggregator
        qdb = QuestDBClient(cfg)
        agg = BronzeToSilverAggregator(qdb)
        print(f"Aggregation for {args.date} — (provide nightshade_ids via config)")

    elif args.command == "features":
        from layer1b.questdb_client import QuestDBClient
        from layer1b.feature_jobs import SilverToGoldFeatureComputer
        qdb = QuestDBClient(cfg)
        feat = SilverToGoldFeatureComputer(qdb)
        print(f"Feature computation for {args.date} — (provide nightshade_ids via config)")

    elif args.command == "gaps":
        from layer1b.gap_tracker import GapTracker
        from layer0.alerts import AlertManager
        alert = AlertManager(cfg)
        gaps = GapTracker(alert_manager=alert)
        if args.list:
            print(json.dumps(gaps.get_open_gaps(), indent=2))
        elif args.fill:
            import getpass
            from layer0.secrets import SecretsManager
            password = getpass.getpass("Master password: ")
            sm = SecretsManager(password)
            key = sm.get("polygon.api_key")
            result = gaps.run_fill_cycle(key)
            print(json.dumps(result, indent=2))

    elif args.command == "health":
        from layer1b.questdb_client import QuestDBClient
        from layer1b.redis_client import RedisStreamClient
        qdb = QuestDBClient(cfg)
        redis = RedisStreamClient(cfg)
        print("QuestDB health:", json.dumps(qdb.health_check(), indent=2))
        print("Redis health:", json.dumps(redis.health_check(), indent=2))


if __name__ == "__main__":
    main()
