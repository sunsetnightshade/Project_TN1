"""Layer 1A — CLI entry point.

Usage:
    python -m layer1a.cli <command> [options]

All subcommands prompt for master password via getpass.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys

from layer0.config import ConfigRegistry
from layer0.logging_config import configure_logging
from layer0.alerts import AlertManager
from layer0.secrets import SecretsManager
from layer1a.security_master import SecurityMaster
from layer1a.universe import UniverseManager
from layer1a.corporate_actions import CorporateActionsManager
from layer1a.bootstrap import Bootstrap


def _build_components(password: str):
    cfg = ConfigRegistry()
    configure_logging(cfg)
    sm_obj = SecretsManager(password)
    alert = AlertManager(cfg, sm_obj)
    sm = SecurityMaster(cfg, alert)
    um = UniverseManager(sm)
    ca = CorporateActionsManager(sm)
    return cfg, sm, um, ca, alert


def cmd_bootstrap(args, password: str) -> None:
    cfg, sm, um, ca, alert = _build_components(password)
    b = Bootstrap(sm, um, alert_manager=alert)
    result = b.run(force=args.force)
    print(json.dumps(result, indent=2))


def cmd_verify(args, password: str) -> None:
    cfg, sm, um, ca, alert = _build_components(password)
    b = Bootstrap(sm, um, alert_manager=alert)
    ok = b.verify()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def cmd_stats(args, password: str) -> None:
    cfg, sm, um, ca, alert = _build_components(password)
    print(json.dumps(sm.get_statistics(), indent=2))


def cmd_resolve(args, password: str) -> None:
    cfg, sm, um, ca, alert = _build_components(password)
    nid = sm.resolve(args.source, args.id)
    print(json.dumps(sm.get_instrument(nid), indent=2))


def cmd_search(args, password: str) -> None:
    cfg, sm, um, ca, alert = _build_components(password)
    results = sm.search_instruments(args.query)
    print(json.dumps(results, indent=2))


def cmd_add_action(args, password: str) -> None:
    cfg, sm, um, ca, alert = _build_components(password)
    action_id = ca.add_action(
        nightshade_id=args.nightshade_id,
        action_type=args.type,
        ex_date=args.ex_date,
        adjustment_factor=float(args.factor),
        data_source=args.source,
    )
    print(f"Created action_id={action_id}")


def cmd_unapplied_actions(args, password: str) -> None:
    from datetime import datetime, timezone
    cfg, sm, um, ca, alert = _build_components(password)
    today = datetime.now(timezone.utc).date().isoformat()
    actions = ca.get_unapplied_actions(today)
    print(json.dumps(actions, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="layer1a.cli", description="Nightshade Security Master CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bs = sub.add_parser("bootstrap", help="Run bootstrap loader")
    p_bs.add_argument("--force", action="store_true")

    sub.add_parser("verify", help="Verify bootstrap integrity")
    sub.add_parser("stats", help="Print SecurityMaster statistics")
    sub.add_parser("unapplied-actions", help="List unapplied corporate actions")

    p_res = sub.add_parser("resolve", help="Resolve external symbol → nightshade_id")
    p_res.add_argument("--source", required=True)
    p_res.add_argument("--id", required=True)

    p_search = sub.add_parser("search", help="Search instruments by name")
    p_search.add_argument("--query", required=True)

    p_action = sub.add_parser("add-action", help="Add corporate action")
    p_action.add_argument("--nightshade-id", dest="nightshade_id", required=True)
    p_action.add_argument("--type", required=True)
    p_action.add_argument("--ex-date", dest="ex_date", required=True)
    p_action.add_argument("--factor", required=True)
    p_action.add_argument("--source", required=True)

    args = parser.parse_args()
    password = getpass.getpass("Master password: ")

    dispatch = {
        "bootstrap": cmd_bootstrap,
        "verify": cmd_verify,
        "stats": cmd_stats,
        "resolve": cmd_resolve,
        "search": cmd_search,
        "add-action": cmd_add_action,
        "unapplied-actions": cmd_unapplied_actions,
    }
    dispatch[args.command](args, password)


if __name__ == "__main__":
    main()
