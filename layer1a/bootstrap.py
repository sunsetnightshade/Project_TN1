"""Layer 1A — Bootstrap.

Idempotent loader for the 30-instrument universe from data/instruments.json
and data/initial_universe.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from layer0.logging_config import get_logger
from layer1a.security_master import SecurityMaster, SymbolNotFoundError
from layer1a.universe import UniverseManager

if TYPE_CHECKING:
    from layer0.alerts import AlertManager

logger = get_logger(__name__)


class Bootstrap:
    """Idempotent bootstrap loader for the initial instrument universe."""

    def __init__(
        self,
        security_master: SecurityMaster,
        universe_manager: UniverseManager,
        instruments_file: str | Path = "data/instruments.json",
        universe_file: str | Path = "data/initial_universe.json",
        alert_manager: Optional["AlertManager"] = None,
    ) -> None:
        self._sm = security_master
        self._um = universe_manager
        self._instruments_file = Path(instruments_file)
        self._universe_file = Path(universe_file)
        self._alert = alert_manager

    def run(self, force: bool = False) -> dict:
        """Run bootstrap.  Returns stats dict."""
        instruments_added = 0
        instruments_skipped = 0
        mappings_added = 0
        universe_memberships_added = 0
        errors: list[str] = []

        instruments = json.loads(self._instruments_file.read_text(encoding="utf-8"))
        universes_data = json.loads(self._universe_file.read_text(encoding="utf-8"))

        # Map ticker → nightshade_id
        ticker_to_id: dict[str, str] = {}

        for instr in instruments:
            ticker = instr["ticker"]
            try:
                existing_id = self._sm.resolve(source="yfinance", external_id=ticker)
                if force:
                    self._sm.update_instrument(
                        existing_id,
                        name=instr["name"],
                        sector=instr.get("sector"),
                        industry=instr.get("industry"),
                    )
                ticker_to_id[ticker] = existing_id
                instruments_skipped += 1
                logger.debug("bootstrap: skip existing %s", ticker)
            except SymbolNotFoundError:
                # Add fresh
                try:
                    nid = self._sm.add_instrument(
                        instrument_type="EQUITY",
                        primary_exchange=instr["exchange"],
                        currency=instr["currency"],
                        name=instr["name"],
                        sector=instr.get("sector"),
                        industry=instr.get("industry"),
                    )
                    # Add yfinance mapping
                    self._sm.add_symbol_mapping(nid, "yfinance", ticker)
                    # Add ISIN mapping
                    self._sm.add_symbol_mapping(nid, "isin", instr["isin"])
                    ticker_to_id[ticker] = nid
                    instruments_added += 1
                    mappings_added += 2
                    logger.debug("bootstrap: added %s → %s", ticker, nid)
                except Exception as exc:
                    errors.append(f"{ticker}: {exc}")
                    logger.error("bootstrap error for %s: %s", ticker, exc)

        # Populate universes
        for universe in universes_data["universes"]:
            uname = universe["name"]
            added_date = universe["added_date"]
            for ticker in universe["tickers"]:
                nid = ticker_to_id.get(ticker)
                if nid is None:
                    errors.append(f"Universe {uname}: no ID for {ticker}")
                    continue
                try:
                    self._um.add_to_universe(uname, nid, added_date)
                    universe_memberships_added += 1
                except Exception:
                    pass  # Already in universe (idempotent)

        return {
            "instruments_added": instruments_added,
            "instruments_skipped": instruments_skipped,
            "mappings_added": mappings_added,
            "universe_memberships_added": universe_memberships_added,
            "errors": errors,
        }

    def verify(self) -> bool:
        """Verify all 30 instruments exist, are in ≥1 universe, and resolve correctly."""
        instruments = json.loads(self._instruments_file.read_text(encoding="utf-8"))
        all_ok = True

        for instr in instruments:
            ticker = instr["ticker"]
            try:
                nid = self._sm.resolve("yfinance", ticker)
                universes = [
                    u for u in self._um.list_universes()
                    if nid in self._um.get_current_universe(u)
                ]
                if not universes:
                    logger.error("bootstrap verify: %s not in any universe", ticker)
                    all_ok = False
            except Exception as exc:
                logger.error("bootstrap verify: %s failed: %s", ticker, exc)
                all_ok = False

        if not all_ok and self._alert:
            self._alert.send_critical(
                "Bootstrap",
                "Bootstrap verification failed — some instruments missing or not in universe",
            )
        return all_ok
