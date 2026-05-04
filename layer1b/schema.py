"""Layer 1B — QuestDB Schema Manager.

Defines Bronze, Silver, and Gold table DDL and manages schema lifecycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from layer0.logging_config import get_logger

if TYPE_CHECKING:
    from layer1b.questdb_client import QuestDBClient

logger = get_logger(__name__)

TICKS_BRONZE_DDL = """
CREATE TABLE IF NOT EXISTS ticks_bronze (
  ts_event         TIMESTAMP,
  ts_recv          LONG,
  ts_db_write      LONG,
  nightshade_id    SYMBOL CAPACITY 256 CACHE INDEX,
  source           SYMBOL CAPACITY 16  CACHE INDEX,
  price_fixed      LONG,
  size             INT,
  exchange         SYMBOL CAPACITY 64  CACHE INDEX,
  conditions       INT,
  data_quality_score BYTE
) TIMESTAMP(ts_event) PARTITION BY DAY WAL;
"""

BARS_SILVER_DDL = """
CREATE TABLE IF NOT EXISTS bars_silver (
  ts_bar_open      TIMESTAMP,
  ts_bar_close     LONG,
  nightshade_id    SYMBOL CAPACITY 256 CACHE INDEX,
  bar_interval     SYMBOL CAPACITY 8   CACHE INDEX,
  open_fixed       LONG,
  high_fixed       LONG,
  low_fixed        LONG,
  close_fixed      LONG,
  volume           LONG,
  vwap_fixed       LONG,
  trade_count      INT,
  data_quality_score BYTE,
  is_complete      BOOLEAN,
  source_row_count INT
) TIMESTAMP(ts_bar_open) PARTITION BY MONTH WAL;
"""

FEATURES_GOLD_DDL = """
CREATE TABLE IF NOT EXISTS features_gold (
  ts_feature       TIMESTAMP,
  nightshade_id    SYMBOL CAPACITY 256 CACHE INDEX,
  feature_name     SYMBOL CAPACITY 128 CACHE INDEX,
  feature_value    DOUBLE,
  lookback_days    INT,
  is_valid         BOOLEAN
) TIMESTAMP(ts_feature) PARTITION BY MONTH WAL;
"""

METRICS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS metrics_layer2 (
  ts           TIMESTAMP,
  component    SYMBOL CAPACITY 64  CACHE INDEX,
  metric_name  SYMBOL CAPACITY 256 CACHE INDEX,
  metric_value DOUBLE,
  host         SYMBOL CAPACITY 8   CACHE INDEX,
  environment  SYMBOL CAPACITY 4   CACHE INDEX,
  tags         STRING
) TIMESTAMP(ts) PARTITION BY DAY WAL;
"""

HEALTH_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS health_history (
  ts                   TIMESTAMP,
  component            SYMBOL CAPACITY 64 CACHE INDEX,
  health_state         SYMBOL CAPACITY 8  CACHE INDEX,
  health_score         INT,
  consecutive_failures INT,
  message              STRING,
  host                 SYMBOL CAPACITY 8  CACHE INDEX
) TIMESTAMP(ts) PARTITION BY DAY WAL;
"""

ALL_DDL = [
    TICKS_BRONZE_DDL,
    BARS_SILVER_DDL,
    FEATURES_GOLD_DDL,
    METRICS_TABLE_DDL,
    HEALTH_TABLE_DDL,
]


class SchemaManager:
    """Executes QuestDB DDL statements to create all tables."""

    def __init__(self, questdb_client: "QuestDBClient") -> None:
        self._qdb = questdb_client

    def create_all_tables(self) -> None:
        """Create all Bronze, Silver, Gold, and observability tables."""
        for ddl in ALL_DDL:
            try:
                self._qdb.query(ddl.strip())
                logger.debug("SchemaManager: executed DDL:\n%s", ddl[:80])
            except Exception as exc:
                logger.error("SchemaManager DDL failed: %s\nSQL: %s", exc, ddl[:80])

    def verify_tables(self) -> dict[str, bool]:
        """Check that all expected tables exist."""
        expected = ["ticks_bronze", "bars_silver", "features_gold", "metrics_layer2", "health_history"]
        result: dict[str, bool] = {}
        for table in expected:
            try:
                rows = self._qdb.query(f"SELECT count() FROM {table}")
                result[table] = True
            except Exception:
                result[table] = False
        return result
