# NIGHTSHADE QUANTITATIVE TRADING SYSTEM
## Complete Architecture & Build Reference

---

## FOUNDATIONAL PHILOSOPHY

Two parallel tracks share one data foundation but are otherwise independent.

**Track A — Research Track:** Discover whether an algorithm works. Slow, thorough, uses historical data, produces a verdict: deploy or discard.

**Track B — Production Track:** A validated algorithm runs live. Fast, hardened, monitored, connected to execution. Nothing enters Track B without graduating from Track A.

> **Current state:** No data, no API keys, no capital. You are building the plumbing. Every decision made in the next month is about plumbing standards, not math. The math comes later.

---

## BUILD ORDER OVERVIEW

| Phase | Layer | Name | Status Gate |
|-------|-------|------|-------------|
| 1 | Layer 0 | Foundation — Secrets, Config, Logging | Today, no keys needed |
| 2 | Layer 1A | Security Master — Identity | Today, no keys needed |
| 3 | Layer 1B | Data Lake — Storage & Ingestion | Today (free Polygon tier) |
| 4 | Layer 1C | Live Data Layer — Hardened Ingestor | Today (free Polygon tier) |
| 5 | Layer 2 | Observability — Operations Center | Today, no keys needed |
| 6 | Layer 3 | Research Track — Alpha Engines | After Databento data arrives |
| 7 | Layer 4 | Risk Shield | After Engine 1 backtests pass |
| 8 | Layer 5 | Order Management System | After Risk Shield is verified |
| 9 | Layer 6 | Execution — Paper Trading | After OMS is verified |
| 10 | Layer 7 | Paper Trading Monitor | After 30-day paper record |

---

## PHASE 0 — LAYER 0: FOUNDATION

### Purpose
Pre-exists every other component. Unglamorous but catastrophic when absent. Provides secrets, configuration, logging, and alerting to every module in the system.

### Project Structure
```
nightshade/
├── layer0/
│   ├── __init__.py
│   ├── secrets.py
│   ├── config.py
│   ├── logging_config.py
│   └── alerts.py
├── tests/layer0/
│   ├── __init__.py
│   ├── test_secrets.py
│   ├── test_config.py
│   ├── test_logging_config.py
│   └── test_alerts.py
├── config.yaml
├── .gitignore
├── requirements.txt
└── README.md
```

### Component 1: Secrets Manager (`layer0/secrets.py`)

**Vault location:** `~/.nightshade/vault.enc` (path from `config.yaml`)  
**Encryption:** Fernet symmetric encryption (`cryptography` library)  
**Key derivation:** PBKDF2HMAC + SHA256 + 480,000 iterations  
**Salt:** Stored separately at `~/.nightshade/vault.salt`  
**Master password:** Never stored — accepted at runtime only

**Class:** `SecretsManager`

| Method | Behaviour |
|--------|-----------|
| `__init__(master_password)` | Derives Fernet key from password + salt. Creates salt (16 bytes, permissions 600) if absent. Creates empty vault if absent. Raises `SecretsVaultCorruptedError` on bad password. Raises `SecretsVaultPermissionError` if file permissions ≠ 600. |
| `get(key)` | Returns stored secret. Raises `SecretNotFoundError` if missing. |
| `set(key, value)` | Stores secret. Atomic write (temp-file → rename). |
| `delete(key)` | Removes secret. Raises `SecretNotFoundError` if missing. |
| `list_keys()` | Returns all key names without values. |
| `rotate_master_password(new_password)` | Re-encrypts with new password + new salt. Backs up old files with timestamp suffix. |

**Custom Exceptions (all inherit `SecretsError`):** `SecretsVaultCorruptedError`, `SecretsVaultPermissionError`, `SecretNotFoundError`

**Rules:** Never log secret values. Key names may be logged at DEBUG. Uses logging from `layer0/logging_config.py`.

---

### Component 2: Configuration Registry (`layer0/config.py`)

**Source file:** `config.yaml` (default: `./config.yaml`)

**Class:** `ConfigRegistry`

| Method | Behaviour |
|--------|-----------|
| `__init__(config_path)` | Loads and validates YAML. Raises `ConfigFileNotFoundError` / `ConfigValidationError`. |
| `get(key_path, default=None)` | Dot-notation access, e.g. `"database.questdb.host"`. Returns `default` if path absent and default provided. |
| `get_required(key_path)` | Same as `get` but always raises `ConfigKeyNotFoundError` if absent. Never returns None. |
| `reload()` | Reloads config from disk at runtime. |
| `as_dict()` | Returns deep copy of entire config dict. |

**Custom Exceptions:** `ConfigError` (base), `ConfigFileNotFoundError`, `ConfigValidationError`, `ConfigKeyNotFoundError`

**Complete `config.yaml` structure:**

```yaml
# Nightshade Quantitative Trading System - Master Configuration

secrets:
  vault_path: "~/.nightshade/vault.enc"       # Path to encrypted secrets vault
  salt_path: "~/.nightshade/vault.salt"        # Path to PBKDF2 salt file

logging:
  level: "INFO"                                # Global log level: DEBUG/INFO/WARNING/ERROR/CRITICAL
  log_dir: "~/.nightshade/logs"               # Directory for rotating log files
  max_bytes: 10485760                          # Max log file size in bytes (10MB)
  backup_count: 90                             # Number of daily log files to retain
  structured: true                             # If true, emit JSON-structured log lines

alerting:
  method: "file"                               # Alert delivery method: "file" or "email"
  alert_log_path: "~/.nightshade/alerts.log"  # Path to dedicated alert log file
  email:
    smtp_host: "smtp.gmail.com"               # SMTP server hostname
    smtp_port: 587                            # SMTP server port
    sender_address: ""                        # Loaded from secrets at runtime
    recipient_address: ""                     # Recipient email address

database:
  questdb:
    host: "localhost"
    http_port: 9000                           # QuestDB HTTP/REST port
    ilp_port: 9009                            # Influx Line Protocol ingestion port
    pg_port: 8812                             # PostgreSQL wire protocol port
  redis:
    host: "localhost"
    port: 6379
    db: 0
    max_stream_len: 100000                    # Maximum messages per instrument Redis stream

system:
  environment: "paper"                        # "paper" or "live"
  timezone: "UTC"
  heartbeat_interval_seconds: 30

research:
  data_lookback_days: 504                     # 2 years in trading days
  rolling_window_short: 20
  rolling_window_long: 252
  min_data_quality_score: 2

risk:
  daily_loss_limit_pct: 0.045                # Hard kill trigger
  soft_stop_pct: 0.040                       # Soft stop trigger
  warning_pct: 0.035                         # Warning trigger
  max_single_name_pct: 0.05                  # Max single-instrument gross exposure
  max_sector_pct: 0.30                       # Max sector gross exposure
  target_portfolio_volatility: 0.12          # Annualized vol target
  vpin_suspension_threshold: 0.70
  correlation_concentration_limit: 0.70
  max_adv_fraction: 0.01                     # Max order size as fraction of ADV

execution:
  paper_slippage_bps: 5.0
  transaction_cost_bps: 10.0
  twap_duration_minutes: 30
  order_timeout_seconds: 300
  fill_timeout_seconds: 60
```

---

### Component 3: Logging Configuration (`layer0/logging_config.py`)

**Public function:** `configure_logging(config: ConfigRegistry) -> logging.Logger`

**Handlers:** `StreamHandler` (stdout) + `TimedRotatingFileHandler` (daily rotation, 90-day retention)

**Log file name:** `nightshade_{date}.log`

**Structured format** (when `logging.structured = true`): Single-line JSON with fields `ts` (ISO 8601 UTC microsecond), `level`, `module`, `msg`, `pid`, `extra` (exceptions included as `exc_type`, `exc_value`, `exc_traceback`)

**Human-readable format:** `{ts} | {level:<8} | {module:<30} | {msg}`

**Convenience function:** `get_logger(name: str) -> logging.Logger` — wraps `logging.getLogger(name)`. All modules call `get_logger(__name__)`. No module ever calls `logging.getLogger` directly. No module uses `print()` in production.

---

### Component 4: Alerting System (`layer0/alerts.py`)

**Enum:** `AlertSeverity` — `INFO`, `WARNING`, `CRITICAL`

**Dataclass:** `Alert` — `alert_id` (UUID, auto), `ts_utc` (auto), `severity`, `component`, `message`, `data` (dict)

**Class:** `AlertManager`

| Method | Behaviour |
|--------|-----------|
| `__init__(config, secrets_manager)` | If method is `"email"`, loads sender address from secrets. Falls back to file if secret missing. |
| `send(severity, component, message, data)` | Creates Alert, writes to log file always, sends email for WARNING/CRITICAL if method is "email". Never raises. Returns Alert. |
| `send_info/warning/critical(...)` | Convenience wrappers. |
| `get_recent_alerts(n, severity_filter)` | Reads last N alerts from log, optionally filtered by severity. Returns sorted by timestamp descending. |

**Storage:** Newline-delimited JSON file. Opened in append mode. Atomic writes via `fcntl.flock`. Email via `smtplib` only (no third-party libs). SMTP password loaded from secrets at send time. Subject: `[NIGHTSHADE {severity}] {component}: {message[:50]}`.

---

### Component 5: `.gitignore`

Excludes: `*.enc`, `*.salt`, `__pycache__`, `*.pyc`, `*.pyo`, `.pytest_cache`, `venv/`, `.venv/`, `env/`, `.idea/`, `.vscode/`, `logs/`, `*.log`, `secrets.yaml`, `secrets.json`, `.env`, `.DS_Store`, `Thumbs.db`

---

### Layer 0 Test Coverage

| File | Scenarios |
|------|-----------|
| `test_secrets.py` | New vault creation, get/set/delete, wrong password → error, file permissions → error, key listing, password rotation, recycled ticker handling |
| `test_config.py` | Valid load, dot-notation nesting, missing key with/without default, missing file, `reload()`, `as_dict()` immutability |
| `test_logging_config.py` | Structured JSON validity, required fields, exception capture, non-structured format, file creation, `get_logger` name |
| `test_alerts.py` | INFO write to file, CRITICAL email fallback to file on SMTP fail, `get_recent_alerts` count and order, severity filter, JSON validity, concurrent write safety |

**Dependencies:** `cryptography`, `PyYAML`, `pytest`, `pytest-mock`, `freezegun` — all pinned exact versions, Python 3.11, Ubuntu 24.

---

## PHASE 1A — LAYER 1A: SECURITY MASTER (IDENTITY LAYER)

### Purpose
Solves three problems that destroy quantitative research:
1. **Identity fragmentation** — same instrument, different symbols across sources (Infosys = INFY.NS, INFY, INE009A01021, 456788108…)
2. **Survivorship bias** — delisted instruments must persist in the historical record
3. **Corporate actions corruption** — splits/dividends cause spurious price signals if untracked

### Project Structure Addition
```
nightshade/
├── layer1a/
│   ├── __init__.py
│   ├── security_master.py
│   ├── universe.py
│   ├── corporate_actions.py
│   └── bootstrap.py
├── data/
│   ├── instruments.json      # 30 instruments with real ISINs
│   └── initial_universe.json # 2 universes
├── tests/layer1a/
│   ├── __init__.py
│   ├── test_security_master.py
│   ├── test_universe.py
│   └── test_corporate_actions.py
```

### Database: SQLite (`~/.nightshade/security_master.db`)

**PRAGMAs:** `journal_mode=WAL`, `foreign_keys=ON`, `synchronous=FULL`

#### Table 1: `instruments`
| Column | Type | Notes |
|--------|------|-------|
| `nightshade_id` | TEXT PK | UUID4, permanent, never recycled |
| `instrument_type` | TEXT NOT NULL | EQUITY, ETF, FUTURE, CRYPTO, INDEX |
| `primary_exchange` | TEXT NOT NULL | ISO 10383 MIC (XNYS, XNAS, XNSE…) |
| `currency` | TEXT NOT NULL | ISO 4217 (USD, INR…) |
| `name` | TEXT NOT NULL | Full legal name |
| `sector` | TEXT | GICS sector (nullable for non-equity) |
| `industry` | TEXT | GICS industry (nullable) |
| `is_active` | INT DEFAULT 1 | 1=active, 0=delisted/suspended |
| `listed_date` | TEXT | ISO 8601 date |
| `delisted_date` | TEXT | ISO 8601 date (null if active) |
| `created_at` / `updated_at` | TEXT | ISO 8601 UTC datetime |

#### Table 2: `symbol_mappings`
| Column | Type | Notes |
|--------|------|-------|
| `mapping_id` | INT PK AUTOINCREMENT | |
| `nightshade_id` | TEXT FK | → instruments |
| `source` | TEXT NOT NULL | "polygon", "databento", "yfinance", "nse", "isin", "cusip", "figi"… |
| `external_id` | TEXT NOT NULL | |
| `effective_from` / `effective_to` | TEXT | UTC datetime range (null to = currently active) |
| `created_at` | TEXT | |

**Constraints:** UNIQUE on `(source, external_id, effective_from)`. Index on `(source, external_id)`.

#### Table 3: `universe_memberships`
| Column | Type | Notes |
|--------|------|-------|
| `membership_id` | INT PK | |
| `universe_name` | TEXT NOT NULL | "SP500", "NIGHTSHADE_RESEARCH"… |
| `nightshade_id` | TEXT FK | |
| `added_date` | TEXT NOT NULL | ISO 8601 date |
| `removed_date` | TEXT | null = still member |
| `removal_reason` | TEXT | DELISTED, RECONSTITUTION, MANUAL_REMOVAL |
| `created_at` | TEXT | |

**Constraints:** UNIQUE on `(universe_name, nightshade_id, added_date)`. Index on `(universe_name, added_date, removed_date)`.

#### Table 4: `corporate_actions`
| Column | Type | Notes |
|--------|------|-------|
| `action_id` | INT PK | |
| `nightshade_id` | TEXT FK | |
| `action_type` | TEXT NOT NULL | SPLIT, REVERSE_SPLIT, DIVIDEND_CASH, DIVIDEND_STOCK, MERGER_ACQUIRED, MERGER_ACQUIRER, SPINOFF, NAME_CHANGE, TICKER_CHANGE |
| `ex_date` | TEXT NOT NULL | ISO 8601 date |
| `record_date` / `pay_date` | TEXT | nullable |
| `adjustment_factor` | REAL NOT NULL | Multiplicative. Split 4:1 → 0.25. RevereSplit 1:4 → 4.0 |
| `raw_value` / `raw_value_unit` | REAL/TEXT | Announced value + unit (RATIO, USD, INR, SHARES) |
| `notes` | TEXT | Free text for edge cases |
| `data_source` | TEXT NOT NULL | |
| `is_applied` | INT DEFAULT 0 | 1 when pipeline has retroactively adjusted historical data |
| `created_at` / `updated_at` | TEXT | |

**Indexes:** `(nightshade_id, ex_date)`, `(is_applied, ex_date)`

---

### Component 2: SecurityMaster (`layer1a/security_master.py`)

**Class:** `SecurityMaster`

| Method | Behaviour |
|--------|-----------|
| `__init__(config, alert_manager)` | Opens SQLite, creates all tables+indexes+PRAGMAs, sets `row_factory=sqlite3.Row`, validates schema. |
| `add_instrument(...)` | Validates type, exchange (4-letter uppercase), currency (3-letter uppercase). Inserts, returns `nightshade_id`. |
| `get_instrument(nightshade_id)` | Returns dict. Raises `InstrumentNotFoundError`. |
| `update_instrument(nightshade_id, **kwargs)` | Only allows: name, sector, industry, is_active, delisted_date. Protected fields raise `InvalidFieldError`. Always updates `updated_at`. |
| `add_symbol_mapping(nightshade_id, source, external_id, effective_from)` | Closes any existing active mapping for same `(source, external_id)` before inserting. Handles ticker recycling. |
| `resolve(source, external_id, at_time)` | **The most important method.** Returns `nightshade_id` respecting `effective_from/to`. Raises `SymbolNotFoundError` / `AmbiguousSymbolError`. |
| `get_all_mappings(nightshade_id)` | All mappings across all sources and time periods. |
| `search_instruments(query, instrument_type, exchange, active_only)` | LIKE search on name. |
| `deactivate_instrument(nightshade_id, delisted_date, reason)` | Sets `is_active=0`, closes all active mappings, sends INFO alert. |
| `get_statistics()` | Returns counts: total/active/inactive instruments, total/active mappings, breakdowns by type and exchange. |

**Custom Exceptions (base `SecurityMasterError`):** `SecurityMasterSchemaError`, `InstrumentNotFoundError`, `InvalidInstrumentTypeError`, `InvalidExchangeCodeError`, `InvalidCurrencyCodeError`, `InvalidFieldError`, `SymbolNotFoundError`, `AmbiguousSymbolError`

---

### Component 3: UniverseManager (`layer1a/universe.py`)

**Class:** `UniverseManager` — shares SecurityMaster's SQLite connection (no new connection, prevents locking conflicts)

| Method | Behaviour |
|--------|-----------|
| `add_to_universe(universe_name, nightshade_id, added_date)` | Validates instrument exists, not already active member. Raises `AlreadyInUniverseError`. |
| `remove_from_universe(universe_name, nightshade_id, removed_date, reason)` | Closes membership. Raises `NotInUniverseError`. Sends WARNING alert if reason is DELISTED. |
| `get_universe_at_date(universe_name, as_of_date)` | **Point-in-time query.** Returns IDs where `added_date <= as_of_date AND (removed_date > as_of_date OR removed_date IS NULL)`. Prevents survivorship bias. String-based ISO 8601 comparison (lexicographically correct). |
| `get_current_universe(universe_name)` | All IDs with null `removed_date`. |
| `get_universe_history(universe_name)` | Full membership history sorted by added_date. |
| `get_membership_record(universe_name, nightshade_id)` | All membership records including past ones. |
| `list_universes()` | All universe names ever defined. |
| `get_universe_size_over_time(universe_name, start_date, end_date, frequency)` | List of `{date, size}` dicts. frequency: "daily"/"weekly"/"monthly". |

**Custom Exceptions:** `UniverseError` (base), `AlreadyInUniverseError`, `NotInUniverseError`, `UniverseNotFoundError`

---

### Component 4: CorporateActionsManager (`layer1a/corporate_actions.py`)

**Class:** `CorporateActionsManager` — shares SecurityMaster's SQLite connection

| Method | Behaviour |
|--------|-----------|
| `add_action(nightshade_id, action_type, ex_date, adjustment_factor, ...)` | Validates nightshade_id exists, action_type allowed, factor is positive float. For SPLIT: factor must be < 1.0. For REVERSE_SPLIT: factor must be > 1.0. Checks for duplicates `(nightshade_id, action_type, ex_date)`. Sends WARNING alert (every action needs human verification). Returns `action_id`. |
| `get_unapplied_actions(as_of_date)` | Returns all `is_applied=0` actions with `ex_date <= as_of_date`. Sorted chronologically. |
| `mark_action_applied(action_id)` | Sets `is_applied=1`, updates `updated_at`. Raises `CorporateActionNotFoundError`. |
| `get_cumulative_adjustment_factor(nightshade_id, from_date, to_date)` | Product of all adjustment factors for actions in range. Returns 1.0 if no actions. |
| `get_actions_for_instrument(nightshade_id, action_type, start_date, end_date)` | Optionally filtered. Sorted by ex_date. |
| `check_for_missed_actions(nightshade_id, price_series)` | Heuristic: detects overnight price changes > 40% not explained by a known action on that date. Returns suspect dates. Results sent as WARNING alert for manual investigation. |

**Custom Exceptions:** `CorporateActionsError` (base), `DuplicateCorporateActionError`, `CorporateActionNotFoundError`, `InvalidActionTypeError`, `InvalidAdjustmentFactorError`

---

### Component 5: Bootstrap (`layer1a/bootstrap.py`)

**Reads from:** `data/instruments.json` (30 instruments with real ISINs) + `data/initial_universe.json` (2 universes)

**30 Instruments:**
- **US Equities (20, all XNAS/USD):** AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSLA, AVGO, ADBE, TXN, QCOM, AMAT, INTU, AMD, CRM, INTC, PYPL, NFLX, ORCL, IBM
- **Indian Equities (10, all XNSE/INR):** INFY.NS, TCS.NS, HCLTECH.NS, TECHM.NS, WIPRO.NS, LTIM.NS, PERSISTENT.NS, COFORGE.NS, MPHASIS.NS, OFSS.NS

**Universes:**
- `NIGHTSHADE_US_TECH` — 20 US instruments, `added_date: 2024-01-01`
- `NIGHTSHADE_NIFTY_IT` — 10 Indian instruments, `added_date: 2024-01-01`

**Class:** `Bootstrap`

| Method | Behaviour |
|--------|-----------|
| `run(force=False)` | Idempotent. Checks existence via `resolve(source="yfinance", ...)`. Skips if found (or updates if `force=True`). Returns `{instruments_added, instruments_skipped, mappings_added, universe_memberships_added, errors}`. |
| `verify()` | Checks all 30 instruments exist, all in ≥1 universe, all mappings resolve. Returns True/False. Sends CRITICAL alert if any check fails. |

---

### Component 6: `config.yaml` Additions (Layer 1A)

```yaml
security_master:
  db_path: "~/.nightshade/security_master.db"
  allowed_instrument_types: [EQUITY, ETF, FUTURE, CRYPTO, INDEX]
  allowed_action_types: [SPLIT, REVERSE_SPLIT, DIVIDEND_CASH, DIVIDEND_STOCK, MERGER_ACQUIRED, MERGER_ACQUIRER, SPINOFF, NAME_CHANGE, TICKER_CHANGE]
  price_change_alert_threshold: 0.40     # Overnight change fraction triggering missed-action check
  bootstrap:
    instruments_file: "data/instruments.json"
    universe_file: "data/initial_universe.json"
```

---

### Component 7: CLI (`layer1a/cli.py`)

All subcommands ask for master password via `getpass`. Uses `argparse` only.

| Command | Action |
|---------|--------|
| `python -m layer1a.cli bootstrap [--force]` | Runs bootstrap |
| `python -m layer1a.cli verify` | Runs verification |
| `python -m layer1a.cli stats` | Prints `get_statistics()` |
| `python -m layer1a.cli resolve --source S --id ID` | Resolves external ID → nightshade_id + full record |
| `python -m layer1a.cli search --query Q` | Searches by name |
| `python -m layer1a.cli add-action --nightshade-id ID --type TYPE --ex-date DATE --factor F --source S` | Adds corporate action |
| `python -m layer1a.cli unapplied-actions` | Lists all unapplied actions |

---

### Layer 1A Test Coverage

**SQLite connection:** `check_same_thread=False`, explicit transaction management via context managers. All datetimes UTC. All date comparisons string-based ISO 8601.

| File | Scenarios |
|------|-----------|
| `test_security_master.py` | Valid add returns UUID, invalid type/exchange → error, resolve with effective dates, time-based resolve, recycled ticker closes old mapping, ambiguous symbol → error, deactivation closes mappings, name search, statistics accuracy |
| `test_universe.py` | Add succeeds, duplicate raises error, remove sets date+reason, remove non-member raises error, `get_universe_at_date` point-in-time correctness, `get_current_universe` null filter, size-over-time counts |
| `test_corporate_actions.py` | Valid split (factor<1), valid reverse split (factor>1), split with factor>1 → error, duplicate → error, unapplied filter by date, mark_applied, cumulative factor multiplication, missed-action heuristic detection |

---

## PHASE 1B — LAYER 1B: DATA LAKE

### Purpose
Institutional-quality data storage. Three-tier architecture eliminates bad ticks, gaps, float errors, and look-ahead bias.

**Bronze → raw ticks (immutable ground truth)**
**Silver → OHLCV bars (computed from Bronze)**
**Gold → pre-computed features (computed from Silver)**

### Project Structure Addition
```
nightshade/
├── layer1b/
│   ├── __init__.py
│   ├── questdb_client.py
│   ├── redis_client.py
│   ├── schema.py
│   ├── data_quality.py
│   ├── websocket_ingestor.py
│   ├── gap_tracker.py
│   ├── aggregation_jobs.py
│   ├── feature_jobs.py
│   └── ingestor_cli.py
├── tests/layer1b/
│   ├── __init__.py
│   ├── test_questdb_client.py
│   ├── test_redis_client.py
│   ├── test_data_quality.py
│   ├── test_websocket_ingestor.py
│   ├── test_gap_tracker.py
│   ├── test_aggregation_jobs.py
│   └── test_feature_jobs.py
├── docker-compose.yml
```

### Architectural Decisions

**QuestDB** — nanosecond timestamps, columnar storage, Influx Line Protocol (high-throughput writes), PostgreSQL wire (analytical queries), no cloud dependency. Docker image: `questdb/questdb:8.1.4`.

**Redis** — in-memory speed, Streams with consumer groups (multiple engines consume independently), automatic backpressure via length limits, AOF persistence. Docker image: `redis:7.2.4-alpine`.

**Fixed-point prices** — prices stored as 64-bit integers in units of 1/10000 of the base currency unit ($150.2350 → 1502350). Eliminates floating-point rounding errors.

**Bi-temporal model** — every tick has three timestamps:
- `ts_event` — exchange timestamp (nanoseconds, when the trade happened)
- `ts_recv` — ingestor received (nanoseconds, network + exchange clock skew)
- `ts_db_write` — committed to QuestDB (microseconds, processing + write latency)

Backtests must exclude rows where `ts_db_write > simulation_time` to prevent look-ahead bias.

---

### Component 1: Docker Setup (`docker-compose.yml`)

**QuestDB service:**
- Image: `questdb/questdb:8.1.4`
- Ports: `9000` (HTTP console), `9009` (ILP ingestion), `8812` (PostgreSQL)
- Volume: `questdb_data:/root/.questdb`
- Env: `QDB_CAIRO_COMMIT_LAG=1000`, `QDB_LINE_TCP_MAINTENANCE_JOB_INTERVAL=100`
- Restart: `unless-stopped`

**Redis service:**
- Image: `redis:7.2.4-alpine`
- Port: `6379`
- Volume: `redis_data:/data`
- Command: `redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru`
- Restart: `unless-stopped`

---

### Component 2: QuestDB Client (`layer1b/questdb_client.py`)

**Class:** `QuestDBClient` — wraps ILP (writes) + psycopg2 (queries). Connections opened lazily. Background thread health-checks every 60 seconds.

| Method | Behaviour |
|--------|-----------|
| `write_tick(tick)` | Single tick via ILP. Never raises — catches all exceptions, logs ERROR, sends CRITICAL alert. |
| `write_ticks_batch(ticks)` | Batched ILP write. Returns count written. Critical for throughput. |
| `query(sql, params)` | PostgreSQL wire. Returns list[dict]. One reconnect attempt on failure. |
| `query_ticks(nightshade_id, start_ts_ns, end_ts_ns, min_quality_score)` | Bronze tier query. Ordered by `ts_event` asc. Parameterized. |
| `query_bars(nightshade_id, bar_interval, start_date, end_date)` | Silver tier query. |
| `query_features(nightshade_id, feature_names, start_date, end_date)` | Gold tier query. |
| `get_latest_tick_timestamp(nightshade_id)` | Most recent `ts_event` or None. |
| `get_table_row_count(table_name)` | O(1) count via QuestDB `tables()` function. |
| `health_check()` | Returns `{ilp_healthy, pg_healthy, ilp_latency_ms, pg_latency_ms}`. |

**ILP field mapping for ticks:** `nightshade_id`, `source`, `exchange` → SYMBOL columns. All others → integer fields.

**Custom Exceptions:** `QuestDBError` (base), `QuestDBConnectionError`, `QuestDBWriteError`, `QuestDBQueryError`

---

### Component 3: Schema Manager (`layer1b/schema.py`)

**Class:** `SchemaManager`

#### Bronze Table: `ticks_bronze`
```sql
CREATE TABLE IF NOT EXISTS ticks_bronze (
  ts_event         TIMESTAMP,          -- designated timestamp, nanosecond precision
  ts_recv          LONG,
  ts_db_write      LONG,
  nightshade_id    SYMBOL CAPACITY 256 CACHE INDEX,
  source           SYMBOL CAPACITY 16  CACHE INDEX,
  price_fixed      LONG,               -- fixed-point, never float
  size             INT,
  exchange         SYMBOL CAPACITY 64  CACHE INDEX,
  conditions       INT,                -- bitmask
  data_quality_score BYTE
) TIMESTAMP(ts_event) PARTITION BY DAY WAL;
```

#### Silver Table: `bars_silver`
```sql
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
```

#### Gold Table: `features_gold`
```sql
CREATE TABLE IF NOT EXISTS features_gold (
  ts_feature       TIMESTAMP,
  nightshade_id    SYMBOL CAPACITY 256 CACHE INDEX,
  feature_name     SYMBOL CAPACITY 128 CACHE INDEX,
  feature_value    DOUBLE,
  lookback_days    INT,
  is_valid         BOOLEAN
) TIMESTAMP(ts_feature) PARTITION BY MONTH WAL;
```

> **Narrow key-value structure** — one row per feature per instrument per timestamp. Trivially adds new features without schema alteration.

---

### Component 4: Redis Stream Client (`layer1b/redis_client.py`)

**Class:** `RedisStreamClient` — connection pool (`max_connections=20`, `socket_connect_timeout=5`, `decode_responses=True`). Background health check every 30 seconds.

| Method | Behaviour |
|--------|-----------|
| `write_tick_to_stream(tick)` | XADD to `ticks:{nightshade_id}` with `MAXLEN ~ {max_stream_len}`. Never raises. Returns stream entry ID or empty string. |
| `read_ticks_from_stream(nightshade_id, consumer_group, consumer_name, count, block_ms)` | XREADGROUP. Returns list with `stream_id` added. |
| `acknowledge_ticks(nightshade_id, consumer_group, stream_ids)` | XACK. Returns acknowledged count. |
| `create_consumer_group(nightshade_id, consumer_group, start_from)` | Creates group if absent. `start_from="$"` (new only) or `"0"` (all history). |
| `get_stream_info(nightshade_id)` | `{length, first_entry_id, last_entry_id, consumer_groups}` with lag. |
| `get_pending_count(nightshade_id, consumer_group)` | Delivered-but-unacknowledged count. |
| `get_all_stream_keys()` | All keys matching `ticks:*`. |
| `flush_stream(nightshade_id)` | Deletes all entries. Sends WARNING in non-test environments. |
| `health_check()` | `{connected, latency_ms, memory_used_mb, total_streams, total_stream_entries}`. |

**Custom Exceptions:** `RedisClientError` (base), `RedisConnectionError`, `RedisStreamError`

---

### Component 5: Data Quality Module (`layer1b/data_quality.py`)

**Class:** `DataQualityScorer` — maintains internal price history cache (last 10 prices per instrument, `collections.deque`) and timestamp cache (last `ts_event` per instrument).

**Main method:** `score(raw_message, source) -> tuple[dict, int]` — returns (normalized_tick_dict, quality_score 0–4)

**Normalization by source:**
- `polygon_ws`: fields `ev`, `sym`, `p` (float→fixed), `s`, `t` (ms→ns), `x` (exchange ID), `c` (conditions list)
- `databento_hist`: fields `ts_event` (ns), `ts_recv` (ns), `instrument_id`, `price` (already fixed-point), `size`, `action`, `side`, `flags`
- `yfinance_hist`: fields `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `ticker`

**Scoring rules (applied in order, stop at first match):**

| Score | Condition |
|-------|-----------|
| **0 — Reject** | nightshade_id unresolvable; price_fixed ≤ 0; size ≤ 0; ts_event > 60s in future; ts_event > 86400s in past; polygon_ws and ev ≠ "T" |
| **1 — Suspicious** | Price deviates > 15% from previous; ts_event > 10s behind ts_recv; conditions include code 12 (odd lot), 41 (extended hours), or 52 (sold out of sequence) |
| **2 — Marginal** | ts_event 2–10s behind ts_recv; size < 10 shares (equity); price deviation 5–15% |
| **3 — Good** | ts_event 500ms–2s behind ts_recv |
| **4 — Clean** | All checks pass |

**Score 0:** logged + discarded. **Score 1:** stored + flagged. **Score 2+:** stored, available to engines.

**Module-level functions:**
- `convert_float_to_fixed(price_float) -> int` — multiply by 10000, round. Raises `InvalidPriceError` for NaN/infinity.
- `convert_fixed_to_float(price_fixed) -> float` — display only.

**Custom Exceptions:** `DataQualityError` (base), `InvalidPriceError`, `InvalidTimestampError`, `UnresolvableSymbolError`

---

### Component 6: WebSocket Ingestor (`layer1b/websocket_ingestor.py`)

**Class:** `PolygonWebSocketIngestor`

**Dependencies injected:** `ConfigRegistry`, `SecretsManager`, `SecurityMaster`, `QuestDBClient`, `RedisStreamClient`, `DataQualityScorer`, `GapTracker`, `AlertManager`

**API key:** Read from `secrets_manager.get("polygon.api_key")`. If missing, logs exact command to add it and exits cleanly.

| Method | Behaviour |
|--------|-----------|
| `start()` | Enters async event loop, calls `_connect_with_retry()`. |
| `_connect_with_retry()` | Exponential backoff: 1s initial, doubles, caps at 60s. CRITICAL alert after 10 failures. Runs indefinitely. |
| `_connect()` | Opens `wss://socket.polygon.io/stocks`. Auth handshake. Subscribes `T.*`. Calls `_message_loop()`. |
| `_message_loop()` | Records `ts_recv` immediately. Parses JSON array. Routes to `_process_event()`. Checks sequence gaps. |
| `_process_event(event, ts_recv)` | Scores via DataQualityScorer. Score 0 → reject+count. Score ≥1 → add to write buffer + Redis stream. Flush when buffer=500 or 100ms elapsed. |
| `_flush_buffer()` | Batch writes to QuestDB. Sets `ts_db_write` just before write. Retains buffer on failure. |
| `_handle_disconnect(reason)` | Logs WARNING. Records last seq in GapTracker. Sends WARNING alert. Returns to retry loop. |
| `get_statistics()` | `{connection_status, ticks_received_today, ticks_written_today, ticks_rejected_today, buffer_current_size, last_tick_ts_recv, reconnection_count, current_backoff_seconds}` |
| `stop()` | Flushes buffer. Closes WebSocket. Closes QuestDB. |

**Custom Exceptions:** `IngestorError` (base), `AuthenticationError`, `ConnectionError`, `SubscriptionError`

---

### Component 7: Gap Tracker (`layer1b/gap_tracker.py`)

**Class:** `GapTracker` — SQLite gap registry at `~/.nightshade/gap_registry.db`

**Gap table columns:** `gap_id`, `nightshade_id`, `source`, `gap_start_ts_ns`, `gap_end_ts_ns`, `detected_at`, `status` (OPEN/FILL_PENDING/FILLED/UNFILLABLE), `fill_attempts`, `filled_at`

| Method | Behaviour |
|--------|-----------|
| `record_gap(nightshade_id, source, gap_start_ts_ns, gap_end_ts_ns)` | Inserts gap with status OPEN. Sends WARNING. Returns `gap_id`. |
| `close_gap(gap_id, gap_end_ts_ns)` | Sets `gap_end_ts_ns` for an OPEN gap. |
| `get_open_gaps()` | Returns OPEN + FILL_PENDING gaps. |
| `attempt_gap_fill(gap_id, polygon_api_key)` | Queries Polygon REST `GET /v3/trades/{ticker}`. Writes recovered ticks. Increments `fill_attempts`. FILL_PENDING if ≤3 attempts, UNFILLABLE if >3. CRITICAL alert for UNFILLABLE. |
| `run_fill_cycle(polygon_api_key)` | Processes all open/pending gaps. Returns `{gaps_filled, gaps_failed, gaps_unfillable}`. |
| `detect_historical_gaps(nightshade_id, start_date, end_date)` | Identifies calendar days with no data where surrounding data exists. Used during Databento historical load. |

---

### Component 8: Bronze-to-Silver Aggregator (`layer1b/aggregation_jobs.py`)

**Class:** `BronzeToSilverAggregator` — checkpoint SQLite at `~/.nightshade/aggregation_state.db`

**Checkpoint table:** `(nightshade_id, bar_interval, last_aggregated_ts_ns, last_run_at)` — prevents re-processing.

| Method | Behaviour |
|--------|-----------|
| `run_aggregation(nightshade_ids, bar_intervals)` | Default intervals: `["1m", "5m", "1d"]`. For each combo calls `_aggregate_instrument_interval()`. |
| `_aggregate_instrument_interval(nightshade_id, bar_interval)` | Reads from Bronze from checkpoint. Groups into bars. Computes open/high/low/close/volume/vwap (fixed-point throughout)/trade_count/data_quality_score (min of constituents)/is_complete. Writes to Silver. Updates checkpoint. |
| `_compute_bar_boundaries(ts_ns, bar_interval)` | Returns `(bar_start_ns, bar_end_ns)`. Bars computed from `ts_event`, not `ts_recv`. 1m bars at whole minutes. 5m bars at :00/:05/…/:55. 1d bars at exchange open → close. |
| `run_integrity_check(nightshade_id, bar_interval, start_date, end_date)` | Verifies no gaps in Silver. Returns `{expected_bar_count, actual_bar_count, missing_bars, duplicate_bars}`. |

**Edge case:** Bar boundary during no-trades period → create bar with zero volume, `trade_count=0`, `is_complete=True`.

**Idempotency:** Running twice produces identical results with no duplicate rows.

---

### Component 9: Silver-to-Gold Feature Computer (`layer1b/feature_jobs.py`)

**Class:** `SilverToGoldFeatureComputer` — checkpoint SQLite at `~/.nightshade/feature_state.db`

**Runs after market close. Only uses completed bars. Zero look-ahead bias possible.**

**Feature groups:**

| Group | Feature Names | Description |
|-------|--------------|-------------|
| Returns | `log_return_1d`, `log_return_5d`, `log_return_21d` | ln(close_t / close_{t-N}) |
| Rolling Stats (20d & 252d) | `rolling_mean_close`, `rolling_std_close`, `rolling_zscore_close`, `realized_volatility` | Point-in-time stats using only historical data. Vol = annualized stddev × √252 |
| VPIN | `vpin_bucket_25`, `vpin_bucket_50` | Volume-weighted probability of informed trading at two bucket sizes |

**`is_valid = False`** when fewer than `lookback_days` of data available.

**`_write_feature()`** — stores 0.0 with `is_valid=False` for NaN/infinity values (WARNING logged).

**`get_feature_coverage(nightshade_id, as_of_date)`** — map of feature_name → available+valid status. Used by alpha engines before running.

---

### Component 10: `config.yaml` Additions (Layer 1B)

```yaml
data_lake:
  bronze:
    table_name: "ticks_bronze"
    write_batch_size: 500              # Ticks buffered before QuestDB flush
    write_batch_timeout_ms: 100        # Max ms before flush regardless of batch size
  silver:
    table_name: "bars_silver"
    bar_intervals: ["1m", "5m", "1d"]
    aggregation_run_time: "16:30"      # UTC, after market close
  gold:
    table_name: "features_gold"
    feature_run_time: "17:00"          # UTC
    min_history_days: 252
websocket:
  polygon_url: "wss://socket.polygon.io/stocks"
  subscription_channels: ["T.*"]
  reconnect_initial_delay_seconds: 1
  reconnect_max_delay_seconds: 60
  reconnect_max_attempts_before_critical: 10
gap_tracker:
  db_path: "~/.nightshade/gap_registry.db"
  max_fill_attempts: 3
  polygon_rest_base_url: "https://api.polygon.io"
```

---

### Component 11: Ingestor CLI (`layer1b/ingestor_cli.py`)

| Command | Action |
|---------|--------|
| `python -m layer1b.ingestor_cli start` | Starts ingestor (blocks). Initializes all components in dependency order. |
| `python -m layer1b.ingestor_cli status` | Reads status from Redis key `nightshade:ingestor:status`. |
| `python -m layer1b.ingestor_cli schema --create` | Creates all QuestDB tables. |
| `python -m layer1b.ingestor_cli schema --verify` | Verifies tables exist with correct structure. |
| `python -m layer1b.ingestor_cli aggregate --date DATE` | Runs Bronze→Silver aggregation for date. |
| `python -m layer1b.ingestor_cli features --date DATE` | Runs Silver→Gold feature computation for date. |
| `python -m layer1b.ingestor_cli gaps --list` | Lists open/pending gaps. |
| `python -m layer1b.ingestor_cli gaps --fill` | Runs one gap fill cycle. |
| `python -m layer1b.ingestor_cli health` | Checks QuestDB, Redis, WebSocket health. |

---

### Layer 1B Test Coverage (all mocked — no running QuestDB/Redis required)

| File | Scenarios |
|------|-----------|
| `test_data_quality.py` | Valid Polygon trade→score 4, zero price→score 0, future timestamp→score 0, >15% price deviation→score 1, unresolvable symbol→score 0, float-to-fixed conversion, NaN raises error, fixed-to-float reversal |
| `test_gap_tracker.py` | Record creates OPEN entry, get_open_gaps filters correctly, successful fill→FILLED, 3 failures→UNFILLABLE+CRITICAL alert, detect_historical_gaps identifies missing day |
| `test_aggregation_jobs.py` | 10 ticks same boundary→1 bar, spans 2 boundaries→2 bars, VWAP correct, quality_score=min, idempotent on double run, checkpoint updated |
| `test_feature_jobs.py` | log_return_1d correct, zscore=0 when close=mean, is_valid=False when insufficient history, NaN stores 0.0, no duplicates on double run |

---

## PHASE 1C — LAYER 1C: LIVE DATA LAYER

### Purpose
Elevates Layer 1B's basic ingestor into a hardened, production-grade live data system.

**New capabilities over Layer 1B:**
1. **Multi-source ingestion** — source abstraction layer (Polygon, Databento, NSE, future sources via one interface)
2. **Sequence Tracker** — per-message gap detection (not just connection-level)
3. **Gap Fill Orchestrator** — continuously running, rate-limited, prioritized gap filling
4. **Connection Health Monitor** — real-time health scores per source → Redis → Layer 2
5. **Market Hours Manager** — exchange-aware session classification (PRE_MARKET/REGULAR/POST_MARKET/CLOSED/HOLIDAY)
6. **Tick Normalizer** — separation of normalization from scoring (explicit testable step)

### Project Structure Addition
```
nightshade/
├── layer1c/
│   ├── __init__.py
│   ├── source_protocol.py
│   ├── tick_normalizer.py
│   ├── sequence_tracker.py
│   ├── polygon_adapter.py
│   ├── databento_adapter.py
│   ├── market_hours.py
│   ├── gap_fill_orchestrator.py
│   ├── connection_health_monitor.py
│   ├── ingestor_supervisor.py
│   └── live_cli.py
├── data/
│   └── exchange_hours.yaml
├── tests/layer1c/
│   ├── __init__.py
│   ├── test_tick_normalizer.py
│   ├── test_sequence_tracker.py
│   ├── test_polygon_adapter.py
│   ├── test_market_hours.py
│   ├── test_gap_fill_orchestrator.py
│   └── test_connection_health_monitor.py
```

---

### Component 1: Source Protocol (`layer1c/source_protocol.py`)

**Protocol class `LiveDataSourceProtocol`** — structural subtyping (no inheritance required)

| Method | Signature |
|--------|-----------|
| `connect()` | Idempotent. Sets state to CONNECTED. |
| `disconnect()` | Flushes pending messages, sets DISCONNECTED. |
| `subscribe(nightshade_ids)` | Resolves to external symbols via SecurityMaster. Raises `SubscriptionError`. |
| `get_source_name()` | Returns canonical source string ("polygon_ws", "databento_ws"…) |
| `get_connection_state()` | DISCONNECTED / CONNECTING / CONNECTED / RECONNECTING / ERROR |
| `get_health_metrics()` | Dict with: `source_name`, `connection_state`, `messages_received_total`, `messages_received_last_minute`, `sequence_gaps_detected`, `last_message_ts_recv`, `current_latency_ms`, `uptime_seconds` |

**Dataclass `RawMessageProtocol`:** `source`, `raw_payload`, `ts_recv_ns`, `sequence_number`

**Custom Exceptions:** `SourceProtocolError` (base), `ConnectionError`, `SubscriptionError`, `AuthenticationError`, `SourceUnavailableError`

---

### Component 2: Tick Normalizer (`layer1c/tick_normalizer.py`)

**Class:** `TickNormalizer` — registry of normalization functions keyed by source name. Registered at init for polygon_ws, databento_ws, yfinance_hist.

**`normalize(raw_message) -> dict | None`** — looks up normalizer, calls it, returns canonical tick or None. Never raises. Logs failures at DEBUG (non-trade events are correctly discarded constantly).

**Extension point:** `register_normalizer(source_name, normalizer_fn)` — new sources need only implement the normalizer function.

**Canonical tick dict keys:** `nightshade_id`, `ts_event_ns`, `ts_recv_ns`, `source`, `price_fixed`, `size`, `exchange`, `conditions_bitmask`, `raw_sequence_number`, `instrument_type`

**Built-in normalizers:**

| Normalizer | Input fields | Key conversions |
|-----------|-------------|-----------------|
| `_normalize_polygon_ws` | `ev`, `sym`, `p` (float), `s`, `t` (ms), `x` (exchange ID int), `c` (conditions list), `z` (tape) | p→fixed, t ms→ns, x→MIC code (hardcoded map), c list→bitmask, sym→nightshade_id via SecurityMaster. Returns None if ev≠"T". |
| `_normalize_databento_ws` | `ts_event` (ns), `ts_recv` (ns), `instrument_id`, `price` (Databento 1e-9 scale), `size`, `action`, `side`, `flags` | Convert 1e-9 price scale to Nightshade 1e-4. Resolve instrument_id via SecurityMaster. Returns None if action≠"T". |
| `_normalize_yfinance_hist` | `Date`, `Open/High/Low/Close`, `Volume`, `ticker` | ts_event_ns = market close (16:00 ET→UTC). price_fixed from Close. exchange = XNAS (US) or XNSE (.NS). |

**Module-level functions:**
- `get_polygon_exchange_map() -> dict[int, str]` — complete Polygon exchange ID → MIC mapping (IDs 1–21)
- `conditions_list_to_bitmask(conditions) -> int` — bit N set if condition N in list. Handles None/empty → 0.

---

### Component 3: Sequence Tracker (`layer1c/sequence_tracker.py`)

**Class:** `SequenceTracker` — per-`(source_name, session_id)` state tracking

**Dataclass `SequenceGap`:** `source_name`, `session_id`, `gap_start_sequence`, `gap_end_sequence`, `estimated_missing_count`, `detected_at_ts_recv_ns`

**Dataclass `SequenceState`:** `last_sequence`, `first_sequence`, `total_messages`, `total_gaps`, `session_start_ts_recv_ns`, `last_message_ts_recv_ns`

| Method | Behaviour |
|--------|-----------|
| `record_message(source_name, session_id, sequence_number, ts_recv_ns)` | First message → init state, return None. Seq = last+1 → update, return None. Seq > last+1 → return SequenceGap. Seq ≤ last → out-of-order, log DEBUG, return None. |
| `reset_session(source_name, session_id)` | Archives old state, clears for new session (reconnect). |
| `get_session_statistics(source_name, session_id)` | SequenceState dict + computed `gap_rate`, `coverage_pct`. |
| `get_all_gaps(source_name)` | All detected gaps, optionally filtered by source. |

---

### Component 4: Polygon Adapter (`layer1c/polygon_adapter.py`)

Supersedes `PolygonWebSocketIngestor` from Layer 1B. Satisfies `LiveDataSourceProtocol`.

**Class:** `PolygonWebSocketAdapter`

**Key behaviours:**
- API key read from `secrets_manager.get("polygon.api_key")`. If missing, logs exact command (`python -m layer0.secrets set polygon.api_key YOUR_KEY_HERE`) then raises `AuthenticationError`.
- Three-message Polygon auth handshake: server connected → client sends auth → server sends auth_success or auth_failed.
- `subscribe()` batches all ticker subscriptions in one message (`T.AAPL,T.MSFT,...`) — individual messages trigger Polygon's rate limiter.
- `_process_single_event()`: Normalizer → DataQualityScorer → SequenceTracker → (if gap) GapTracker → write buffer → Redis stream.
- 500ms flush timer enforced in message loop.
- `_flush_write_buffer()`: sets `ts_db_write = time.time_ns() // 1000` (microseconds) immediately before batch write. Buffer retained on failure.
- Exponential backoff reconnection: 1s → doubles → caps 60s → CRITICAL after 10 failures.

---

### Component 5: Databento Adapter (`layer1c/databento_adapter.py`)

**Class:** `DatabentоWebSocketAdapter` — satisfies `LiveDataSourceProtocol`

**Key differences from Polygon:**
- Binary DBN encoding (not JSON) — handled by `databento` library
- Auth via API key as URL query parameter (not post-connection message)
- Schema: MBP-1 (Market By Price, top of book)
- Datasets: `XNAS.ITCH` (US), `XNSE.ITCH` (India) — both read from config
- Resolves `instrument_id` via SecurityMaster for source="databento"
- Databento price scale 1e-9 → Nightshade 1e-4 conversion

**Graceful degradation:** If `databento` library not installed, module is still importable but raises `SourceUnavailableError` with pip install command when `connect()` is called.

---

### Component 6: Market Hours Manager (`layer1c/market_hours.py`)

**Class:** `MarketHoursManager` — reads `data/exchange_hours.yaml`, caches schedule in memory, reloads at midnight UTC daily.

| Method | Behaviour |
|--------|-----------|
| `get_session_state(exchange_mic, ts_utc_ns)` | Returns PRE_MARKET / REGULAR / POST_MARKET / CLOSED / HOLIDAY. Checks: day-of-week → holidays → pre-market → regular → post-market windows. |
| `is_regular_session(exchange_mic, ts_utc_ns)` | True only if REGULAR. |
| `get_next_market_open(exchange_mic, ts_utc_ns)` | Next regular open timestamp (ns UTC). Skips weekends + holidays. |
| `get_session_boundaries(exchange_mic, date_str)` | Dict of `pre_market_open_utc_ns`, `regular_open_utc_ns`, `regular_close_utc_ns`, `post_market_close_utc_ns`. None for non-existent sessions. |

**`data/exchange_hours.yaml`** — complete data for:
| Exchange | Timezone | Regular Hours | Pre/Post Market |
|----------|----------|---------------|-----------------|
| XNAS (NASDAQ) | US/Eastern | 09:30–16:00 | 04:00–09:30 / 16:00–20:00 |
| XNYS (NYSE) | US/Eastern | 09:30–16:00 | Same as NASDAQ |
| XNSE (NSE India) | Asia/Kolkata (UTC+5:30) | 09:15–15:30 | None |
| XBSE (BSE India) | Asia/Kolkata | 09:15–15:30 | None |

**Includes:** Complete 2024 + 2025 US market holiday calendars + NSE holiday calendars. Early close days (day before Thanksgiving, Christmas Eve): US markets close at 13:00 ET.

---

### Component 7: Gap Fill Orchestrator (`layer1c/gap_fill_orchestrator.py`)

**Class:** `GapFillOrchestrator` — wraps GapTracker with rate limiting, prioritization, progress reporting. Background `_fill_loop()` runs every 10 seconds.

**Inner class `TokenBucket`** (thread-safe via `threading.Lock`):

| Method | Behaviour |
|--------|-----------|
| `add_tokens()` | Adds tokens based on elapsed time. Rate = `rate_per_minute / 60` tokens/second. |
| `consume(count=1)` | Returns True + decrements if available. Returns False if insufficient. |

**Priority scoring formula:**
- Recency: gap < 1 hour → +100; < 1 day → +50; older → +10
- Importance: instrument in active universe → +50
- Size: < 1000 missing messages → +30; else → +10

**`_fetch_gap_from_polygon(gap)`:** Calls `GET /v3/trades/{ticker}?timestamp.gte=...&timestamp.lte=...`. Follows `next_url` pagination (max 10 pages). Converts via TickNormalizer. Returns normalized ticks.

**Rate limits:** Default 5 req/min (free tier), configurable for paid tiers.

**Statistics:** `{total_gaps_processed, total_gaps_filled, total_gaps_failed, total_gaps_unfillable, tokens_available, queue_depth, last_fill_ts, fill_rate_per_hour}`

---

### Component 8: Connection Health Monitor (`layer1c/connection_health_monitor.py`)

**Class:** `ConnectionHealthMonitor` — polls all registered adapters every 10 seconds. Writes health scores to Redis.

**Health score (0–100):** Start 100. Deduct:
- −20 if `connection_state ≠ "CONNECTED"`
- −10 if `messages_received_last_minute = 0` during market hours
- −10 per sequence gap in last minute (max −30)
- −20 if latency > 500ms
- −10 if latency > 200ms

**Redis keys:** `nightshade:health:{source_name}` — JSON with `{health_score, metrics, ts_written_utc}`. TTL: 60 seconds.

**Alert transitions:**
- First drop below 50 (healthy → degraded): WARNING
- First drop below 20 (degraded → failed): CRITICAL
- Rise above 80 after being below 50 (recovery): INFO
- No repeated alerts for unchanged state.

---

### Component 9: Ingestor Supervisor (`layer1c/ingestor_supervisor.py`)

**Class:** `IngestorSupervisor` — top-level orchestrator for the entire live data layer.

**Initialization order (strict):** `SecurityMaster → QuestDBClient → RedisStreamClient → SchemaManager (create_all_tables) → DataQualityScorer → GapTracker → TickNormalizer → SequenceTracker → MarketHoursManager → PolygonWebSocketAdapter → DatabentоWebSocketAdapter (optional) → GapFillOrchestrator → ConnectionHealthMonitor`

**`start()`:**
1. Subscribes all adapters to current universe
2. Connects all adapters
3. Starts GapFillOrchestrator
4. Starts status writer background thread (writes to `nightshade:supervisor:status` every 10s)
5. Blocks main thread until SIGINT/SIGTERM → calls `stop()` and waits for clean shutdown

**Status dict in Redis:** `{ts_utc, adapters: {source_name: {health_score, connection_state}}, gaps: open_count, ticks_written_today, ticks_rejected_today}`

**`handle_universe_change(added_ids, removed_ids)`:** Sends updated subscriptions for added instruments. Does not unsubscribe removed instruments (they are still scored and stored, just not in active universe).

---

### Component 10: Layer 1C CLI (`layer1c/live_cli.py`)

| Command | Action |
|---------|--------|
| `python -m layer1c.live_cli start` | Starts IngestorSupervisor (blocks) |
| `python -m layer1c.live_cli status` | Reads `nightshade:supervisor:status` from Redis |
| `python -m layer1c.live_cli health` | Reads all `nightshade:health:*` keys, prints dashboard |
| `python -m layer1c.live_cli gaps --list` | Lists open gaps sorted by priority descending |
| `python -m layer1c.live_cli gaps --fill` | Triggers immediate fill cycle |
| `python -m layer1c.live_cli subscribe --source S --symbols SYM1,SYM2` | Writes command to `nightshade:supervisor:commands`, polled every 5s |
| `python -m layer1c.live_cli market-hours --exchange MIC --date DATE` | Prints session boundaries |

---

### Component 11: `config.yaml` Additions (Layer 1C)

```yaml
live_data:
  polygon:
    ws_url: "wss://socket.polygon.io/stocks"
    rest_base_url: "https://api.polygon.io"
    rest_rate_limit_per_minute: 5           # Free tier; paid: 100
    ws_reconnect_initial_delay_seconds: 1
    ws_reconnect_max_delay_seconds: 60
    ws_reconnect_critical_after_attempts: 10
    write_buffer_size: 500
    write_buffer_timeout_ms: 500
  databento:
    ws_gateway_url: "wss://live.databento.com/v0/live"
    dataset_us_equities: "XNAS.ITCH"
    dataset_india_equities: "XNSE.ITCH"
    schema: "trades"
sequence_tracker:
  gap_alert_threshold: 1
  gap_critical_count: 10
health_monitor:
  check_interval_seconds: 10
  degraded_threshold: 50
  failed_threshold: 20
  redis_key_expiry_seconds: 60
supervisor:
  status_write_interval_seconds: 10
  command_poll_interval_seconds: 5
exchange_hours:
  config_file: "data/exchange_hours.yaml"
  reload_time_utc: "00:00"
```

---

### Layer 1C Test Coverage (all mocked)

| File | Scenarios |
|------|-----------|
| `test_tick_normalizer.py` | Valid Polygon trade normalizes correctly, ev="Q"→None, unresolvable symbol→None, Databento 1e-9→1e-4 conversion, conditions_list_to_bitmask([12,41]) correct bits, empty list→0, custom normalizer registration |
| `test_sequence_tracker.py` | First message→None, seq+1→None, seq+5→SequenceGap(missing=4), out-of-order→None, reset_session archives and clears, gap_rate+coverage_pct correct |
| `test_polygon_adapter.py` | Missing API key→AuthenticationError+correct command, auth_failed→AuthenticationError, score=0→reject counter incremented, SequenceGap→GapTracker.record_gap called, successful flush clears buffer, failed flush retains buffer |
| `test_market_hours.py` | XNAS 14:00 UTC Tuesday→REGULAR, 09:00 UTC Tuesday→PRE_MARKET, 21:00 UTC→CLOSED, Saturday→CLOSED, holiday→HOLIDAY, XNSE 04:30 UTC Tuesday→REGULAR, next_market_open skips weekend, XNSE pre_market_open→None |
| `test_gap_fill_orchestrator.py` | TokenBucket consume→True decrements, empty bucket→False, health score=100 for healthy metrics, −20 for disconnected, pagination followed correctly, recent gap scored higher |

---

## PHASE 2 — LAYER 2: OBSERVABILITY LAYER

### Purpose
Without this layer, the system is a black box. Data flows in, signals flow out, and when something breaks you have no idea why.

**Four mechanisms:**
1. **Metrics Collector** — receives metrics from all modules via Unix domain socket → QuestDB
2. **Health Monitor** — polls all components via `health_check()` → state machine → alerts
3. **Audit Log** — append-only SQLite for every significant decision (signals, risk, orders, kills)
4. **Dashboard Writer** — ASCII dashboard to local file, monitorable with `tail -f`

### Project Structure Addition
```
nightshade/
├── layer2/
│   ├── __init__.py
│   ├── metrics_schema.py
│   ├── metrics_collector.py
│   ├── metrics_emitter.py
│   ├── health_monitor.py
│   ├── audit_log.py
│   ├── dashboard_writer.py
│   ├── system_resource_monitor.py
│   └── observability_cli.py
├── tests/layer2/
│   ├── __init__.py
│   ├── test_metrics_collector.py
│   ├── test_metrics_emitter.py
│   ├── test_health_monitor.py
│   ├── test_audit_log.py
│   └── test_dashboard_writer.py
```

---

### Component 1: Metrics Schema (`layer2/metrics_schema.py`)

**Class:** `MetricsSchemaManager`

#### `metrics_layer2` table (QuestDB)
```sql
CREATE TABLE IF NOT EXISTS metrics_layer2 (
  ts           TIMESTAMP,
  component    SYMBOL CAPACITY 64  CACHE INDEX,
  metric_name  SYMBOL CAPACITY 256 CACHE INDEX,
  metric_value DOUBLE,
  host         SYMBOL CAPACITY 8   CACHE INDEX,
  environment  SYMBOL CAPACITY 4   CACHE INDEX,
  tags         STRING                          -- JSON string of key-value metadata
) TIMESTAMP(ts) PARTITION BY DAY WAL;
```

#### `health_history` table (QuestDB)
```sql
CREATE TABLE IF NOT EXISTS health_history (
  ts                  TIMESTAMP,
  component           SYMBOL CAPACITY 64 CACHE INDEX,
  health_state        SYMBOL CAPACITY 8  CACHE INDEX,  -- HEALTHY/DEGRADED/FAILED/UNKNOWN
  health_score        INT,
  consecutive_failures INT,
  message             STRING,
  host                SYMBOL CAPACITY 8  CACHE INDEX
) TIMESTAMP(ts) PARTITION BY DAY WAL;
```

**Dataclasses:**
- `Metric`: `ts_utc_ns`, `component`, `metric_name`, `metric_value`, `host`, `environment`, `tags` (dict). Method `to_ilp_dict()`.
- `HealthCheckResult`: `component`, `health_state`, `health_score`, `consecutive_failures`, `message`, `ts_utc_ns`, `response_time_ms`

---

### Component 2: Metrics Collector (`layer2/metrics_collector.py`)

**Class:** `MetricsCollector` — Unix domain socket server. Two background threads: `_accept_loop` + `_flush_loop`.

**Socket:** `/tmp/nightshade_metrics.sock` (default from config). Permissions `0o666` (any local process can write). Removed on clean shutdown. If exists on startup (unclean previous shutdown) → delete and recreate.

**Protocol:** Newline-delimited JSON. One JSON object per line. Each object is a serialized `Metric` dict.

| Method | Behaviour |
|--------|-----------|
| `start()` | Binds socket, starts threads. |
| `_accept_loop()` | Accepts connections, spawns per-connection thread. |
| `_handle_connection(conn)` | Reads lines, validates required fields, appends to buffer under lock. Discards malformed lines at DEBUG. |
| `_flush_loop()` | Every `flush_interval_seconds`, acquires lock, copies+clears buffer, releases lock, calls `_write_batch()`. |
| `_write_batch(metrics)` | Writes to QuestDB via `write_ticks_batch()`. On failure: logs ERROR + WARNING alert. Does not re-queue (best-effort). Records its own write latency as a metric. |
| `stop()` | Stops threads. Flushes remaining buffer. Closes + removes socket file. |
| `health_check()` | HEALTHY if socket bound + last flush succeeded. DEGRADED if last flush failed or buffer > 80% full. FAILED if socket not bound. |

---

### Component 3: Metrics Emitter (`layer2/metrics_emitter.py`)

**Class:** `MetricsEmitter` — the client side. Every module imports and uses this.

**Cardinal rule: NEVER raises. NEVER blocks > 10ms. Socket timeout = 10ms.**

| Method | Behaviour |
|--------|-----------|
| `emit(metric_name, metric_value, tags)` | Creates Metric, serializes to JSON + newline, writes to socket. On failure: increments dropped counter, attempts reconnect next call. Logs at DEBUG only. |
| `emit_counter(metric_name, increment, tags)` | Suffixes `.count` if not present. |
| `emit_gauge(metric_name, value, tags)` | Suffixes `.gauge` if not present. |
| `emit_timing(metric_name, duration_ms, tags)` | Suffixes `.ms` if not present. |
| `timing_context(metric_name, tags)` | Context manager. Measures wall-clock time, calls `emit_timing()` on exit. Does not suppress exceptions inside context. |
| `get_dropped_count()` | Count of emission failures since init. |

**Module-level function:** `get_emitter(component, config) -> MetricsEmitter` — singleton per component name. Thread-safe. All modules call this once at init.

---

### Component 4: Health Monitor (`layer2/health_monitor.py`)

**Class:** `HealthMonitor` — polling-based (not push). Polling is authority: crashed component stops responding = failure signal.

**`HealthStateRecord`:** `component`, `current_state`, `previous_state`, `consecutive_failures`, `consecutive_successes`, `last_check_ts`, `last_state_change_ts`, `total_checks`, `total_failures`, `alert_sent_for_current_state`

**State machine:**
```
UNKNOWN → result immediately
HEALTHY → DEGRADED (after degraded_threshold consecutive non-HEALTHY)
DEGRADED → FAILED (after failure_threshold consecutive non-HEALTHY)
DEGRADED → HEALTHY (after 2 consecutive HEALTHY)
FAILED → DEGRADED (after 1 HEALTHY) → HEALTHY (after 2 more HEALTHY)
```
Sustained recovery required before declaring HEALTHY — prevents flapping alerts.

**Alert transitions:** HEALTHY→DEGRADED: WARNING. DEGRADED→FAILED: CRITICAL. Any→HEALTHY: INFO. One alert per state (no repeated alerts for same state).

**Each health_check runs in its own thread with enforced timeout** — slow checks don't block monitoring of other components.

**Redis key `nightshade:health:system`:** JSON system health summary. TTL 90 seconds. Individual component keys: `nightshade:health:component:{name}`.

**Components to register** (done by top-level startup):
- QuestDBClient, RedisStreamClient, PolygonWebSocketAdapter, DatabentоWebSocketAdapter (if instantiated), GapFillOrchestrator, MetricsCollector, SystemResourceMonitor, DashboardWriter

---

### Component 5: Audit Log (`layer2/audit_log.py`)

**Database:** SQLite at `~/.nightshade/audit.db`. PRAGMAs: WAL, foreign_keys, synchronous=FULL, `check_same_thread=False`.

**`AuditEventType` enum:**
```
SIGNAL_GENERATED, RISK_CHECK_PASSED, RISK_CHECK_FAILED,
ORDER_SUBMITTED, ORDER_FILLED, ORDER_CANCELLED, ORDER_REJECTED,
POSITION_OPENED, POSITION_CLOSED,
ENGINE_STARTED, ENGINE_SUSPENDED, ENGINE_RESUMED, ENGINE_STOPPED,
KILL_SWITCH_ARMED, KILL_SWITCH_TRIGGERED, KILL_SWITCH_RESET,
DATA_GAP_DETECTED, DATA_GAP_FILLED, DATA_GAP_UNFILLABLE,
UNIVERSE_CHANGED, CORPORATE_ACTION_APPLIED,
SYSTEM_STARTUP, SYSTEM_SHUTDOWN,
HEALTH_STATE_CHANGED, ALERT_SENT
```

**`audit_events` table schema:**
| Column | Notes |
|--------|-------|
| `audit_id` | UUID4 PK |
| `ts_utc` | ISO 8601 UTC microsecond precision |
| `event_type` | AuditEventType string |
| `engine_id` | nullable |
| `nightshade_id` | nullable |
| `component` | NOT NULL |
| `severity` | NOT NULL |
| `message` | NOT NULL |
| `payload` | JSON blob NOT NULL |
| `session_id` | UUID generated once per process startup (all entries in same run share session_id) |
| `host` | NOT NULL |
| `environment` | NOT NULL |

**Indexes:** `ts_utc`, `event_type`, `nightshade_id`, `engine_id`, `session_id`

| Method | Behaviour |
|--------|-----------|
| `log(event_type, component, message, payload, ...)` | Thread-safe (threading.Lock). Returns `audit_id`. Never raises — on DB failure logs CRITICAL to application log. |
| `query(event_type, engine_id, nightshade_id, component, start_ts, end_ts, session_id, limit)` | All filters optional, combined with AND. Parameterized. Returns list[dict] ordered by ts_utc descending. |
| `get_session_summary(session_id)` | `{total_events, events_by_type, events_by_severity, signals_generated, orders_submitted, orders_filled, kill_switch_triggers, health_state_changes, session_start_ts, session_end_ts}` |
| `get_engine_audit_trail(engine_id, start_ts, end_ts)` | Chronological audit for one engine. Used in post-mortems. |
| `export_session_to_json(session_id, output_path)` | Newline-delimited JSON. Returns event count. For regulatory record-keeping. |

---

### Component 6: System Resource Monitor (`layer2/system_resource_monitor.py`)

**Class:** `SystemResourceMonitor` — uses `psutil`. Background sampling thread. Alert thresholds from config.

**System metrics emitted:** `system.cpu.percent_total`, `system.cpu.percent_per_core` (tagged `core_id`), `system.memory.used_mb`, `system.memory.available_mb`, `system.memory.percent`, `system.swap.used_mb`, `system.swap.percent`, `system.disk.used_gb`, `system.disk.free_gb`, `system.disk.percent`, `system.network.bytes_sent_per_sec`, `system.network.bytes_recv_per_sec`, `system.load_avg_1m/5m/15m`

**Process metrics emitted:** `process.cpu.percent`, `process.memory.rss_mb`, `process.memory.vms_mb`, `process.threads.count`, `process.open_files.count`, `process.connections.count`

**Thresholds (configurable):** CPU 85%, Memory 90%, Disk 85%. Sends WARNING alert when exceeded.

**`health_check()`:** HEALTHY if all below thresholds. DEGRADED if any above. FAILED if psutil raises exception.

**Note:** If psutil not installed, module raises `ImportError` at import time with clear message — it is a required dependency.

---

### Component 7: Dashboard Writer (`layer2/dashboard_writer.py`)

**Class:** `DashboardWriter` — writes ASCII dashboard to `~/.nightshade/dashboard.txt`. Atomic writes (temp → rename). Default refresh: 30 seconds. Monitor via `tail -f`.

**Dashboard sections:**

| Section | Content |
|---------|---------|
| **Header** | "NIGHTSHADE TRADING SYSTEM — OPERATIONS DASHBOARD", UTC timestamp, environment (PAPER/LIVE), uptime |
| **System Health** | Overall state (HEALTHY/DEGRADED/FAILED), table of components with state, score, consecutive failures, time since last change. ASCII symbols: ✓ HEALTHY, ⚠ DEGRADED, ✗ FAILED |
| **Live Data Ingestion** | Ticks received/rejected per minute per source, quality score distribution %, sequence gaps, buffer depth, write latency ms |
| **Data Lake** | Bronze row count, Silver bar count, Gold feature count, most recent tick/bar timestamps, open gap count, oldest gap age |
| **System Resources** | CPU/memory/disk with ASCII bars, thread count, file count |
| **Recent Alerts** | Last 10 alerts: timestamp, severity, component, message (truncated to 80 chars) |
| **Footer** | Dashboard file path, refresh interval, `tail -f` instruction |

**Helper methods:**
- `_format_ascii_bar(percent, width=20)` — e.g. 75% → `[############### ] 75.0%`
- `_format_table(headers, rows, col_widths)` — fixed-width ASCII table, auto-width if not specified, header separator line

---

### Component 8: Observability CLI (`layer2/observability_cli.py`)

| Command | Action |
|---------|--------|
| `python -m layer2.observability_cli start` | Starts full observability stack (blocks) |
| `python -m layer2.observability_cli health` | Reads `nightshade:health:system` from Redis, prints table |
| `python -m layer2.observability_cli metrics --component C --metric M --last N` | Queries QuestDB, prints time series with ASCII chart |
| `python -m layer2.observability_cli audit --event-type TYPE --last HOURS` | Queries audit log, prints formatted blocks |
| `python -m layer2.observability_cli audit --session SESSION_ID` | Prints session summary |
| `python -m layer2.observability_cli audit --export --session ID --output PATH` | Exports session to JSON |
| `python -m layer2.observability_cli dashboard` | Prints current dashboard file to stdout |
| `python -m layer2.observability_cli resources` | Prints current resource snapshot from QuestDB |

---

### Component 9: `config.yaml` Additions (Layer 2)

```yaml
observability:
  metrics_collector:
    socket_path: "/tmp/nightshade_metrics.sock"
    batch_size: 200
    flush_interval_seconds: 5
    max_buffer_size: 10000
  health_monitor:
    check_interval_seconds: 30
    failure_threshold: 3
    degraded_threshold: 2
    component_timeout_seconds: 5.0
    redis_key_expiry_seconds: 90
  system_resources:
    sample_interval_seconds: 15
    cpu_alert_pct: 85
    memory_alert_pct: 90
    disk_alert_pct: 85
    data_dir: "~/.nightshade"
  audit_log:
    db_path: "~/.nightshade/audit.db"
    write_lock_timeout_seconds: 5.0
  dashboard:
    output_path: "~/.nightshade/dashboard.txt"
    refresh_interval_seconds: 30
    ascii_bar_width: 25
    alerts_to_display: 10
    max_line_width: 100
  metrics_table_name: "metrics_layer2"
  health_table_name: "health_history"
```

---

### Component 10: Standard Metrics Catalogue

**Layer 1B metrics:**
| Module | Metric Names |
|--------|-------------|
| QuestDBClient | `questdb.write.latency_ms`, `questdb.write.batch_size`, `questdb.write.errors.count`, `questdb.query.latency_ms`, `questdb.health.ilp_score`, `questdb.health.pg_score` |
| RedisStreamClient | `redis.stream.write_latency_ms`, `redis.stream.pending_count` (tagged `nightshade_id`), `redis.health.connected` |
| DataQualityScorer | `dq.score.distribution.0–4`, `dq.rejection.count` (tagged `reason`) |
| GapTracker | `gaps.open.count`, `gaps.fill_attempts.count`, `gaps.filled.count`, `gaps.unfillable.count` |

**Layer 1C metrics:**
| Module | Metric Names |
|--------|-------------|
| PolygonWebSocketAdapter | `polygon.messages.received.count`, `polygon.messages.rejected.count`, `polygon.write_buffer.size`, `polygon.latency.mean_ms`, `polygon.sequence.gaps.count`, `polygon.reconnections.count` |
| DatabentоWebSocketAdapter | Same with `databento.` prefix |
| GapFillOrchestrator | `gap_fill.token_bucket.available`, `gap_fill.queue.depth`, `gap_fill.fills.succeeded.count`, `gap_fill.fills.failed.count` |
| ConnectionHealthMonitor | `connection_health.score` (tagged `source_name`) |

**Layer 2 metrics:**
| Module | Metric Names |
|--------|-------------|
| MetricsCollector | `collector.received.count`, `collector.written.count`, `collector.dropped.count`, `collector.buffer.size`, `collector.write.latency_ms` |
| HealthMonitor | `health.check.latency_ms` (tagged `component`), `health.component.score` (tagged `component`) |
| SystemResourceMonitor | All `system.*` and `process.*` metrics |

---

### Component 11: Integration Wiring Guide

**Inject MetricsEmitter post-initialization** (no constructor changes to Layer 1B/1C):

Every Layer 1B and 1C class adds:
```python
def set_metrics_emitter(self, emitter: MetricsEmitter) -> None:
    self._emitter = emitter
# Emit pattern: if self._emitter: self._emitter.emit(...)
```

`IngestorSupervisor.start()` calls `set_metrics_emitter()` on every component immediately after `MetricsCollector` starts, before starting any other component.

**AuditLog accessibility:**
- `IngestorSupervisor` → logs `SYSTEM_STARTUP` + `SYSTEM_SHUTDOWN`
- Layer 3 alpha engines → log `SIGNAL_GENERATED`
- Layer 4 risk shield → logs `RISK_CHECK_PASSED` + `RISK_CHECK_FAILED`

---

### Layer 2 Test Coverage (all mocked)

| File | Scenarios |
|------|-----------|
| `test_metrics_collector.py` | Socket created on start, valid JSON line parsed+buffered, malformed JSON discarded, missing required field discarded, buffer flushed after interval, QuestDB exception → logged not crashed, stop flushes remaining, statistics accurate |
| `test_metrics_emitter.py` | emit() sends correct JSON, no exception if socket unavailable, dropped counter increments on failure, timing_context measures time correctly, timing_context doesn't suppress exceptions, singleton per component name, different component→different instance |
| `test_health_monitor.py` | HEALTHY result recorded correctly, FAILED × degraded_threshold→DEGRADED, FAILED consistently→FAILED, FAILED then HEALTHY→DEGRADED before HEALTHY (sustained recovery), exception→FAILED, timeout→FAILED, WARNING sent once on HEALTHY→DEGRADED, CRITICAL on DEGRADED→FAILED, get_system_health FAILED if any component FAILED |
| `test_audit_log.py` | log() inserts UUID+correct timestamp, no raise on DB failure, query by event_type, query by time range, multi-filter AND, same-process entries share session_id, session summary event-type counts, export writes valid NDJSON, 10-thread concurrent log safety |
| `test_dashboard_writer.py` | ascii_bar(75%, 20) correct width+15 fills, ascii_bar(0%) no fill, ascii_bar(100%) full, format_table correct column count, _build_dashboard non-empty when Redis valid, no exception when Redis returns None, stop terminates thread within 5s |

---

## LAYERS 3–7: ARCHITECTURE REFERENCE

### Layer 3: Research Track (Alpha Engine Development)

#### Alpha Registry
Plugin system. All engines inherit from `AlphaEngine` base class.

**Required interface:**
```python
def initialize(universe, config): ...
def on_bar(bar_data): ...
def get_signals(): ...
def get_health_metrics(): ...
def shutdown(): ...
```

#### Signal Contract (standardized output format)
```python
@dataclass
class Signal:
    engine_id: str
    nightshade_id: str
    signal_time: datetime
    direction: str             # "long", "short", "flat"
    predicted_return_bps: float   # normalized to 1-day holding period
    confidence: float          # [0,1] = signal_strength / (2 × historical_std)
    holding_period_days: float
    universe_version: str
    model_version: str
```

#### Engine 1: StatArb Cointegration Engine
- **Method:** Engle-Granger two-step, rolling OLS hedge ratio, rolling Z-score spread signal
- **Pair registry:** cointegrated pairs with hedge ratio, Z-score, cointegration health status
- **Health recomputation:** Weekly (full history) + daily (60-day rolling)
- **Suspension trigger:** Rolling ADF p-value > 0.10 for 5 consecutive days → WARNING
- **`get_health_metrics()` returns:** ADF p-value, spread Z-score, 60-day rolling Sharpe, active pairs count

#### Engine 2: PCA-Kalman Residual Engine
- **PCA:** Rolling window standardization (eliminates look-ahead bias), top K components via Gavish-Donoho optimal hard threshold (not arbitrary variance %)
- **Residuals:** Project returns onto PCA basis to extract per-instrument residuals
- **Kalman Filter:** Unscented variant (handles non-Gaussian financial returns). Q and R noise matrices calibrated via MLE on 252-day rolling window, recalibrated every 5 trading days
- **Signal:** Residual > 2 std from Kalman-tracked mean → reversion signal

#### Engine 3: DMD Spectral Engine (built after Engines 1+2 in production)
- **Data:** 5-minute OHLCV bars (respects Nyquist constraint)
- **Pipeline:** Overlapping snapshot matrices → SVD with Gavish-Donoho thresholding → reduced subspace projection → eigendecomposition → continuous-time eigenvalues
- **Alpha window filter:** Real part ∈ (−0.15, −0.02), imaginary part > 0.05
- **Consistency Filter:** Mode must appear in ≥3 consecutive rolling calculations with cosine similarity > 0.85 to generate signal
- **Position sizing:** Half-Kelly criterion
- **Extension:** Hankel DMD with 5 time delays (nonlinear memory effects)

#### Backtesting Framework
- Strict event-driven simulation
- Bi-temporal enforcement: excludes rows where `ts_db_write > simulation_time`
- Transaction costs: 10 bps/trade simulated
- Slippage: Almgren-Chriss model (linear function of order size / ADV)
- **Full performance report:** CAGR, Sharpe, Sortino, Max Drawdown, Calmar, Beta, Alpha, Hit Rate, Return Autocorrelation at lags 1/5/21 days

#### Model Validation Gate (mandatory, no exceptions)
| Criterion | Threshold |
|-----------|-----------|
| Out-of-sample Sharpe | > 1.5 |
| Maximum drawdown | < 10% |
| Portfolio beta | ∈ (−0.1, 0.1) — confirms market neutrality |
| Return autocorrelation (lag 1) | < 0.2 — confirms Sharpe not overstated |
| Look-ahead bias check | Run with bi-temporal filter; if bi-temporal significantly worse than naive → bias present |
| Paper trading record | ≥ 30 trading days positive |

Every validation run logged in audit trail with pass/fail + specific metrics.

---

### Layer 4: Risk Shield

**Enforcement:** Execution layer has NO reference to any alpha engine. Only to Risk Shield's output queue.

#### Pre-Trade Hard Limit Checks (synchronous, < 1ms, binary pass/fail)
| Check | Limit | Rejection Reason |
|-------|-------|-----------------|
| Instrument trading status | Not halted | — |
| Single-name gross exposure | Max 5% of capital | — |
| Sector gross exposure | Max 30% | — |
| Daily P&L | > −3.5%: WARNING; > −4.0%: Soft Stop; > −4.5%: Hard Kill | — |
| Universe membership | Signal's universe version = current | — |
| Data freshness | Most recent data < 5 minutes old | — |

#### Liquidity Filter
Uses Almgren-Chriss model with 20-day ADV and current bid-ask spread. Rejects if estimated market impact (bps) > predicted return (bps). Rejection reason: `INSUFFICIENT_ALPHA_AFTER_IMPACT`

#### VPIN Filter
Reads VPIN from Gold tier. VPIN > 0.70 → reject. Reason: `TOXIC_FLOW_DETECTED`. VPIN bucket size: 50 std units of volume imbalance.

#### Portfolio Correlation Check
Reads 60-day rolling return correlations from Gold tier. If new position correlation > 0.70 with any existing position AND cluster concentration > 5% single-name limit → reject. Reason: `CORRELATION_CONCENTRATION`

#### Volatility-Targeted Position Sizing (transformation, not reject/pass)
```
target_vol = 12% annualized
scaling_factor = target_vol / realized_vol (last 20 trading days)
max scale-up: 2x
max scale-down: 0.25x
```

#### VaR and CVaR Check
10,000-path Monte Carlo with Cholesky-correlated random returns from empirical distribution (fat tails, not Gaussian). If 99% one-day CVaR > 3% of capital → reject. Reason: `VAR_BREACH`

#### Three Kill Switch States
| State | Trigger | Behaviour |
|-------|---------|-----------|
| NORMAL | Default / market open reset | All signals pass |
| SOFT_STOP | Daily P&L < −4.0% | No new positions; existing may close |
| HARD_KILL | Daily P&L < −4.5% | Emergency close all positions; all engines suspended |

Kill switch state written to Redis every second. Reset to NORMAL at market open daily.

---

### Layer 5: Order Management System

**Single source of truth for all positions.** Only the OMS reads from/writes to the positions table.

#### Positions Table (SQLite — ACID compliant, local)
| Column | Notes |
|--------|-------|
| `nightshade_id` | |
| `direction` | long or short |
| `quantity` | |
| `average_entry_price` | |
| `current_mark_price` | |
| `unrealized_pnl` | |
| `realized_pnl` | |
| `entry_time` | |
| `last_update_time` | |
| `engine_id` | Which engine owns this position |
| `status` | open, closing, closed |

#### Orders Table (full lifecycle tracking)
**States:** PENDING_SUBMIT → SUBMITTED → PARTIALLY_FILLED → FILLED / CANCELLED / REJECTED

| Column | Notes |
|--------|-------|
| `order_id` | UUID |
| `nightshade_id` | |
| `direction` | |
| `quantity` | |
| `order_type` | market, limit, TWAP |
| `limit_price` | null for market orders |
| `submitted_time` / `filled_time` | |
| `average_fill_price` | |
| `total_commission` | |
| `slippage_bps` | Computed at fill: (fill_price − signal_price) in bps |
| `engine_id` | |
| `status` | |

#### Position Reconciliation
On every startup: queries broker position API, compares to local state, flags discrepancies as CRITICAL. Accepts broker's version as authoritative, logs discrepancy in audit trail.

#### Implementation Shortfall Tracking
Every fill: `shortfall_bps = avg_fill_price − signal_price` (in bps). Fed back to Layer 3 backtesting for transaction cost calibration. If realized shortfall consistently exceeds modeled cost → engine's backtest parameters flagged for recalibration.

---

### Layer 6: Execution Layer (Paper Trading)

**Interface design:** When real trading begins, only the broker adapter changes. Everything above is identical.

#### Broker Adapter Interface (abstract base class)
```python
def submit_order(order): ...
def cancel_order(order_id): ...
def get_position(nightshade_id): ...
def get_account_balance(): ...
def get_order_status(order_id): ...
```

Concrete implementations: `AlpacaPaperAdapter`, `ZerodhaAdapter` — substituted without changing any other code.

#### Execution Algorithm Selection
| Order Size (fraction of ADV) | Algorithm |
|------------------------------|-----------|
| < 0.1% | Single limit order at current mid-price |
| 0.1% – 1% | TWAP over 30 minutes |
| > 1% | Reject — notify Risk Shield to reduce position size |

#### Fill Simulator (paper mode)
Limit orders filled when price crosses limit in tick stream. Market orders filled at next tick + log-normal slippage (calibrated to historical bid-ask spread).

#### Watchdog (separate process)
Checks every 10 seconds: broker API reachable, last fill confirmed within 60s of submission, no order in SUBMITTED state > 5 minutes. Violations → alert + automatic cancellation for stale orders.

---

### Layer 7: Paper Trading Performance Monitor

#### Paper Portfolio Tracker (daily job)
Reads all fills from OMS orders table. Reconstructs complete P&L history. Writes to performance table: daily return, daily P&L ($), trailing 20-day Sharpe, running max drawdown, gross/net exposure, open positions count.

#### Promotion Checklist (after 30 trading days)
| Criterion | Threshold |
|-----------|-----------|
| Paper Sharpe | > 1.5 |
| Paper max drawdown | < 8% (tighter than backtest — paper should be cleaner) |
| Portfolio beta | ∈ (−0.1, 0.1) |
| ENGINE_SUSPENDED events | Zero during 30-day period |

If passed → promotion recommendation logged in audit trail. Human must manually set engine status to "promoted" in Alpha Registry config file before production instantiation.

---

## TESTING STRATEGY (ALL LAYERS)

### Unit Tests
Every function with a calculation has a unit test. Run automatically before every commit via pre-commit git hook.

### Integration Tests
Test harness: local QuestDB + synthetic data. Runs full pipeline from data quality check → OMS order creation. Verifies audit trail contains expected event sequence. Run nightly.

### Chaos Tests (monthly)
1. Kill WebSocket mid-session → verify: gap detected, gap-fill initiated, data continuity maintained
2. Write bad tick (price=0) → verify: scored quality 0, rejected
3. Corrupt Redis stream → verify: OMS falls back to QuestDB historical record

### Paper Trading Shadow Mode
Before promoting engine: run 2 weeks in shadow mode — signals generated, full Risk Shield + OMS pipeline executed, but execution layer logs orders without submitting. Compare shadow portfolio Sharpe to backtest Sharpe. If divergence > 20% → investigate before promotion.

---

## CRITICAL DESIGN PRINCIPLES

### What This Architecture Explicitly Does NOT Have
- Co-location (requires capital)
- FIX protocol connectivity (requires legal structures)
- Sub-millisecond latency (requires hardware)
- MEV infrastructure (requires engineering maturity)
- DeFi / Uniswap integration (requires capital + legal)

These are absent because the prerequisites are absent. **Every one of them can be added as a new adapter in Layer 6 or a new engine in Layer 3 without touching the rest of the system.** The architecture above is the prerequisite for all of them.

### Non-Negotiable Rules
1. **No API key ever appears in a config file, script, git commit, or log file.** All go through `SecretsManager`.
2. **No float prices in any database write.** All prices in fixed-point integers.
3. **No module uses `print()` in production code.** All logging via `get_logger(__name__)`.
4. **Nothing bypasses the Risk Shield.** Execution layer has no reference to any alpha engine.
5. **Nothing enters the Production Track without graduating from the Research Track.** The Model Validation Gate has no exceptions.
6. **The OMS is the single source of truth for positions.** All modules call `oms.get_position()`.
7. **Every audit entry is immutable once written.** Audit log is append-only.
8. **Secrets file is never committed to git.** Only `config.yaml` is version-controlled.

### Data Quality Hierarchy
```
Score 0 → Reject (never stored)
Score 1 → Store + Flag (suspicious)
Score 2 → Store + Available (marginal)
Score 3 → Store + Available (good)
Score 4 → Store + Available (clean)
```
Each alpha engine sets its own minimum quality threshold.

### Timestamp Priority for Backtesting
```
ts_event  = when trade happened (exchange clock)
ts_recv   = when ingestor received it (network + clock skew)
ts_db_write = when stored in QuestDB (processing latency)
```
**Backtest must use `ts_db_write` as the simulation time boundary.** Using `ts_event` introduces look-ahead bias.

---

## BUILD TIMELINE

### Today (No API Keys, No Capital)
- Layer 0 — secrets, config, logging
- Layer 1A — Security Master (manual 30-instrument list)
- Layer 1B — QuestDB schema, Data Quality Module (synthetic data), WebSocket skeleton (Polygon free tier)
- Layer 1C — Hardened live data layer
- Layer 2 — Observability layer, health monitor, alerting

### In Two Weeks (Databento Access)
- Databento batch ingestor + historical load
- Verify bi-temporal timestamps (spot-check 100 random rows)
- Verify Silver OHLCV matches public historical data for same instruments

### Month After Data Arrives
- Engine 1 (StatArb Cointegration)
- Backtesting framework
- Model Validation Gate
- Begin 30-day paper trading record

### Month After Paper Trading Begins
- Layer 4 (Risk Shield) + Layer 5 (OMS)
- Connect to Engine 1 signal output
- Chaos tests for kill switches

### After 30-Day Paper Trading Passes Promotion Checklist
- Switch paper broker adapter to real prop firm challenge account
- Engine 2 (PCA-Kalman) enters research simultaneously
- Engine 3 (DMD Spectral) enters research 6 months later

---

*Document compiled from: Nightshade_Final_Overall, Nightshade_Final_0, Nightshade_Final_1A, Nightshade_Final_1B, Nightshade_Final_1C, Nightshade_Final_2*
