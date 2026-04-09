from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta


# ---------------------------------------------------------------------------
# Ticker Universe — 30 US-only Nasdaq-100 Technology stocks
# ---------------------------------------------------------------------------

NASDAQ_30: list[str] = [
    # Mega-cap / FAANG+ core
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # Nvidia
    "GOOGL", # Alphabet
    "META",  # Meta Platforms
    "AMZN",  # Amazon
    "TSLA",  # Tesla
    "AVGO",  # Broadcom
    "ADBE",  # Adobe
    "NFLX",  # Netflix
    # Semiconductors
    "TXN",   # Texas Instruments
    "QCOM",  # Qualcomm
    "AMAT",  # Applied Materials
    "AMD",   # Advanced Micro Devices
    "LRCX",  # Lam Research
    "KLAC",  # KLA Corporation
    "SNPS",  # Synopsys
    "CDNS",  # Cadence Design
    "MRVL",  # Marvell Technology
    # Enterprise Software / SaaS
    "INTU",  # Intuit
    "CSCO",  # Cisco
    "CRWD",  # CrowdStrike
    "PANW",  # Palo Alto Networks
    "FTNT",  # Fortinet
    "WDAY",  # Workday
    "DDOG",  # Datadog
    "ZS",    # Zscaler
    "TEAM",  # Atlassian
    "MDB",   # MongoDB
    "ON",    # ON Semiconductor
]

assert len(NASDAQ_30) == 30, (
    f"NASDAQ_30 must have exactly 30 tickers, got {len(NASDAQ_30)}"
)
assert len(set(NASDAQ_30)) == 30, (
    f"NASDAQ_30 contains duplicates: {[t for t in NASDAQ_30 if NASDAQ_30.count(t) > 1]}"
)

# 5 reserve / backup tickers — used for zombie ticker replacement only
RESERVE_BENCH: list[str] = [
    "INTC",   # Intel
    "PYPL",   # PayPal
    "CRM",    # Salesforce
    "ADSK",   # Autodesk
    "ISRG",   # Intuitive Surgical
]

assert len(RESERVE_BENCH) == 5, (
    f"RESERVE_BENCH must have exactly 5 tickers, got {len(RESERVE_BENCH)}"
)

PRIMARY_TICKERS: list[str] = NASDAQ_30
ALL_TICKERS: list[str] = PRIMARY_TICKERS + RESERVE_BENCH


# ---------------------------------------------------------------------------
# Date defaults — rolling 2-year lookback from today
# ---------------------------------------------------------------------------

def _today() -> date:
    return date.today()


END_DATE: date = _today()
START_DATE: date = END_DATE - timedelta(days=730)


# ---------------------------------------------------------------------------
# Live ingest settings
# ---------------------------------------------------------------------------

LIVE_PROVIDER_CHOICES: tuple[str, str] = ("twelvedata", "polygon")
LIVE_BAR_SINK_CHOICES: tuple[str, str] = ("redis", "zeromq")
PRICE_SCALE_1E9: int = 1_000_000_000


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {name}: {raw}") from exc


@dataclass(frozen=True)
class PipelineConfig:
    start_date: date = START_DATE
    end_date: date = END_DATE
    primary_tickers: tuple[str, ...] = tuple(PRIMARY_TICKERS)
    reserve_tickers: tuple[str, ...] = tuple(RESERVE_BENCH)


@dataclass(frozen=True)
class LiveIngestConfig:
    enabled: bool
    provider: str
    bar_sink: str
    symbols: tuple[str, ...]
    heartbeat_seconds: float
    pong_timeout_seconds: float
    reconnect_backoff_seconds: float
    latency_alert_ms: float
    redis_host: str
    redis_port: int
    redis_db: int
    redis_stream_60: str
    redis_stream_500: str
    redis_maxlen_60: int
    redis_maxlen_500: int
    zmq_endpoint: str
    enable_hourly_janitor: bool
    janitor_interval_seconds: float
    janitor_lookback_hours: int
    type2_parquet_root: str
    price_scale: int
    twelvedata_api_key: str | None
    polygon_api_key: str | None


def get_live_ingest_config(
    *,
    provider_override: str | None = None,
    symbols_override: list[str] | None = None,
) -> LiveIngestConfig:
    enabled = os.getenv("QM_ENABLE_LIVE_INGEST", "0").strip() == "1"
    provider = (provider_override or os.getenv("QM_LIVE_PROVIDER", "twelvedata")).strip().lower()
    if provider not in LIVE_PROVIDER_CHOICES:
        allowed = ", ".join(LIVE_PROVIDER_CHOICES)
        raise ValueError(f"Unsupported live provider '{provider}'. Allowed: {allowed}")

    bar_sink = os.getenv("QM_BAR_SINK", "redis").strip().lower()
    if bar_sink not in LIVE_BAR_SINK_CHOICES:
        allowed = ", ".join(LIVE_BAR_SINK_CHOICES)
        raise ValueError(f"Unsupported bar sink '{bar_sink}'. Allowed: {allowed}")

    symbols = tuple(symbols_override or PRIMARY_TICKERS)
    redis_maxlen_60_default = max(1, 60 * len(symbols))
    redis_maxlen_500_default = max(1, 500 * len(symbols))

    return LiveIngestConfig(
        enabled=enabled,
        provider=provider,
        bar_sink=bar_sink,
        symbols=symbols,
        heartbeat_seconds=_env_float("QM_LIVE_HEARTBEAT_SECONDS", 10.0),
        pong_timeout_seconds=_env_float("QM_LIVE_PONG_TIMEOUT_SECONDS", 3.0),
        reconnect_backoff_seconds=_env_float("QM_LIVE_RECONNECT_BACKOFF_SECONDS", 2.0),
        latency_alert_ms=_env_float("QM_LIVE_LATENCY_ALERT_MS", 100.0),
        redis_host=os.getenv("QM_REDIS_HOST", "127.0.0.1"),
        redis_port=_env_int("QM_REDIS_PORT", 6379),
        redis_db=_env_int("QM_REDIS_DB", 0),
        redis_stream_60=os.getenv("QM_REDIS_STREAM_60", "type1:bars:1m:60"),
        redis_stream_500=os.getenv("QM_REDIS_STREAM_500", "type1:bars:1m:500"),
        redis_maxlen_60=_env_int("QM_REDIS_MAXLEN_60", redis_maxlen_60_default),
        redis_maxlen_500=_env_int("QM_REDIS_MAXLEN_500", redis_maxlen_500_default),
        zmq_endpoint=os.getenv("QM_ZMQ_ENDPOINT", "tcp://127.0.0.1:5555"),
        enable_hourly_janitor=os.getenv("QM_ENABLE_HOURLY_JANITOR", "1").strip() == "1",
        janitor_interval_seconds=_env_float("QM_JANITOR_INTERVAL_SECONDS", 3600.0),
        janitor_lookback_hours=_env_int("QM_JANITOR_LOOKBACK_HOURS", 2),
        type2_parquet_root=os.getenv("QM_TYPE2_PARQUET_ROOT", "outputs/live/parquet"),
        price_scale=_env_int("QM_PRICE_SCALE", PRICE_SCALE_1E9),
        twelvedata_api_key=os.getenv("TWELVEDATA_API_KEY"),
        polygon_api_key=os.getenv("POLYGON_API_KEY"),
    )


def validate_live_ingest_config(
    cfg: LiveIngestConfig,
    *,
    require_provider_keys: bool = True,
) -> None:
    if cfg.heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be > 0")
    if cfg.pong_timeout_seconds <= 0:
        raise ValueError("pong_timeout_seconds must be > 0")
    if cfg.reconnect_backoff_seconds <= 0:
        raise ValueError("reconnect_backoff_seconds must be > 0")
    if cfg.latency_alert_ms <= 0:
        raise ValueError("latency_alert_ms must be > 0")
    if cfg.price_scale <= 0:
        raise ValueError("price_scale must be > 0")
    if cfg.janitor_interval_seconds <= 0:
        raise ValueError("janitor_interval_seconds must be > 0")
    if cfg.janitor_lookback_hours < 1:
        raise ValueError("janitor_lookback_hours must be >= 1")
    if cfg.redis_maxlen_60 < len(cfg.symbols):
        raise ValueError("redis_maxlen_60 must be at least number of symbols")
    if cfg.redis_maxlen_500 < len(cfg.symbols):
        raise ValueError("redis_maxlen_500 must be at least number of symbols")

    if require_provider_keys:
        if cfg.provider == "twelvedata" and not cfg.twelvedata_api_key:
            raise ValueError("TWELVEDATA_API_KEY is required for provider 'twelvedata'")
        if cfg.provider == "polygon" and not cfg.polygon_api_key:
            raise ValueError("POLYGON_API_KEY is required for provider 'polygon'")
