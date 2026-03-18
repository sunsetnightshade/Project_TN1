from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


NIFTY_10: list[str] = [
    "INFY.NS",
    "TCS.NS",
    "HCLTECH.NS",
    "TECHM.NS",
    "WIPRO.NS",
    "LTIM.NS",
    "PERSISTENT.NS",
    "COFORGE.NS",
    "MPHASIS.NS",
    "OFSS.NS",
]

NASDAQ_20: list[str] = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "META",
    "AMZN",
    "TSLA",
    "AVGO",
    "ADBE",
    "TXN",
    "QCOM",
    "AMAT",
    "INTU",
    "CSCO",
    "NFLX",
    "PEP",
    "COST",
    "TMUS",
    "CMCSA",
    "AMD",
]

RESERVE_BENCH: list[str] = ["INTC", "PYPL", "CRM", "ADSK", "ISRG", "GILD"]


def _today() -> date:
    return date.today()


END_DATE: date = _today()
START_DATE: date = END_DATE - timedelta(days=730)


PRIMARY_TICKERS: list[str] = NIFTY_10 + NASDAQ_20
ALL_TICKERS: list[str] = PRIMARY_TICKERS + RESERVE_BENCH


@dataclass(frozen=True)
class PipelineConfig:
    start_date: date = START_DATE
    end_date: date = END_DATE
    primary_tickers: tuple[str, ...] = tuple(PRIMARY_TICKERS)
    reserve_tickers: tuple[str, ...] = tuple(RESERVE_BENCH)

