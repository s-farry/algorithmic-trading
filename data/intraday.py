"""Intraday market data from FMP.

Provides minute/hourly bar data for backtesting intraday strategies and
running live day-trading signals.

FMP endpoint used:
    GET /historical-chart/{interval}/{symbol}?from=YYYY-MM-DD&to=YYYY-MM-DD

Supported intervals: 1min, 5min, 15min, 30min, 1hour, 4hour

FMP returns bars in US Eastern Time (no timezone suffix).  Market-hours
filtering uses simple string comparison on "HH:MM" so no tz library is needed.

History limits (approximate, varies by plan):
    1min   — ~30 days  (Starter), ~2 years (Premium)
    5min   — ~6 months (Starter), ~5 years (Premium)
    15min  — ~1 year   (Starter), ~5 years (Premium)
    1hour  — ~2 years  (Starter), ~15 years (Premium)
"""

import logging
import time
from datetime import date

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

INTRADAY_CACHE_DIR = config.RESULTS_DIR / "cache" / "intraday"
INTRADAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VALID_INTERVALS = {"1min", "5min", "15min", "30min", "1hour", "4hour"}

# Regular market hours (US Eastern Time)
_MARKET_OPEN = "09:30"
_MARKET_CLOSE = "16:00"


def _vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative intraday VWAP, reset at the start of each trading day."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    day = df["datetime"].dt.date.astype(str)
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum()
    return (cum_pv / cum_vol.replace(0, float("nan"))).round(4)


def get_intraday_prices(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "5min",
    market_hours_only: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch intraday OHLCV bars for a single symbol from FMP.

    Args:
        symbol: Ticker symbol (e.g. "AAPL").
        start_date: Inclusive start date "YYYY-MM-DD".
        end_date: Inclusive end date "YYYY-MM-DD".
        interval: Bar size — one of 1min, 5min, 15min, 30min, 1hour, 4hour.
        market_hours_only: Strip pre/post-market bars (keep 09:30–16:00 ET).
        use_cache: Serve from local cache when available.  Pass False for
                   today's live data (incomplete sessions should not be cached).

    Returns:
        DataFrame sorted ascending by datetime with columns:
            datetime, open, high, low, close, volume, vwap
        Returns an empty DataFrame if no data is available.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {VALID_INTERVALS}, got '{interval}'")

    suffix = "" if market_hours_only else "_full"
    cache_file = INTRADAY_CACHE_DIR / f"{symbol}_{interval}_{start_date}_{end_date}{suffix}.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["datetime"])
        return df

    logger.info(f"Fetching intraday {symbol} ({interval}) {start_date} → {end_date}")
    url = (
        f"{config.FMP_BASE_URL}/historical-chart/{interval}/{symbol}"
        f"?from={start_date}&to={end_date}&apikey={config.FMP_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Intraday fetch failed for {symbol}: {e}")
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "vwap"])

    if not data or not isinstance(data, list):
        logger.warning(f"No intraday data returned for {symbol} ({interval})")
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "vwap"])

    df = pd.DataFrame(data)
    df = df.rename(columns={"date": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df[["datetime", "open", "high", "low", "close", "volume"]]

    if market_hours_only:
        # FMP returns US ET times as naive datetimes — filter by time string
        hhmm = df["datetime"].dt.strftime("%H:%M")
        df = df[(hhmm >= _MARKET_OPEN) & (hhmm < _MARKET_CLOSE)].reset_index(drop=True)

    df["vwap"] = _vwap(df)

    # Only cache if end_date is in the past (completed sessions)
    if use_cache and end_date < date.today().isoformat():
        df.to_csv(cache_file, index=False)

    return df


def get_bulk_intraday_prices(
    symbols: list[str],
    start_date: str,
    end_date: str,
    interval: str = "5min",
    market_hours_only: bool = True,
    use_cache: bool = True,
    delay: float = 0.25,
) -> dict[str, pd.DataFrame]:
    """Fetch intraday prices for multiple symbols.

    Returns a dict of {symbol: DataFrame}.  Symbols with no data are omitted.

    A slightly longer delay (0.25s) is used compared to daily fetches because
    intraday requests consume more FMP API credits.
    """
    prices: dict[str, pd.DataFrame] = {}
    total = len(symbols)
    suffix = "" if market_hours_only else "_full"

    for i, symbol in enumerate(symbols, 1):
        cache_file = INTRADAY_CACHE_DIR / f"{symbol}_{interval}_{start_date}_{end_date}{suffix}.csv"
        served_from_cache = use_cache and cache_file.exists()

        logger.info(f"[{i}/{total}] {symbol} ({'cache' if served_from_cache else 'fetch'})")
        try:
            df = get_intraday_prices(
                symbol, start_date, end_date, interval,
                market_hours_only=market_hours_only, use_cache=use_cache,
            )
            if not df.empty:
                prices[symbol] = df
        except Exception as e:
            logger.warning(f"Skipping {symbol}: {e}")

        if not served_from_cache:
            time.sleep(delay)

    logger.info(f"Fetched intraday data for {len(prices)}/{total} symbols ({interval})")
    return prices
