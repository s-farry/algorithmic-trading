"""Intraday market data from Polygon.io.

Drop-in replacement for data/intraday.py — exposes the same two functions:
    get_intraday_prices(symbol, start_date, end_date, ...)
    get_bulk_intraday_prices(symbols, start_date, end_date, ...)

Polygon advantages over FMP for intraday backtesting:
    - Free tier includes 2+ years of historical 1-min / 5-min data
    - Cleaner, more consistent data quality
    - VWAP included in the response (used as-is instead of recomputed)
    - Rate limit: 5 requests/minute on free tier (use delay >= 12s between calls)

Polygon.io endpoint used:
    GET /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
    ?adjusted=true&sort=asc&limit=50000&apiKey=...

Timestamps in the response are Unix milliseconds in UTC.
They are converted to US Eastern Time before market-hours filtering so
the same "09:30–16:00" window used by the FMP module applies correctly,
including automatic DST handling via pytz / zoneinfo.

Set POLYGON_API_KEY in your .env file to use this module.
"""

import logging
import time
from datetime import date

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

_CACHE_DIR = config.RESULTS_DIR / "cache" / "polygon_intraday"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VALID_INTERVALS = {"1min", "5min", "15min", "30min", "1hour", "4hour"}

_MARKET_OPEN = "09:30"
_MARKET_CLOSE = "16:00"

# Map our interval strings to Polygon (multiplier, timespan) pairs
_INTERVAL_MAP = {
    "1min":  (1,  "minute"),
    "5min":  (5,  "minute"),
    "15min": (15, "minute"),
    "30min": (30, "minute"),
    "1hour": (1,  "hour"),
    "4hour": (4,  "hour"),
}


def _to_eastern(utc_ms_series: pd.Series) -> pd.Series:
    """Convert a Series of UTC millisecond timestamps to naive US/Eastern datetimes."""
    utc = pd.to_datetime(utc_ms_series, unit="ms", utc=True)
    eastern = utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return eastern


def _vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative intraday VWAP reset each session (fallback if Polygon omits vw)."""
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
    """Fetch intraday OHLCV bars for a single symbol from Polygon.io.

    Args:
        symbol:           Ticker symbol (e.g. "USO", "AAPL").
        start_date:       Inclusive start date "YYYY-MM-DD".
        end_date:         Inclusive end date "YYYY-MM-DD".
        interval:         Bar size — one of 1min, 5min, 15min, 30min, 1hour, 4hour.
        market_hours_only: Strip pre/post-market bars (keep 09:30–16:00 ET).
        use_cache:        Serve from local CSV cache when available.  Always pass
                          False for today's data (incomplete session).

    Returns:
        DataFrame sorted ascending by datetime with columns:
            datetime, open, high, low, close, volume, vwap
        Returns an empty DataFrame on error or no data.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {VALID_INTERVALS}, got '{interval}'")

    if not config.POLYGON_API_KEY:
        raise ValueError(
            "POLYGON_API_KEY is not set. Add it to your .env file.\n"
            "Sign up free at https://polygon.io — the free tier covers 2+ years of history."
        )

    suffix = "" if market_hours_only else "_full"
    cache_file = _CACHE_DIR / f"{symbol}_{interval}_{start_date}_{end_date}{suffix}.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["datetime"])
        return df

    multiplier, timespan = _INTERVAL_MAP[interval]
    url = (
        f"{config.POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range"
        f"/{multiplier}/{timespan}/{start_date}/{end_date}"
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": config.POLYGON_API_KEY,
    }

    logger.info(f"Fetching intraday {symbol} ({interval}) {start_date} → {end_date} [Polygon]")

    rows = []
    next_url = None

    try:
        while True:
            if next_url:
                # Polygon pagination: follow next_url (already includes apiKey)
                resp = requests.get(next_url, timeout=30)
            else:
                resp = requests.get(url, params=params, timeout=30)

            if resp.status_code == 403:
                logger.error(
                    f"Polygon API key invalid or insufficient permissions for {symbol}. "
                    f"Check POLYGON_API_KEY in .env."
                )
                break

            if resp.status_code == 429:
                logger.warning("Polygon rate limit hit — waiting 60s...")
                time.sleep(60)
                continue

            resp.raise_for_status()
            payload = resp.json()

            status = payload.get("status", "")
            if status not in ("OK", "DELAYED"):
                logger.warning(
                    f"Polygon returned status '{status}' for {symbol}: "
                    f"{payload.get('message', '')}"
                )
                break

            results = payload.get("results") or []
            rows.extend(results)

            next_url = payload.get("next_url")
            if not next_url:
                break

            # Small pause between paginated requests to respect rate limits
            time.sleep(0.5)

    except Exception as e:
        logger.warning(f"Polygon intraday fetch failed for {symbol}: {e}")
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "vwap"])

    if not rows:
        logger.warning(f"No intraday data returned for {symbol} ({interval}) from Polygon")
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "vwap"])

    df = pd.DataFrame(rows)

    # Polygon field names: t=timestamp_ms, o=open, h=high, l=low, c=close, v=volume, vw=vwap
    df["datetime"] = _to_eastern(df["t"])
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})

    df = df.sort_values("datetime").reset_index(drop=True)
    df = df[["datetime", "open", "high", "low", "close", "volume"] +
            (["vw"] if "vw" in df.columns else [])]

    if market_hours_only:
        hhmm = df["datetime"].dt.strftime("%H:%M")
        df = df[(hhmm >= _MARKET_OPEN) & (hhmm < _MARKET_CLOSE)].reset_index(drop=True)

    # Use Polygon's pre-computed VWAP if available, else compute it
    if "vw" in df.columns:
        df = df.rename(columns={"vw": "vwap"})
    else:
        df["vwap"] = _vwap(df)

    # Only cache completed (past) sessions
    if use_cache and end_date < date.today().isoformat():
        df.to_csv(cache_file, index=False)
        logger.debug(f"Cached {len(df)} bars for {symbol} to {cache_file.name}")

    return df


def get_bulk_intraday_prices(
    symbols: list[str],
    start_date: str,
    end_date: str,
    interval: str = "5min",
    market_hours_only: bool = True,
    use_cache: bool = True,
    delay: float = 12.0,
) -> dict[str, pd.DataFrame]:
    """Fetch intraday prices for multiple symbols from Polygon.io.

    Returns a dict of {symbol: DataFrame}.  Symbols with no data are omitted.

    The default delay is 12s between API calls to stay within the free tier
    rate limit of 5 requests/minute.  Set delay=0.5 if you have a paid plan.
    """
    prices: dict[str, pd.DataFrame] = {}
    total = len(symbols)
    suffix = "" if market_hours_only else "_full"

    for i, symbol in enumerate(symbols, 1):
        cache_file = _CACHE_DIR / f"{symbol}_{interval}_{start_date}_{end_date}{suffix}.csv"
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

    logger.info(f"Fetched intraday data for {len(prices)}/{total} symbols ({interval}) [Polygon]")
    return prices
