import logging
import time
from pathlib import Path

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

CACHE_DIR = config.RESULTS_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _fetch_constituents(endpoint, label, use_cache=True):
    """Generic constituent list fetcher."""
    cache_file = CACHE_DIR / f"{endpoint}_constituents.csv"

    if use_cache and cache_file.exists():
        logger.info(f"Loading {label} constituents from cache")
        return pd.read_csv(cache_file)

    logger.info(f"Fetching {label} constituents from FMP")
    url = f"{config.FMP_BASE_URL}/{endpoint}?apikey={config.FMP_API_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    df.to_csv(cache_file, index=False)
    return df


def get_sp500_constituents(use_cache=True):
    """Fetch the current S&P 500 constituent list from FMP."""
    return _fetch_constituents("sp500_constituent", "S&P 500", use_cache)


def get_sp500_changes(use_cache=True):
    """Fetch historical S&P 500 additions and removals from FMP.

    Each row represents a change: a stock was added (symbol) and optionally
    another was removed (removedTicker) on a given date.

    Returns a DataFrame with columns:
        date, symbol, addedSecurity, removedTicker, removedSecurity, reason
    """
    cache_file = CACHE_DIR / "sp500_historical_changes.csv"

    if use_cache and cache_file.exists():
        logger.info("Loading S&P 500 historical changes from cache")
        df = pd.read_csv(cache_file, parse_dates=["date"])
        return df

    logger.info("Fetching S&P 500 historical changes from FMP")
    url = (
        f"{config.FMP_BASE_URL}/historical/sp500_constituent"
        f"?apikey={config.FMP_API_KEY}"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    logger.info(f"Fetched {len(df)} S&P 500 constituent changes")
    return df


def get_sp500_constituents_at(as_of_date, use_cache=True):
    """Reconstruct the S&P 500 membership at a specific historical date.

    Works backwards from today's list: for each change after as_of_date,
    reverse the addition (remove the added stock) and reverse the removal
    (add back the removed stock).

    Args:
        as_of_date: Date string (YYYY-MM-DD) or datetime-like.

    Returns:
        Sorted list of ticker symbols that were in the S&P 500 on that date.
    """
    as_of = pd.Timestamp(as_of_date)

    # Start with today's constituents
    current = get_sp500_constituents(use_cache=use_cache)
    members = set(current["symbol"].tolist())

    # Get all changes and find those after our target date
    changes = get_sp500_changes(use_cache=use_cache)
    future_changes = changes[changes["date"] > as_of].sort_values("date", ascending=False)

    # Reverse each change that happened after as_of_date
    for _, row in future_changes.iterrows():
        added = row["symbol"]
        removed = row.get("removedTicker", "")

        # Undo the addition
        if added and added in members:
            members.discard(added)

        # Undo the removal (add back the removed stock)
        if removed and isinstance(removed, str) and removed.strip():
            members.add(removed.strip())

    result = sorted(members)
    logger.info(f"S&P 500 at {as_of.date()}: {len(result)} constituents")
    return result


def get_sp500_all_historical(start_date, end_date=None, use_cache=True):
    """Get every stock that was in the S&P 500 at any point during a period.

    This is the survivorship-bias-free universe: it includes stocks that
    were later removed due to delisting, acquisition, or market cap decline.

    Args:
        start_date: Period start (YYYY-MM-DD).
        end_date: Period end (YYYY-MM-DD). Defaults to today.

    Returns:
        Sorted list of all ticker symbols that were members during the period.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

    # Start with everyone in the index at the start of the period
    members_at_start = set(get_sp500_constituents_at(start_date, use_cache=use_cache))

    # Add every stock that was added during the period
    changes = get_sp500_changes(use_cache=use_cache)
    period_changes = changes[(changes["date"] >= start) & (changes["date"] <= end)]

    all_members = set(members_at_start)
    for _, row in period_changes.iterrows():
        added = row["symbol"]
        removed = row.get("removedTicker", "")
        if added:
            all_members.add(added)
        if removed and isinstance(removed, str) and removed.strip():
            all_members.add(removed.strip())

    result = sorted(all_members)
    logger.info(
        f"S&P 500 survivorship-free universe ({start.date()} to {end.date()}): "
        f"{len(result)} total symbols "
        f"({len(result) - len(members_at_start)} removed/added during period)"
    )
    return result


def get_nasdaq_constituents(use_cache=True):
    """Fetch the current NASDAQ-100 constituent list from FMP."""
    return _fetch_constituents("nasdaq_constituent", "NASDAQ-100", use_cache)


def get_dowjones_constituents(use_cache=True):
    """Fetch the current Dow Jones constituent list from FMP."""
    return _fetch_constituents("dowjones_constituent", "Dow Jones", use_cache)


def get_midcap_stocks(use_cache=True):
    """Fetch US mid-cap stocks ($2B-$10B market cap) via FMP screener."""
    return _get_screener_stocks(
        "midcap", "US Mid-Cap",
        cap_more_than=2_000_000_000, cap_less_than=10_000_000_000,
        limit=400, use_cache=use_cache,
    )


def _get_screener_stocks(cache_name, label, cap_more_than, cap_less_than=None,
                          limit=2500, use_cache=True):
    """Fetch US stocks via FMP screener filtered by market cap range."""
    cache_file = CACHE_DIR / f"{cache_name}_constituents.csv"

    if use_cache and cache_file.exists():
        logger.info(f"Loading {label} stocks from cache")
        return pd.read_csv(cache_file)

    logger.info(f"Fetching {label} stocks from FMP screener")
    url = (
        f"{config.FMP_BASE_URL}/stock-screener"
        f"?marketCapMoreThan={cap_more_than}"
        f"&country=US"
        f"&exchange=NYSE,NASDAQ"
        f"&isEtf=false"
        f"&isFund=false"
        f"&isActivelyTrading=true"
        f"&limit={limit}"
        f"&apikey={config.FMP_API_KEY}"
    )
    if cap_less_than is not None:
        url += f"&marketCapLowerThan={cap_less_than}"

    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    if "companyName" in df.columns:
        df = df.rename(columns={"companyName": "name"})
    df.to_csv(cache_file, index=False)
    logger.info(f"Fetched {len(df)} {label} stocks")
    return df


def get_russell1000_stocks(use_cache=True):
    """Approximate Russell 1000 — largest ~1000 US stocks by market cap (>$2B)."""
    return _get_screener_stocks(
        "russell1000", "Russell 1000",
        cap_more_than=2_000_000_000, limit=1000, use_cache=use_cache,
    )


def get_russell2000_stocks(use_cache=True):
    """Approximate Russell 2000 — US small-cap stocks ($300M–$2B market cap)."""
    return _get_screener_stocks(
        "russell2000", "Russell 2000",
        cap_more_than=300_000_000, cap_less_than=2_000_000_000,
        limit=2000, use_cache=use_cache,
    )


UNIVERSES = {
    "sp500": ("S&P 500", get_sp500_constituents),
    "nasdaq": ("NASDAQ-100", get_nasdaq_constituents),
    "dowjones": ("Dow Jones 30", get_dowjones_constituents),
    "midcap": ("US Mid-Cap", get_midcap_stocks),
    "russell1000": ("Russell 1000", get_russell1000_stocks),
    "russell2000": ("Russell 2000", get_russell2000_stocks),
}


def get_universe(name: str, use_cache=True, start_date=None, end_date=None) -> list[str]:
    """Return a list of ticker symbols for a given universe.

    Args:
        name: One of 'sp500', 'nasdaq', 'dowjones', 'midcap', 'all',
              or 'sp500_historical' (survivorship-bias-free).
        start_date: Required for 'sp500_historical'. Period start (YYYY-MM-DD).
        end_date: Optional for 'sp500_historical'. Period end (YYYY-MM-DD).

    Returns:
        Deduplicated list of ticker symbols.
    """
    if name == "sp500_historical":
        if not start_date:
            raise ValueError("start_date is required for 'sp500_historical' universe")
        return get_sp500_all_historical(start_date, end_date, use_cache=use_cache)

    if name == "all":
        all_symbols = set()
        for key in UNIVERSES:
            _, fetcher = UNIVERSES[key]
            df = fetcher(use_cache=use_cache)
            all_symbols.update(df["symbol"].tolist())
        symbols = sorted(all_symbols)
        logger.info(f"Universe 'all': {len(symbols)} unique symbols")
        return symbols

    if name not in UNIVERSES:
        raise ValueError(f"Unknown universe '{name}'. Choose from: {list(UNIVERSES.keys()) + ['all', 'sp500_historical']}")

    label, fetcher = UNIVERSES[name]
    df = fetcher(use_cache=use_cache)
    symbols = df["symbol"].tolist()
    logger.info(f"Universe '{name}' ({label}): {len(symbols)} symbols")
    return symbols


def get_historical_prices(symbol, start_date, end_date, use_cache=True):
    """Fetch daily OHLCV data for a single symbol from FMP.

    Returns a DataFrame sorted by date ascending with columns:
    date, open, high, low, close, volume
    """
    cache_file = CACHE_DIR / f"{symbol}_{start_date}_{end_date}.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["date"])
        return df

    logger.info(f"Fetching historical prices for {symbol}")
    url = (
        f"{config.FMP_BASE_URL}/historical-price-full/{symbol}"
        f"?from={start_date}&to={end_date}&apikey={config.FMP_API_KEY}"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    if "historical" not in data:
        logger.warning(f"No historical data returned for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(data["historical"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[["date", "open", "high", "low", "close", "volume"]]

    df.to_csv(cache_file, index=False)
    return df


def get_bulk_prices(symbols, start_date, end_date, use_cache=True, delay=0.15):
    """Fetch historical prices for a list of symbols.

    Returns a dict of {symbol: DataFrame}.
    """
    prices = {}
    total = len(symbols)

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{total}] Fetching {symbol}")
        df = get_historical_prices(symbol, start_date, end_date, use_cache=use_cache)
        if not df.empty:
            prices[symbol] = df

        if not use_cache or not (CACHE_DIR / f"{symbol}_{start_date}_{end_date}.csv").exists():
            time.sleep(delay)

    logger.info(f"Fetched price data for {len(prices)}/{total} symbols")
    return prices
