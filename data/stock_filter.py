import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass
class StockProfile:
    """Describes what kind of stocks a strategy works best with."""
    # Volatility: "high", "low", or "any"
    volatility: str = "any"
    # Trend strength: "trending", "mean_reverting", or "any"
    regime: str = "any"
    # Minimum average daily volume
    min_avg_volume: float = 0


def compute_stock_metrics(price_data: dict[str, pd.DataFrame], start_date: str) -> pd.DataFrame:
    """Compute filtering metrics for each stock using warmup-period data.

    Uses only data BEFORE start_date (the warmup window) to avoid look-ahead bias.

    Returns a DataFrame indexed by symbol with columns:
        - annualised_vol: annualised volatility of daily returns
        - trend_strength: absolute value of returns autocorrelation at lag 1
                          (positive autocorr = trending, negative = mean-reverting)
        - autocorr: raw lag-1 autocorrelation of daily returns
        - avg_volume: average daily volume
    """
    start_ts = pd.Timestamp(start_date)
    records = []

    for symbol, df in price_data.items():
        # Use only warmup data (before simulation start) for metric computation
        warmup = df[df["date"] < start_ts]
        if len(warmup) < 60:  # Need at least ~3 months of data
            continue

        returns = warmup["close"].pct_change().dropna()
        if len(returns) < 30 or returns.std() == 0:
            continue

        annualised_vol = returns.std() * np.sqrt(252)
        autocorr = returns.autocorr(lag=1)
        if np.isnan(autocorr):
            autocorr = 0.0

        avg_volume = warmup["volume"].mean()

        records.append({
            "symbol": symbol,
            "annualised_vol": annualised_vol,
            "trend_strength": abs(autocorr),
            "autocorr": autocorr,
            "avg_volume": avg_volume,
        })

    if not records:
        logger.warning("No stocks had enough warmup data to compute metrics")
        return pd.DataFrame(columns=["annualised_vol", "trend_strength", "autocorr", "avg_volume"])

    metrics = pd.DataFrame(records).set_index("symbol")
    logger.info(f"Computed metrics for {len(metrics)} stocks")
    return metrics


def filter_stocks(
    price_data: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    profile: StockProfile,
    top_n: int = None,
) -> dict[str, pd.DataFrame]:
    """Filter price_data to stocks matching the given profile.

    Args:
        price_data: Full price data dict {symbol: DataFrame}
        metrics: Stock metrics DataFrame from compute_stock_metrics
        profile: StockProfile describing desired stock characteristics
        top_n: Max number of stocks to return (default from config)

    Returns:
        Filtered price_data dict containing only matching symbols.
    """
    top_n = top_n or config.FILTER_TOP_N
    candidates = metrics.copy()

    # Volume filter
    if profile.min_avg_volume > 0:
        candidates = candidates[candidates["avg_volume"] >= profile.min_avg_volume]

    # Volatility filter + ranking
    if profile.volatility == "high":
        # Take the most volatile stocks
        candidates = candidates.sort_values("annualised_vol", ascending=False)
    elif profile.volatility == "low":
        # Take the least volatile stocks
        candidates = candidates.sort_values("annualised_vol", ascending=True)

    # Regime filter + ranking
    if profile.regime == "trending":
        # Positive autocorrelation = trending behavior
        # Sort by autocorrelation descending (most trending first)
        candidates = candidates[candidates["autocorr"] > -0.05]
        candidates = candidates.sort_values("autocorr", ascending=False)
    elif profile.regime == "mean_reverting":
        # Negative autocorrelation = mean-reverting behavior
        # Sort by autocorrelation ascending (most mean-reverting first)
        candidates = candidates[candidates["autocorr"] < 0.05]
        candidates = candidates.sort_values("autocorr", ascending=True)

    # Take top N
    selected = candidates.head(top_n)
    selected_symbols = set(selected.index)

    filtered = {s: df for s, df in price_data.items() if s in selected_symbols}

    logger.info(
        f"Filtered {len(price_data)} -> {len(filtered)} stocks "
        f"(vol={profile.volatility}, regime={profile.regime})"
    )

    return filtered
