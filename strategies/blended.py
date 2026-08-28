import logging

import numpy as np
import pandas as pd

from data.stock_filter import StockProfile
from strategies.base import Strategy
from strategies.moving_average import MovingAverageCrossover
from strategies.mean_reversion import MeanReversionBollinger
from strategies.momentum import RSIMACDMomentum

logger = logging.getLogger(__name__)


class BlendedStrategy(Strategy):
    """Assigns each stock to its best-fitting strategy based on metrics.

    Classification rules (using warmup-period data):
      - High volatility + positive autocorrelation -> MA Crossover (trending)
      - High volatility + high volume             -> RSI+MACD Momentum
      - Low volatility + negative autocorrelation  -> Bollinger Mean Reversion

    Stocks are scored and assigned to whichever strategy they fit best.
    """

    name = "blended"
    stock_profile = StockProfile()  # No filtering - uses all stocks

    def __init__(self):
        self.ma = MovingAverageCrossover()
        self.mr = MeanReversionBollinger()
        self.mom = RSIMACDMomentum()
        self.warmup_days = max(self.ma.warmup_days, self.mr.warmup_days, self.mom.warmup_days)

        # Populated by set_assignments() before simulation
        self._assignments: dict[str, Strategy] = {}

    def set_assignments(self, price_data: dict[str, pd.DataFrame], start_date: str):
        """Classify each stock and assign it to the best sub-strategy.

        Must be called before generate_signals().
        """
        start_ts = pd.Timestamp(start_date)
        counts = {"ma_crossover": 0, "mean_reversion": 0, "momentum": 0}

        for symbol, df in price_data.items():
            warmup = df[df["date"] < start_ts]
            if len(warmup) < 60:
                self._assignments[symbol] = self.mr  # Default fallback
                counts["mean_reversion"] += 1
                continue

            returns = warmup["close"].pct_change().dropna()
            if len(returns) < 30 or returns.std() == 0:
                self._assignments[symbol] = self.mr
                counts["mean_reversion"] += 1
                continue

            vol = returns.std() * np.sqrt(252)
            autocorr = returns.autocorr(lag=1)
            if np.isnan(autocorr):
                autocorr = 0.0
            avg_volume = warmup["volume"].mean()

            median_vol = 0.30  # ~30% annualised vol as threshold

            if vol > median_vol and autocorr > 0.02:
                # High vol + trending -> MA Crossover
                self._assignments[symbol] = self.ma
                counts["ma_crossover"] += 1
            elif vol > median_vol and avg_volume > 1_000_000:
                # High vol + liquid -> Momentum
                self._assignments[symbol] = self.mom
                counts["momentum"] += 1
            else:
                # Low vol or mean-reverting -> Bollinger Bands
                self._assignments[symbol] = self.mr
                counts["mean_reversion"] += 1

        logger.info(
            f"Blended assignments: "
            f"MA Crossover={counts['ma_crossover']}, "
            f"Mean Reversion={counts['mean_reversion']}, "
            f"Momentum={counts['momentum']}"
        )

    def generate_signals(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Delegate to the assigned sub-strategy for this symbol."""
        if symbol and symbol in self._assignments:
            return self._assignments[symbol].generate_signals(df)

        # Fallback: use mean reversion (safest default)
        return self.mr.generate_signals(df)
