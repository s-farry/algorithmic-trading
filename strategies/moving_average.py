import pandas as pd

import config
from data.stock_filter import StockProfile
from strategies.base import Strategy


class MovingAverageCrossover(Strategy):
    """Golden Cross / Death Cross strategy using two SMAs.

    Buy when short MA crosses above long MA.
    Sell when short MA crosses below long MA.

    Works best on trending, higher-volatility stocks where
    price moves sustain direction after a crossover.
    """

    name = "ma_crossover"
    stock_profile = StockProfile(volatility="high", regime="trending")

    def __init__(self, short_window=None, long_window=None):
        self.short_window = short_window or config.MA_SHORT_WINDOW
        self.long_window = long_window or config.MA_LONG_WINDOW
        self.warmup_days = self.long_window + 1

    def generate_signals(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        df = df.copy()
        df["sma_short"] = df["close"].rolling(window=self.short_window).mean()
        df["sma_long"] = df["close"].rolling(window=self.long_window).mean()

        df["signal"] = 0

        # Crossover detection: compare current and previous relative positions
        short_above = (df["sma_short"] > df["sma_long"]).astype(bool)
        short_above_prev = short_above.shift(1).fillna(False).astype(bool)

        # Golden cross: short crosses above long
        df.loc[short_above & ~short_above_prev, "signal"] = 1
        # Death cross: short crosses below long
        df.loc[~short_above & short_above_prev, "signal"] = -1

        return df
