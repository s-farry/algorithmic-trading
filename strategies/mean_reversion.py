import pandas as pd

import config
from data.stock_filter import StockProfile
from strategies.base import Strategy


class MeanReversionBollinger(Strategy):
    """Mean reversion strategy using Bollinger Bands.

    Buy when price closes below the lower band (oversold).
    Sell when price closes above the upper band (overbought).

    Works best on low-volatility, range-bound stocks that
    oscillate around a stable mean.
    """

    name = "mean_reversion"
    stock_profile = StockProfile(volatility="low", regime="mean_reverting")

    def __init__(self, window=None, num_std=None):
        self.window = window or config.BB_WINDOW
        self.num_std = num_std or config.BB_NUM_STD
        self.warmup_days = self.window + 1

    def generate_signals(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        df = df.copy()
        df["bb_mid"] = df["close"].rolling(window=self.window).mean()
        df["bb_std"] = df["close"].rolling(window=self.window).std()
        df["bb_upper"] = df["bb_mid"] + self.num_std * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - self.num_std * df["bb_std"]

        df["signal"] = 0
        df.loc[df["close"] < df["bb_lower"], "signal"] = 1   # Oversold -> buy
        df.loc[df["close"] > df["bb_upper"], "signal"] = -1  # Overbought -> sell

        return df
