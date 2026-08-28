import pandas as pd

import config
from data.stock_filter import StockProfile
from strategies.base import Strategy


class RSIMACDMomentum(Strategy):
    """Momentum strategy combining RSI and MACD.

    Buy when RSI < oversold threshold AND MACD line crosses above signal line.
    Sell when RSI > overbought threshold AND MACD line crosses below signal line.

    Works best on volatile, liquid stocks with clear momentum swings.
    """

    name = "momentum"
    stock_profile = StockProfile(volatility="high", regime="any", min_avg_volume=1_000_000)

    def __init__(
        self,
        rsi_period=None,
        rsi_oversold=None,
        rsi_overbought=None,
        macd_fast=None,
        macd_slow=None,
        macd_signal=None,
    ):
        self.rsi_period = rsi_period or config.RSI_PERIOD
        self.rsi_oversold = rsi_oversold or config.RSI_OVERSOLD
        self.rsi_overbought = rsi_overbought or config.RSI_OVERBOUGHT
        self.macd_fast = macd_fast or config.MACD_FAST
        self.macd_slow = macd_slow or config.MACD_SLOW
        self.macd_signal = macd_signal or config.MACD_SIGNAL
        self.warmup_days = self.macd_slow + self.macd_signal + 1

    def _compute_rsi(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        df = df.copy()

        # RSI
        df["rsi"] = self._compute_rsi(df["close"])

        # MACD
        ema_fast = df["close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.macd_slow, adjust=False).mean()
        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd_line"].ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_hist"] = df["macd_line"] - df["macd_signal"]

        # MACD crossover detection
        macd_above = (df["macd_line"] > df["macd_signal"]).astype(bool)
        macd_above_prev = macd_above.shift(1).fillna(False).astype(bool)
        macd_cross_up = macd_above & ~macd_above_prev
        macd_cross_down = ~macd_above & macd_above_prev

        df["signal"] = 0
        # Buy: RSI oversold AND MACD crosses up
        df.loc[(df["rsi"] < self.rsi_oversold) & macd_cross_up, "signal"] = 1
        # Sell: RSI overbought AND MACD crosses down
        df.loc[(df["rsi"] > self.rsi_overbought) & macd_cross_down, "signal"] = -1

        return df
