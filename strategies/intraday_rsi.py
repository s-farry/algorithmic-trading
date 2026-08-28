import pandas as pd

from data.stock_filter import StockProfile
from strategies.base import Strategy


class IntradayRSI(Strategy):
    """Intraday RSI mean-reversion strategy.

    Uses a short RSI period (default 7) and tighter thresholds than the
    daily RSI+MACD strategy, since intraday swings are smaller in magnitude.

    Buy when RSI drops below `rsi_oversold` (default 30).
    Sell when RSI rises above `rsi_overbought` (default 70).

    Optionally adds a VWAP directional filter: only take buy signals when
    price is below VWAP (mean-reversion context) and sell signals when
    above VWAP.  Enable with `vwap_filter=True`.

    Designed for 5-min bars.  For 1-min bars reduce rsi_period to 5.
    """

    name = "intraday_rsi"
    stock_profile = StockProfile(volatility="any", regime="any", min_avg_volume=500_000)

    def __init__(
        self,
        rsi_period: int = 7,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        vwap_filter: bool = True,
    ):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.vwap_filter = vwap_filter
        self.warmup_bars = rsi_period + 2
        self.warmup_days = 0

    def _rsi(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def generate_signals(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        df = df.copy()

        df["rsi"] = self._rsi(df["close"])

        buy_cond = df["rsi"] < self.rsi_oversold
        sell_cond = df["rsi"] > self.rsi_overbought

        # Optional VWAP directional filter
        if self.vwap_filter and "vwap" in df.columns:
            # Only buy when price is below VWAP (reversion is likely)
            buy_cond = buy_cond & (df["close"] < df["vwap"])
            # Only sell when price is above VWAP (reversion is likely)
            sell_cond = sell_cond & (df["close"] > df["vwap"])

        df["signal"] = 0
        df.loc[buy_cond, "signal"] = 1
        df.loc[sell_cond & ~buy_cond, "signal"] = -1

        return df
