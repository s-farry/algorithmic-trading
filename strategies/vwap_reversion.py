import pandas as pd

from data.stock_filter import StockProfile
from strategies.base import Strategy


class VWAPReversion(Strategy):
    """Intraday VWAP mean-reversion strategy.

    Buy when price dips more than `vwap_threshold`% below VWAP and RSI
    is oversold — signalling temporary weakness rather than a sustained
    downtrend.

    Sell when price recovers back above VWAP, or RSI becomes overbought.

    Works best on liquid large-cap stocks that oscillate around VWAP
    intraday (most S&P 500 stocks on normal trading days).

    Requires the 'vwap' column produced by data.intraday.get_intraday_prices().
    Designed for 5-min bars; increase vwap_threshold slightly for 1-min bars.
    """

    name = "vwap_reversion"
    stock_profile = StockProfile(volatility="any", regime="any", min_avg_volume=1_000_000)

    def __init__(
        self,
        vwap_threshold: float = 0.003,
        rsi_period: int = 7,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
    ):
        # vwap_threshold: fractional deviation, e.g. 0.003 = 0.3% below VWAP
        self.vwap_threshold = vwap_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.warmup_bars = rsi_period + 2   # bars needed before RSI stabilises
        self.warmup_days = 0                # no prior-day daily data needed

    def _rsi(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        # Use EWM (Wilder smoothing) — more responsive on short intraday windows
        avg_gain = gain.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def generate_signals(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        df = df.copy()

        if "vwap" not in df.columns:
            raise ValueError(
                "VWAPReversion requires a 'vwap' column. "
                "Fetch data with data.intraday.get_intraday_prices()."
            )

        df["rsi"] = self._rsi(df["close"])
        df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"]

        df["signal"] = 0

        # Buy: price > threshold below VWAP AND RSI oversold
        buy = (df["vwap_dev"] < -self.vwap_threshold) & (df["rsi"] < self.rsi_oversold)
        df.loc[buy, "signal"] = 1

        # Sell: price back above VWAP (target hit) OR RSI overbought (momentum gone)
        sell = (df["vwap_dev"] > 0) | (df["rsi"] > self.rsi_overbought)
        df.loc[sell & ~buy, "signal"] = -1

        return df
