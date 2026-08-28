import pandas as pd

from data.stock_filter import StockProfile
from strategies.base import Strategy


class OpeningRangeBreakout(Strategy):
    """Opening Range Breakout (ORB) intraday strategy.

    The opening range is the high/low established during the first
    `opening_bars` bars of the session (e.g. 6 × 5-min bars = 30 minutes).

    Buy when a bar closes above the opening high — the breakout signals
    directional momentum for the rest of the day.

    Sell when price closes back below the opening high (breakout fails)
    or below the opening low (hard stop).

    Works best on high-volatility stocks on days with a clear directional
    catalyst (earnings, news, macro events).  Combine with a stock screener
    that pre-selects the top movers on gap-up days.

    Designed for 5-min bars with opening_bars=6 (first 30 minutes).
    For 15-min bars use opening_bars=2.
    """

    name = "opening_range_breakout"
    stock_profile = StockProfile(volatility="high", regime="any", min_avg_volume=500_000)

    def __init__(self, opening_bars: int = 6):
        # opening_bars: number of bars that define the opening range
        # 6 × 5-min = 30-minute ORB (standard institutional reference)
        self.opening_bars = opening_bars
        self.warmup_bars = opening_bars + 1  # need the range to be set first
        self.warmup_days = 0

    def generate_signals(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        df = df.copy()

        dt_col = "datetime" if "datetime" in df.columns else "date"
        dates = pd.to_datetime(df[dt_col]).dt.date.astype(str)

        # Bar index within each trading day (0-indexed)
        df["_bar_in_day"] = df.groupby(dates).cumcount()

        # Opening range: high/low of the first `opening_bars` bars each day
        is_opening = df["_bar_in_day"] < self.opening_bars
        opening_stats = (
            df[is_opening]
            .groupby(dates[is_opening])
            .agg(opening_high=("high", "max"), opening_low=("low", "min"))
        )

        df["_date_str"] = dates
        df = df.merge(opening_stats, left_on="_date_str", right_index=True, how="left")

        df["signal"] = 0

        # Only act after the opening range is established
        post_open = df["_bar_in_day"] >= self.opening_bars

        # Buy: close breaks above the opening high
        df.loc[post_open & (df["close"] > df["opening_high"]), "signal"] = 1

        # Sell: close falls back below the opening high (breakout failed)
        #       or breaks below the opening low (hard stop / short trigger)
        df.loc[
            post_open & (df["close"] < df["opening_high"]),
            "signal",
        ] = -1

        # Clean up helper columns
        df = df.drop(columns=["_bar_in_day", "_date_str"])

        return df
