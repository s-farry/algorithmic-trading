import pandas as pd

from data.stock_filter import StockProfile
from strategies.base import Strategy


class CommodityORB(Strategy):
    """Opening Range Breakout adapted for commodity ETFs (USO, GLD, UNG, etc.).

    Long-only. Enters once per session when a valid breakout fires; holds
    until price fails back below the opening high or EOD forced close.

    Filters:
        1. EIA inventory report block (Wed 10:00–11:00 ET)
        2. Volume confirmation — breakout bar >= volume_multiplier × avg opening vol
        3. Late-session cutoff — no new positions after 15:00 ET
        4. Breakout threshold — close must clear opening_high by >= breakout_pct (0.5%)
    """

    name = "commodity_orb"
    stock_profile = StockProfile(volatility="high", regime="any", min_avg_volume=500_000)

    def __init__(
        self,
        opening_bars: int = 8,
        volume_multiplier: float = 1.5,
        late_cutoff_hour: int = 15,
        eia_block_start: str = "10:00",
        eia_block_end: str = "11:00",
        breakout_pct: float = 0.005,
    ):
        self.opening_bars = opening_bars
        self.volume_multiplier = volume_multiplier
        self.late_cutoff_hour = late_cutoff_hour
        self.breakout_pct = breakout_pct

        h0, m0 = map(int, eia_block_start.split(":"))
        h1, m1 = map(int, eia_block_end.split(":"))
        self._eia_start_mins = h0 * 60 + m0
        self._eia_end_mins = h1 * 60 + m1

        self.warmup_bars = opening_bars + 1
        self.warmup_days = 0

    def generate_signals(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        df = df.copy()

        dt_col = "datetime" if "datetime" in df.columns else "date"
        dt = pd.to_datetime(df[dt_col])
        dates = dt.dt.date.astype(str)

        df["_bar_in_day"] = df.groupby(dates).cumcount()
        df["_date_str"] = dates

        # --- Opening range ---
        is_opening = df["_bar_in_day"] < self.opening_bars
        opening_stats = (
            df[is_opening]
            .groupby(dates[is_opening])
            .agg(
                opening_high=("high", "max"),
                opening_low=("low", "min"),
                avg_opening_vol=("volume", "mean"),
            )
        )
        df = df.merge(opening_stats, left_on="_date_str", right_index=True, how="left")

        # --- Time filters ---
        bar_mins = dt.dt.hour * 60 + dt.dt.minute
        in_eia_window = (
            (dt.dt.weekday == 2) &
            (bar_mins >= self._eia_start_mins) &
            (bar_mins < self._eia_end_mins)
        )
        is_late = dt.dt.hour >= self.late_cutoff_hour

        # --- Volume confirmation ---
        avg_vol = df["avg_opening_vol"].fillna(0)
        vol_ok = (avg_vol == 0) | (df["volume"] >= self.volume_multiplier * avg_vol)

        # --- Signal assembly ---
        post_open = df["_bar_in_day"] >= self.opening_bars
        tradeable = post_open & ~in_eia_window & ~is_late

        df["signal"] = 0
        df.loc[in_eia_window | is_late, "signal"] = -1

        # Buy: close clears opening_high by breakout_pct with volume
        breakout_level = df["opening_high"] * (1 + self.breakout_pct)
        df.loc[tradeable & (df["close"] > breakout_level) & vol_ok, "signal"] = 1

        # Sell: close falls back below opening high
        df.loc[tradeable & (df["close"] < df["opening_high"]), "signal"] = -1

        df = df.drop(columns=["_bar_in_day", "_date_str"])
        return df
