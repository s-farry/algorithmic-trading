import pandas as pd

from data.stock_filter import StockProfile
from strategies.base import Strategy


class VolumeSurge(Strategy):
    """Volume-primary intraday strategy for commodities.

    Core idea: unusually large volume with clear directional conviction
    (close near the high or low of the bar) represents institutional
    order flow.  That kind of flow tends to have follow-through.

    Entry:
        A "surge bar" is one where volume exceeds `surge_multiple` ×
        the same-time-of-day rolling average.  Direction is determined
        by the close ratio: (close - low) / (high - low).

            close_ratio >= close_ratio_threshold  →  bullish surge  →  go long
            close_ratio <= 1 - close_ratio_threshold  →  bearish surge  →  go short

        Using same-time-of-day averages matters because 9:30am naturally
        has 3-5× more volume than 2pm — a flat rolling window would make
        every open look like a surge.

    Session direction lock:
        Once the first valid surge fires, only that direction is allowed
        for the rest of the session.  This prevents flip-flopping from
        multiple conflicting surges in the same day.

    Exit:
        - Close falls back through the VWAP against the position, BUT
          only after holding for at least `min_hold_bars` bars.
        - An opposite-direction surge flips the position (ignores lock
          if the counter-signal is a strong surge while holding <
          min_hold_bars, this is suppressed).
        - EIA report window (Wed 10:00–11:00 ET) — forced flat.
        - Late session cutoff (15:00 ET) — forced flat.
        - EOD forced close handled by the simulator.

    Signal convention (bidirectional = True):
        1  = want to be long  (enter if flat, cover if short)
        -1 = want to be short (enter if flat, close if long)
        0  = want to be flat  (close any open position)
    """

    name = "volume_surge"
    bidirectional = True
    stock_profile = StockProfile(volatility="high", regime="any", min_avg_volume=500_000)

    def __init__(
        self,
        surge_multiple: float = 3.5,
        close_ratio_threshold: float = 0.70,
        sessions_lookback: int = 10,
        min_bar_range_pct: float = 0.001,
        min_hold_bars: int = 6,
        late_cutoff_hour: int = 15,
        eia_block_start: str = "10:00",
        eia_block_end: str = "11:00",
    ):
        self.surge_multiple = surge_multiple
        self.close_ratio_threshold = close_ratio_threshold
        self.sessions_lookback = sessions_lookback
        self.min_bar_range_pct = min_bar_range_pct
        self.min_hold_bars = min_hold_bars       # bars to hold before VWAP exit can trigger
        self.late_cutoff_hour = late_cutoff_hour

        h0, m0 = map(int, eia_block_start.split(":"))
        h1, m1 = map(int, eia_block_end.split(":"))
        self._eia_start_mins = h0 * 60 + m0
        self._eia_end_mins = h1 * 60 + m1

        bars_per_session = 78
        self.warmup_bars = sessions_lookback * bars_per_session
        self.warmup_days = sessions_lookback

    def generate_signals(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        df = df.copy()

        dt_col = "datetime" if "datetime" in df.columns else "date"
        dt = pd.to_datetime(df[dt_col])
        dates = dt.dt.date.astype(str)

        # --- Same-time-of-day rolling average volume ---
        df["_time"] = dt.dt.strftime("%H:%M")
        tod_avg = (
            df.groupby("_time")["volume"]
            .transform(
                lambda s: s.shift(1)
                           .rolling(self.sessions_lookback, min_periods=max(3, self.sessions_lookback // 3))
                           .mean()
            )
        )
        simple_avg = df["volume"].shift(1).rolling(20, min_periods=5).mean()
        avg_vol = tod_avg.fillna(simple_avg)

        # --- Surge detection ---
        is_surge = df["volume"] > self.surge_multiple * avg_vol

        # --- Directional conviction ---
        bar_range = df["high"] - df["low"]
        has_range = bar_range > self.min_bar_range_pct * df["close"]
        close_ratio = (df["close"] - df["low"]) / bar_range.replace(0, float("nan"))

        bullish_surge = is_surge & has_range & (close_ratio >= self.close_ratio_threshold)
        bearish_surge = is_surge & has_range & (close_ratio <= 1.0 - self.close_ratio_threshold)

        # --- Time filters ---
        bar_mins = dt.dt.hour * 60 + dt.dt.minute
        in_eia = (
            (dt.dt.weekday == 2) &
            (bar_mins >= self._eia_start_mins) &
            (bar_mins < self._eia_end_mins)
        )
        is_late = dt.dt.hour >= self.late_cutoff_hour
        blocked = (in_eia | is_late).values

        # --- VWAP for exit reference ---
        vwap = df["vwap"].values if "vwap" in df.columns else df["close"].values

        close_vals = df["close"].values
        bullish = bullish_surge.values
        bearish = bearish_surge.values
        date_vals = dates.values

        # --- Stateful signal computation ---
        signal = [0] * len(df)
        current = 0          # current desired position
        prev_date = None
        session_direction = 0   # direction locked for the session (0 = no lock yet)
        bars_in_position = 0    # how many bars we've held the current position

        for i in range(len(df)):
            today = date_vals[i]

            # Session boundary — reset everything
            if today != prev_date:
                current = 0
                session_direction = 0
                bars_in_position = 0
                prev_date = today

            # Track consecutive bars in position
            if current != 0:
                bars_in_position += 1
            else:
                bars_in_position = 0

            # Blocked bar — force flat
            if blocked[i]:
                current = 0
                bars_in_position = 0
                signal[i] = 0
                continue

            # New surge: only enter if it matches session direction lock (or no lock yet)
            if bullish[i] and (session_direction >= 0):
                if current != 1:
                    current = 1
                    bars_in_position = 0
                    session_direction = 1  # lock long for the day
            elif bearish[i] and (session_direction <= 0):
                if current != -1:
                    current = -1
                    bars_in_position = 0
                    session_direction = -1  # lock short for the day
            else:
                # Hold, then check VWAP exit only after min_hold_bars
                if bars_in_position >= self.min_hold_bars:
                    if current == 1 and close_vals[i] < vwap[i]:
                        current = 0
                    elif current == -1 and close_vals[i] > vwap[i]:
                        current = 0

            signal[i] = current

        df["signal"] = signal
        df = df.drop(columns=["_time"])
        return df
