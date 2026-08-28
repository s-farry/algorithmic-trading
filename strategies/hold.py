import pandas as pd

from strategies.base import Strategy


class BuyAndHold(Strategy):
    """Buy-and-hold benchmark strategy.

    Emits a single buy signal on the first trading day and holds for the
    entire period. The simulator will close all positions on the last day,
    giving an equal-weight buy-and-hold return across whatever stocks are
    provided.
    """

    name = "hold"
    warmup_days = 0

    def generate_signals(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        # Mark every row as a buy signal — the simulator will only act on the
        # first one since it skips symbols already held. This ensures the buy
        # fires on the first day within the simulation window regardless of
        # how much warmup data precedes it.
        df["signal"] = 1
        return df
