from abc import ABC, abstractmethod

import pandas as pd

from data.stock_filter import StockProfile


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    name: str = "base"
    warmup_days: int = 0  # Trading days of prior data needed for indicators
    stock_profile: StockProfile = StockProfile()  # Default: no filtering

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Add indicator columns and a 'signal' column to the price DataFrame.

        Args:
            df: DataFrame with columns [date, open, high, low, close, volume],
                sorted by date ascending.
            symbol: Optional stock symbol (used by BlendedStrategy to delegate).

        Returns:
            The same DataFrame with additional indicator columns and a 'signal'
            column where: 1 = buy, -1 = sell, 0 = hold.
        """
        ...
