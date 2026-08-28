from abc import ABC, abstractmethod

from engine.portfolio import Position


class Broker(ABC):
    """Abstract base class for broker integrations.

    Provides a common interface so the simulator can switch between
    simulation mode and live trading with any supported broker.
    """

    name: str = "base"

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the broker."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close the broker connection."""
        ...

    @abstractmethod
    def get_account_info(self) -> dict:
        """Return account details.

        Returns:
            dict with keys: 'cash', 'portfolio_value', 'currency'
        """
        ...

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Return current open positions keyed by symbol."""
        ...

    @abstractmethod
    def buy(self, symbol: str, quantity: int, price: float) -> dict | None:
        """Place a buy order.

        Args:
            symbol: Ticker symbol.
            quantity: Number of shares.
            price: Limit price (or reference price for market orders).

        Returns:
            Order details dict on success, None on failure.
        """
        ...

    @abstractmethod
    def sell(self, symbol: str, quantity: int, price: float) -> dict | None:
        """Place a sell order.

        Args:
            symbol: Ticker symbol.
            quantity: Number of shares.
            price: Limit price (or reference price for market orders).

        Returns:
            Order details dict on success, None on failure.
        """
        ...

    @abstractmethod
    def get_current_price(self, symbol: str) -> float | None:
        """Get the latest price for a symbol."""
        ...
