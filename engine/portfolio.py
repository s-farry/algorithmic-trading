import logging
from dataclasses import dataclass
from datetime import date

import config

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_cost: float
    entry_date: date


class Portfolio:
    """Tracks cash, long positions, short positions, and portfolio value over time."""

    def __init__(self, initial_capital=None, commission=None):
        self.initial_capital = initial_capital or config.DEFAULT_CAPITAL
        self.cash = self.initial_capital
        self.commission = commission if commission is not None else config.COMMISSION_PER_TRADE
        self.positions: dict[str, Position] = {}        # long positions
        self.short_positions: dict[str, Position] = {}  # short positions
        self.value_history: list[tuple[date, float]] = []

    def buy(self, symbol, quantity, price, trade_date):
        """Execute a buy order. Returns total cost or None if insufficient cash."""
        cost = quantity * price + self.commission
        if cost > self.cash:
            return None
        self.cash -= cost
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_qty = pos.quantity + quantity
            pos.avg_cost = (pos.avg_cost * pos.quantity + price * quantity) / total_qty
            pos.quantity = total_qty
        else:
            self.positions[symbol] = Position(
                symbol=symbol, quantity=quantity, avg_cost=price, entry_date=trade_date,
            )
        logger.debug(f"BUY {quantity} {symbol} @ {price:.2f} (cost: {cost:.2f})")
        return cost

    def sell(self, symbol, quantity, price, trade_date):
        """Close a long position. Returns (proceeds, pnl) or None if no position."""
        if symbol not in self.positions:
            return None
        pos = self.positions[symbol]
        quantity = min(quantity, pos.quantity)
        proceeds = quantity * price - self.commission
        pnl = proceeds - quantity * pos.avg_cost
        self.cash += proceeds
        pos.quantity -= quantity
        if pos.quantity == 0:
            del self.positions[symbol]
        logger.debug(f"SELL {quantity} {symbol} @ {price:.2f} (proceeds: {proceeds:.2f}, pnl: {pnl:.2f})")
        return proceeds, pnl

    def short(self, symbol, quantity, price, trade_date):
        """Enter a short position. Returns proceeds received or None if already short.

        Shorting: we borrow and sell shares, receiving cash immediately.
        The liability (cost to cover) is tracked separately via short_positions.
        """
        if symbol in self.short_positions:
            return None  # already short this symbol
        proceeds = quantity * price - self.commission
        self.cash += proceeds
        self.short_positions[symbol] = Position(
            symbol=symbol, quantity=quantity, avg_cost=price, entry_date=trade_date,
        )
        logger.debug(f"SHORT {quantity} {symbol} @ {price:.2f} (proceeds: {proceeds:.2f})")
        return proceeds

    def cover(self, symbol, quantity, price, trade_date):
        """Close a short position. Returns (cost, pnl) or None if no short position.

        Covering: we buy shares back. Profit when exit_price < entry_price.
        """
        if symbol not in self.short_positions:
            return None
        pos = self.short_positions[symbol]
        quantity = min(quantity, pos.quantity)
        cost = quantity * price + self.commission
        self.cash -= cost
        # Short P&L: (entry_price - exit_price) × qty minus the cover commission
        pnl = (pos.avg_cost - price) * quantity - self.commission
        pos.quantity -= quantity
        if pos.quantity == 0:
            del self.short_positions[symbol]
        logger.debug(f"COVER {quantity} {symbol} @ {price:.2f} (cost: {cost:.2f}, pnl: {pnl:.2f})")
        return cost, pnl

    def get_total_value(self, current_prices: dict[str, float]) -> float:
        """Calculate total portfolio value (cash + longs - short cover cost)."""
        long_value = sum(
            pos.quantity * current_prices.get(pos.symbol, pos.avg_cost)
            for pos in self.positions.values()
        )
        # Short liability: what it would cost to cover all shorts right now.
        # Cash already includes the proceeds received when shorts were opened,
        # so subtracting the current cover cost gives the correct net value.
        short_cover_cost = sum(
            pos.quantity * current_prices.get(pos.symbol, pos.avg_cost)
            for pos in self.short_positions.values()
        )
        return self.cash + long_value - short_cover_cost

    def record_value(self, trade_date, current_prices: dict[str, float]):
        """Snapshot portfolio value for the given date."""
        self.value_history.append((trade_date, self.get_total_value(current_prices)))

    def get_position_count(self) -> int:
        return len(self.positions) + len(self.short_positions)
