import logging
from datetime import date

import pandas as pd

import config
from engine.portfolio import Portfolio
from strategies.base import Strategy

logger = logging.getLogger(__name__)


class Simulator:
    """Core simulation engine that runs a strategy over historical data.

    In simulation mode (default), trades are executed against the in-memory
    Portfolio. In live mode, trades are routed to a real broker.
    """

    def __init__(
        self,
        strategy: Strategy,
        price_data: dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        initial_capital: float = None,
        broker=None,
        commission: float = None,
        min_hold_days: int = 0,
        slippage_pct: float = 0.0,
    ):
        self.strategy = strategy
        self.price_data = price_data
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.broker = broker
        self.live_mode = broker is not None
        self.min_hold_days = min_hold_days
        self.slippage_pct = slippage_pct
        self.portfolio = Portfolio(initial_capital=initial_capital, commission=commission)
        self.transactions: list[dict] = []

        # In live mode, sync portfolio state from broker
        if self.live_mode:
            self._sync_from_broker()

        # Pre-compute signals for all symbols
        self.signals: dict[str, pd.DataFrame] = {}
        for symbol, df in self.price_data.items():
            if len(df) < 2:
                continue
            sig_df = self.strategy.generate_signals(df, symbol=symbol)
            sig_df = sig_df[
                (sig_df["date"] >= self.start_date)
                & (sig_df["date"] <= self.end_date)
            ]
            self.signals[symbol] = sig_df

    def _sync_from_broker(self):
        """Sync local portfolio state from broker's actual positions."""
        account = self.broker.get_account_info()
        self.portfolio.cash = account.get("cash", self.portfolio.cash)
        broker_positions = self.broker.get_positions()
        self.portfolio.positions = broker_positions
        logger.info(
            f"Synced from {self.broker.name}: "
            f"cash=${self.portfolio.cash:,.2f}, "
            f"{len(broker_positions)} positions"
        )

    def run(self):
        """Execute the simulation day by day."""
        mode = f"LIVE ({self.broker.name})" if self.live_mode else "simulation"

        # Collect all trading dates across all symbols
        all_dates = set()
        for df in self.signals.values():
            all_dates.update(df["date"].dt.date.tolist())
        all_dates = sorted(all_dates)

        logger.info(
            f"Running {self.strategy.name} {mode}: "
            f"{len(all_dates)} trading days, {len(self.signals)} symbols"
        )

        for trade_date in all_dates:
            self._process_day(trade_date)

        # Close all remaining positions on the last day (sim mode only)
        if all_dates and not self.live_mode:
            self._close_all_positions(all_dates[-1])

        logger.info(
            f"{mode.capitalize()} complete: {len(self.transactions)} transactions, "
            f"final value: ${self.portfolio.cash:,.2f}"
        )

    def _process_day(self, trade_date: date):
        """Process signals and execute trades for a single day."""
        buy_signals = []
        sell_signals = []
        current_prices = {}

        for symbol, sig_df in self.signals.items():
            day_data = sig_df[sig_df["date"].dt.date == trade_date]
            if day_data.empty:
                continue

            row = day_data.iloc[-1]
            current_prices[symbol] = row["close"]
            signal = row["signal"]

            if signal == 1:
                buy_signals.append((symbol, row["close"]))
            elif signal == -1 and symbol in self.portfolio.positions:
                # Enforce minimum holding period
                if self.min_hold_days > 0:
                    pos = self.portfolio.positions[symbol]
                    held_days = (trade_date - pos.entry_date).days
                    if held_days < self.min_hold_days:
                        continue
                sell_signals.append((symbol, row["close"]))

        # Apply slippage: buys fill at ask (higher), sells at bid (lower)
        half_spread = self.slippage_pct / 2

        # Execute sells first to free up cash
        for symbol, price in sell_signals:
            pos = self.portfolio.positions.get(symbol)
            if pos is None:
                continue

            sell_qty = pos.quantity
            sell_cost_basis = sell_qty * pos.avg_cost
            fill_price = price * (1 - half_spread)  # Sell at bid

            if self.live_mode:
                broker_result = self.broker.sell(symbol, sell_qty, price)
                if broker_result is None:
                    logger.warning(f"Live SELL {symbol} failed, skipping")
                    continue
                fill_price = broker_result.get("price", price)
                self.portfolio.sell(symbol, sell_qty, fill_price, trade_date)
                proceeds = sell_qty * fill_price
                pnl = proceeds - sell_cost_basis
            else:
                result = self.portfolio.sell(symbol, sell_qty, fill_price, trade_date)
                if not result:
                    continue
                proceeds, pnl = result

            self.transactions.append({
                "date": trade_date,
                "symbol": symbol,
                "action": "SELL",
                "quantity": sell_qty,
                "price": fill_price,
                "commission": self.portfolio.commission,
                "cost_basis": sell_cost_basis,
                "proceeds": proceeds,
                "pnl": pnl,
                "pnl_pct": pnl / sell_cost_basis * 100 if sell_cost_basis else 0,
            })

        # Execute buys with position sizing
        if buy_signals:
            available = self.portfolio.cash
            max_per_position = min(
                available / len(buy_signals),
                self.portfolio.get_total_value(current_prices) * config.MAX_POSITION_PCT,
            )

            for symbol, price in buy_signals:
                if symbol in self.portfolio.positions:
                    continue  # Already holding

                fill_price = price * (1 + half_spread)  # Buy at ask
                quantity = int(max_per_position / fill_price)
                if quantity < 1:
                    continue

                if self.live_mode:
                    broker_result = self.broker.buy(symbol, quantity, price)
                    if broker_result is None:
                        logger.warning(f"Live BUY {symbol} failed, skipping")
                        continue
                    fill_price = broker_result.get("price", price)
                    cost = self.portfolio.buy(symbol, quantity, fill_price, trade_date)
                else:
                    cost = self.portfolio.buy(symbol, quantity, fill_price, trade_date)

                if cost is not None:
                    self.transactions.append({
                        "date": trade_date,
                        "symbol": symbol,
                        "action": "BUY",
                        "quantity": quantity,
                        "price": fill_price,
                        "commission": self.portfolio.commission,
                        "cost_basis": cost,
                        "proceeds": 0.0,
                        "pnl": 0.0,
                        "pnl_pct": 0.0,
                    })

        # Record daily portfolio value
        self.portfolio.record_value(trade_date, current_prices)

    def _close_all_positions(self, trade_date: date):
        """Liquidate all open positions at current prices."""
        symbols_to_close = list(self.portfolio.positions.keys())

        for symbol in symbols_to_close:
            pos = self.portfolio.positions[symbol]
            sig_df = self.signals.get(symbol)
            if sig_df is None or sig_df.empty:
                continue

            last_row = sig_df.iloc[-1]
            price = last_row["close"] * (1 - self.slippage_pct / 2)

            close_qty = pos.quantity
            close_cost_basis = close_qty * pos.avg_cost
            result = self.portfolio.sell(symbol, close_qty, price, trade_date)
            if result:
                proceeds, pnl = result
                self.transactions.append({
                    "date": trade_date,
                    "symbol": symbol,
                    "action": "SELL (CLOSE)",
                    "quantity": close_qty,
                    "price": price,
                    "commission": self.portfolio.commission,
                    "cost_basis": close_cost_basis,
                    "proceeds": proceeds,
                    "pnl": pnl,
                    "pnl_pct": pnl / close_cost_basis * 100 if close_cost_basis else 0,
                })
