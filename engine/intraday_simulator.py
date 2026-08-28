"""Intraday simulation engine.

Runs a bar-by-bar backtest of intraday strategies against historical
minute/5-minute/hourly data.

Key differences from the daily Simulator:
- Iterates through bar timestamps in chronological order (not day by day).
- Skips the first `strategy.warmup_bars` bars per symbol before acting on
  signals (intraday indicators need a few bars to stabilise).
- Forces-close all open positions at the last bar of each trading session
  (simulates day-trading — no overnight carries).
- Tracks per-session P&L so you can see the daily breakdown.
"""

import logging
from datetime import date, datetime

import pandas as pd

import config
from engine.portfolio import Portfolio
from strategies.base import Strategy

logger = logging.getLogger(__name__)


class IntradaySimulator:
    """Bar-by-bar intraday simulation engine.

    Args:
        strategy:        An intraday Strategy instance.
        price_data:      Dict of {symbol: DataFrame} with columns
                         [datetime, open, high, low, close, volume, vwap].
                         Sorted ascending by datetime.
        start_date:      Simulation start date "YYYY-MM-DD".
        end_date:        Simulation end date "YYYY-MM-DD".
        initial_capital: Starting cash (default: config.DEFAULT_CAPITAL).
        commission:      Fixed commission per trade in $ (default: 0 for
                         intraday — most brokers charge per-share for DT).
        slippage_pct:    Simulated round-trip bid-ask spread as a fraction
                         (e.g. 0.002 = 0.2 %).  Half applied to each leg.
    """

    def __init__(
        self,
        strategy: Strategy,
        price_data: dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        initial_capital: float = None,
        commission: float = 0.0,
        slippage_pct: float = 0.0,
    ):
        self.strategy = strategy
        self.price_data = price_data
        self.start_dt = pd.Timestamp(start_date)
        self.end_dt = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59)
        self.slippage_pct = slippage_pct
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            commission=commission if commission is not None else 0.0,
        )
        self.transactions: list[dict] = []
        self.session_pnl: list[dict] = []     # per-session summary rows
        self.value_history: list[tuple] = []  # (datetime, total_value)
        self.bidirectional = getattr(strategy, "bidirectional", False)

        # Pre-compute signals for all symbols over the full date range
        self.signals: dict[str, pd.DataFrame] = {}
        for symbol, df in price_data.items():
            if len(df) < 2:
                continue
            sig_df = strategy.generate_signals(df, symbol=symbol)
            sig_df = sig_df[
                (sig_df["datetime"] >= self.start_dt)
                & (sig_df["datetime"] <= self.end_dt)
            ].copy()
            if not sig_df.empty:
                self.signals[symbol] = sig_df

        logger.info(
            f"IntradaySimulator: {strategy.name} | "
            f"{len(self.signals)} symbols | {start_date} → {end_date}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """Execute the simulation bar by bar, session by session."""
        if not self.signals:
            logger.warning("No signal data — nothing to simulate.")
            return

        # Collect all unique bar timestamps across all symbols, sorted
        all_timestamps: set[pd.Timestamp] = set()
        for sig_df in self.signals.values():
            all_timestamps.update(sig_df["datetime"].tolist())
        all_timestamps_sorted = sorted(all_timestamps)

        # Group timestamps by calendar date (trading session)
        sessions: dict[date, list[pd.Timestamp]] = {}
        for ts in all_timestamps_sorted:
            d = ts.date()
            sessions.setdefault(d, []).append(ts)

        # Per-symbol warmup counter: track how many bars have been seen
        bars_seen: dict[str, int] = {sym: 0 for sym in self.signals}
        warmup = getattr(self.strategy, "warmup_bars", 0)

        logger.info(
            f"Running {len(sessions)} sessions | "
            f"{len(all_timestamps_sorted)} bars | warmup={warmup} bars"
        )

        for session_date, session_bars in sorted(sessions.items()):
            self._run_session(session_date, session_bars, bars_seen, warmup)

        logger.info(
            f"Simulation complete | {len(self.transactions)} transactions | "
            f"final portfolio value: ${self.portfolio.get_total_value({}):,.2f}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_session(
        self,
        session_date: date,
        session_bars: list[pd.Timestamp],
        bars_seen: dict[str, int],
        warmup: int,
    ):
        """Process all bars for a single trading session."""
        n_bars = len(session_bars)
        half_spread = self.slippage_pct / 2

        # Snapshot cash + position values at session open (for P&L tracking)
        session_start_value = self._portfolio_value_at_bar(session_bars[0])

        for bar_idx, ts in enumerate(session_bars):
            is_last_bar = bar_idx == n_bars - 1

            # Collect signals for all symbols at this timestamp
            close_candidates = []   # exit longs (unidirectional) or force-flat (bidirectional)
            cover_candidates = []   # cover shorts (bidirectional only)
            buy_candidates = []
            short_candidates = []   # short entries (bidirectional only)
            current_prices: dict[str, float] = {}

            for symbol, sig_df in self.signals.items():
                bar_rows = sig_df[sig_df["datetime"] == ts]
                if bar_rows.empty:
                    continue

                row = bar_rows.iloc[0]
                price = float(row["close"])
                current_prices[symbol] = price

                bars_seen[symbol] += 1
                if bars_seen[symbol] <= warmup:
                    continue

                signal = int(row["signal"])
                long_pos = symbol in self.portfolio.positions
                short_pos = symbol in self.portfolio.short_positions

                if is_last_bar:
                    # Force flat at EOD — close longs and cover shorts
                    if long_pos:
                        close_candidates.append((symbol, price, "SELL (EOD)"))
                    if short_pos:
                        cover_candidates.append((symbol, price, "COVER (EOD)"))

                elif self.bidirectional:
                    # signal 1 = want long, -1 = want short, 0 = want flat
                    if signal == 1:
                        if short_pos:
                            cover_candidates.append((symbol, price, "COVER"))
                        elif not long_pos:
                            buy_candidates.append((symbol, price))
                    elif signal == -1:
                        if long_pos:
                            close_candidates.append((symbol, price, "SELL"))
                        elif not short_pos:
                            short_candidates.append((symbol, price))
                    elif signal == 0:
                        # Force flat
                        if long_pos:
                            close_candidates.append((symbol, price, "SELL"))
                        if short_pos:
                            cover_candidates.append((symbol, price, "COVER"))

                else:
                    # Unidirectional: 1 = buy, -1 = sell existing long, 0 = hold
                    if signal == 1 and not long_pos:
                        buy_candidates.append((symbol, price))
                    elif signal == -1 and long_pos:
                        close_candidates.append((symbol, price, "SELL"))

            # --- Close longs first ---
            for symbol, price, action in close_candidates:
                pos = self.portfolio.positions.get(symbol)
                if pos is None:
                    continue
                fill_price = price * (1 - half_spread)
                qty = pos.quantity
                cost_basis = qty * pos.avg_cost
                result = self.portfolio.sell(symbol, qty, fill_price, session_date)
                if not result:
                    continue
                proceeds, pnl = result
                self.transactions.append({
                    "datetime": ts, "date": session_date, "symbol": symbol,
                    "action": action, "quantity": qty, "price": fill_price,
                    "commission": self.portfolio.commission, "cost_basis": cost_basis,
                    "proceeds": proceeds, "pnl": pnl,
                    "pnl_pct": pnl / cost_basis * 100 if cost_basis else 0.0,
                })

            # --- Cover shorts ---
            for symbol, price, action in cover_candidates:
                pos = self.portfolio.short_positions.get(symbol)
                if pos is None:
                    continue
                fill_price = price * (1 + half_spread)  # cover at ask (unfavourable)
                qty = pos.quantity
                entry_value = qty * pos.avg_cost
                result = self.portfolio.cover(symbol, qty, fill_price, session_date)
                if not result:
                    continue
                cost, pnl = result
                self.transactions.append({
                    "datetime": ts, "date": session_date, "symbol": symbol,
                    "action": action, "quantity": qty, "price": fill_price,
                    "commission": self.portfolio.commission, "cost_basis": entry_value,
                    "proceeds": entry_value + pnl, "pnl": pnl,
                    "pnl_pct": pnl / entry_value * 100 if entry_value else 0.0,
                })

            # --- Enter longs ---
            if buy_candidates and not is_last_bar:
                available = self.portfolio.cash
                total_value = self.portfolio.get_total_value(current_prices)
                max_per_position = min(
                    available / len(buy_candidates),
                    total_value * config.MAX_POSITION_PCT,
                )
                remaining_cash = available
                for symbol, price in buy_candidates:
                    if symbol in self.portfolio.positions:
                        continue
                    fill_price = price * (1 + half_spread)
                    affordable = min(remaining_cash, max_per_position)
                    qty = int(affordable / fill_price)
                    if qty < 1:
                        continue
                    cost = self.portfolio.buy(symbol, qty, fill_price, session_date)
                    if cost is not None:
                        remaining_cash -= cost
                        self.transactions.append({
                            "datetime": ts, "date": session_date, "symbol": symbol,
                            "action": "BUY", "quantity": qty, "price": fill_price,
                            "commission": self.portfolio.commission, "cost_basis": cost,
                            "proceeds": 0.0, "pnl": 0.0, "pnl_pct": 0.0,
                        })

            # --- Enter shorts ---
            if short_candidates and not is_last_bar:
                available = self.portfolio.cash
                total_value = self.portfolio.get_total_value(current_prices)
                max_per_position = min(
                    available / len(short_candidates),
                    total_value * config.MAX_POSITION_PCT,
                )
                remaining_cash = available
                for symbol, price in short_candidates:
                    if symbol in self.portfolio.short_positions:
                        continue
                    fill_price = price * (1 - half_spread)  # short at bid
                    affordable = min(remaining_cash, max_per_position)
                    qty = int(affordable / fill_price)
                    if qty < 1:
                        continue
                    proceeds = self.portfolio.short(symbol, qty, fill_price, session_date)
                    if proceeds is not None:
                        remaining_cash = self.portfolio.cash  # cash increased after short
                        entry_value = qty * fill_price
                        self.transactions.append({
                            "datetime": ts, "date": session_date, "symbol": symbol,
                            "action": "SHORT", "quantity": qty, "price": fill_price,
                            "commission": self.portfolio.commission, "cost_basis": entry_value,
                            "proceeds": 0.0, "pnl": 0.0, "pnl_pct": 0.0,
                        })

            # Snapshot portfolio value at each bar (use last bar for session)
            if is_last_bar:
                session_end_value = self.portfolio.get_total_value(current_prices)
                self.value_history.append((session_date, session_end_value))

        # --- Record per-session P&L ---
        session_end_value = self.portfolio.get_total_value({})
        session_pnl = session_end_value - session_start_value
        self.session_pnl.append({
            "date": session_date,
            "start_value": round(session_start_value, 2),
            "end_value": round(session_end_value, 2),
            "pnl": round(session_pnl, 2),
            "pnl_pct": round(session_pnl / session_start_value * 100, 4)
                       if session_start_value else 0.0,
        })

    def _portfolio_value_at_bar(self, ts: pd.Timestamp) -> float:
        """Portfolio value at the start of a session (no current prices yet)."""
        # Use avg_cost for positions since we don't have current prices yet
        return self.portfolio.get_total_value({})
