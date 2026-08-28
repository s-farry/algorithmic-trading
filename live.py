#!/usr/bin/env python3
"""Live Trading Runner

Runs once, generates today's signals, and executes trades via a broker.
Designed to be called daily via cron or manually.

Usage:
    python live.py --strategy momentum --broker etoro
    python live.py --strategy momentum --broker etoro --demo
    python live.py --strategy blended --broker etoro --symbols AAPL MSFT GOOGL
    python live.py --strategy momentum --broker etoro --dry-run

Cron example (9:45am EST, Mon-Fri):
    45 14 * * 1-5  cd /home/sfarry/algorithmic-trading && python live.py --strategy momentum --broker etoro
"""

import argparse
import csv
import logging
import sys
from datetime import datetime, timedelta, date

import pandas as pd

import config
from brokers import BROKERS
from data.sp500 import get_universe, get_bulk_prices
from data.stock_filter import compute_stock_metrics, filter_stocks
from engine.portfolio import Position
from strategies import STRATEGIES

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

LIVE_LOG_DIR = config.RESULTS_DIR / "live"
LIVE_LOG_DIR.mkdir(exist_ok=True)


def append_transactions(transactions: list[dict], strategy_name: str):
    """Append today's transactions to a persistent CSV log."""
    if not transactions:
        return

    log_file = LIVE_LOG_DIR / f"{strategy_name}_live_log.csv"
    file_exists = log_file.exists()

    fieldnames = [
        "date", "symbol", "action", "quantity", "price",
        "commission", "cost_basis", "proceeds", "pnl", "pnl_pct",
    ]

    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(transactions)

    logger.info(f"Appended {len(transactions)} transactions to {log_file}")


def get_known_holdings(strategy_name: str) -> set[str]:
    """Read the local transaction log to determine what we currently hold.

    This acts as a safety net when the broker's get_positions() fails to
    return positions (e.g. due to instrument ID mapping issues).
    """
    log_file = LIVE_LOG_DIR / f"{strategy_name}_live_log.csv"
    if not log_file.exists():
        return set()

    try:
        df = pd.read_csv(log_file)
    except Exception:
        return set()

    if df.empty or "symbol" not in df.columns or "action" not in df.columns:
        return set()

    holdings = {}  # symbol -> net quantity
    for _, row in df.iterrows():
        sym = row["symbol"]
        qty = row.get("quantity", 0)
        if row["action"] == "BUY":
            holdings[sym] = holdings.get(sym, 0) + qty
        elif row["action"].startswith("SELL"):
            holdings[sym] = holdings.get(sym, 0) - qty

    return {sym for sym, qty in holdings.items() if qty > 0}


def get_todays_signal(strategy, df, symbol=None) -> int:
    """Generate signals and return only the most recent one."""
    sig_df = strategy.generate_signals(df, symbol=symbol)
    if sig_df.empty:
        return 0
    return int(sig_df.iloc[-1]["signal"])


def main():
    parser = argparse.ArgumentParser(description="Live Trading Runner")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=[k for k in STRATEGIES.keys() if k != "hold"],
        help="Strategy to run",
    )
    parser.add_argument(
        "--broker",
        type=str,
        required=True,
        choices=["etoro", "ibkr"],
        help="Broker to trade with",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Use broker's demo/paper trading mode (eToro only)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="*",
        default=None,
        help="Specific symbols to trade (default: filtered S&P 500)",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="sp500",
        choices=["sp500", "nasdaq", "dowjones", "midcap", "russell1000", "russell2000", "all"],
        help="Stock universe to trade (default: sp500)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        default=False,
        help="Disable stock filtering",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Generate signals and show what would be traded, but don't execute",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Cap the portfolio value used for position sizing (default: use broker's full balance)",
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=None,
        help=f"Max portfolio %% per position (default: {config.MAX_POSITION_PCT * 100:.0f}%%)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=config.COMMISSION_PER_TRADE,
        help=f"Commission per trade in $ (default: {config.COMMISSION_PER_TRADE:.2f})",
    )
    parser.add_argument(
        "--min-hold-days",
        type=int,
        default=0,
        help="Minimum days to hold a position before selling (default: 0)",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Sell any position that has fallen more than this %% below its average cost "
             "(e.g. --stop-loss 8 exits when down 8%%). Disabled by default.",
    )
    parser.add_argument(
        "--max-hold-days",
        type=int,
        default=None,
        help="Force-sell any position held longer than this many calendar days, "
             "regardless of strategy signal (e.g. --max-hold-days 30). Disabled by default.",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        default=False,
        help="Skip confirmation prompt (for cron/automation)",
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to predictions CSV (required for 'prediction' strategy). "
             "Multiple files can be passed; first file takes precedence on overlaps.",
    )
    parser.add_argument(
        "--buy-threshold",
        type=float,
        default=None,
        help=f"Buy when predicted uplift >= threshold (default: {config.PREDICTION_BUY_THRESHOLD})",
    )
    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=None,
        help=f"Sell when predicted uplift <= threshold (default: {config.PREDICTION_SELL_THRESHOLD})",
    )

    args = parser.parse_args()

    # Validate prediction strategy args
    if args.strategy == "prediction" and not args.predictions_file:
        parser.error("--predictions-file is required when using the 'prediction' strategy")

    today = date.today()
    logger.info(f"Live trading run: {today}")

    # Initialise strategy
    strategy_cls = STRATEGIES[args.strategy]
    strategy_kwargs = {}
    if args.predictions_file:
        strategy_kwargs["predictions_files"] = args.predictions_file
    if args.buy_threshold is not None:
        strategy_kwargs["buy_threshold"] = args.buy_threshold
    if args.sell_threshold is not None:
        strategy_kwargs["sell_threshold"] = args.sell_threshold
    strategy = strategy_cls(**strategy_kwargs)

    max_pos_pct = args.max_position_pct / 100 if args.max_position_pct else config.MAX_POSITION_PCT

    # Connect to broker
    broker_cls = BROKERS[args.broker]
    if args.broker == "etoro":
        broker = broker_cls(demo=args.demo)
    else:
        broker = broker_cls()

    mode_label = "DRY RUN" if args.dry_run else ("DEMO" if args.demo else "LIVE")
    capital_label = f" | Capital: ${args.capital:,.0f}" if args.capital else ""
    print(f"\n{'=' * 60}")
    print(f"  {mode_label} TRADING — {args.broker.upper()}")
    print(f"  Strategy: {args.strategy} | Date: {today}{capital_label}")
    print(f"{'=' * 60}")

    if not args.dry_run and not args.yes:
        confirm = input("  Continue? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

    broker.connect()

    # Pre-populate broker instrument cache from known holdings so
    # get_positions() can reverse-lookup instrument IDs to symbols
    local_holdings = get_known_holdings(args.strategy)
    if local_holdings and hasattr(broker, "warm_cache"):
        broker.warm_cache(local_holdings)

    # Get account state
    account = broker.get_account_info()
    cash = account.get("cash", 0.0)
    portfolio_value = account.get("portfolio_value", 0.0)
    current_positions = broker.get_positions()

    # Merge with local transaction log as safety net against broker API gaps
    broker_symbols = set(current_positions.keys())
    missing_from_broker = local_holdings - broker_symbols
    if missing_from_broker:
        logger.warning(
            f"Broker returned {len(broker_symbols)} positions but local log "
            f"shows {len(local_holdings)} holdings. "
            f"Missing from broker: {sorted(missing_from_broker)}"
        )
        # Add missing symbols so we don't re-buy them
        for sym in missing_from_broker:
            current_positions[sym] = Position(
                symbol=sym, quantity=0, avg_cost=0.0, entry_date=None
            )

    # Apply capital cap if specified
    if args.capital:
        positions_value = sum(
            pos.quantity * pos.avg_cost for pos in current_positions.values()
        )
        portfolio_value = min(portfolio_value, args.capital)
        cash = min(cash, max(0, args.capital - positions_value))
        logger.info(
            f"Capital cap: ${args.capital:,.2f} "
            f"(positions: ${positions_value:,.2f}, available: ${cash:,.2f})"
        )

    logger.info(
        f"Account: cash=${cash:,.2f}, portfolio=${portfolio_value:,.2f}, "
        f"{len(current_positions)} positions"
    )

    # Determine symbols
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = get_universe(args.universe)

    # Fetch historical data with warmup
    # Need enough data for: strategy indicators + 60 trading days for stock metrics
    min_history_days = max(int(strategy.warmup_days * 1.5), 120) + 30
    fetch_start = (datetime.now() - timedelta(days=min_history_days)).strftime("%Y-%m-%d")
    fetch_end = today.strftime("%Y-%m-%d")

    logger.info(f"Fetching price data: {fetch_start} to {fetch_end} for {len(symbols)} symbols")
    price_data = get_bulk_prices(symbols, fetch_start, fetch_end, use_cache=False)

    if not price_data:
        logger.error("No price data fetched. Exiting.")
        broker.disconnect()
        sys.exit(1)

    # Apply stock filtering
    if not args.no_filter and not args.symbols:
        # Save price data for held positions BEFORE filtering — we must always be
        # able to check sell signals for stocks we hold, even if they no longer
        # pass the buy filter (volume dropped, regime changed, etc.)
        held_price_data = {
            sym: price_data[sym] for sym in current_positions if sym in price_data
        }
        # Use today as the cutoff — metrics are computed from all data before it
        metrics_cutoff = fetch_end
        metrics = compute_stock_metrics(price_data, metrics_cutoff)
        if not metrics.empty:
            price_data = filter_stocks(price_data, metrics, strategy.stock_profile)
            logger.info(f"Filtered to {len(price_data)} stocks")
            # Restore held positions that were filtered out
            restored = [sym for sym in held_price_data if sym not in price_data]
            if restored:
                logger.info(
                    f"Restoring {len(restored)} held positions to price_data for sell checks: {sorted(restored)}"
                )
                for sym in restored:
                    price_data[sym] = held_price_data[sym]

    # Set up blended strategy assignments
    if hasattr(strategy, "set_assignments"):
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        strategy.set_assignments(price_data, cutoff)

    # Generate today's signals
    buy_signals = []
    sell_signals = []

    for symbol, df in price_data.items():
        if len(df) < 2:
            continue

        signal = get_todays_signal(strategy, df, symbol=symbol)
        latest_price = df.iloc[-1]["close"]

        holding = symbol in current_positions or symbol in local_holdings
        if signal == 1 and not holding:
            buy_signals.append((symbol, latest_price))
        elif holding:
            # Check stop-loss before strategy signal (overrides min-hold-days)
            stop_loss_triggered = False
            if args.stop_loss is not None:
                pos = current_positions.get(symbol)
                if pos and pos.avg_cost > 0:
                    drawdown_pct = (latest_price - pos.avg_cost) / pos.avg_cost * 100
                    if drawdown_pct <= -args.stop_loss:
                        logger.warning(
                            f"STOP-LOSS {symbol}: down {drawdown_pct:.1f}% "
                            f"(cost ${pos.avg_cost:.2f}, now ${latest_price:.2f})"
                        )
                        sell_signals.append((symbol, latest_price))
                        stop_loss_triggered = True

            # Max hold-days: force-exit stale positions
            if not stop_loss_triggered and args.max_hold_days is not None:
                pos = current_positions.get(symbol)
                held_days = (today - pos.entry_date).days if (pos and pos.entry_date) else 0
                if held_days >= args.max_hold_days:
                    logger.info(
                        f"MAX-HOLD {symbol}: held {held_days}d >= {args.max_hold_days}d limit"
                    )
                    sell_signals.append((symbol, latest_price))
                    continue

            if not stop_loss_triggered and signal == -1:
                # Enforce minimum holding period for strategy-driven sells
                if args.min_hold_days > 0:
                    pos = current_positions.get(symbol)
                    held_days = (today - pos.entry_date).days if (pos and pos.entry_date) else 999
                    if held_days < args.min_hold_days:
                        logger.info(f"Skipping SELL {symbol}: held {held_days}d < {args.min_hold_days}d minimum")
                        continue
                sell_signals.append((symbol, latest_price))

    logger.info(f"Signals: {len(buy_signals)} buys, {len(sell_signals)} sells")

    # Diagnostic: for held positions with no sell signal, show indicator values
    held_no_signal = [
        sym for sym in current_positions
        if sym not in [s for s, _ in sell_signals]
        and sym in price_data
    ]
    if held_no_signal:
        logger.info("Held positions (no sell signal today):")
        for symbol in sorted(held_no_signal):
            df = price_data[symbol]
            if len(df) < 2:
                continue
            sig_df = strategy.generate_signals(df, symbol=symbol)
            last = sig_df.iloc[-1]
            pos = current_positions.get(symbol)
            held_days = (today - pos.entry_date).days if (pos and pos.entry_date) else "?"
            # Show whatever indicator columns the strategy adds
            indicator_cols = [c for c in ["rsi", "macd_line", "macd_signal", "vwap_dev", "signal"]
                              if c in last.index]
            indicator_str = "  ".join(f"{c}={last[c]:.2f}" for c in indicator_cols)
            logger.info(f"  {symbol:<8} held={held_days}d  {indicator_str}")

    # Fetch live broker prices for signal symbols (more accurate for execution)
    if hasattr(broker, "get_current_price"):
        signal_symbols = [s for s, _ in buy_signals] + [s for s, _ in sell_signals]
        if signal_symbols:
            logger.info(f"Fetching live prices from {args.broker} for {len(signal_symbols)} symbols...")
        live_prices = {}
        for sym in signal_symbols:
            live_price = broker.get_current_price(sym)
            if live_price is not None:
                live_prices[sym] = live_price

        if live_prices:
            logger.info(f"Got live prices for {len(live_prices)}/{len(signal_symbols)} symbols")
            buy_signals = [(sym, live_prices.get(sym, fmp_price)) for sym, fmp_price in buy_signals]
            sell_signals = [(sym, live_prices.get(sym, fmp_price)) for sym, fmp_price in sell_signals]

    # Print signal summary
    if buy_signals:
        print(f"\n  BUY signals:")
        for sym, price in sorted(buy_signals):
            print(f"    {sym:<8} @ ${price:,.2f}")

    if sell_signals:
        print(f"\n  SELL signals:")
        for sym, price in sorted(sell_signals):
            pos = current_positions.get(sym)
            qty = pos.quantity if pos else "?"
            print(f"    {sym:<8} @ ${price:,.2f}  (holding {qty} shares)")

    if not buy_signals and not sell_signals:
        print("\n  No trades today.")
        broker.disconnect()
        return

    transactions = []

    # Execute sells first
    for symbol, price in sell_signals:
        pos = current_positions.get(symbol)
        if pos is None:
            continue

        sell_qty = pos.quantity
        cost_basis = sell_qty * pos.avg_cost

        if args.dry_run:
            logger.info(f"[DRY RUN] Would SELL {sell_qty} {symbol} @ ~${price:,.2f}")
            continue

        result = broker.sell(symbol, sell_qty, price)
        if result:
            actual_price = result.get("price", price)
            proceeds = sell_qty * actual_price - args.commission
            pnl = proceeds - cost_basis
            transactions.append({
                "date": today.isoformat(),
                "symbol": symbol,
                "action": "SELL",
                "quantity": sell_qty,
                "price": actual_price,
                "commission": args.commission,
                "cost_basis": round(cost_basis, 2),
                "proceeds": round(proceeds, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / cost_basis * 100, 2) if cost_basis else 0,
            })
            print(f"  SOLD {sell_qty} {symbol} @ ${actual_price:,.2f} (P&L: ${pnl:,.2f})")
        else:
            logger.error(f"Failed to sell {symbol}")

    # Refresh cash after sells
    if transactions and not args.dry_run:
        account = broker.get_account_info()
        cash = account.get("cash", cash)

        # Re-apply capital cap to refreshed cash
        if args.capital:
            current_positions = broker.get_positions()
            positions_value = sum(
                pos.quantity * pos.avg_cost for pos in current_positions.values()
            )
            portfolio_value = min(account.get("portfolio_value", portfolio_value), args.capital)
            cash = min(cash, max(0, args.capital - positions_value))

    # Execute buys with position sizing
    if buy_signals:
        total_value = portfolio_value if portfolio_value > 0 else cash
        max_per_position = total_value * max_pos_pct

        remaining_cash = cash
        logger.info(
            f"Buy sizing: {len(buy_signals)} signals, "
            f"max/position=${max_per_position:,.2f}, available=${remaining_cash:,.2f}"
        )

        for symbol, price in buy_signals:
            # Use max_per_position capped by whatever cash remains
            affordable = min(remaining_cash, max_per_position)
            quantity = int(affordable / price)
            if quantity < 1:
                logger.warning(
                    f"Skipping {symbol}: need ${price:,.2f}/share, "
                    f"${remaining_cash:,.2f} remaining (max/pos ${max_per_position:,.2f})"
                )
                continue

            if args.dry_run:
                cost = quantity * price
                logger.info(f"[DRY RUN] Would BUY {quantity} {symbol} @ ~${price:,.2f} (${cost:,.2f})")
                remaining_cash -= cost
                continue

            result = broker.buy(symbol, quantity, price)
            if result:
                actual_price = result.get("price", price)
                cost = quantity * actual_price + args.commission
                remaining_cash -= cost
                transactions.append({
                    "date": today.isoformat(),
                    "symbol": symbol,
                    "action": "BUY",
                    "quantity": quantity,
                    "price": actual_price,
                    "commission": args.commission,
                    "cost_basis": round(cost, 2),
                    "proceeds": 0.0,
                    "pnl": 0.0,
                    "pnl_pct": 0.0,
                })
                print(f"  BOUGHT {quantity} {symbol} @ ${actual_price:,.2f} (${cost:,.2f}, ${remaining_cash:,.2f} left)")
            else:
                logger.error(f"Failed to buy {symbol}")

    # Log transactions
    if transactions:
        append_transactions(transactions, args.strategy)

    # Print summary
    print(f"\n{'-' * 60}")
    if args.dry_run:
        print(f"  DRY RUN complete — no trades executed")
    else:
        buys = [t for t in transactions if t["action"] == "BUY"]
        sells = [t for t in transactions if t["action"] == "SELL"]
        total_pnl = sum(t["pnl"] for t in sells)
        print(f"  Executed: {len(buys)} buys, {len(sells)} sells")
        if sells:
            print(f"  Realised P&L: ${total_pnl:,.2f}")
    print(f"{'=' * 60}\n")

    broker.disconnect()


if __name__ == "__main__":
    main()
