#!/usr/bin/env python3
"""Algorithmic Trading Simulator

Run trading strategy backtests against historical price data from
Financial Modeling Prep. Also supports live broker execution.

Usage:
    python simulate.py --strategy ma_crossover --start-date 2024-01-01 --end-date 2025-01-01
    python simulate.py --strategy all --start-date 2023-01-01 --end-date 2025-01-01
    python simulate.py --strategy momentum --symbols AAPL MSFT GOOGL
    python simulate.py --strategy prediction --predictions-file predictions.csv --start-date 2024-01-01 --end-date 2025-01-01
    python simulate.py --strategy momentum --broker etoro --demo --symbols AAPL --start-date 2025-01-01 --end-date 2025-02-01
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

import config

from data.sp500 import get_universe, get_bulk_prices
from data.stock_filter import compute_stock_metrics, filter_stocks
from engine.simulator import Simulator
from reporting.transaction_log import save_transactions, compute_summary, print_summary
from strategies import STRATEGIES

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def run_strategy(strategy_name, strategy_cls, price_data, start_date, end_date, capital,
                  broker=None, commission=None, min_hold_days=0, slippage_pct=0.0,
                  strategy_kwargs=None):
    """Run a single strategy and report results."""
    mode = f"LIVE ({broker.name})" if broker else "SIM"
    logger.info(f"{'=' * 40}")
    logger.info(f"Running strategy: {strategy_name} [{mode}]")
    logger.info(f"{'=' * 40}")

    strategy = strategy_cls(**(strategy_kwargs or {}))

    # Blended strategy needs to classify stocks before signal generation
    if hasattr(strategy, "set_assignments"):
        strategy.set_assignments(price_data, start_date)

    sim = Simulator(
        strategy=strategy,
        price_data=price_data,
        start_date=start_date,
        end_date=end_date,
        initial_capital=capital,
        broker=broker,
        commission=commission,
        min_hold_days=min_hold_days,
        slippage_pct=slippage_pct,
    )
    sim.run()

    # Save and report
    save_transactions(sim.transactions, strategy_name)
    summary = compute_summary(
        sim.transactions,
        sim.portfolio.value_history,
        capital,
        strategy_name,
    )
    print_summary(summary)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Algorithmic Trading Simulator")
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=list(STRATEGIES.keys()) + ["all"],
        help="Strategy to run (default: all)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Simulation start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Simulation end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Starting capital (default: 100000)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="*",
        default=None,
        help="Specific symbols to trade (overrides --universe)",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="sp500",
        choices=["sp500", "sp500_historical", "nasdaq", "dowjones", "midcap", "russell1000", "russell2000", "all"],
        help="Stock universe to trade (default: sp500). "
             "Use 'sp500_historical' for survivorship-bias-free backtesting.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        default=False,
        help="Disable stock filtering (use all symbols for every strategy)",
    )
    parser.add_argument(
        "--broker",
        type=str,
        default="sim",
        choices=["sim", "etoro", "ibkr"],
        help="Broker to use: sim (default), etoro, or ibkr",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Use broker's demo/paper trading mode (eToro only)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=None,
        help=f"Commission per trade in $ (default: {config.COMMISSION_PER_TRADE:.2f})",
    )
    parser.add_argument(
        "--min-hold-days",
        type=int,
        default=0,
        help="Minimum days to hold a position before selling (default: 0)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0,
        help="Simulated bid-ask spread as %% (e.g. 0.2 for 0.2%% round-trip spread). "
             "Models broker spread costs like eToro's.",
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

    # Build strategy kwargs for prediction strategy
    strategy_kwargs = {}
    if args.predictions_file:
        strategy_kwargs["predictions_files"] = args.predictions_file
    if args.buy_threshold is not None:
        strategy_kwargs["buy_threshold"] = args.buy_threshold
    if args.sell_threshold is not None:
        strategy_kwargs["sell_threshold"] = args.sell_threshold

    # Determine symbols
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = get_universe(
            args.universe, start_date=args.start_date, end_date=args.end_date
        )

    logger.info(f"Trading {len(symbols)} symbols from {args.start_date} to {args.end_date}")

    # Determine strategies and compute warmup period
    strategies_to_run = (
        STRATEGIES if args.strategy == "all" else {args.strategy: STRATEGIES[args.strategy]}
    )

    # Instantiate strategies to get warmup_days, then compute fetch start date
    max_warmup = max(cls().warmup_days for cls in strategies_to_run.values())
    # Convert trading days to calendar days (~1.4x for weekends + holidays)
    warmup_calendar_days = int(max_warmup * 1.5)
    fetch_start = (
        datetime.strptime(args.start_date, "%Y-%m-%d") - timedelta(days=warmup_calendar_days)
    ).strftime("%Y-%m-%d")

    logger.info(
        f"Fetching data from {fetch_start} ({max_warmup} trading days warmup) "
        f"to {args.end_date}"
    )

    # Fetch price data with warmup included (cached after first fetch)
    price_data = get_bulk_prices(symbols, fetch_start, args.end_date)

    if not price_data:
        logger.error("No price data fetched. Check API key and network connection.")
        sys.exit(1)

    # Set up broker (if live trading)
    broker = None
    if args.broker != "sim":
        from brokers import BROKERS
        broker_cls = BROKERS[args.broker]

        if args.broker == "etoro":
            broker = broker_cls(demo=args.demo)
        else:
            broker = broker_cls()

        # Safety confirmation for live trading
        mode_label = "DEMO" if args.demo else "REAL MONEY"
        strategy_label = args.strategy
        num_symbols = len(symbols)
        print(f"\n{'!' * 60}")
        print(f"  WARNING: LIVE TRADING MODE ({args.broker.upper()}) — {mode_label}")
        print(f"  Strategy: {strategy_label} | Capital: ${args.capital:,.0f} | Symbols: {num_symbols}")
        print(f"{'!' * 60}")
        confirm = input("  Continue? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

        broker.connect()
        logger.info(f"Broker connected: {args.broker}")

    # Compute stock metrics for filtering (uses warmup data only, no look-ahead bias)
    use_filter = not args.no_filter and not args.symbols
    metrics = None
    if use_filter:
        metrics = compute_stock_metrics(price_data, args.start_date)

    summaries = []
    for name, cls in strategies_to_run.items():
        # Pass strategy kwargs only to the matching strategy
        kwargs = strategy_kwargs if name == "prediction" else {}
        strategy_instance = cls(**kwargs)

        if use_filter and metrics is not None and not metrics.empty and name != "hold":
            filtered_data = filter_stocks(
                price_data, metrics, strategy_instance.stock_profile
            )
            selected = sorted(filtered_data.keys())
            logger.info(
                f"Strategy {name}: trading {len(filtered_data)} filtered stocks "
                f"(profile: vol={strategy_instance.stock_profile.volatility}, "
                f"regime={strategy_instance.stock_profile.regime})"
            )
            logger.info(f"  Selected: {', '.join(selected)}")
        else:
            filtered_data = price_data

        summary = run_strategy(
            name, cls, filtered_data, args.start_date, args.end_date, args.capital,
            broker=broker, commission=args.commission, min_hold_days=args.min_hold_days,
            slippage_pct=args.slippage / 100,
            strategy_kwargs=kwargs,
        )
        summaries.append(summary)

    # Disconnect broker when done
    if broker:
        broker.disconnect()

    # Comparison table if running multiple strategies
    if len(summaries) > 1:
        print("\n" + "=" * 70)
        print("  STRATEGY COMPARISON")
        print("=" * 70)
        print(f"  {'Strategy':<20} {'Return %':>10} {'Sharpe':>8} {'Max DD %':>10} {'Win Rate %':>10}")
        print("-" * 70)
        for s in summaries:
            print(
                f"  {s.get('strategy', ''):<20} "
                f"{s.get('total_return_pct', 0):>10.2f} "
                f"{s.get('sharpe_ratio', 0):>8.2f} "
                f"{s.get('max_drawdown_pct', 0):>10.2f} "
                f"{s.get('win_rate_pct', 0):>10.2f}"
            )
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
