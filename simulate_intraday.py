#!/usr/bin/env python3
"""Intraday Trading Simulator

Backtest intraday strategies against minute/5-minute bar data.
All positions are forced-closed at the end of each trading session (day-trading mode).

Data sources:
    --data-source fmp      Financial Modeling Prep (default, limited free history)
    --data-source polygon  Polygon.io (recommended, 2+ years free history)

Usage:
    python simulate_intraday.py --strategy vwap_reversion --symbols AAPL MSFT --start-date 2024-11-01 --end-date 2024-11-30
    python simulate_intraday.py --strategy commodity_orb --symbols USO --start-date 2026-03-01 --end-date 2026-05-13 --data-source polygon
    python simulate_intraday.py --strategy opening_range_breakout --symbols AAPL --interval 5min --capital 50000
    python simulate_intraday.py --strategy intraday_rsi --symbols AAPL TSLA NVDA --start-date 2024-10-01 --end-date 2024-12-31 --data-source polygon
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

import pandas as pd

import config
from engine.intraday_simulator import IntradaySimulator
from reporting.transaction_log import save_transactions, compute_summary, print_summary
from strategies import INTRADAY_STRATEGIES

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def print_session_breakdown(session_pnl: list[dict]):
    """Print a per-session (daily) P&L table."""
    if not session_pnl:
        return

    print("\n" + "=" * 55)
    print("  DAILY SESSION BREAKDOWN")
    print("=" * 55)
    print(f"  {'Date':<12}  {'Start Value':>12}  {'End Value':>12}  {'P&L':>10}  {'%':>7}")
    print("-" * 55)

    for row in session_pnl:
        pnl = row["pnl"]
        sign = "+" if pnl >= 0 else ""
        print(
            f"  {str(row['date']):<12}  "
            f"${row['start_value']:>11,.2f}  "
            f"${row['end_value']:>11,.2f}  "
            f"{sign}${pnl:>9,.2f}  "
            f"{sign}{row['pnl_pct']:>6.2f}%"
        )

    total_pnl = sum(r["pnl"] for r in session_pnl)
    sign = "+" if total_pnl >= 0 else ""
    print("-" * 55)
    print(f"  {'TOTAL':<12}  {'':>12}  {'':>12}  {sign}${total_pnl:>9,.2f}")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Intraday Trading Simulator")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=list(INTRADAY_STRATEGIES.keys()),
        help="Intraday strategy to run",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        required=True,
        help="Ticker symbols to trade (e.g. AAPL MSFT TSLA)",
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
        "--interval",
        type=str,
        default="5min",
        choices=["1min", "5min", "15min", "30min", "1hour"],
        help="Bar interval (default: 5min)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=50_000.0,
        help="Starting capital in $ (default: 50000)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.1,
        help="Simulated bid-ask spread as %% of price, round-trip (default: 0.1)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0,
        help="Commission per trade in $ (default: 0.0)",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="fmp",
        choices=["fmp", "polygon"],
        help="Intraday data source: fmp (default, limited history) or polygon "
             "(recommended for backtests, 2+ years free, requires POLYGON_API_KEY in .env)",
    )
    parser.add_argument(
        "--polygon-delay",
        type=float,
        default=12.0,
        help="Seconds between Polygon API calls (default: 12 to respect free-tier "
             "5 req/min limit). Set to 0.5 if you have a paid Polygon plan.",
    )

    args = parser.parse_args()

    strategy_cls = INTRADAY_STRATEGIES[args.strategy]
    strategy = strategy_cls()

    logger.info(
        f"Strategy: {args.strategy} | Symbols: {args.symbols} | "
        f"Interval: {args.interval} | {args.start_date} → {args.end_date} | "
        f"Data: {args.data_source}"
    )

    # Warmup: fetch extra bars before start_date so indicators have time to stabilise.
    # warmup_bars at ~78 bars/day for 5min (6.5h × 12) → 1 day covers most strategies.
    # Use 2 extra calendar days to be safe.
    warmup_bars = getattr(strategy, "warmup_bars", 0)
    extra_days = max(2, (warmup_bars // 78) + 1)
    fetch_start = (
        datetime.strptime(args.start_date, "%Y-%m-%d") - timedelta(days=extra_days)
    ).strftime("%Y-%m-%d")

    if fetch_start != args.start_date:
        logger.info(f"Fetching {extra_days} warmup days from {fetch_start}")

    # Select data source
    if args.data_source == "polygon":
        from data.polygon_intraday import get_bulk_intraday_prices
    else:
        from data.intraday import get_bulk_intraday_prices

    # Fetch intraday data
    fetch_kwargs = dict(
        symbols=args.symbols,
        start_date=fetch_start,
        end_date=args.end_date,
        interval=args.interval,
        market_hours_only=True,
        use_cache=True,
    )
    if args.data_source == "polygon":
        fetch_kwargs["delay"] = args.polygon_delay

    price_data = get_bulk_intraday_prices(**fetch_kwargs)

    if not price_data:
        logger.error("No intraday data returned. Check API key and symbols.")
        sys.exit(1)

    logger.info(f"Loaded intraday data for {len(price_data)} symbols")

    # Run simulation
    sim = IntradaySimulator(
        strategy=strategy,
        price_data=price_data,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
        commission=args.commission,
        slippage_pct=args.slippage / 100,
    )
    sim.run()

    # Save transactions
    tag = f"intraday_{args.strategy}"
    save_transactions(sim.transactions, tag)

    # Print session breakdown
    print_session_breakdown(sim.session_pnl)

    # Overall summary (reuse daily reporting module)
    summary = compute_summary(
        sim.transactions,
        sim.value_history,
        args.capital,
        tag,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
