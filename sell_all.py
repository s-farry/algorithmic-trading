#!/usr/bin/env python3
"""Sell all open positions for any supported broker.

Usage:
    python sell_all.py --broker etoro --demo        # eToro demo account
    python sell_all.py --broker etoro               # eToro real account
    python sell_all.py --broker ibkr                # IBKR
    python sell_all.py --broker etoro --demo -y     # skip confirmation prompt
"""

import argparse
import logging
import sys

from brokers import BROKERS
from live import get_known_holdings

log_fmt = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Sell all open broker positions")
    parser.add_argument(
        "--broker",
        type=str,
        required=True,
        choices=list(BROKERS.keys()),
        help="Broker to use",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Use demo/paper trading mode (eToro only)",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        default=False,
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="momentum",
        help="Strategy name used to read local transaction log for cache warming (default: momentum)",
    )
    args = parser.parse_args()

    broker_cls = BROKERS[args.broker]
    if args.broker == "etoro":
        broker = broker_cls(demo=args.demo)
    else:
        broker = broker_cls()

    broker.connect()

    # eToro needs instrument cache warmed from local log for ID->symbol resolution
    if hasattr(broker, "warm_cache"):
        local_holdings = get_known_holdings(args.strategy)
        if local_holdings:
            broker.warm_cache(local_holdings)

    positions = broker.get_positions()

    if not positions:
        print("No open positions found.")
        broker.disconnect()
        return

    # Fetch live prices where available for a more accurate value estimate
    current_prices = {}
    if hasattr(broker, "get_current_price"):
        logger.info(f"Fetching live prices for {len(positions)} positions...")
        for symbol in positions:
            price = broker.get_current_price(symbol)
            if price:
                current_prices[symbol] = price

    mode = "DEMO" if args.demo else "REAL MONEY"
    print(f"\n{'=' * 58}")
    print(f"  {args.broker.upper()} {mode} — Positions to close:")
    print(f"{'=' * 58}")
    total_value = 0.0
    for symbol, pos in sorted(positions.items()):
        price = current_prices.get(symbol, pos.avg_cost)
        value = pos.quantity * price
        total_value += value
        live_tag = " (live)" if symbol in current_prices else " (cost)"
        print(
            f"  {symbol:<8}  {pos.quantity:>6} shares  "
            f"@ ${price:>8.2f}{live_tag}  ~${value:>10,.2f}"
        )
    print(f"{'=' * 58}")
    print(f"  Total positions: {len(positions)}   Est. value: ~${total_value:,.2f}")
    print()

    if not args.yes:
        confirm = input("  Sell ALL of the above? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            broker.disconnect()
            sys.exit(0)

    sold = []
    failed = []

    for symbol, pos in sorted(positions.items()):
        price = current_prices.get(symbol, pos.avg_cost)
        logger.info(f"Closing {symbol} ({pos.quantity} shares @ ~${price:.2f})...")
        result = broker.sell(symbol, pos.quantity, price)
        if result:
            sold.append(symbol)
            print(f"  SOLD    {symbol}")
        else:
            failed.append(symbol)
            print(f"  FAILED  {symbol}")

    print(f"\n{'=' * 58}")
    print(f"  Closed: {len(sold)}  Failed: {len(failed)}")
    if failed:
        print(f"  Failed symbols: {', '.join(failed)}")
    print(f"{'=' * 58}\n")

    broker.disconnect()


if __name__ == "__main__":
    main()
