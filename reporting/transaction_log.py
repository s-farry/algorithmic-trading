import logging
from datetime import date

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def save_transactions(transactions: list[dict], strategy_name: str) -> pd.DataFrame:
    """Save all transactions to a CSV file.

    Returns the transactions DataFrame.
    """
    if not transactions:
        logger.warning("No transactions to save")
        return pd.DataFrame()

    df = pd.DataFrame(transactions)
    output_path = config.RESULTS_DIR / f"{strategy_name}_transactions.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} transactions to {output_path}")
    return df


def compute_summary(
    transactions: list[dict],
    value_history: list[tuple[date, float]],
    initial_capital: float,
    strategy_name: str,
) -> dict:
    """Compute and save performance summary statistics."""
    summary = {"strategy": strategy_name, "initial_capital": initial_capital}

    if not transactions:
        return summary

    tx_df = pd.DataFrame(transactions)

    # Overall returns
    sells = tx_df[tx_df["action"].str.startswith("SELL")]
    buys = tx_df[tx_df["action"] == "BUY"]

    total_invested = buys["cost_basis"].sum() if not buys.empty else 0
    total_proceeds = sells["proceeds"].sum() if not sells.empty else 0
    total_pnl = sells["pnl"].sum() if not sells.empty else 0

    summary["total_trades"] = len(tx_df)
    summary["total_buys"] = len(buys)
    summary["total_sells"] = len(sells)
    summary["total_invested"] = round(total_invested, 2)
    summary["total_proceeds"] = round(total_proceeds, 2)
    summary["total_pnl"] = round(total_pnl, 2)

    # Win/loss stats
    if not sells.empty:
        wins = sells[sells["pnl"] > 0]
        losses = sells[sells["pnl"] <= 0]

        summary["winning_trades"] = len(wins)
        summary["losing_trades"] = len(losses)
        summary["win_rate_pct"] = round(len(wins) / len(sells) * 100, 2)
        summary["avg_win"] = round(wins["pnl"].mean(), 2) if not wins.empty else 0
        summary["avg_loss"] = round(losses["pnl"].mean(), 2) if not losses.empty else 0

    # Portfolio value metrics from history
    if value_history:
        values = pd.DataFrame(value_history, columns=["date", "value"])
        final_value = values["value"].iloc[-1]
        summary["final_value"] = round(final_value, 2)
        summary["total_return_pct"] = round(
            (final_value - initial_capital) / initial_capital * 100, 2
        )

        # Annualised return
        days = (values["date"].iloc[-1] - values["date"].iloc[0]).days
        if days > 0:
            annualised = ((final_value / initial_capital) ** (365.0 / days) - 1) * 100
            summary["annualised_return_pct"] = round(annualised, 2)

        # Max drawdown
        peak = values["value"].cummax()
        drawdown = (values["value"] - peak) / peak
        summary["max_drawdown_pct"] = round(drawdown.min() * 100, 2)

        # Sharpe ratio (assuming 252 trading days, risk-free rate ~ 0)
        daily_returns = values["value"].pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            summary["sharpe_ratio"] = round(sharpe, 2)

    # Holding period stats: match each sell with its preceding buy
    if not sells.empty and not buys.empty:
        holding_days = []
        for _, sell_row in sells.iterrows():
            symbol = sell_row["symbol"]
            sell_date = pd.Timestamp(sell_row["date"])
            # Find the most recent buy of this symbol before (or on) the sell date
            symbol_buys = buys[
                (buys["symbol"] == symbol)
                & (pd.to_datetime(buys["date"]) <= sell_date)
            ]
            if not symbol_buys.empty:
                buy_date = pd.Timestamp(symbol_buys.iloc[-1]["date"])
                holding_days.append((sell_date - buy_date).days)

        if holding_days:
            holding_arr = np.array(holding_days)
            summary["avg_holding_days"] = round(float(holding_arr.mean()), 1)
            summary["median_holding_days"] = round(float(np.median(holding_arr)), 1)
            summary["min_holding_days"] = int(holding_arr.min())
            summary["max_holding_days"] = int(holding_arr.max())

    # Per-stock breakdown
    if not sells.empty:
        stock_summary = (
            sells.groupby("symbol")
            .agg(
                trades=("pnl", "count"),
                total_pnl=("pnl", "sum"),
                avg_pnl_pct=("pnl_pct", "mean"),
            )
            .sort_values("total_pnl", ascending=False)
        )

        stock_path = config.RESULTS_DIR / f"{strategy_name}_per_stock.csv"
        stock_summary.to_csv(stock_path)
        logger.info(f"Saved per-stock breakdown to {stock_path}")

    # Save summary
    summary_path = config.RESULTS_DIR / f"{strategy_name}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")

    return summary


def print_summary(summary: dict):
    """Pretty-print the strategy performance summary."""
    print("\n" + "=" * 60)
    print(f"  Strategy: {summary.get('strategy', 'N/A')}")
    print("=" * 60)
    print(f"  Initial Capital:     ${summary.get('initial_capital', 0):>12,.2f}")
    print(f"  Final Value:         ${summary.get('final_value', 0):>12,.2f}")
    print(f"  Total P&L:           ${summary.get('total_pnl', 0):>12,.2f}")
    print(f"  Total Return:        {summary.get('total_return_pct', 0):>12.2f}%")
    print(f"  Annualised Return:   {summary.get('annualised_return_pct', 0):>12.2f}%")
    print(f"  Max Drawdown:        {summary.get('max_drawdown_pct', 0):>12.2f}%")
    print(f"  Sharpe Ratio:        {summary.get('sharpe_ratio', 0):>12.2f}")
    print("-" * 60)
    print(f"  Total Trades:        {summary.get('total_trades', 0):>12}")
    print(f"  Winning Trades:      {summary.get('winning_trades', 0):>12}")
    print(f"  Losing Trades:       {summary.get('losing_trades', 0):>12}")
    print(f"  Win Rate:            {summary.get('win_rate_pct', 0):>12.2f}%")
    print(f"  Avg Win:             ${summary.get('avg_win', 0):>12,.2f}")
    print(f"  Avg Loss:            ${summary.get('avg_loss', 0):>12,.2f}")
    if "avg_holding_days" in summary:
        print("-" * 60)
        print(f"  Avg Holding Period:  {summary['avg_holding_days']:>10.1f} days")
        print(f"  Median Hold:         {summary['median_holding_days']:>10.1f} days")
        print(f"  Min / Max Hold:      {summary['min_holding_days']:>4} / {summary['max_holding_days']} days")
    print("=" * 60 + "\n")
