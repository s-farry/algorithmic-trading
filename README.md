# Algorithmic Trading

A Python framework for backtesting and running systematic equity/commodity trading strategies, with optional live execution through eToro or Interactive Brokers (IBKR).

It covers the full loop: pull historical or intraday price data, run a strategy against it in a simulated portfolio, review the resulting P&L/summary stats, and — if you want — point the same strategy at a real (or demo) broker account for daily live trading.

## Features

- **Daily-bar backtesting** (`simulate.py`) across an S&P 500 / NASDAQ / Dow / Russell universe or a custom symbol list, with stock filtering, commission, slippage, and minimum-hold-period modeling.
- **Intraday backtesting** (`simulate_intraday.py`) on 1min–1hour bars, with all positions force-closed at end of session (day-trading mode) and a per-session P&L breakdown.
- **Live trading** (`live.py`) that generates today's signals and places real orders, intended to be run once a day (e.g. via cron).
- **Position flattening** (`sell_all.py`) to liquidate all open broker positions in one command.
- **Pluggable strategies** — moving average crossover, Bollinger mean reversion, RSI/MACD momentum, buy-and-hold, a blended multi-strategy allocator, an external-prediction-driven strategy, and several intraday strategies (VWAP reversion, opening range breakout, intraday RSI, commodity ORB, volume surge).
- **Pluggable brokers** — a built-in simulator, eToro (demo or live), and Interactive Brokers (via `ibapi`).
- **Pluggable data sources** — Financial Modeling Prep (daily + limited intraday history) and Polygon.io (deeper intraday history).

## Project layout

```
config.py              Central config: API keys, capital/commission defaults, indicator params
simulate.py             Daily-bar backtest runner
simulate_intraday.py    Intraday backtest runner
live.py                 Generates today's signals and executes them via a broker
sell_all.py              Liquidates all open positions for a broker
strategies/              Strategy implementations (see strategies/base.py for the interface)
brokers/                 Broker integrations (sim, eToro, IBKR) — see brokers/base.py
engine/                  Portfolio and simulator engines (daily + intraday)
data/                    Price data fetching, caching, and stock universe/filtering
reporting/                Transaction logging and summary stat computation
results/                 Backtest output (transactions, per-stock and summary CSVs) — gitignored
```

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with the API keys/credentials you need:

```bash
fmd_api_key=your_financial_modeling_prep_key
POLYGON_API_KEY=your_polygon_key          # optional, for deeper intraday history

# eToro (optional, for live/demo trading)
ETORO_API_KEY=your_etoro_api_key
ETORO_USER_KEY=your_etoro_user_key

# IBKR (optional, for live trading via TWS/Gateway)
IBKR_HOST=127.0.0.1
IBKR_PORT=7496
IBKR_CLIENT_ID=1
```

`.env` is gitignored — never commit real API keys or credentials.

## Usage

### Backtest a daily strategy

```bash
python simulate.py --strategy ma_crossover --start-date 2024-01-01 --end-date 2025-01-01
python simulate.py --strategy all --start-date 2023-01-01 --end-date 2025-01-01
python simulate.py --strategy momentum --symbols AAPL MSFT GOOGL --start-date 2024-01-01 --end-date 2025-01-01
```

### Backtest an intraday strategy

```bash
python simulate_intraday.py --strategy vwap_reversion --symbols AAPL MSFT --start-date 2024-11-01 --end-date 2024-11-30
python simulate_intraday.py --strategy commodity_orb --symbols USO --start-date 2026-03-01 --end-date 2026-05-13 --data-source polygon
```

### Run live (paper or real)

```bash
python live.py --strategy momentum --broker etoro --demo   # eToro demo account
python live.py --strategy momentum --broker etoro          # eToro real account
python live.py --strategy blended --broker ibkr
```

Live trading requires explicit confirmation at the prompt before any order is placed, and `simulate.py` can also route through a live broker (`--broker etoro|ibkr`) for paper-testing execution against real market prices.

### Flatten all positions

```bash
python sell_all.py --broker etoro --demo
```

Run any script with `--help` for the full list of options.

## Disclaimer

This project is for research and educational purposes. Nothing here is financial advice, and past backtest performance is not indicative of future results. Live trading involves real financial risk — use demo/paper modes and small position sizes until you're confident in a strategy.

## License

MIT — see [LICENSE](LICENSE).
