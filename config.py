import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FMP_API_KEY = os.environ.get("fmd_api_key")
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# Portfolio defaults
DEFAULT_CAPITAL = 100_000.0
COMMISSION_PER_TRADE = 1.00  # $1 flat per trade
MAX_POSITION_PCT = 0.15      # Max 5% of portfolio in a single stock

# Stock filtering
FILTER_TOP_N = 50            # Max stocks per strategy after filtering

# Moving Average Crossover
MA_SHORT_WINDOW = 50
MA_LONG_WINDOW = 200

# Bollinger Bands / Mean Reversion
BB_WINDOW = 20
BB_NUM_STD = 2.0

# RSI + MACD Momentum
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Prediction strategy
PREDICTION_BUY_THRESHOLD = 1.04   # Buy if predicted uplift >= 5%
PREDICTION_SELL_THRESHOLD = 0.96  # Sell if predicted uplift <= -5%

# Polygon.io (intraday data, free tier: 2+ years history, 5 req/min)
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
POLYGON_BASE_URL = "https://api.polygon.io"

# eToro broker
ETORO_API_KEY = os.environ.get("ETORO_API_KEY")        # x-api-key header
ETORO_USER_KEY = os.environ.get("ETORO_USER_KEY")      # x-user-key header

# IBKR broker
IBKR_HOST = os.environ.get("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.environ.get("IBKR_PORT", "7496"))
IBKR_CLIENT_ID = int(os.environ.get("IBKR_CLIENT_ID", "1"))
