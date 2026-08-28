import logging
from pathlib import Path

import pandas as pd

import config
from strategies.base import Strategy

logger = logging.getLogger(__name__)


class PredictionStrategy(Strategy):
    """Strategy that trades based on external ML model predictions.

    Reads a predictions CSV produced by predict_technical_model.py or
    predict_fundamental_model.py and generates buy/sell signals based
    on the predicted price movement.

    Buy when predicted uplift >= buy_threshold (e.g. 1.05 = +5%).
    Sell when predicted uplift <= sell_threshold (e.g. 0.95 = -5%).
    """

    name = "prediction"
    warmup_days = 0  # No warmup — signals come from external predictions

    def __init__(self, predictions_files=None, buy_threshold=None, sell_threshold=None):
        self.buy_threshold = buy_threshold or config.PREDICTION_BUY_THRESHOLD
        self.sell_threshold = sell_threshold or config.PREDICTION_SELL_THRESHOLD
        self.predictions = None

        if predictions_files:
            if isinstance(predictions_files, str):
                predictions_files = [predictions_files]
            self.load_predictions(predictions_files)

    def _read_predictions_file(self, filepath):
        """Read and normalise a single predictions CSV file.

        Supports two formats:
          - Technical: columns [date, symbol, Uplift, close, close_pred]
          - Fundamental: columns [date, Symbol, y_pred, close, close_pred]

        Returns a DataFrame with columns [date, symbol, uplift].
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Predictions file not found: {filepath}")

        df = pd.read_csv(filepath, parse_dates=["date"])

        # Normalise to common format: date, symbol, uplift
        if "Uplift" in df.columns and "symbol" in df.columns:
            df = df.rename(columns={"Uplift": "uplift"})
        elif "y_pred" in df.columns and "Symbol" in df.columns:
            df = df.rename(columns={"y_pred": "uplift", "Symbol": "symbol"})
        elif "y_pred" in df.columns and "symbol" in df.columns:
            df = df.rename(columns={"y_pred": "uplift"})
        elif "Uplift" in df.columns and "Symbol" in df.columns:
            df = df.rename(columns={"Uplift": "uplift", "Symbol": "symbol"})
        else:
            raise ValueError(
                f"Predictions file '{filepath}' must contain an uplift column "
                f"('Uplift' or 'y_pred') and a symbol column ('symbol' or 'Symbol'). "
                f"Found columns: {list(df.columns)}"
            )

        required = {"date", "symbol", "uplift"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Predictions file '{filepath}' missing required columns: {missing}"
            )

        if df.empty:
            raise ValueError(f"Predictions file '{filepath}' is empty")

        logger.info(
            f"  {path.name}: {len(df)} rows, {df['symbol'].nunique()} symbols, "
            f"{df['date'].nunique()} dates "
            f"({df['date'].min().date()} to {df['date'].max().date()})"
        )

        return df[["date", "symbol", "uplift"]]

    def load_predictions(self, filepaths):
        """Load one or more predictions CSV files.

        Files are processed in order. For overlapping (symbol, date) pairs,
        the first file takes precedence.
        """
        logger.info(f"Loading predictions from {len(filepaths)} file(s):")

        frames = []
        for filepath in filepaths:
            frames.append(self._read_predictions_file(filepath))

        combined = pd.concat(frames, ignore_index=True)

        # First file takes precedence: drop_duplicates keeps the first occurrence
        combined = combined.drop_duplicates(subset=["symbol", "date"], keep="first")

        n_symbols = combined["symbol"].nunique()
        n_dates = combined["date"].nunique()
        logger.info(
            f"Combined predictions: {len(combined)} rows, {n_symbols} symbols, "
            f"{n_dates} dates"
        )
        logger.info(
            f"Thresholds: buy >= {self.buy_threshold}, sell <= {self.sell_threshold}"
        )

        self.predictions = combined.set_index(["symbol", "date"])["uplift"]

    def generate_signals(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["uplift"] = float("nan")

        if self.predictions is None:
            logger.warning("No predictions loaded — all signals will be hold")
            return df

        if symbol is None:
            return df

        # Check if we have predictions for this symbol
        if symbol not in self.predictions.index.get_level_values(0):
            return df

        # Merge predictions onto the price dataframe by date
        symbol_preds = self.predictions.loc[symbol]
        date_to_uplift = symbol_preds.to_dict()

        df["uplift"] = df["date"].map(date_to_uplift)

        # Generate signals
        df.loc[df["uplift"] >= self.buy_threshold, "signal"] = 1
        df.loc[df["uplift"] <= self.sell_threshold, "signal"] = -1

        return df
