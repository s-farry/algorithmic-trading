from strategies.moving_average import MovingAverageCrossover
from strategies.mean_reversion import MeanReversionBollinger
from strategies.momentum import RSIMACDMomentum
from strategies.blended import BlendedStrategy
from strategies.hold import BuyAndHold
from strategies.prediction import PredictionStrategy
from strategies.vwap_reversion import VWAPReversion
from strategies.opening_range_breakout import OpeningRangeBreakout
from strategies.intraday_rsi import IntradayRSI
from strategies.commodity_orb import CommodityORB
from strategies.volume_surge import VolumeSurge

STRATEGIES = {
    "ma_crossover": MovingAverageCrossover,
    "mean_reversion": MeanReversionBollinger,
    "momentum": RSIMACDMomentum,
    "blended": BlendedStrategy,
    "hold": BuyAndHold,
    "prediction": PredictionStrategy,
    # Intraday strategies (use with simulate_intraday.py / live_intraday.py)
    "vwap_reversion": VWAPReversion,
    "opening_range_breakout": OpeningRangeBreakout,
    "intraday_rsi": IntradayRSI,
}

INTRADAY_STRATEGIES = {
    "vwap_reversion": VWAPReversion,
    "opening_range_breakout": OpeningRangeBreakout,
    "intraday_rsi": IntradayRSI,
    "commodity_orb": CommodityORB,
    "volume_surge": VolumeSurge,
}
