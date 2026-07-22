"""SP2L (Spike-2Leg) strategy engine and backtester.

Implements Mohammad Ali Poursamadi's SP2L price-action setup:
spike detection -> pullback -> second-leg breakout entry, with
SL behind the spike origin and TP at a configurable R:R.
"""

from .models import Bar, Config, Direction, Setup, Trade
from .strategy import SP2LStrategy
from .backtest import Backtester, BacktestResult

__all__ = [
    "Bar",
    "Config",
    "Direction",
    "Setup",
    "Trade",
    "SP2LStrategy",
    "Backtester",
    "BacktestResult",
]

__version__ = "0.1.0"
