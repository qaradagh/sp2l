"""Poursamadi price-action setups: SP2L and Pro BTB, with a backtester.

SP2L: spike -> pullback -> second-leg breakout entry (stop at level B).
Pro BTB: spike breaks a key level -> retest of the level (limit at L).
Both share the same spike-detection engine and can run combined.
"""

from .models import Bar, Config, Direction, Setup, Trade
from .strategy import SP2LStrategy
from .btb import BTBStrategy
from .backtest import Backtester, BacktestResult

__all__ = [
    "Bar",
    "Config",
    "Direction",
    "Setup",
    "Trade",
    "SP2LStrategy",
    "BTBStrategy",
    "Backtester",
    "BacktestResult",
]

__version__ = "0.1.0"
