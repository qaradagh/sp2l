"""Data models for the SP2L strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Optional, Tuple


class Direction(Enum):
    LONG = 1
    SHORT = -1


@dataclass(frozen=True)
class Bar:
    """A single OHLCV candle."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def body_ratio(self) -> float:
        return self.body / self.range if self.range > 0 else 0.0

    @property
    def is_bull(self) -> bool:
        return self.close > self.open

    @property
    def is_bear(self) -> bool:
        return self.close < self.open


@dataclass
class Config:
    """SP2L parameters. Defaults follow docs/BLUEPRINT.md section 6."""

    # Spike detection
    min_spike_bars: int = 3
    min_body_ratio: float = 0.5
    doji_tolerance: int = 1
    max_doji_body_ratio: float = 0.25
    require_gap: bool = True
    power_mult: float = 1.5
    atr_len: int = 14

    # Pullback / setup lifecycle
    max_retrace: float = 1.0
    max_wait_bars: int = 20

    # Trade management
    rr: float = 1.0
    sl_buffer_pct: float = 0.0
    scale_in: bool = False

    # Session filter (UTC)
    session_filter: bool = False
    session_start: time = time(13, 30)
    session_end: time = time(20, 0)

    # Money management
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.01

    # Pro BTB (Back To Breakeven) — docs/BTB_BLUEPRINT.md
    btb_level_lookback: int = 10
    btb_rr: float = 2.0
    btb_max_wait_bars: int = 30

    # Which setups the backtester runs, in priority order (first wins when
    # several signal on the same bar).
    enabled_setups: Tuple[str, ...] = ("btb", "sp2l")

    def validate(self) -> None:
        if self.min_spike_bars < 2:
            raise ValueError("min_spike_bars must be >= 2")
        if not 0 < self.min_body_ratio <= 1:
            raise ValueError("min_body_ratio must be in (0, 1]")
        if self.rr <= 0 or self.btb_rr <= 0:
            raise ValueError("rr and btb_rr must be positive")
        if self.max_retrace <= 0:
            raise ValueError("max_retrace must be positive")
        if self.btb_level_lookback < 1:
            raise ValueError("btb_level_lookback must be >= 1")
        unknown = set(self.enabled_setups) - {"sp2l", "btb"}
        if unknown:
            raise ValueError(f"unknown setups: {sorted(unknown)}")


@dataclass
class Setup:
    """A confirmed spike waiting for pullback + breakout."""

    direction: Direction
    spike_start_idx: int
    spike_end_idx: int
    point_a: float  # spike origin extreme (SL anchor)
    point_b: float  # last spike candle extreme (entry level)
    created_idx: int
    in_pullback: bool = False
    pullback_start_idx: Optional[int] = None
    level: Optional[float] = None  # BTB: the broken key level (retest target)

    @property
    def spike_len(self) -> float:
        return abs(self.point_b - self.point_a)


@dataclass
class Trade:
    """An executed (open or closed) trade."""

    direction: Direction
    entry_idx: int
    entry_ts: datetime
    entry: float
    stop: float
    target: float
    tag: str = "sp2l"
    size: float = 0.0
    scale_in_price: Optional[float] = None
    scaled_in: bool = False
    exit_idx: Optional[int] = None
    exit_ts: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_idx is None

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop)


@dataclass
class Signal:
    """Emitted by the strategy when a breakout entry triggers."""

    direction: Direction
    entry: float
    stop: float
    target: float
    tag: str = "sp2l"
    setup: Setup = field(repr=False, default=None)
