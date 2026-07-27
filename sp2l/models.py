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

    # Partial take-profit: close partial_pct of the position at partial_rr R.
    partial_enabled: bool = False
    partial_rr: float = 1.0
    partial_pct: float = 0.5

    # Breakeven (risk-free): move SL to entry + be_offset_r * R once triggered.
    # Modes: "off", "rr" (price reaches be_trigger_rr R), "after_partial".
    be_mode: str = "off"
    be_trigger_rr: float = 1.0
    be_offset_r: float = 0.0

    # ATR trailing stop (0 = off). Trails at close -/+ mult * ATR, only ever
    # tightening the stop.
    trail_atr_mult: float = 0.0

    # SP2L pullback quality: minimum bars between pullback start and breakout,
    # and minimum pullback depth as a fraction of the spike (0 = off).
    min_pullback_bars: int = 1
    min_retrace: float = 0.0

    # Round-trip trading cost per trade, in R (spread + commission).
    cost_r: float = 0.0

    # Daily risk guard: block new entries after this many trades per day or
    # once the day's net R hits -max_daily_loss_r (0 = off).
    max_trades_per_day: int = 0
    max_daily_loss_r: float = 0.0

    # Take every signal even while already in a trade / holding a pending
    # order (disables the "skip because in trade" rule).
    allow_concurrent: bool = False

    # SP2L entry trigger. The four supported combinations are:
    #   break_confirm="touch", entry_mode="market" -> option 1
    #   break_confirm="touch", entry_mode="limit"  -> option 2
    #   break_confirm="close", entry_mode="market" -> option 3 (default)
    #   break_confirm="close", entry_mode="limit"  -> option 4
    #
    # break_confirm: how the break of level B counts
    #   "touch" - any trade through B (shadow or close)
    #   "close" - the bar must close beyond B
    # entry_mode: how we enter once the break is confirmed
    #   "market" - market order filled at the OPEN of the next bar
    #   "limit"  - a limit resting at B; only activates if price returns to B
    #              within limit_wait_bars, otherwise the order is cancelled
    #              and never becomes a trade
    break_confirm: str = "close"
    entry_mode: str = "market"
    limit_wait_bars: int = 10

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
        if self.partial_rr <= 0:
            raise ValueError("partial_rr must be positive")
        if not 0 < self.partial_pct < 1:
            raise ValueError("partial_pct must be in (0, 1)")
        if self.be_mode not in ("off", "rr", "after_partial"):
            raise ValueError("be_mode must be off, rr or after_partial")
        if self.entry_mode not in ("market", "limit"):
            raise ValueError("entry_mode must be market or limit")
        if self.break_confirm not in ("touch", "close"):
            raise ValueError("break_confirm must be touch or close")
        if self.limit_wait_bars < 1:
            raise ValueError("limit_wait_bars must be >= 1")
        for name in ("trail_atr_mult", "min_retrace", "cost_r",
                     "max_trades_per_day", "max_daily_loss_r"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.min_pullback_bars < 1:
            raise ValueError("min_pullback_bars must be >= 1")
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
    pullback_extreme: Optional[float] = None  # deepest pullback price so far
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
    init_stop: Optional[float] = None  # stop at entry time (R anchor)
    scale_in_price: Optional[float] = None
    scaled_in: bool = False
    partial_done: bool = False
    be_done: bool = False
    trailed: bool = False
    realized_pnl: float = 0.0  # banked by the partial exit
    exit_idx: Optional[int] = None
    exit_ts: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0

    def __post_init__(self) -> None:
        if self.init_stop is None:
            self.init_stop = self.stop

    @property
    def is_open(self) -> bool:
        return self.exit_idx is None

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.init_stop)

    @property
    def r_multiple(self) -> float:
        """Trade result measured in R (initial risk units)."""
        risk = self.risk_per_unit * self.size
        return self.pnl / risk if risk > 0 else 0.0


@dataclass
class Signal:
    """Emitted by a strategy when an entry triggers.

    entry_type decides how the backtester turns this into a trade:
      "now"       - fill immediately at `entry` on the signal bar
      "next_open" - fill at the OPEN of the next bar (entry/target recomputed)
      "limit"     - rest a limit at `entry`; fill only if price returns to it
    """

    direction: Direction
    entry: float
    stop: float
    target: float
    tag: str = "sp2l"
    entry_type: str = "now"
    rr: float = 1.0  # lets the backtester recompute the target after a fill
    setup: Setup = field(repr=False, default=None)
