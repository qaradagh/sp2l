"""SP2L state machine: spike detection -> pullback -> breakout signal.

Long setup (short is the exact mirror):
  1. Spike: >= min_spike_bars consecutive bullish candles with strong bodies
     (limited doji tolerance), containing a price gap / FVG, and long enough
     relative to ATR (movement-power filter).
  2. Pullback: a candle whose low breaks the previous candle's low.
  3. Entry: breakout of the last spike candle's high (point B).
     SL behind the spike origin (point A), TP at rr * risk.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Sequence

from .models import Bar, Config, Direction, Setup, Signal


class State(Enum):
    IDLE = "idle"
    SPIKE_CONFIRMED = "spike_confirmed"
    PULLBACK = "pullback"


def true_ranges(bars: Sequence[Bar]) -> List[float]:
    out: List[float] = []
    for i, b in enumerate(bars):
        if i == 0:
            out.append(b.range)
        else:
            prev_close = bars[i - 1].close
            out.append(max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close)))
    return out


class SpikeDetectorBase:
    """Shared spike-detection engine used by SP2L and Pro BTB.

    Subclasses call _append_bar() at the top of on_bar() and use
    _detect_spike() plus the helper predicates to drive their own lifecycle.
    """

    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or Config()
        self.cfg.validate()
        self._bars: List[Bar] = []
        self._trs: List[float] = []
        self._ema: Optional[float] = None

    def _append_bar(self, bar: Bar) -> int:
        """Track the bar, its true range and the trend EMA; returns the index."""
        self._bars.append(bar)
        idx = len(self._bars) - 1
        if idx == 0:
            self._trs.append(bar.range)
        else:
            prev_close = self._bars[idx - 1].close
            self._trs.append(
                max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close))
            )
        if self.cfg.trend_ema_len > 0:
            if self._ema is None:
                self._ema = bar.close
            else:
                alpha = 2.0 / (self.cfg.trend_ema_len + 1)
                self._ema += alpha * (bar.close - self._ema)
        return idx

    def _trend_ok(self, direction: Direction, bar: Bar) -> bool:
        if self.cfg.trend_ema_len <= 0 or self._ema is None:
            return True
        return bar.close > self._ema if direction is Direction.LONG else bar.close < self._ema

    # ------------------------------------------------------------------ utils

    def _atr(self, idx: int) -> Optional[float]:
        """Simple ATR over the last atr_len true ranges ending at idx (exclusive
        of future bars). Returns None when there is not enough history."""
        n = self.cfg.atr_len
        if idx + 1 < n:
            return None
        window = self._trs[idx + 1 - n : idx + 1]
        return sum(window) / n

    def _in_session(self, bar: Bar) -> bool:
        if not self.cfg.session_filter:
            return True
        t = bar.ts.time()
        start, end = self.cfg.session_start, self.cfg.session_end
        if start <= end:
            return start <= t <= end
        return t >= start or t <= end  # session spans midnight

    # -------------------------------------------------------- spike detection

    def _candle_fits_spike(self, bar: Bar, direction: Direction) -> Optional[str]:
        """Returns 'strong', 'doji' or None if the candle breaks the run."""
        if bar.body_ratio <= self.cfg.max_doji_body_ratio:
            return "doji"  # tolerated regardless of its direction
        if direction is Direction.LONG and not bar.is_bull:
            return None
        if direction is Direction.SHORT and not bar.is_bear:
            return None
        if bar.body_ratio >= self.cfg.min_body_ratio:
            return "strong"
        return None

    def _find_spike_run(self, end_idx: int, direction: Direction) -> Optional[int]:
        """Walk backwards from end_idx; return start index of a valid run."""
        cfg = self.cfg
        dojis = 0
        strong = 0
        i = end_idx
        start = None
        while i >= 0:
            kind = self._candle_fits_spike(self._bars[i], direction)
            if kind is None:
                break
            if kind == "doji":
                dojis += 1
                if dojis > cfg.doji_tolerance:
                    break
            else:
                strong += 1
            start = i
            i -= 1
        if start is None:
            return None
        run_len = end_idx - start + 1
        # The last candle of the spike defines the entry level; require it strong.
        if run_len < cfg.min_spike_bars or strong < cfg.min_spike_bars - cfg.doji_tolerance:
            return None
        if self._candle_fits_spike(self._bars[end_idx], direction) != "strong":
            return None
        return start

    def _has_gap(self, start: int, end: int, direction: Direction) -> bool:
        """FVG (3-candle imbalance) anywhere across the spike run."""
        if not self.cfg.require_gap:
            return True
        for k in range(max(start + 1, 2), end + 1):
            a, c = self._bars[k - 2], self._bars[k]
            if direction is Direction.LONG and c.low > a.high:
                return True
            if direction is Direction.SHORT and c.high < a.low:
                return True
        return False

    def _power_ok(self, start: int, end: int, direction: Direction) -> bool:
        if self.cfg.power_mult <= 0:
            return True
        atr = self._atr(end)
        if atr is None or atr <= 0:
            return True  # not enough history to judge; don't block
        if direction is Direction.LONG:
            move = self._bars[end].high - self._bars[start].low
        else:
            move = self._bars[start].high - self._bars[end].low
        return move >= self.cfg.power_mult * atr

    def _detect_spike(self, idx: int) -> Optional[Setup]:
        for direction in (Direction.LONG, Direction.SHORT):
            start = self._find_spike_run(idx, direction)
            if start is None:
                continue
            if not self._trend_ok(direction, self._bars[idx]):
                continue
            if not self._has_gap(start, idx, direction):
                continue
            if not self._power_ok(start, idx, direction):
                continue
            if direction is Direction.LONG:
                point_a = self._bars[start].low
                point_b = self._bars[idx].high
            else:
                point_a = self._bars[start].high
                point_b = self._bars[idx].low
            return Setup(
                direction=direction,
                spike_start_idx=start,
                spike_end_idx=idx,
                point_a=point_a,
                point_b=point_b,
                created_idx=idx,
            )
        return None

class SP2LStrategy(SpikeDetectorBase):
    """Feed bars one by one via on_bar(); returns a Signal when entry triggers."""

    tag = "sp2l"

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.state = State.IDLE
        self.setup: Optional[Setup] = None

    # ------------------------------------------------------------- lifecycle

    def _invalidated(self, bar: Bar) -> bool:
        s = self.setup
        limit = s.spike_len * self.cfg.max_retrace
        if s.direction is Direction.LONG:
            return bar.low <= s.point_b - limit
        return bar.high >= s.point_b + limit

    def _expired(self, idx: int) -> bool:
        return idx - self.setup.spike_end_idx > self.cfg.max_wait_bars

    def _try_extend_spike(self, idx: int) -> bool:
        """While waiting for pullback, a contiguous strong candle extends the
        spike and moves the entry level (point B) forward."""
        s = self.setup
        bar = self._bars[idx]
        if idx != s.spike_end_idx + 1:
            return False
        if self._candle_fits_spike(bar, s.direction) != "strong":
            return False
        s.spike_end_idx = idx
        s.point_b = bar.high if s.direction is Direction.LONG else bar.low
        return True

    def _pullback_started(self, idx: int) -> bool:
        """Correction starts when a candle trades through the previous
        candle's extreme against the spike, or simply closes against it."""
        bar, prev = self._bars[idx], self._bars[idx - 1]
        if self.setup.direction is Direction.LONG:
            return bar.low < prev.low or bar.close < bar.open
        return bar.high > prev.high or bar.close > bar.open

    def _breakout_signal(self, idx: int) -> Optional[Signal]:
        s = self.setup
        bar = self._bars[idx]
        cfg = self.cfg
        buffer = s.spike_len * cfg.sl_buffer_pct
        if s.direction is Direction.LONG and bar.high > s.point_b:
            entry = max(s.point_b, bar.open)  # gap-open fills at open
            stop = s.point_a - buffer
            target = entry + cfg.rr * (entry - stop)
            return Signal(Direction.LONG, entry, stop, target, setup=s)
        if s.direction is Direction.SHORT and bar.low < s.point_b:
            entry = min(s.point_b, bar.open)
            stop = s.point_a + buffer
            target = entry - cfg.rr * (stop - entry)
            return Signal(Direction.SHORT, entry, stop, target, setup=s)
        return None

    def _reset(self) -> None:
        self.state = State.IDLE
        self.setup = None

    # ------------------------------------------------------------------ main

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        """Process the next bar; returns a Signal when an entry triggers."""
        idx = self._append_bar(bar)

        if self.state in (State.SPIKE_CONFIRMED, State.PULLBACK):
            if self._expired(idx) or self._invalidated(bar):
                self._reset()

        if self.state is State.SPIKE_CONFIRMED:
            if self._try_extend_spike(idx):
                pass  # spike keeps running; entry level updated
            elif idx > 0 and self._pullback_started(idx):
                self.state = State.PULLBACK
                self.setup.in_pullback = True
                self.setup.pullback_start_idx = idx
                self.setup.pullback_extreme = (
                    bar.low if self.setup.direction is Direction.LONG else bar.high
                )

        if self.state is State.PULLBACK:
            s = self.setup
            if s.direction is Direction.LONG:
                s.pullback_extreme = min(s.pullback_extreme, bar.low)
            else:
                s.pullback_extreme = max(s.pullback_extreme, bar.high)
            # Entry only after the pullback matured (the intrabar order of a
            # bar that both corrects and breaks out is unknowable).
            if idx >= s.pullback_start_idx + self.cfg.min_pullback_bars:
                signal = self._breakout_signal(idx)
                if signal is not None:
                    depth_ok = True
                    if self.cfg.min_retrace > 0 and s.spike_len > 0:
                        depth = abs(s.point_b - s.pullback_extreme) / s.spike_len
                        depth_ok = depth >= self.cfg.min_retrace
                    in_session = self._in_session(bar)
                    self._reset()  # the breakout consumes the setup either way
                    if depth_ok and in_session:
                        return signal
                    return None

        if self.state is State.IDLE:
            setup = self._detect_spike(idx)
            if setup is not None:
                self.setup = setup
                self.state = State.SPIKE_CONFIRMED
        return None
