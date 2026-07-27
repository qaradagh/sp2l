"""Pro BTB (Professional Back To Breakeven) state machine.

Long setup (short is the exact mirror):
  1. Spike (same detection as SP2L) that breaks a key level L — the highest
     high of the btb_level_lookback bars before the spike — with A < L < B.
  2. Retest: price returns to L; a limit order at L fills there
     (or at the open when the bar gaps below L but above the stop).
  3. SL behind the spike origin (A), TP at btb_rr * risk (default 2R).

See docs/BTB_BLUEPRINT.md for the rules and the assumptions made.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from .models import Bar, Config, Direction, Setup, Signal
from .strategy import SpikeDetectorBase


class BTBState(Enum):
    IDLE = "idle"
    AWAIT_RETEST = "await_retest"


class BTBStrategy(SpikeDetectorBase):
    """Feed bars via on_bar(); returns a Signal when the retest entry fills."""

    tag = "btb"

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.state = BTBState.IDLE
        self.setup: Optional[Setup] = None

    # ------------------------------------------------------------------ level

    def _find_level(self, setup: Setup) -> Optional[float]:
        """The broken key level: prior-range extreme inside the spike body."""
        start = setup.spike_start_idx
        window = self._bars[max(0, start - self.cfg.btb_level_lookback) : start]
        if not window:
            return None
        if setup.direction is Direction.LONG:
            level = max(b.high for b in window)
            if setup.point_a < level < setup.point_b:
                return level
        else:
            level = min(b.low for b in window)
            if setup.point_b < level < setup.point_a:
                return level
        return None

    # -------------------------------------------------------------- lifecycle

    def _reset(self) -> None:
        self.state = BTBState.IDLE
        self.setup = None

    def _retest_signal(self, bar: Bar) -> Optional[Signal]:
        """Emits the limit-fill signal, or resets on a gap through the stop.
        Returns None when the level was not touched or the setup died."""
        s = self.setup
        cfg = self.cfg
        buffer = s.spike_len * cfg.sl_buffer_pct
        if s.direction is Direction.LONG:
            if bar.low > s.level:
                return None
            stop = s.point_a - buffer
            if bar.open <= stop:  # gapped through the setup: assume no fill
                self._reset()
                return None
            entry = min(s.level, bar.open)
            target = entry + cfg.btb_rr * (entry - stop)
            return Signal(Direction.LONG, entry, stop, target, tag=self.tag,
                          entry_type="now", rr=cfg.btb_rr, setup=s)
        if bar.high < s.level:
            return None
        stop = s.point_a + buffer
        if bar.open >= stop:
            self._reset()
            return None
        entry = max(s.level, bar.open)
        target = entry - cfg.btb_rr * (stop - entry)
        return Signal(Direction.SHORT, entry, stop, target, tag=self.tag,
                      entry_type="now", rr=cfg.btb_rr, setup=s)

    # ------------------------------------------------------------------ main

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        idx = self._append_bar(bar)

        if self.state is BTBState.AWAIT_RETEST:
            s = self.setup
            if idx - s.spike_end_idx > self.cfg.btb_max_wait_bars:
                self._reset()
            elif (
                idx == s.spike_end_idx + 1
                and self._candle_fits_spike(bar, s.direction) == "strong"
            ):
                # Contiguous continuation extends the spike; L and A stay put.
                s.spike_end_idx = idx
                s.point_b = bar.high if s.direction is Direction.LONG else bar.low
            else:
                signal = self._retest_signal(bar)
                if signal is not None:
                    in_session = self._in_session(bar)
                    self._reset()
                    if in_session:
                        return signal

        if self.state is BTBState.IDLE:
            setup = self._detect_spike(idx)
            if setup is not None:
                level = self._find_level(setup)
                if level is not None:
                    setup.level = level
                    self.setup = setup
                    self.state = BTBState.AWAIT_RETEST
        return None
