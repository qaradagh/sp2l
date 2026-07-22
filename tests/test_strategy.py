"""Unit tests for SP2L spike detection, setup lifecycle and signals."""

from datetime import datetime, timedelta, timezone

import pytest

from sp2l.models import Bar, Config, Direction
from sp2l.strategy import SP2LStrategy, State

T0 = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)


def mk_bars(specs):
    """specs: list of (open, high, low, close) tuples -> bars, 1 min apart."""
    return [
        Bar(T0 + timedelta(minutes=i), o, h, l, c) for i, (o, h, l, c) in enumerate(specs)
    ]


def base_config(**overrides) -> Config:
    defaults = dict(power_mult=0.0, require_gap=False, min_spike_bars=3)
    defaults.update(overrides)
    return Config(**defaults)


def feed(strategy, bars):
    signals = []
    for b in bars:
        s = strategy.on_bar(b)
        if s is not None:
            signals.append(s)
    return signals


# Bullish spike: three strong candles 100->103, then pullback, then breakout.
SPIKE3 = [
    (100.0, 101.05, 99.95, 101.0),
    (101.0, 102.05, 100.95, 102.0),
    (102.0, 103.05, 101.95, 103.0),
]


class TestSpikeDetection:
    def test_bull_spike_confirmed(self):
        st = SP2LStrategy(base_config())
        feed(st, mk_bars(SPIKE3))
        assert st.state is State.SPIKE_CONFIRMED
        assert st.setup.direction is Direction.LONG
        assert st.setup.point_a == pytest.approx(99.95)
        assert st.setup.point_b == pytest.approx(103.05)

    def test_too_few_bars_no_setup(self):
        st = SP2LStrategy(base_config(min_spike_bars=4))
        feed(st, mk_bars(SPIKE3))
        assert st.state is State.IDLE

    def test_weak_bodies_no_setup(self):
        weak = [
            (100.0, 101.5, 99.5, 100.4),  # big wicks, small bodies
            (100.4, 101.9, 99.9, 100.8),
            (100.8, 102.3, 100.3, 101.2),
        ]
        st = SP2LStrategy(base_config())
        feed(st, mk_bars(weak))
        assert st.state is State.IDLE

    def test_gap_requirement(self):
        # Overlapping candles (low[2] <= high[0]): no 3-candle FVG.
        overlapping = [
            (100.0, 101.1, 99.95, 101.0),
            (101.0, 102.1, 100.5, 102.0),
            (102.0, 103.0, 101.1, 102.95),
        ]
        st = SP2LStrategy(base_config(require_gap=True))
        feed(st, mk_bars(overlapping))
        assert st.state is State.IDLE

        # Push candles apart so low[2] > high[0] -> FVG exists.
        gapped = [
            (100.0, 100.6, 99.95, 100.5),
            (100.8, 101.4, 100.75, 101.3),
            (101.6, 102.2, 101.55, 102.1),
        ]
        st2 = SP2LStrategy(base_config(require_gap=True))
        feed(st2, mk_bars(gapped))
        assert st2.state is State.SPIKE_CONFIRMED

    def test_bear_spike_confirmed(self):
        bear = [(o, h, l, c) for (c, h, l, o) in SPIKE3]  # mirror: swap open/close
        st = SP2LStrategy(base_config())
        feed(st, mk_bars(bear))
        assert st.state is State.SPIKE_CONFIRMED
        assert st.setup.direction is Direction.SHORT


class TestLifecycle:
    def test_pullback_then_breakout_signal(self):
        bars = mk_bars(
            SPIKE3
            + [
                (103.0, 103.02, 102.4, 102.5),  # pullback: breaks prev low
                (102.5, 102.9, 102.3, 102.8),
                (102.8, 103.5, 102.7, 103.4),  # breaks point B (103.05) -> signal
            ]
        )
        st = SP2LStrategy(base_config())
        signals = feed(st, bars)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction is Direction.LONG
        assert sig.entry == pytest.approx(103.05)
        assert sig.stop == pytest.approx(99.95)
        assert sig.target == pytest.approx(103.05 + (103.05 - 99.95))

    def test_invalidation_behind_point_a(self):
        bars = mk_bars(
            SPIKE3
            + [
                (103.0, 103.02, 102.4, 102.5),  # pullback starts
                (102.5, 102.6, 99.5, 99.8),  # dives below A (99.95): invalid
                (99.8, 104.0, 99.7, 103.9),  # late breakout must NOT signal
            ]
        )
        st = SP2LStrategy(base_config())
        signals = feed(st, bars)
        assert signals == []

    def test_expiry_after_max_wait_bars(self):
        flat = [(102.6, 102.75, 102.45, 102.5)] * 12  # drifting sideways
        bars = mk_bars(SPIKE3 + [(103.0, 103.02, 102.4, 102.5)] + flat)
        st = SP2LStrategy(base_config(max_wait_bars=5))
        signals = feed(st, bars)
        assert signals == []
        assert st.state is State.IDLE

    def test_spike_extension_moves_entry_level(self):
        bars = mk_bars(
            SPIKE3
            + [
                (103.0, 104.05, 102.95, 104.0),  # 4th strong candle extends spike
                (104.0, 104.02, 103.4, 103.5),  # pullback
                (103.5, 104.5, 103.4, 104.4),  # breaks new B = 104.05
            ]
        )
        st = SP2LStrategy(base_config())
        signals = feed(st, bars)
        assert len(signals) == 1
        assert signals[0].entry == pytest.approx(104.05)

    def test_session_filter_blocks_out_of_session_entry(self):
        cfg = base_config(session_filter=True)  # default 13:30-20:00 UTC
        bars = mk_bars(
            SPIKE3
            + [
                (103.0, 103.02, 102.4, 102.5),
                (102.5, 103.5, 102.4, 103.4),
            ]
        )
        # Shift all bars to 22:00 UTC (outside the session).
        late = [
            Bar(b.ts.replace(hour=22), b.open, b.high, b.low, b.close) for b in bars
        ]
        st = SP2LStrategy(cfg)
        assert feed(st, late) == []


class TestPowerFilter:
    def test_small_spike_rejected_by_atr_power(self):
        # 20 quiet bars to build ATR ~0.6, then a tiny 3-candle "spike"
        # whose total move is below power_mult * ATR.
        quiet = [(100.0 + 0.01 * i, 100.3 + 0.01 * i, 99.7 + 0.01 * i, 100.05 + 0.01 * i) for i in range(20)]
        tiny = [
            (100.2, 100.32, 100.18, 100.3),
            (100.3, 100.42, 100.28, 100.4),
            (100.4, 100.52, 100.38, 100.5),
        ]
        st = SP2LStrategy(base_config(power_mult=1.5))
        feed(st, mk_bars(quiet + tiny))
        assert st.state is State.IDLE
