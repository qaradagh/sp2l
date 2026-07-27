"""Tests for the Pro BTB strategy and the combined SP2L+BTB backtest."""

from datetime import datetime, timedelta, timezone

import pytest

from sp2l.backtest import Backtester
from sp2l.btb import BTBState, BTBStrategy
from sp2l.models import Bar, Config, Direction

T0 = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)


def mk_bars(specs):
    return [
        Bar(T0 + timedelta(minutes=i), o, h, l, c) for i, (o, h, l, c) in enumerate(specs)
    ]


def cfg(**overrides) -> Config:
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


# Alternating pre-range so SP2L sees no spike there; resistance = 100.0.
PRE_RANGE = [
    (99.9, 99.95, 99.4, 99.5),
    (99.5, 100.0, 99.45, 99.9),
    (99.9, 99.95, 99.4, 99.5),
    (99.5, 100.0, 99.45, 99.9),
    (99.9, 99.95, 99.4, 99.5),
    (99.5, 99.95, 99.45, 99.9),
]

# Bull spike from 99.5 through the 100.0 level: A = 99.45, B = 102.55.
SPIKE = [
    (99.5, 100.55, 99.45, 100.5),
    (100.5, 101.55, 100.45, 101.5),
    (101.5, 102.55, 101.45, 102.5),
]

RETEST = [
    (102.5, 102.52, 101.8, 101.9),   # correction starts
    (101.9, 101.95, 99.98, 100.4),   # touches L = 100.0 -> limit fill
]


class TestBTBSetup:
    def test_level_found_and_await_retest(self):
        st = BTBStrategy(cfg())
        feed(st, mk_bars(PRE_RANGE + SPIKE))
        assert st.state is BTBState.AWAIT_RETEST
        assert st.setup.direction is Direction.LONG
        assert st.setup.level == pytest.approx(100.0)
        assert st.setup.point_a == pytest.approx(99.45)
        assert st.setup.point_b == pytest.approx(102.55)

    def test_no_prior_range_no_btb(self):
        # Spike starting at bar 0: nothing was broken -> no BTB setup.
        st = BTBStrategy(cfg())
        feed(st, mk_bars(SPIKE))
        assert st.state is BTBState.IDLE

    def test_level_outside_spike_body_rejected(self):
        # Prior highs below the spike origin (99.45): A < L fails. Strong
        # alternating bodies ending bearish so the spike run can't absorb them.
        low_range = [(98.6, 99.1, 98.55, 99.0), (99.0, 99.05, 98.5, 98.6)] * 3
        st = BTBStrategy(cfg())
        feed(st, mk_bars(low_range + SPIKE))
        assert st.state is BTBState.IDLE


class TestBTBEntry:
    def test_retest_limit_fill_levels(self):
        st = BTBStrategy(cfg())
        signals = feed(st, mk_bars(PRE_RANGE + SPIKE + RETEST))
        assert len(signals) == 1
        sig = signals[0]
        assert sig.tag == "btb"
        assert sig.entry == pytest.approx(100.0)      # filled at L
        assert sig.stop == pytest.approx(99.45)       # behind spike origin
        assert sig.target == pytest.approx(100.0 + 2.0 * (100.0 - 99.45))

    def test_gap_through_stop_invalidates_without_fill(self):
        gap_bar = [(99.3, 100.4, 99.2, 100.3)]  # opens below stop 99.45
        st = BTBStrategy(cfg())
        signals = feed(st, mk_bars(PRE_RANGE + SPIKE + RETEST[:1] + gap_bar))
        assert signals == []
        assert st.state is BTBState.IDLE

    def test_expiry_without_retest(self):
        drift = [(102.5, 102.7, 102.4, 102.6), (102.6, 102.8, 102.5, 102.7)] * 4
        st = BTBStrategy(cfg(btb_max_wait_bars=5))
        signals = feed(st, mk_bars(PRE_RANGE + SPIKE + drift))
        assert signals == []
        assert st.state is BTBState.IDLE

    def test_spike_extension_keeps_level(self):
        ext = [(102.5, 103.55, 102.45, 103.5)]  # contiguous strong candle
        st = BTBStrategy(cfg())
        feed(st, mk_bars(PRE_RANGE + SPIKE + ext))
        assert st.state is BTBState.AWAIT_RETEST
        assert st.setup.point_b == pytest.approx(103.55)
        assert st.setup.level == pytest.approx(100.0)


class TestCombined:
    def test_btb_trade_wins_2r(self):
        rally = [(100.4, 101.3, 100.35, 101.2)]  # reaches TP = 101.1
        result = Backtester(cfg()).run(mk_bars(PRE_RANGE + SPIKE + RETEST + rally))
        assert result.total_trades == 1
        t = result.closed[0]
        assert t.tag == "btb"
        assert t.exit_reason == "target"
        # 1% of 10k risked at 2R -> ~+200.
        assert t.pnl == pytest.approx(200.0, rel=1e-6)

    def test_sp2l_still_fires_when_no_level(self):
        # The classic SP2L flow (spike at bar 0, shallow pullback, breakout):
        # BTB has no prior range, SP2L takes the trade. The last bar is the
        # market fill bar (entry = its open).
        bars = mk_bars(
            [
                (100.0, 101.05, 99.95, 101.0),
                (101.0, 102.05, 100.95, 102.0),
                (102.0, 103.05, 101.95, 103.0),
                (103.0, 103.02, 102.4, 102.5),
                (102.5, 103.5, 102.4, 103.4),
                (103.05, 103.15, 102.95, 103.10),
            ]
        )
        result = Backtester(cfg()).run(bars)
        assert result.total_trades == 1
        assert result.closed[0].tag == "sp2l"

    def test_summary_reports_per_setup(self):
        rally = [(100.4, 101.3, 100.35, 101.2)]
        sp2l_flow = [
            (105.0, 106.05, 104.95, 106.0),
            (106.0, 107.05, 105.95, 107.0),
            (107.0, 108.05, 106.95, 108.0),
            (108.0, 108.02, 107.4, 107.5),
            (107.5, 108.5, 107.4, 108.4),
            (108.4, 112.0, 108.3, 111.9),  # TP for the SP2L trade
        ]
        result = Backtester(cfg()).run(
            mk_bars(PRE_RANGE + SPIKE + RETEST + rally + sp2l_flow)
        )
        tags = {t.tag for t in result.closed}
        assert "btb" in tags
        assert "Per setup:" in result.summary()

    def test_single_position_at_a_time(self):
        rally = [(100.4, 101.3, 100.35, 101.2)]
        result = Backtester(cfg()).run(mk_bars(PRE_RANGE + SPIKE + RETEST + rally))
        # Trades never overlap: each entry comes at or after the prior exit.
        closed = sorted(result.closed, key=lambda t: t.entry_idx)
        for a, b in zip(closed, closed[1:]):
            assert b.entry_idx >= a.exit_idx
