"""Tests for allow_concurrent (take all trades) and market/limit entry mode."""

from datetime import datetime, timedelta, timezone

import pytest

from sp2l.backtest import Backtester
from sp2l.models import Bar, Config, Direction

T0 = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)


def mk_bars(specs, start=T0):
    return [
        Bar(start + timedelta(minutes=i), o, h, l, c) for i, (o, h, l, c) in enumerate(specs)
    ]


def cfg(**overrides) -> Config:
    defaults = dict(
        power_mult=0.0, require_gap=False, min_spike_bars=3, enabled_setups=("sp2l",)
    )
    defaults.update(overrides)
    return Config(**defaults)


# One SP2L long episode: entry at breakout of B=103.05, SL=99.95.
EPISODE = [
    (100.0, 101.05, 99.95, 101.0),
    (101.0, 102.05, 100.95, 102.0),
    (102.0, 103.05, 101.95, 103.0),
    (103.0, 103.02, 102.4, 102.5),   # pullback
    (102.5, 103.5, 102.4, 103.4),    # breakout -> entry
]
# Two bearish separators so a following episode is a fresh spike run.
SEP = [(103.4, 103.45, 102.9, 103.0), (103.0, 103.05, 102.5, 102.6)]


def long_episode(base):
    """An SP2L long episode built on a rising base price for uniqueness."""
    return [
        (base + 0.0, base + 1.05, base - 0.05, base + 1.0),
        (base + 1.0, base + 2.05, base + 0.95, base + 2.0),
        (base + 2.0, base + 3.05, base + 1.95, base + 3.0),
        (base + 3.0, base + 3.02, base + 2.4, base + 2.5),
        (base + 2.5, base + 3.5, base + 2.4, base + 3.4),
    ]


class TestConcurrency:
    def _overlapping(self):
        # Second spike starts while the first trade (wide target) is still open.
        first = long_episode(100.0)
        hold = [(103.4, 103.6, 103.3, 103.5)]  # first trade still running
        second = long_episode(104.0)  # new spike on top, first still open
        run_on = [(107.5, 112.0, 107.4, 111.9)]  # push both to their targets
        return mk_bars(first + hold + second + run_on)

    def test_skips_second_when_not_concurrent(self):
        result = Backtester(cfg(rr=5.0, allow_concurrent=False)).run(self._overlapping())
        assert result.total_trades == 1

    def test_takes_both_when_concurrent(self):
        result = Backtester(cfg(rr=5.0, allow_concurrent=True)).run(self._overlapping())
        assert result.total_trades == 2
        # Both trades overlap in time (second opens before first closes).
        a, b = sorted(result.closed, key=lambda t: t.entry_idx)
        assert b.entry_idx < a.exit_idx


class TestMarketEntry:
    def test_market_fills_at_breakout(self):
        result = Backtester(cfg(rr=1.0, entry_mode="market")).run(
            mk_bars(EPISODE + [(103.4, 107.0, 103.3, 106.9)])
        )
        assert result.total_trades == 1
        assert result.closed[0].entry == pytest.approx(103.05)
        assert result.cancelled_limits == 0


class TestLimitEntry:
    def test_limit_fills_on_pullback(self):
        # After the breakout, price pulls back to B (103.05) -> limit fills,
        # then rallies to the 1R target.
        bars = EPISODE + [
            (103.4, 103.45, 102.9, 103.0),   # pulls back through B -> fill at 103.05
            (103.0, 106.5, 102.95, 106.3),   # rally to target
        ]
        result = Backtester(cfg(rr=1.0, entry_mode="limit")).run(mk_bars(bars))
        assert result.total_trades == 1
        t = result.closed[0]
        assert t.entry == pytest.approx(103.05)   # filled exactly at the level
        assert t.exit_reason == "target"
        assert result.cancelled_limits == 0

    def test_limit_cancelled_when_no_pullback(self):
        # Price runs away and never returns to B within the window.
        runaway = [(103.4 + i * 0.5, 104.0 + i * 0.5, 103.3 + i * 0.5, 103.9 + i * 0.5)
                   for i in range(6)]
        result = Backtester(cfg(rr=1.0, entry_mode="limit", limit_wait_bars=3)).run(
            mk_bars(EPISODE + runaway)
        )
        assert result.total_trades == 0
        assert result.cancelled_limits == 1

    def test_market_default_unaffected(self):
        # entry_mode defaults to market -> no limit bookkeeping.
        result = Backtester(cfg(rr=1.0)).run(mk_bars(EPISODE + [(103.4, 107.0, 103.3, 106.9)]))
        assert result.total_trades == 1
        assert result.cancelled_limits == 0
