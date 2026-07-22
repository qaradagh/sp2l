"""Tests for the trend filter, trailing stop, costs, daily guard and
pullback-quality filters."""

from datetime import datetime, timedelta, timezone

import pytest

from sp2l.backtest import Backtester
from sp2l.models import Bar, Config

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


# SP2L long flow: entry 103.05, SL 99.95, R = 3.10.
FLOW = [
    (100.0, 101.05, 99.95, 101.0),
    (101.0, 102.05, 100.95, 102.0),
    (102.0, 103.05, 101.95, 103.0),
    (103.0, 103.02, 102.4, 102.5),   # pullback (depth 0.65 -> ~21% of spike)
    (102.5, 103.5, 102.4, 103.4),    # breakout
]
WIN = [(103.4, 107.0, 103.3, 106.9)]  # reaches 1R target 106.15


class TestTrendFilter:
    # Flat bars far above the spike price keep the EMA high, so a long
    # spike at ~100 sits below trend.
    HIGH_RANGE = [(200.0, 200.6, 199.4, 200.1), (200.1, 200.7, 199.5, 200.0)] * 15

    def test_long_below_ema_blocked(self):
        result = Backtester(cfg(trend_ema_len=20)).run(mk_bars(self.HIGH_RANGE + FLOW + WIN))
        assert result.total_trades == 0

    def test_disabled_filter_allows_trade(self):
        result = Backtester(cfg(trend_ema_len=0)).run(mk_bars(self.HIGH_RANGE + FLOW + WIN))
        assert result.total_trades == 1


class TestTrailingStop:
    def test_trail_locks_profit(self):
        # Far target keeps the trade open; the rally drags the trail up and
        # the drop then exits at a profit via the trailed stop.
        bars = FLOW + [
            (103.4, 106.0, 103.35, 105.9),
            (105.9, 108.0, 105.85, 107.9),
            (107.9, 107.95, 104.0, 104.1),
        ]
        result = Backtester(cfg(rr=10.0, trail_atr_mult=1.0)).run(mk_bars(bars))
        assert result.total_trades == 1
        t = result.closed[0]
        assert t.trailed
        assert t.exit_reason == "trail-stop"
        assert t.pnl > 0

    def test_trail_never_loosens(self):
        bars = FLOW + [(103.4, 103.45, 99.5, 99.6)]  # straight dump
        result = Backtester(cfg(rr=10.0, trail_atr_mult=50.0)).run(mk_bars(bars))
        t = result.closed[0]
        # A huge ATR multiple would put the trail below the initial stop;
        # the stop must never move down.
        assert t.exit_price == pytest.approx(99.95)


class TestCost:
    def test_cost_reduces_pnl(self):
        result = Backtester(cfg(rr=1.0, cost_r=0.1)).run(mk_bars(FLOW + WIN))
        t = result.closed[0]
        assert t.pnl == pytest.approx(90.0, rel=1e-6)   # +1R - 0.1R on 100 risk
        assert t.r_multiple == pytest.approx(0.9, rel=1e-6)


DAY2_SEP = [(101.0, 101.05, 100.4, 100.5), (100.5, 100.55, 99.9, 100.0)]
# Hits the SL but closes off the low (body < 50% of range) so the dump candle
# cannot seed a bearish spike run of its own.
LOSS = [(103.4, 103.45, 99.5, 101.5)]


class TestDailyGuard:
    def _two_episode_day(self):
        # Two full SP2L episodes inside one day; both would trade unguarded.
        return mk_bars(FLOW + LOSS + DAY2_SEP + FLOW + WIN)

    def test_unguarded_trades_twice(self):
        result = Backtester(cfg(rr=1.0)).run(self._two_episode_day())
        assert result.total_trades == 2

    def test_max_trades_per_day(self):
        result = Backtester(cfg(rr=1.0, max_trades_per_day=1)).run(self._two_episode_day())
        assert result.total_trades == 1

    def test_max_daily_loss_blocks_after_losing_r(self):
        result = Backtester(cfg(rr=1.0, max_daily_loss_r=0.5)).run(self._two_episode_day())
        # First trade loses 1R -> guard trips -> second entry blocked.
        assert result.total_trades == 1
        assert result.closed[0].pnl < 0

    def test_guard_resets_next_day(self):
        day1 = mk_bars(FLOW + LOSS)
        day2 = mk_bars(DAY2_SEP + FLOW + WIN, start=T0 + timedelta(days=1))
        result = Backtester(cfg(rr=1.0, max_daily_loss_r=0.5)).run(day1 + day2)
        assert result.total_trades == 2


class TestPullbackQuality:
    def test_min_retrace_blocks_shallow_pullback(self):
        # FLOW pullback depth ~21% of the spike; a 30% floor blocks it.
        result = Backtester(cfg(min_retrace=0.30)).run(mk_bars(FLOW + WIN))
        assert result.total_trades == 0

    def test_min_retrace_pass(self):
        result = Backtester(cfg(min_retrace=0.15)).run(mk_bars(FLOW + WIN))
        assert result.total_trades == 1

    def test_min_pullback_bars_blocks_fast_breakout(self):
        # Breakout arrives one bar after the pullback started.
        result = Backtester(cfg(min_pullback_bars=3)).run(mk_bars(FLOW + WIN))
        assert result.total_trades == 0
