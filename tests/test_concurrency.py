"""Tests for the four SP2L entry-trigger options and allow_concurrent.

Entry options (break_confirm x entry_mode):
  1. touch + market -> break by shadow or close, fill at next bar's open
  2. touch + limit  -> break by shadow or close, limit resting at B
  3. close + market -> break must close beyond B, fill at next bar's open
  4. close + limit  -> break must close beyond B, limit resting at B
"""

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


# Spike to B = 103.05 (A = 99.95), then a pullback.
SPIKE = [
    (100.0, 101.05, 99.95, 101.0),
    (101.0, 102.05, 100.95, 102.0),
    (102.0, 103.05, 101.95, 103.0),
]
PULLBACK = [(103.0, 103.02, 102.4, 102.5)]

# Pokes above B = 103.05 with the shadow but closes back below it.
SHADOW_BREAK = [(102.5, 103.20, 102.4, 102.90)]
# Closes above B.
CLOSE_BREAK = [(102.5, 103.50, 102.4, 103.40)]
# Opens exactly at B: market entries fill here at 103.05; a limit resting at
# B also fills here because the bar trades down to 102.95.
FILL = [(103.05, 103.15, 102.95, 103.10)]
NEUTRAL = [(103.10, 103.20, 103.00, 103.15)]


class TestOption1TouchMarket:
    def test_shadow_break_fills_next_open(self):
        bars = mk_bars(SPIKE + PULLBACK + SHADOW_BREAK + FILL + NEUTRAL)
        result = Backtester(cfg(break_confirm="touch", entry_mode="market")).run(bars)
        assert result.total_trades == 1
        t = result.closed[0]
        assert t.entry == pytest.approx(103.05)   # the fill bar's open
        assert t.entry_idx == 5                   # the bar after the break


class TestOption2TouchLimit:
    def test_shadow_break_rests_limit_at_level(self):
        bars = mk_bars(SPIKE + PULLBACK + SHADOW_BREAK + FILL + NEUTRAL)
        result = Backtester(cfg(break_confirm="touch", entry_mode="limit")).run(bars)
        assert result.total_trades == 1
        assert result.closed[0].entry == pytest.approx(103.05)  # filled at B
        assert result.cancelled_limits == 0

    def test_limit_cancelled_if_price_never_returns(self):
        runaway = [(103.4 + i * 0.5, 104.0 + i * 0.5, 103.35 + i * 0.5, 103.9 + i * 0.5)
                   for i in range(6)]
        bars = mk_bars(SPIKE + PULLBACK + SHADOW_BREAK + runaway)
        result = Backtester(
            cfg(break_confirm="touch", entry_mode="limit", limit_wait_bars=3)
        ).run(bars)
        assert result.total_trades == 0
        assert result.cancelled_limits == 1


class TestOption3CloseMarket:
    def test_close_break_fills_next_open(self):
        bars = mk_bars(SPIKE + PULLBACK + CLOSE_BREAK + FILL + NEUTRAL)
        result = Backtester(cfg(break_confirm="close", entry_mode="market")).run(bars)
        assert result.total_trades == 1
        t = result.closed[0]
        assert t.entry == pytest.approx(103.05)
        assert t.entry_idx == 5

    def test_shadow_only_break_is_ignored(self):
        # Same bars as option 1, but close-confirm rejects the shadow poke.
        # The FILL bar closes at 103.10 (above B) so the break is only
        # confirmed there, and the entry lands one bar later.
        bars = mk_bars(SPIKE + PULLBACK + SHADOW_BREAK + FILL + NEUTRAL)
        result = Backtester(cfg(break_confirm="close", entry_mode="market")).run(bars)
        assert result.total_trades == 1
        assert result.closed[0].entry_idx == 6  # not 5: shadow poke skipped

    def test_no_trade_when_close_never_confirms(self):
        stall = [(102.9, 103.00, 102.5, 102.7)] * 3  # stays below B
        bars = mk_bars(SPIKE + PULLBACK + SHADOW_BREAK + stall)
        result = Backtester(cfg(break_confirm="close", entry_mode="market")).run(bars)
        assert result.total_trades == 0


class TestOption4CloseLimit:
    def test_close_break_then_limit_fill_at_level(self):
        bars = mk_bars(SPIKE + PULLBACK + CLOSE_BREAK + FILL + NEUTRAL)
        result = Backtester(cfg(break_confirm="close", entry_mode="limit")).run(bars)
        assert result.total_trades == 1
        assert result.closed[0].entry == pytest.approx(103.05)

    def test_limit_expires_without_a_return(self):
        runaway = [(103.5 + i * 0.5, 104.1 + i * 0.5, 103.45 + i * 0.5, 104.0 + i * 0.5)
                   for i in range(6)]
        bars = mk_bars(SPIKE + PULLBACK + CLOSE_BREAK + runaway)
        result = Backtester(
            cfg(break_confirm="close", entry_mode="limit", limit_wait_bars=3)
        ).run(bars)
        assert result.total_trades == 0
        assert result.cancelled_limits == 1


def long_episode(base):
    """A full SP2L long episode (spike, pullback, confirmed break, fill bar)."""
    return [
        (base + 0.00, base + 1.05, base - 0.05, base + 1.00),
        (base + 1.00, base + 2.05, base + 0.95, base + 2.00),
        (base + 2.00, base + 3.05, base + 1.95, base + 3.00),
        (base + 3.00, base + 3.02, base + 2.40, base + 2.50),   # pullback
        (base + 2.50, base + 3.50, base + 2.40, base + 3.40),   # confirmed break
        (base + 3.05, base + 3.15, base + 2.95, base + 3.10),   # market fill
    ]


class TestConcurrency:
    def _overlapping(self):
        # A second episode starts while the first trade (far target) is open.
        first = long_episode(100.0)
        second = long_episode(104.0)
        run_on = [(107.5, 112.0, 107.4, 111.9)]  # pushes both to their targets
        return mk_bars(first + second + run_on)

    def test_skips_second_when_not_concurrent(self):
        result = Backtester(cfg(rr=5.0, allow_concurrent=False)).run(self._overlapping())
        assert result.total_trades == 1

    def test_takes_both_when_concurrent(self):
        result = Backtester(cfg(rr=5.0, allow_concurrent=True)).run(self._overlapping())
        assert result.total_trades == 2
        # The trades overlap: the second opens before the first one closes.
        a, b = sorted(result.closed, key=lambda t: t.entry_idx)
        assert b.entry_idx < a.exit_idx
