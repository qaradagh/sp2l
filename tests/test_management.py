"""Tests for partial take-profit, breakeven and the daily report."""

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


# SP2L long flow: entry 103.05, initial SL 99.95 -> R = 3.10.
# The market entry fills at the OPEN of the bar after the confirmed break, so
# the last bar here opens exactly at B (103.05) to keep the R maths round.
FLOW = [
    (100.0, 101.05, 99.95, 101.0),
    (101.0, 102.05, 100.95, 102.0),
    (102.0, 103.05, 101.95, 103.0),
    (103.0, 103.02, 102.4, 102.5),      # pullback
    (102.5, 103.5, 102.4, 103.4),       # confirmed break (closes above B)
    (103.05, 103.15, 102.95, 103.10),   # fill bar: entry = open = 103.05
]
R = 3.10
ENTRY = 103.05
TP1 = ENTRY + R          # 106.15 (partial at 1R)
TP2 = ENTRY + 2 * R      # 109.25 (final at 2R)


class TestPartial:
    def test_partial_then_stop_nets_zero(self):
        bars = FLOW + [
            (103.4, 106.2, 103.3, 106.0),   # hits TP1 -> bank 50% at +1R
            (106.0, 106.1, 99.5, 99.6),     # dumps through the original SL
        ]
        result = Backtester(cfg(rr=2.0, partial_enabled=True)).run(mk_bars(bars))
        t = result.closed[0]
        assert t.partial_done
        assert t.exit_reason == "stop"
        # +1R on half, -1R on half -> net ~0.
        assert t.pnl == pytest.approx(0.0, abs=1e-9)
        assert t.r_multiple == pytest.approx(0.0, abs=1e-9)

    def test_partial_then_final_target(self):
        bars = FLOW + [
            (103.4, 106.2, 103.3, 106.0),   # partial at +1R
            (106.0, 109.3, 105.9, 109.2),   # final TP at +2R
        ]
        result = Backtester(cfg(rr=2.0, partial_enabled=True)).run(mk_bars(bars))
        t = result.closed[0]
        assert t.exit_reason == "target"
        # +1R * 0.5 + 2R * 0.5 = +1.5R -> 150 on a 100 risk.
        assert t.pnl == pytest.approx(150.0, rel=1e-6)
        assert t.r_multiple == pytest.approx(1.5, rel=1e-6)

    def test_no_partial_on_entry_bar(self):
        # Entry bar high already exceeds TP1, but management starts next bar.
        bars = FLOW[:-1] + [(102.5, 106.5, 102.4, 106.3)] + [
            (106.3, 106.4, 99.5, 99.6),
        ]
        result = Backtester(cfg(rr=2.0, partial_enabled=True)).run(mk_bars(bars))
        t = result.closed[0]
        assert not t.partial_done
        assert t.pnl == pytest.approx(-100.0, rel=1e-6)


class TestBreakeven:
    def test_rr_trigger_moves_stop_to_entry(self):
        bars = FLOW + [
            (103.4, 106.2, 103.3, 106.0),   # reaches +1R -> SL moves to entry
            (106.0, 106.1, 102.0, 102.1),   # falls back to entry -> be-stop
        ]
        result = Backtester(cfg(rr=3.0, be_mode="rr")).run(mk_bars(bars))
        t = result.closed[0]
        assert t.be_done
        assert t.exit_reason == "be-stop"
        assert t.pnl == pytest.approx(0.0, abs=1e-9)

    def test_be_offset_locks_profit(self):
        bars = FLOW + [
            (103.4, 106.2, 103.3, 106.0),
            (106.0, 106.1, 102.0, 102.1),
        ]
        result = Backtester(cfg(rr=3.0, be_mode="rr", be_offset_r=0.25)).run(mk_bars(bars))
        t = result.closed[0]
        assert t.exit_reason == "be-stop"
        assert t.pnl == pytest.approx(25.0, rel=1e-6)  # +0.25R locked

    def test_be_after_partial(self):
        bars = FLOW + [
            (103.4, 106.2, 103.3, 106.0),   # partial banks +0.5R, SL -> entry
            (106.0, 106.1, 102.0, 102.1),   # back to entry: keep the +0.5R
        ]
        result = Backtester(
            cfg(rr=3.0, partial_enabled=True, be_mode="after_partial")
        ).run(mk_bars(bars))
        t = result.closed[0]
        assert t.partial_done and t.be_done
        assert t.exit_reason == "be-stop"
        assert t.pnl == pytest.approx(50.0, rel=1e-6)

    def test_be_never_loosens_stop(self):
        # Negative offset would place the stop below the current one; ignored.
        bars = FLOW + [
            (103.4, 106.2, 103.3, 106.0),
            (106.0, 106.1, 99.5, 99.6),
        ]
        result = Backtester(
            cfg(rr=3.0, be_mode="rr", be_offset_r=-5.0)
        ).run(mk_bars(bars))
        t = result.closed[0]
        assert t.stop >= t.init_stop


class TestDailyReport:
    def test_trades_grouped_by_exit_day(self):
        day1 = FLOW + [(103.4, 107.0, 103.3, 106.9)]  # TP same day (rr=1 -> 106.15)
        # Second episode the next day, ends at SL. Two bearish separators
        # keep day 1's rally candle from chaining into day 2's spike run.
        day2 = [
            (101.0, 101.05, 100.4, 100.5),
            (100.5, 100.55, 99.9, 100.0),
            (100.0, 101.05, 99.95, 101.0),
            (101.0, 102.05, 100.95, 102.0),
            (102.0, 103.05, 101.95, 103.0),
            (103.0, 103.02, 102.4, 102.5),
            (102.5, 103.5, 102.4, 103.4),
            (103.4, 103.45, 99.5, 99.6),
        ]
        bars = mk_bars(day1) + mk_bars(day2, start=T0 + timedelta(days=1))
        result = Backtester(cfg(rr=1.0)).run(bars)
        rows = result.daily_summary()
        assert len(rows) == 2
        d1, d2 = rows
        assert d1[0] == T0.date() and d1[1] == 1 and d1[4] == pytest.approx(1.0, rel=1e-6)
        assert d2[1] == 1 and d2[4] == pytest.approx(-1.0, rel=1e-6)
