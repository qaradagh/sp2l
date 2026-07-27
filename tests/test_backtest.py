"""Backtester tests: fills, exits, money management, end-to-end demo run."""

from datetime import datetime, timedelta, timezone

import pytest

from sp2l.backtest import Backtester
from sp2l.data import synthetic_bars
from sp2l.models import Bar, Config

T0 = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)


def mk_bars(specs):
    return [
        Bar(T0 + timedelta(minutes=i), o, h, l, c) for i, (o, h, l, c) in enumerate(specs)
    ]


def cfg(**overrides) -> Config:
    defaults = dict(power_mult=0.0, require_gap=False, min_spike_bars=3)
    defaults.update(overrides)
    return Config(**defaults)


SPIKE3 = [
    (100.0, 101.05, 99.95, 101.0),
    (101.0, 102.05, 100.95, 102.0),
    (102.0, 103.05, 101.95, 103.0),
]
PULLBACK = [(103.0, 103.02, 102.4, 102.5)]
BREAKOUT = [(102.5, 103.5, 102.4, 103.4)]
# Market entries fill at the OPEN of the bar after the confirmed break; this
# bar opens exactly at B (103.05) so the R maths below stays round.
FILL = [(103.05, 103.15, 102.95, 103.10)]


class TestTradeFlow:
    def test_winning_long_trade(self):
        # Entry 103.05, SL 99.95 (risk 3.10), TP 106.15; rally reaches it.
        rally = [
            (103.4, 105.0, 103.3, 104.9),
            (104.9, 106.5, 104.8, 106.3),
        ]
        result = Backtester(cfg()).run(mk_bars(SPIKE3 + PULLBACK + BREAKOUT + FILL + rally))
        assert result.total_trades == 1
        t = result.closed[0]
        assert t.exit_reason == "target"
        assert t.exit_price == pytest.approx(106.15)
        # 1% risk on 10k = 100 risked; RR 1:1 -> ~+100 profit.
        assert t.pnl == pytest.approx(100.0, rel=1e-6)
        assert result.win_rate == 1.0

    def test_losing_long_trade(self):
        dump = [
            (103.4, 103.45, 101.0, 101.1),
            (101.1, 101.2, 99.5, 99.6),  # through SL at 99.95
        ]
        result = Backtester(cfg()).run(mk_bars(SPIKE3 + PULLBACK + BREAKOUT + FILL + dump))
        assert result.total_trades == 1
        t = result.closed[0]
        assert t.exit_reason == "stop"
        assert t.pnl == pytest.approx(-100.0, rel=1e-6)

    def test_same_bar_sl_and_tp_counts_stop_first(self):
        # One huge bar sweeps both SL and TP after entry: conservative = stop.
        sweep = [(103.4, 107.0, 99.0, 100.0)]
        result = Backtester(cfg()).run(mk_bars(SPIKE3 + PULLBACK + BREAKOUT + FILL + sweep))
        assert result.total_trades == 1
        assert result.closed[0].exit_reason == "stop"

    def test_open_trade_closed_at_end(self):
        result = Backtester(cfg()).run(mk_bars(SPIKE3 + PULLBACK + BREAKOUT + FILL))
        assert result.total_trades == 1
        assert result.closed[0].exit_reason in ("eod", "stop", "target")

    def test_equity_curve_length_matches_bars(self):
        bars = mk_bars(SPIKE3 + PULLBACK + BREAKOUT + FILL)
        result = Backtester(cfg()).run(bars)
        assert len(result.equity_curve) == len(bars)


class TestScaleIn:
    def test_scale_in_halves_entry_distance(self):
        # After entry at 103.05 (SL 99.95), price dips to the 50% level
        # (101.50) then rallies to target.
        dip_and_rally = [
            (103.0, 103.1, 101.4, 101.6),  # touches scale-in at 101.50
            (101.6, 106.5, 101.5, 106.3),
        ]
        result = Backtester(cfg(scale_in=True)).run(
            mk_bars(SPIKE3 + PULLBACK + BREAKOUT + FILL + dip_and_rally)
        )
        assert result.total_trades == 1
        t = result.closed[0]
        assert t.scaled_in
        assert t.entry == pytest.approx((103.05 + 101.50) / 2)


class TestEndToEnd:
    def test_demo_backtest_produces_trades(self):
        bars = synthetic_bars(n=3000, seed=7)
        result = Backtester(Config()).run(bars)  # real defaults incl. gap+power
        assert result.total_trades > 5
        assert 0.0 <= result.win_rate <= 1.0
        assert result.max_drawdown < 0.5
        summary = result.summary()
        assert "Win rate" in summary and "Profit factor" in summary
