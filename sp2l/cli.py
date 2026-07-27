"""Command-line backtest runner.

Usage:
    python -m sp2l --demo                 # backtest on synthetic data
    python -m sp2l path/to/ohlcv.csv      # backtest on your own data
    python -m sp2l data.csv --rr 1.5 --min-spike-bars 4 --session-filter
"""

from __future__ import annotations

import argparse
from datetime import time

from .backtest import Backtester
from .data import load_csv, synthetic_bars
from .models import Config


def _parse_time(raw: str) -> time:
    h, m = raw.split(":")
    return time(int(h), int(m))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sp2l", description="SP2L strategy backtester")
    p.add_argument("csv", nargs="?", help="OHLCV CSV file (time,open,high,low,close[,volume])")
    p.add_argument("--demo", action="store_true", help="run on synthetic demo data")
    p.add_argument("--demo-bars", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--min-spike-bars", type=int, default=3)
    p.add_argument("--min-body-ratio", type=float, default=0.5)
    p.add_argument("--doji-tolerance", type=int, default=1)
    p.add_argument("--max-doji-body-ratio", type=float, default=0.25)
    p.add_argument("--no-gap", action="store_true", help="disable the P-Gap/FVG requirement")
    p.add_argument("--power-mult", type=float, default=1.5, help="0 disables the ATR power filter")
    p.add_argument("--atr-len", type=int, default=14)
    p.add_argument("--max-retrace", type=float, default=1.0)
    p.add_argument("--max-wait-bars", type=int, default=20)
    p.add_argument("--rr", type=float, default=1.0)
    p.add_argument("--sl-buffer-pct", type=float, default=0.0)
    p.add_argument("--scale-in", action="store_true")
    p.add_argument("--setup", choices=("both", "sp2l", "btb"), default="both",
                   help="which setups to trade (both = BTB priority, then SP2L)")
    p.add_argument("--partial", action="store_true",
                   help="take a partial profit at --partial-rr")
    p.add_argument("--partial-rr", type=float, default=1.0)
    p.add_argument("--partial-pct", type=float, default=0.5,
                   help="fraction of the position closed at the partial (0-1)")
    p.add_argument("--be", choices=("off", "rr", "after_partial"), default="off",
                   help="breakeven mode: move SL to entry when triggered")
    p.add_argument("--be-trigger-rr", type=float, default=1.0)
    p.add_argument("--be-offset-r", type=float, default=0.0,
                   help="lock this many R when moving to breakeven")
    p.add_argument("--daily", action="store_true", help="print the per-day report")
    p.add_argument("--trail-atr", type=float, default=0.0,
                   help="ATR trailing stop multiplier (0 = off)")
    p.add_argument("--min-pullback-bars", type=int, default=1,
                   help="SP2L: min bars between pullback start and breakout")
    p.add_argument("--min-retrace", type=float, default=0.0,
                   help="SP2L: min pullback depth as fraction of spike (0 = off)")
    p.add_argument("--cost-r", type=float, default=0.0,
                   help="round-trip cost per trade, in R")
    p.add_argument("--max-trades-day", type=int, default=0,
                   help="daily guard: max trades per day (0 = off)")
    p.add_argument("--max-daily-loss-r", type=float, default=0.0,
                   help="daily guard: stop trading after losing this many R in a day (0 = off)")
    p.add_argument("--btb-lookback", type=int, default=10,
                   help="bars before the spike to find the broken level")
    p.add_argument("--btb-rr", type=float, default=2.0)
    p.add_argument("--btb-max-wait-bars", type=int, default=30)
    p.add_argument("--session-filter", action="store_true")
    p.add_argument("--session-start", type=_parse_time, default=time(13, 30), metavar="HH:MM")
    p.add_argument("--session-end", type=_parse_time, default=time(20, 0), metavar="HH:MM")
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--risk", type=float, default=0.01, help="risk per trade (fraction of equity)")
    p.add_argument("--trades", action="store_true", help="print individual trades")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.demo and not args.csv:
        build_parser().error("provide a CSV file or use --demo")

    cfg = Config(
        min_spike_bars=args.min_spike_bars,
        min_body_ratio=args.min_body_ratio,
        doji_tolerance=args.doji_tolerance,
        max_doji_body_ratio=args.max_doji_body_ratio,
        require_gap=not args.no_gap,
        power_mult=args.power_mult,
        atr_len=args.atr_len,
        max_retrace=args.max_retrace,
        max_wait_bars=args.max_wait_bars,
        rr=args.rr,
        sl_buffer_pct=args.sl_buffer_pct,
        scale_in=args.scale_in,
        enabled_setups=("btb", "sp2l") if args.setup == "both" else (args.setup,),
        btb_level_lookback=args.btb_lookback,
        btb_rr=args.btb_rr,
        btb_max_wait_bars=args.btb_max_wait_bars,
        partial_enabled=args.partial,
        partial_rr=args.partial_rr,
        partial_pct=args.partial_pct,
        be_mode=args.be,
        be_trigger_rr=args.be_trigger_rr,
        be_offset_r=args.be_offset_r,
        trail_atr_mult=args.trail_atr,
        min_pullback_bars=args.min_pullback_bars,
        min_retrace=args.min_retrace,
        cost_r=args.cost_r,
        max_trades_per_day=args.max_trades_day,
        max_daily_loss_r=args.max_daily_loss_r,
        session_filter=args.session_filter,
        session_start=args.session_start,
        session_end=args.session_end,
        initial_equity=args.equity,
        risk_per_trade=args.risk,
    )

    bars = synthetic_bars(n=args.demo_bars, seed=args.seed) if args.demo else load_csv(args.csv)
    source = "synthetic demo data" if args.demo else args.csv
    print(f"SP2L backtest on {source} ({len(bars)} bars)\n")

    result = Backtester(cfg).run(bars)
    print(result.summary())

    if args.daily:
        print("\nday          trades  wins   net_pnl        R")
        for day, n, wins, pnl, r in result.daily_summary():
            print(f"{day}   {n:<7}{wins:<6}{pnl:>+10.2f}   {r:>+6.2f}R")

    if args.trades:
        print(
            "\n#   setup  dir    entry_ts              entry      stop       target"
            "     exit      pnl          R"
        )
        for i, t in enumerate(result.closed, 1):
            print(
                f"{i:<3} {t.tag:<6} {t.direction.name:<6}"
                f" {t.entry_ts:%Y-%m-%d %H:%M}     "
                f" {t.entry:<10.2f} {t.init_stop:<10.2f} {t.target:<10.2f}"
                f" {t.exit_reason:<9} {t.pnl:+10.2f}  {t.r_multiple:+6.2f}R"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
