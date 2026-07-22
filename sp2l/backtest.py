"""Bar-by-bar backtester for the SP2L / Pro BTB strategies.

Fill model (conservative):
  - Entry fills on the signal bar at the level (or at the open when the bar
    gaps past it). On the entry bar only SL/TP are checked - partial and
    breakeven management starts from the next bar.
  - When both SL and TP are touched inside the same bar, SL is assumed hit
    first. Breakeven moves apply from the bar after the trigger.
  - One position at a time; signals arriving while a trade is open are skipped.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

from .btb import BTBStrategy
from .models import Bar, Config, Direction, Trade
from .strategy import SP2LStrategy

STRATEGY_REGISTRY = {"sp2l": SP2LStrategy, "btb": BTBStrategy}


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: List[float]
    config: Config

    @property
    def closed(self) -> List[Trade]:
        return [t for t in self.trades if not t.is_open]

    @property
    def total_trades(self) -> int:
        return len(self.closed)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.closed if t.pnl > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.closed if t.pnl <= 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades else 0.0

    @property
    def net_pnl(self) -> float:
        return sum(t.pnl for t in self.closed)

    @property
    def gross_profit(self) -> float:
        return sum(t.pnl for t in self.closed if t.pnl > 0)

    @property
    def gross_loss(self) -> float:
        return sum(-t.pnl for t in self.closed if t.pnl < 0)

    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0:
            return float("inf") if self.gross_profit > 0 else 0.0
        return self.gross_profit / self.gross_loss

    @property
    def max_drawdown(self) -> float:
        """Max peak-to-trough drawdown of the equity curve, as a fraction."""
        peak = float("-inf")
        max_dd = 0.0
        for eq in self.equity_curve:
            peak = max(peak, eq)
            if peak > 0:
                max_dd = max(max_dd, (peak - eq) / peak)
        return max_dd

    @property
    def avg_r(self) -> float:
        """Average result per trade measured in R (initial risk units)."""
        rs = [
            t.pnl / (t.risk_per_unit * t.size)
            for t in self.closed
            if t.risk_per_unit > 0 and t.size > 0
        ]
        return sum(rs) / len(rs) if rs else 0.0

    def by_tag(self) -> Dict[str, List[Trade]]:
        out: Dict[str, List[Trade]] = {}
        for t in self.closed:
            out.setdefault(t.tag, []).append(t)
        return out

    def daily(self) -> "OrderedDict[date, List[Trade]]":
        """Closed trades grouped by exit day, in chronological order."""
        out: "OrderedDict[date, List[Trade]]" = OrderedDict()
        for t in sorted(self.closed, key=lambda t: t.exit_ts):
            out.setdefault(t.exit_ts.date(), []).append(t)
        return out

    def daily_summary(self) -> List[Tuple[date, int, int, float, float]]:
        """Per-day rows: (day, trades, wins, net_pnl, net_r)."""
        rows = []
        for day, ts in self.daily().items():
            wins = sum(1 for t in ts if t.pnl > 0)
            rows.append(
                (day, len(ts), wins, sum(t.pnl for t in ts), sum(t.r_multiple for t in ts))
            )
        return rows

    def summary(self) -> str:
        lines = [
            f"Trades          : {self.total_trades}",
            f"Wins / Losses   : {self.wins} / {self.losses}",
            f"Win rate        : {self.win_rate:.1%}",
            f"Net PnL         : {self.net_pnl:,.2f}",
            f"Profit factor   : {self.profit_factor:.2f}",
            f"Avg R per trade : {self.avg_r:+.2f}",
            f"Max drawdown    : {self.max_drawdown:.1%}",
            f"Final equity    : {self.equity_curve[-1]:,.2f}"
            if self.equity_curve
            else "Final equity    : n/a",
        ]
        groups = self.by_tag()
        if len(groups) > 1:
            lines.append("Per setup:")
            for tag in sorted(groups):
                ts = groups[tag]
                wins = sum(1 for t in ts if t.pnl > 0)
                net = sum(t.pnl for t in ts)
                lines.append(
                    f"  {tag:<6}: {len(ts)} trades, "
                    f"win rate {wins / len(ts):.1%}, net {net:+,.2f}"
                )
        return "\n".join(lines)


@dataclass
class Backtester:
    config: Config = field(default_factory=Config)

    def run(self, bars: Sequence[Bar]) -> BacktestResult:
        cfg = self.config
        cfg.validate()
        # Independent state machines; list order = priority when several
        # setups signal on the same bar.
        strategies = [STRATEGY_REGISTRY[name](cfg) for name in cfg.enabled_setups]
        equity = cfg.initial_equity
        trades: List[Trade] = []
        equity_curve: List[float] = []
        open_trade: Optional[Trade] = None

        for idx, bar in enumerate(bars):
            # Manage the open position first (exits before new entries).
            if open_trade is not None:
                self._maybe_scale_in(open_trade, bar)
                exited = self._check_exit(open_trade, bar, idx)
                if exited:
                    equity += open_trade.pnl
                    open_trade = None

            # Every strategy sees every bar so their states stay current.
            signals = [s for s in (st.on_bar(bar) for st in strategies) if s]
            signal = signals[0] if signals else None

            if signal is not None and open_trade is None:
                risk_amount = equity * cfg.risk_per_trade
                per_unit = abs(signal.entry - signal.stop)
                if per_unit > 0:
                    trade = Trade(
                        direction=signal.direction,
                        entry_idx=idx,
                        entry_ts=bar.ts,
                        entry=signal.entry,
                        stop=signal.stop,
                        target=signal.target,
                        tag=signal.tag,
                        size=risk_amount / per_unit,
                    )
                    if cfg.scale_in:
                        half = 0.5 * (signal.entry - signal.stop)
                        trade.scale_in_price = signal.entry - half
                    trades.append(trade)
                    open_trade = trade
                    # The signal bar itself may already reach SL or TP;
                    # partial/BE management starts on the next bar.
                    exited = self._check_exit(open_trade, bar, idx, manage=False)
                    if exited:
                        equity += open_trade.pnl
                        open_trade = None

            equity_curve.append(equity)

        # Mark-to-market close of any position left open at the end.
        if open_trade is not None and bars:
            last = bars[-1]
            self._close(open_trade, len(bars) - 1, last, last.close, "eod")
            equity += open_trade.pnl
            equity_curve[-1] = equity

        return BacktestResult(trades=trades, equity_curve=equity_curve, config=cfg)

    # ------------------------------------------------------------------ fills

    def _maybe_scale_in(self, trade: Trade, bar: Bar) -> None:
        if not self.config.scale_in or trade.scaled_in or trade.scale_in_price is None:
            return
        price = trade.scale_in_price
        touched = (
            bar.low <= price if trade.direction is Direction.LONG else bar.high >= price
        )
        if touched:
            # Second unit of the same size at the better price.
            trade.entry = (trade.entry + price) / 2
            trade.size *= 2
            trade.scaled_in = True

    def _check_exit(self, trade: Trade, bar: Bar, idx: int, manage: bool = True) -> bool:
        cfg = self.config
        sign = 1 if trade.direction is Direction.LONG else -1
        risk = trade.risk_per_unit

        def touched(price: float, from_below: bool) -> bool:
            return bar.high >= price if from_below else bar.low <= price

        # Conservative: the stop is always checked first.
        if touched(trade.stop, from_below=trade.direction is Direction.SHORT):
            stop_recovers = sign * (trade.stop - trade.entry) >= 0
            reason = "be-stop" if trade.be_done and stop_recovers else "stop"
            self._close(trade, idx, bar, trade.stop, reason)
            return True

        if manage and risk > 0:
            # Partial take-profit at partial_rr R.
            if cfg.partial_enabled and not trade.partial_done:
                tp1 = trade.entry + sign * cfg.partial_rr * risk
                if touched(tp1, from_below=trade.direction is Direction.LONG):
                    trade.realized_pnl += sign * (tp1 - trade.entry) * trade.size * cfg.partial_pct
                    trade.partial_done = True
                    if cfg.be_mode == "after_partial" and not trade.be_done:
                        self._move_to_breakeven(trade, sign, risk)
            # Breakeven trigger by R level.
            if cfg.be_mode == "rr" and not trade.be_done:
                trigger = trade.entry + sign * cfg.be_trigger_rr * risk
                if touched(trigger, from_below=trade.direction is Direction.LONG):
                    self._move_to_breakeven(trade, sign, risk)

        if touched(trade.target, from_below=trade.direction is Direction.LONG):
            self._close(trade, idx, bar, trade.target, "target")
            return True
        return False

    def _move_to_breakeven(self, trade: Trade, sign: int, risk: float) -> None:
        """Move the stop to entry + offset, never loosening it."""
        new_stop = trade.entry + sign * self.config.be_offset_r * risk
        trade.stop = max(trade.stop, new_stop) if sign > 0 else min(trade.stop, new_stop)
        trade.be_done = True

    def _close(self, trade: Trade, idx: int, bar: Bar, price: float, reason: str) -> None:
        trade.exit_idx = idx
        trade.exit_ts = bar.ts
        trade.exit_price = price
        trade.exit_reason = reason
        sign = 1 if trade.direction is Direction.LONG else -1
        remaining = 1.0 - self.config.partial_pct if trade.partial_done else 1.0
        trade.pnl = trade.realized_pnl + sign * (price - trade.entry) * trade.size * remaining
