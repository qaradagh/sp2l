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
from .strategy import SP2LStrategy, true_ranges

STRATEGY_REGISTRY = {"sp2l": SP2LStrategy, "btb": BTBStrategy}


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: List[float]
    config: Config
    cancelled_limits: int = 0  # limit orders that expired without filling

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
        if self.cancelled_limits:
            lines.append(f"Limits unfilled : {self.cancelled_limits}")
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
        open_trades: List[Trade] = []
        pending: List[dict] = []  # resting limit orders awaiting a fill
        cancelled_limits = 0

        # Rolling ATR for the trailing stop.
        trs = true_ranges(bars)
        atr: List[float] = []
        for i in range(len(trs)):
            window = trs[max(0, i - cfg.atr_len + 1) : i + 1]
            atr.append(sum(window) / len(window))

        # Daily risk-guard state.
        cur_day = None
        day_r = 0.0
        day_trades = 0

        def open_trade(direction, entry, stop, target, tag, idx, bar):
            nonlocal equity, day_r, day_trades
            per_unit = abs(entry - stop)
            if per_unit <= 0:
                return
            trade = Trade(
                direction=direction, entry_idx=idx, entry_ts=bar.ts,
                entry=entry, stop=stop, target=target, tag=tag,
                size=equity * cfg.risk_per_trade / per_unit,
            )
            if cfg.scale_in:
                trade.scale_in_price = entry - 0.5 * (entry - stop)
            trades.append(trade)
            open_trades.append(trade)
            day_trades += 1
            # The entry bar itself may already reach SL or TP; partial/BE
            # management starts on the next bar.
            if self._check_exit(trade, bar, idx, manage=False):
                equity += trade.pnl
                day_r += trade.r_multiple
                open_trades.remove(trade)

        for idx, bar in enumerate(bars):
            if bar.ts.date() != cur_day:
                cur_day = bar.ts.date()
                day_r = 0.0
                day_trades = 0

            # 1) manage open positions (exits before any new entry)
            for trade in list(open_trades):
                self._maybe_scale_in(trade, bar)
                if self._check_exit(trade, bar, idx, atr_val=atr[idx]):
                    equity += trade.pnl
                    day_r += trade.r_multiple
                    open_trades.remove(trade)

            # 2) fill or cancel resting limit orders placed on earlier bars
            for order in list(pending):
                filled = (
                    bar.low <= order["entry"] if order["direction"] is Direction.LONG
                    else bar.high >= order["entry"]
                )
                if filled:
                    pending.remove(order)
                    open_trade(order["direction"], order["entry"], order["stop"],
                               order["target"], order["tag"], idx, bar)
                elif idx - order["idx"] >= cfg.limit_wait_bars:
                    pending.remove(order)
                    cancelled_limits += 1

            # 3) new signals (every strategy sees every bar to stay current)
            signals = [s for s in (st.on_bar(bar) for st in strategies) if s]
            for signal in signals:
                guard_blocked = (
                    (cfg.max_trades_per_day > 0 and day_trades >= cfg.max_trades_per_day)
                    or (cfg.max_daily_loss_r > 0 and day_r <= -cfg.max_daily_loss_r)
                )
                if guard_blocked:
                    continue
                # "flat" means no open trade AND no resting order.
                if (open_trades or pending) and not cfg.allow_concurrent:
                    continue
                if signal.entry_type == "limit":
                    pending.append(dict(
                        direction=signal.direction, entry=signal.entry,
                        stop=signal.stop, target=signal.target, tag=signal.tag, idx=idx,
                    ))
                else:
                    open_trade(signal.direction, signal.entry, signal.stop,
                               signal.target, signal.tag, idx, bar)

            equity_curve.append(equity)

        # Mark-to-market close of anything still open at the end.
        if bars:
            last = bars[-1]
            for trade in list(open_trades):
                self._close(trade, len(bars) - 1, last, last.close, "eod")
                equity += trade.pnl
            equity_curve[-1] = equity
            cancelled_limits += len(pending)  # never filled by the last bar

        return BacktestResult(
            trades=trades, equity_curve=equity_curve, config=cfg,
            cancelled_limits=cancelled_limits,
        )

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

    def _check_exit(
        self,
        trade: Trade,
        bar: Bar,
        idx: int,
        manage: bool = True,
        atr_val: Optional[float] = None,
    ) -> bool:
        cfg = self.config
        sign = 1 if trade.direction is Direction.LONG else -1
        risk = trade.risk_per_unit

        def touched(price: float, from_below: bool) -> bool:
            return bar.high >= price if from_below else bar.low <= price

        # Conservative: the stop is always checked first.
        if touched(trade.stop, from_below=trade.direction is Direction.SHORT):
            stop_recovers = sign * (trade.stop - trade.entry) >= 0
            if trade.trailed and stop_recovers:
                reason = "trail-stop"
            elif trade.be_done and stop_recovers:
                reason = "be-stop"
            else:
                reason = "stop"
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
            # ATR trailing stop: only ever tightens; takes effect next bar.
            if cfg.trail_atr_mult > 0 and atr_val is not None and atr_val > 0:
                candidate = bar.close - sign * cfg.trail_atr_mult * atr_val
                tightened = (
                    max(trade.stop, candidate) if sign > 0 else min(trade.stop, candidate)
                )
                if tightened != trade.stop:
                    trade.stop = tightened
                    trade.trailed = True

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
        # Round-trip cost, expressed in R.
        trade.pnl -= self.config.cost_r * trade.risk_per_unit * trade.size
