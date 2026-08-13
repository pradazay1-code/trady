"""Event-driven backtester.

Walks bars forward one at a time and only ever shows the strategy data up to the
current bar, so a signal cannot see its own future. Entries fill on the *next*
bar's open, never the signal bar's close — that one detail is the difference
between a backtest and a fantasy.

Exit precedence within a bar is pessimistic: if a bar's range contains both the
stop and the target, the stop is assumed to hit first. Intrabar order is
unknowable from OHLC, so the assumption is the one that hurts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time

import numpy as np
import pandas as pd

from . import strategy as strat
from .brokers import Order, PaperBroker, Position, estimate_fees
from .config import Config
from .indicators import enrich
from .journal import Journal, compute_stats, stats_by
from .risk import AccountState, PDTTracker, RiskManager, trail_stop


@dataclass
class BacktestResult:
    stats: dict
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    by_strategy: pd.DataFrame
    by_symbol: pd.DataFrame
    rejections: dict[str, int]
    config_summary: dict
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        s = self.stats
        lines = [
            "BACKTEST RESULT",
            "=" * 60,
            f"  trades .............. {s['trades']}",
            f"  win rate ............ {s['win_rate']:.1%}  ({s['wins']}W / {s['losses']}L)",
            f"  net P&L ............. ${s['net_pnl']:,.2f}",
            f"  gross P&L ........... ${s['gross_pnl']:,.2f}",
            f"  fees ................ ${s['fees']:,.2f}",
            f"  expectancy/trade .... {s['expectancy']:.4%}",
            f"  average R ........... {s['avg_r']:.3f}",
            f"  profit factor ....... {s['profit_factor']}",
            f"  max drawdown ........ {s['max_drawdown']:.1%}",
            f"  Sharpe (ann.) ....... {s['sharpe']}",
            f"  worst losing streak . {s['max_consecutive_losses']}",
            f"  day trades .......... {s['day_trades']}",
        ]
        if self.rejections:
            lines += ["", "  signals rejected by reason:"]
            for reason, n in sorted(
                self.rejections.items(), key=lambda kv: kv[1], reverse=True
            )[:8]:
                lines.append(f"    {n:>5}  {reason[:70]}")
        if not self.by_strategy.empty:
            lines += ["", "  by strategy:", "  " + self.by_strategy.to_string(index=False).replace("\n", "\n  ")]
        if self.warnings:
            lines += ["", "  warnings:"] + [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)


def _reason_bucket(reason: str) -> str:
    """Collapse rejection messages into countable buckets."""
    r = reason.lower()
    for key, label in (
        ("confluence", "confluence below minimum"),
        ("reward:risk", "reward:risk too low"),
        ("volume confirmation", "no volume confirmation"),
        ("against a", "counter-trend without reversal setup"),
        ("sizing", "position size resolved to zero"),
        ("thin", "insufficient liquidity"),
        ("outside tradable", "price outside band"),
    ):
        if key in r:
            return label
    return reason[:60]


class Backtester:
    def __init__(self, cfg: Config, journal: Journal | None = None):
        self.cfg = cfg
        self.journal = journal

    # -----------------------------------------------------------------
    def run(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        warmup: int = 120,
        record: bool = False,
        verbose: bool = False,
    ) -> BacktestResult:
        cfg = self.cfg
        equity0 = cfg.risk.starting_equity
        broker = PaperBroker(cfg.execution, equity0)
        pdt = PDTTracker(cfg.pdt)
        rm = RiskManager(cfg, pdt)

        enriched = {s: enrich(df) for s, df in frames.items() if len(df) > warmup}
        if not enriched:
            return BacktestResult(
                compute_stats(pd.DataFrame()), pd.DataFrame(), pd.DataFrame(),
                pd.DataFrame(), pd.DataFrame(), {}, {},
                ["no symbol had enough history to backtest"],
            )

        # Unified, sorted timeline across all symbols.
        timeline = sorted(set().union(*[set(df.index) for df in enriched.values()]))
        timeline = [t for t in timeline if t >= min(
            df.index[warmup] for df in enriched.values() if len(df) > warmup
        )]

        open_positions: dict[str, Position] = {}
        pending: list[tuple[str, strat.Signal]] = []
        closed: list[dict] = []
        marks: list[dict] = []
        rejections: dict[str, int] = {}
        warnings: list[str] = []

        current_day = None
        day_start_equity = equity0
        realized_today = 0.0
        trades_today = 0
        consecutive_losses = 0
        peak_equity = equity0
        halted = False
        halt_reason = ""

        def _fees(side, qty, px):
            return estimate_fees(cfg.execution, side, qty, px)

        for ts in timeline:
            day = ts.date()
            if day != current_day:
                # New session: reset daily counters, unhalt daily-scope halts.
                current_day = day
                day_start_equity = broker.equity
                realized_today = 0.0
                trades_today = 0
                # The cool-off is a within-day brake, not a permanent ban. Without
                # this reset a 3-loss day would freeze the agent forever, since no
                # further trade could ever produce the win that clears it.
                consecutive_losses = 0
                if halted and "drawdown" not in halt_reason:
                    halted, halt_reason = False, ""

            # Current bar per symbol.
            bars = {}
            for sym, df in enriched.items():
                if ts in df.index:
                    bars[sym] = df.loc[ts]
            if not bars:
                continue
            broker.mark({s: float(b["close"]) for s, b in bars.items()})

            # ---------------- fill pending entries at this bar's open -----
            for sym, sig in pending:
                bar = bars.get(sym)
                if bar is None:
                    continue
                px = float(bar["open"])
                qty = sig.sizing.shares
                if qty <= 0:
                    continue
                side = "buy" if sig.direction == "long" else "sell_short"
                order = Order(sym, side, qty, "market", note=sig.strategy)
                fill = broker.submit(order, px, ts)
                if fill is None:
                    continue
                pos = Position(
                    sym, sig.direction, qty, fill.price, ts,
                    stop=sig.stop, target=sig.target, strategy=sig.strategy,
                )
                open_positions[sym] = pos
                trades_today += 1
                if record and self.journal:
                    sid = self.journal.record_signal(sig, taken=True)
                    pos.trade_id = self.journal.open_trade(
                        sym, sig.direction, qty, fill.price, ts,
                        strategy=sig.strategy, signal_id=sid,
                        planned_stop=sig.stop, planned_target=sig.target,
                        broker="paper(backtest)",
                    )
            pending.clear()

            # ---------------- manage open positions ----------------------
            for sym in list(open_positions):
                pos = open_positions[sym]
                bar = bars.get(sym)
                if bar is None:
                    continue
                hi, lo, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
                pos.update_excursion(hi, lo)
                pos.bars_held += 1

                exit_px = exit_reason = None
                if pos.direction == "long":
                    if pos.stop is not None and lo <= pos.stop:
                        exit_px, exit_reason = pos.stop, "stop"
                    elif pos.target is not None and hi >= pos.target:
                        exit_px, exit_reason = pos.target, "target"
                else:
                    if pos.stop is not None and hi >= pos.stop:
                        exit_px, exit_reason = pos.stop, "stop"
                    elif pos.target is not None and lo <= pos.target:
                        exit_px, exit_reason = pos.target, "target"

                # Force flat before the close — day trades do not hold overnight.
                if exit_px is None and isinstance(ts, pd.Timestamp):
                    h, m = cfg.session.force_flat_at.split(":")
                    if ts.time() >= time(int(h), int(m)):
                        exit_px, exit_reason = close, "eod_flat"

                if exit_px is None:
                    # Ratchet the stop; move to breakeven after +1R.
                    a = float(bar.get("atr_14") or 0) or close * 0.01
                    risk = abs(pos.entry_price - (pos.stop or pos.entry_price))
                    if risk > 0 and cfg.risk.breakeven_at_r > 0:
                        gain = (close - pos.entry_price) if pos.direction == "long" \
                            else (pos.entry_price - close)
                        if gain >= cfg.risk.breakeven_at_r * risk:
                            pos.stop = (
                                max(pos.stop, pos.entry_price) if pos.direction == "long"
                                else min(pos.stop, pos.entry_price)
                            )
                    if pos.stop is not None:
                        pos.stop = trail_stop(pos.stop, close, a, pos.direction, cfg.risk)
                    continue

                # Close it.
                side = "sell" if pos.direction == "long" else "buy_to_cover"
                order = Order(sym, side, pos.qty, "market")
                fill = broker.submit(order, exit_px, ts)
                fee = fill.fees if fill else _fees(side, pos.qty, exit_px)
                px = fill.price if fill else exit_px

                sign = 1.0 if pos.direction == "long" else -1.0
                gross = (px - pos.entry_price) * pos.qty * sign
                entry_fee = _fees("buy" if pos.direction == "long" else "sell_short",
                                  pos.qty, pos.entry_price)
                net = gross - fee - entry_fee
                risk_ps = abs(pos.entry_price - (pos.stop or pos.entry_price))
                same_day = pos.entry_ts.date() == ts.date()

                closed.append({
                    "symbol": sym, "direction": pos.direction, "strategy": pos.strategy,
                    "qty": pos.qty, "entry_ts": pos.entry_ts, "entry_price": pos.entry_price,
                    "exit_ts": ts, "exit_price": px, "exit_reason": exit_reason,
                    "gross_pnl": gross, "fees": fee + entry_fee, "net_pnl": net,
                    "pnl_pct": net / (pos.entry_price * pos.qty),
                    "r_multiple": net / (risk_ps * pos.qty) if risk_ps > 0 else np.nan,
                    "mae": pos.mae, "mfe": pos.mfe, "bars_held": pos.bars_held,
                    "is_day_trade": int(same_day),
                })
                realized_today += net
                consecutive_losses = consecutive_losses + 1 if net <= 0 else 0
                if same_day:
                    pdt.record(sym, ts.date())
                if record and self.journal and pos.trade_id:
                    self.journal.close_trade(
                        pos.trade_id, px, ts, exit_reason, fees=fee + entry_fee,
                        mae=pos.mae, mfe=pos.mfe, bars_held=pos.bars_held,
                    )
                open_positions.pop(sym, None)

            # ---------------- equity + halts -----------------------------
            eq = broker.equity
            peak_equity = max(peak_equity, eq)
            state = AccountState(
                equity=eq, starting_equity_today=day_start_equity,
                peak_equity=peak_equity, open_positions=len(open_positions),
                gross_exposure=sum(abs(p.qty * float(bars[s]["close"]))
                                   for s, p in open_positions.items() if s in bars),
                realized_pnl_today=realized_today, trades_today=trades_today,
                consecutive_losses=consecutive_losses, halted=halted,
                halt_reason=halt_reason,
            )
            if not halted:
                should, why = rm.should_halt(state)
                if should:
                    halted, halt_reason = True, why
                    state.halted, state.halt_reason = True, why
                    if record and self.journal:
                        self.journal.log("halt", f"{ts}: {why}")

            marks.append({"ts": ts, "equity": eq, "cash": broker.cash,
                          "open_positions": len(open_positions),
                          "realized_pnl_today": realized_today})

            # ---------------- look for new entries -----------------------
            if halted:
                continue
            for sym, df in enriched.items():
                if sym in open_positions or sym not in bars:
                    continue
                hist = df.loc[:ts]
                if len(hist) < warmup:
                    continue
                try:
                    sig = strat.generate(
                        sym, hist, cfg, eq,
                        stats=compute_stats(pd.DataFrame(closed), base_equity=equity0),
                    )
                except Exception as exc:
                    warnings.append(f"{sym} @ {ts}: {type(exc).__name__}: {exc}")
                    continue
                if sig is None:
                    continue
                if sig.rejected_reason:
                    b = _reason_bucket(sig.rejected_reason)
                    rejections[b] = rejections.get(b, 0) + 1
                    continue

                decision = rm.check(
                    state, now=ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts,
                    intends_same_day_exit=True,
                    new_notional=sig.sizing.notional, symbol=sym,
                )
                if not decision.allowed:
                    b = _reason_bucket(decision.reason)
                    rejections[b] = rejections.get(b, 0) + 1
                    continue

                pending.append((sym, sig))
                if verbose:
                    print(f"  {ts} SIGNAL {sig.direction} {sym} score={sig.score:.2f} "
                          f"{sig.strategy}")
                break  # one new entry per bar keeps sizing honest

        # ---------------- assemble ---------------------------------------
        trades_df = pd.DataFrame(closed)
        equity_df = pd.DataFrame(marks)
        stats = compute_stats(trades_df, base_equity=equity0)
        stats["starting_equity"] = equity0
        stats["ending_equity"] = round(broker.equity, 2)
        stats["return_pct"] = round((broker.equity / equity0 - 1.0), 4) if equity0 else 0.0

        if stats["trades"] < 30:
            warnings.append(
                f"only {stats['trades']} trades — too few to conclude anything about edge"
            )
        if not trades_df.empty and trades_df["is_day_trade"].sum() > 0:
            warnings.append(
                "backtest ignores the PDT budget across days unless equity < $25k was "
                "simulated; check pdt settings before trading this live"
            )

        return BacktestResult(
            stats=stats,
            trades=trades_df,
            equity_curve=equity_df,
            by_strategy=stats_by(trades_df, "strategy", base_equity=equity0),
            by_symbol=stats_by(trades_df, "symbol", base_equity=equity0),
            rejections=rejections,
            config_summary={
                "risk_per_trade_pct": cfg.risk.risk_per_trade_pct,
                "max_position_pct": cfg.risk.max_position_pct,
                "min_confluence_score": cfg.signal.min_confluence_score,
                "min_reward_risk": cfg.risk.min_reward_risk,
                "stop_atr_multiple": cfg.risk.stop_atr_multiple,
                "target_atr_multiple": cfg.risk.target_atr_multiple,
            },
            warnings=warnings[:10],
        )


# =====================================================================
#  Walk-forward
# =====================================================================
def walk_forward(
    cfg: Config, frames: dict[str, pd.DataFrame], splits: int = 4, warmup: int = 120
) -> pd.DataFrame:
    """Run the same config over sequential slices.

    A strategy that only works in one slice is curve-fitted, which the books
    call out by name: over-optimisation produces a model that describes the past
    and predicts nothing.
    """
    rows = []
    for sym, df in frames.items():
        n = len(df)
        if n < warmup * 2 + splits * 50:
            continue
        bounds = np.linspace(0, n, splits + 1, dtype=int)
        for i in range(splits):
            lo, hi = bounds[i], bounds[i + 1]
            if hi - lo < warmup + 50:
                continue
            sl = df.iloc[max(0, lo - warmup) : hi]
            res = Backtester(cfg).run({sym: sl}, warmup=warmup)
            rows.append({
                "symbol": sym, "split": i + 1,
                "from": str(sl.index[0])[:16], "to": str(sl.index[-1])[:16],
                **{k: res.stats[k] for k in
                   ("trades", "win_rate", "expectancy", "net_pnl", "max_drawdown", "avg_r")},
            })
    return pd.DataFrame(rows)
