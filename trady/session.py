"""The trading session — what the agent does during a market day.

One `TradingSession` owns a day: it loads data, scans for signals, passes each
through the risk gate, sizes it, routes the order to whichever broker is
configured, manages open positions bar by bar, forces flat before the close, and
writes everything to the journal. At the end of the day it runs the review that
updates its own parameters.

`run_once()` is a single pass, suitable for a cron job every N minutes.
`run_day()` loops until the close. Both are safe to interrupt: state lives in the
journal, so restarting picks up where it left off.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

from . import data as datamod
from . import strategy as strat
from .brokers import Broker, FidelityBridge, Order, PaperBroker, Position, make_broker
from .config import Config
from .indicators import enrich
from .journal import Journal, compute_stats
from .learn import Learner
from .reporting import daily_report, save_daily
from .risk import AccountState, PDTTracker, RiskManager, trail_stop


def _parse_time(hhmm: str) -> time:
    h, m = hhmm.split(":")
    return time(int(h), int(m))


@dataclass
class SessionState:
    day: date
    starting_equity: float
    realized_today: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    peak_equity: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    positions: dict[str, Position] = field(default_factory=dict)


class TradingSession:
    def __init__(
        self,
        cfg: Config,
        journal: Journal | None = None,
        broker: Broker | None = None,
        dry_run: bool = True,
        notifier=None,
    ):
        cfg.ensure_dirs()
        self.cfg = cfg
        self.journal = journal or Journal(cfg.journal_db)
        self.broker = broker or make_broker(cfg, cfg.risk.starting_equity)
        self.dry_run = dry_run
        self.notifier = notifier

        # Whether alerts may be presented as live-tradable, or must carry the
        # PAPER MODE stamp. Decided by measured performance, not by a flag.
        from .notify import AlertGate

        self.gate = AlertGate(cfg, self.journal).evaluate()

        self.pdt = PDTTracker(cfg.pdt)
        self.pdt.load(self.journal.day_trades())
        self.risk = RiskManager(cfg, self.pdt)
        self.learner = Learner(cfg, self.journal)
        self.cache = datamod.BarCache(Path(cfg.data_dir) / "bars.sqlite")

        eq = self._equity()
        self.state = SessionState(
            day=date.today(), starting_equity=eq, peak_equity=eq
        )
        self._restore_open_positions()

    # -----------------------------------------------------------------
    def _equity(self) -> float:
        try:
            return float(self.broker.equity) or self.cfg.risk.starting_equity
        except Exception:
            return self.cfg.risk.starting_equity

    def _restore_open_positions(self) -> None:
        """Re-hydrate positions from the journal after a restart."""
        df = self.journal.open_trades()
        for _, t in df.iterrows():
            self.state.positions[t["symbol"]] = Position(
                symbol=t["symbol"],
                direction=t["direction"],
                qty=int(t["qty"]),
                entry_price=float(t["entry_price"]),
                entry_ts=pd.to_datetime(t["entry_ts"]).to_pydatetime(),
                stop=t["planned_stop"],
                target=t["planned_target"],
                trade_id=int(t["id"]),
                strategy=t["strategy"] or "",
            )
        if len(df):
            self.journal.log(
                "info", f"restored {len(df)} open position(s) from the journal"
            )

    # -----------------------------------------------------------------
    def _roll_day(self, now: datetime) -> None:
        """Start a fresh session when the date changes.

        Daily counters reset, and so does the consecutive-loss cool-off: it is a
        within-day brake on tilt, not a permanent ban. A daily-loss halt clears
        too; a drawdown halt does not, because that one is about the account, not
        the day.
        """
        if now.date() == self.state.day:
            return
        eq = self._equity()
        self.state.day = now.date()
        self.state.starting_equity = eq
        self.state.realized_today = 0.0
        self.state.trades_today = 0
        self.state.consecutive_losses = 0
        self.state.peak_equity = max(self.state.peak_equity, eq)
        if self.state.halted and "drawdown" not in self.state.halt_reason:
            self.state.halted, self.state.halt_reason = False, ""
            self.journal.log("resume", "new session — daily halt cleared")
        self.pdt.load(self.journal.day_trades())

    def load_frames(self, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
        symbols = symbols or self.cfg.watchlist
        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = datamod.load(
                    sym, provider=self.cfg.data_provider, interval=self.cfg.bar_interval,
                    days=self.cfg.history_days, cache=self.cache,
                )
                if len(df) >= 60:
                    out[sym] = enrich(df)
            except Exception as exc:
                self.journal.log("error", f"data load failed for {sym}: {exc}")
        return out

    # -----------------------------------------------------------------
    def account_state(self, prices: dict[str, float] | None = None) -> AccountState:
        prices = prices or {}
        eq = self._equity()
        self.state.peak_equity = max(self.state.peak_equity, eq)
        gross = sum(
            abs(p.qty * prices.get(s, p.entry_price)) for s, p in self.state.positions.items()
        )
        return AccountState(
            equity=eq,
            starting_equity_today=self.state.starting_equity,
            peak_equity=self.state.peak_equity,
            open_positions=len(self.state.positions),
            gross_exposure=gross,
            realized_pnl_today=self.state.realized_today,
            trades_today=self.state.trades_today,
            consecutive_losses=self.state.consecutive_losses,
            halted=self.state.halted,
            halt_reason=self.state.halt_reason,
            open_symbols=list(self.state.positions),
        )

    # -----------------------------------------------------------------
    def manage_positions(self, frames: dict[str, pd.DataFrame], now: datetime) -> list[dict]:
        """Check stops, targets and the force-flat clock on every open position."""
        closed: list[dict] = []
        flat_at = _parse_time(self.cfg.session.force_flat_at)

        for sym in list(self.state.positions):
            pos = self.state.positions[sym]
            df = frames.get(sym)
            if df is None or df.empty:
                continue
            bar = df.iloc[-1]
            hi, lo, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
            pos.update_excursion(hi, lo)

            exit_px = exit_reason = None
            if pos.direction == "long":
                if pos.stop is not None and lo <= pos.stop:
                    exit_px, exit_reason = float(pos.stop), "stop"
                elif pos.target is not None and hi >= pos.target:
                    exit_px, exit_reason = float(pos.target), "target"
            else:
                if pos.stop is not None and hi >= pos.stop:
                    exit_px, exit_reason = float(pos.stop), "stop"
                elif pos.target is not None and lo <= pos.target:
                    exit_px, exit_reason = float(pos.target), "target"

            if exit_px is None and now.time() >= flat_at:
                exit_px, exit_reason = close, "eod_flat"

            if exit_px is None:
                a = float(bar.get("atr_14") or 0) or close * 0.01
                risk = abs(pos.entry_price - (pos.stop or pos.entry_price))
                gain = (close - pos.entry_price) if pos.direction == "long" \
                    else (pos.entry_price - close)
                if risk > 0 and gain >= self.cfg.risk.breakeven_at_r * risk and pos.stop is not None:
                    pos.stop = (max(pos.stop, pos.entry_price) if pos.direction == "long"
                                else min(pos.stop, pos.entry_price))
                if pos.stop is not None:
                    pos.stop = trail_stop(pos.stop, close, a, pos.direction, self.cfg.risk)
                continue

            closed.append(self._close_position(pos, exit_px, exit_reason, now))
        return closed

    def _close_position(
        self, pos: Position, price: float, reason: str, now: datetime
    ) -> dict:
        side = "sell" if pos.direction == "long" else "buy_to_cover"
        order = Order(pos.symbol, side, pos.qty, "market",
                      note=f"exit: {reason}")
        fees = 0.0
        fill_px = price
        if not self.dry_run:
            fill = self.broker.submit(order, price, now)
            if fill:
                fill_px, fees = fill.price, fill.fees
        result = {}
        if pos.trade_id:
            result = self.journal.close_trade(
                pos.trade_id, fill_px, now, reason, fees=fees,
                mae=pos.mae, mfe=pos.mfe, bars_held=pos.bars_held,
            )
            self.learner.after_trade(pos.trade_id)
        net = result.get("net_pnl", 0.0)
        self.state.realized_today += net
        self.state.consecutive_losses = (
            self.state.consecutive_losses + 1 if net <= 0 else 0
        )
        if result.get("is_day_trade"):
            self.pdt.record(pos.symbol, now.date())
        self.state.positions.pop(pos.symbol, None)
        self.journal.log(
            "info",
            f"closed {pos.symbol} {pos.direction} x{pos.qty} @ {fill_px:.2f} "
            f"({reason}) net={net:+.2f}",
        )

        if self.notifier is not None:
            from .notify import exit_alert

            self.notifier.send(
                exit_alert(pos.symbol, pos.direction, pos.qty, fill_px, reason,
                           net, result.get("r_multiple")),
                dedupe_key=f"exit:{pos.symbol}:{now:%Y-%m-%d %H:%M}",
            )
        return {"symbol": pos.symbol, "price": fill_px, "reason": reason, **result}

    # -----------------------------------------------------------------
    def _enter(self, sig: strat.Signal, now: datetime) -> dict | None:
        qty = sig.sizing.shares
        if qty <= 0:
            return None
        side = "buy" if sig.direction == "long" else "sell_short"
        offset = sig.entry * (self.cfg.execution.limit_offset_bps / 10_000.0)
        limit = sig.entry + offset if side == "buy" else sig.entry - offset

        order = Order(
            symbol=sig.symbol, side=side, qty=qty,
            order_type=self.cfg.execution.default_order_type,
            limit_price=round(limit, 2), tif="day",
            note=f"{sig.strategy} | score {sig.score:.2f} | "
                 f"stop {sig.stop:.2f} target {sig.target:.2f}",
        )

        sid = self.journal.record_signal(sig, taken=True)
        fill_px = sig.entry

        if self.dry_run:
            self.journal.log("info", f"DRY RUN — would place: {order.describe()}")
        else:
            fill = self.broker.submit(order, sig.entry, now)
            if fill is None:
                # FidelityBridge returns None by design: a human places the order.
                if isinstance(self.broker, FidelityBridge):
                    self.journal.log("info", f"ticket queued for Fidelity: {order.describe()}")
                else:
                    self.journal.log("info", f"order not filled: {order.describe()}")
                    return None
            else:
                fill_px = fill.price

        trade_id = self.journal.open_trade(
            sig.symbol, sig.direction, qty, fill_px, now,
            strategy=sig.strategy, signal_id=sid,
            planned_stop=sig.stop, planned_target=sig.target,
            broker=self.broker.name, notes=sig.rationale(),
        )
        self.state.positions[sig.symbol] = Position(
            sig.symbol, sig.direction, qty, fill_px, now,
            stop=sig.stop, target=sig.target, trade_id=trade_id, strategy=sig.strategy,
        )
        self.state.trades_today += 1

        if self.notifier is not None:
            from .notify import entry_alert

            self.notifier.send(
                entry_alert(
                    sig, self._equity(),
                    self.pdt.remaining(self._equity(), now.date()),
                    live=self.gate.live_allowed and not self.dry_run,
                ),
                dedupe_key=f"entry:{sig.symbol}:{now:%Y-%m-%d %H:%M}",
            )
        return {"symbol": sig.symbol, "qty": qty, "price": fill_px, "trade_id": trade_id}

    # -----------------------------------------------------------------
    def run_once(self, now: datetime | None = None, verbose: bool = True) -> dict:
        """One scan/manage cycle."""
        now = now or datetime.now()
        self._roll_day(now)
        frames = self.load_frames()
        if not frames:
            return {"error": "no market data available"}

        prices = {s: float(df["close"].iloc[-1]) for s, df in frames.items()}
        if isinstance(self.broker, PaperBroker):
            self.broker.mark(prices)

        closed = self.manage_positions(frames, now)
        state = self.account_state(prices)

        should, why = self.risk.should_halt(state)
        if should and not self.state.halted:
            self.state.halted, self.state.halt_reason = True, why
            self.journal.log("halt", why)
            if self.notifier is not None:
                from .notify import risk_alert

                self.notifier.send(
                    risk_alert(why, "No further entries until the next session."),
                    dedupe_key=f"halt:{now:%Y-%m-%d}",
                )

        taken, considered, blocked = [], [], []
        if not self.state.halted:
            stats = self.journal.stats(
                limit=self.cfg.learn.lookback_trades,
                base_equity=self.cfg.risk.starting_equity,
            )
            signals = strat.scan(
                {s: df for s, df in frames.items() if s not in self.state.positions},
                self.cfg, state.equity, stats=stats,
            )
            for sig in signals:
                considered.append(sig)
                if sig.rejected_reason:
                    self.journal.record_signal(sig, taken=False)
                    continue
                decision = self.risk.check(
                    state, now=now, intends_same_day_exit=True,
                    new_notional=sig.sizing.notional, symbol=sig.symbol,
                )
                if not decision.allowed:
                    sig.rejected_reason = decision.reason
                    self.journal.record_signal(sig, taken=False)
                    blocked.append((sig.symbol, decision.reason))
                    self.journal.log("risk_block", f"{sig.symbol}: {decision.reason}")
                    continue
                res = self._enter(sig, now)
                if res:
                    taken.append(sig)
                    state = self.account_state(prices)
                if len(taken) >= 2:  # pace entries within a cycle
                    break

        eq = self._equity()
        self.journal.mark_equity(
            eq, cash=getattr(self.broker, "cash", 0.0),
            positions_value=getattr(self.broker, "positions_value", 0.0),
            realized_pnl_today=self.state.realized_today,
            open_positions=len(self.state.positions),
            trades_today=self.state.trades_today, ts=now,
        )

        if verbose:
            self._print_cycle(now, eq, taken, closed, blocked, considered)

        return {
            "time": now.isoformat(timespec="seconds"),
            "equity": eq,
            "open_positions": len(self.state.positions),
            "entered": [s.symbol for s in taken],
            "closed": closed,
            "blocked": blocked,
            "considered": len(considered),
            "halted": self.state.halted,
            "halt_reason": self.state.halt_reason,
            "pdt_remaining": self.pdt.remaining(eq, now.date()),
        }

    def _print_cycle(self, now, eq, taken, closed, blocked, considered) -> None:
        print(f"\n[{now:%H:%M:%S}] equity ${eq:,.2f}  "
              f"open {len(self.state.positions)}  "
              f"day P&L {self.state.realized_today:+,.2f}")
        for c in closed:
            print(f"   CLOSED {c['symbol']} @ {c['price']:.2f} ({c['reason']}) "
                  f"net {c.get('net_pnl', 0):+,.2f}")
        for s in taken:
            print(f"   ENTERED {s.direction} {s.symbol} x{s.sizing.shares} @ {s.entry:.2f} "
                  f"stop {s.stop:.2f} target {s.target:.2f}  [{s.strategy}]")
        for sym, why in blocked[:3]:
            print(f"   blocked {sym}: {why[:70]}")
        if self.state.halted:
            print(f"   HALTED: {self.state.halt_reason}")

    # -----------------------------------------------------------------
    def run_day(self, poll_seconds: int = 300, max_cycles: int = 200) -> dict:
        """Loop until the close, then close out and review."""
        close_t = _parse_time(self.cfg.session.market_close)
        cycles = 0
        while cycles < max_cycles:
            now = datetime.now()
            if now.time() >= close_t:
                break
            self.run_once(now)
            cycles += 1
            _time.sleep(poll_seconds)
        return self.end_of_day()

    # -----------------------------------------------------------------
    def end_of_day(self, now: datetime | None = None) -> dict:
        """Force flat, write the day's report, run the review."""
        now = now or datetime.now()
        frames = self.load_frames(list(self.state.positions)) if self.state.positions else {}
        for sym in list(self.state.positions):
            pos = self.state.positions[sym]
            df = frames.get(sym)
            px = float(df["close"].iloc[-1]) if df is not None and not df.empty else pos.entry_price
            self._close_position(pos, px, "eod_flat", now)

        eq = self._equity()
        self.journal.mark_equity(
            eq, realized_pnl_today=self.state.realized_today,
            open_positions=len(self.state.positions),
            trades_today=self.state.trades_today, ts=now, note="end of day",
        )
        report_path = save_daily(self.journal, self.cfg, now.date())
        review = self.learner.review(scope="daily")

        if self.notifier is not None:
            from .notify import summary_alert

            self.notifier.send(
                summary_alert(
                    self.journal.stats(base_equity=self.cfg.risk.starting_equity),
                    eq, self.state.realized_today,
                ),
                dedupe_key=f"summary:{now:%Y-%m-%d}",
            )
        return {
            "date": now.date().isoformat(),
            "equity": eq,
            "realized_pnl": self.state.realized_today,
            "trades": self.state.trades_today,
            "report": str(report_path),
            "review": review,
        }
