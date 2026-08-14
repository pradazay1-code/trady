"""Command line interface.

    python3 -m trady <command> [options]

Run `python3 -m trady --help` for the full list.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from . import data as datamod
from . import patterns as pat
from . import strategy as strat
from . import structure as st
from .backtest import Backtester, walk_forward
from .brokers import FidelityBridge
from .config import Config
from .indicators import enrich
from .journal import Journal, compute_stats, stats_by
from .knowledge import KnowledgeBase, explain
from .learn import Learner, evidence_performance, sweep
from .notify import AlertGate, Notifier
from .reporting import daily_report, html_report, performance_report, save_daily
from .risk import PDTTracker, monte_carlo_ruin, probability_of_ruin
from .session import TradingSession


def _load_frames(
    cfg: Config, symbols: list[str], synthetic: bool, bars: int = 3000
) -> dict[str, pd.DataFrame]:
    provider = "synthetic" if synthetic else cfg.data_provider
    cache = datamod.BarCache(Path(cfg.data_dir) / "bars.sqlite")
    out: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        try:
            df = datamod.load(
                sym, provider=provider, interval=cfg.bar_interval,
                days=cfg.history_days, cache=cache,
            ) if not synthetic else datamod.synthetic(sym, bars=bars, seed=100 + i)
            out[sym] = df
            print(f"  {sym}: {len(df)} bars  {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}")
        except Exception as exc:
            print(f"  {sym}: FAILED — {exc}")
    return out


# =====================================================================
#  Commands
# =====================================================================
def cmd_scan(args, cfg: Config) -> int:
    symbols = args.symbols or cfg.watchlist
    print(f"Loading {len(symbols)} symbols...")
    frames = _load_frames(cfg, symbols, args.synthetic, getattr(args, 'bars', 3000))
    if not frames:
        print("\nNo data. Use --synthetic to test the pipeline offline.")
        return 1

    journal = Journal(cfg.journal_db)
    stats = journal.stats(base_equity=cfg.risk.starting_equity)
    equity = args.equity or cfg.risk.starting_equity

    print(f"\nScanning with equity ${equity:,.0f}...\n")
    signals = strat.scan({s: enrich(d) for s, d in frames.items()}, cfg, equity, stats)

    actionable = [s for s in signals if s.actionable]
    print(f"{len(actionable)} actionable of {len(signals)} evaluated\n")
    for sig in signals[: args.limit]:
        flag = "**" if sig.actionable else "  "
        print(f"{flag} {sig.rationale()}\n")
        if args.record:
            journal.record_signal(sig, taken=False)
    return 0


def cmd_analyze(args, cfg: Config) -> int:
    sym = args.symbol
    df = (datamod.synthetic(sym, bars=getattr(args, 'bars', 3000)) if args.synthetic
          else datamod.load(sym, cfg.data_provider, cfg.bar_interval, cfg.history_days))
    df = enrich(df)
    snap = st.snapshot(df)

    print(f"\n{sym} — {len(df)} bars through {df.index[-1]}")
    print("=" * 66)
    print(f"  price ............ {snap['price']:.2f}")
    print(f"  trend ............ {snap['trend_direction']} "
          f"(strength {snap['trend_strength']:.2f}, phase {snap['trend_phase']})")
    print(f"  ATR .............. {snap['atr']:.3f}  ({snap['atr_pct']:.2%} of price)")
    print(f"  support .......... {snap['support']}")
    print(f"  resistance ....... {snap['resistance']}")
    if snap["recent_breakout"]:
        print(f"  breakout ......... {snap['recent_breakout']}")
    if snap["recent_gap"]:
        print(f"  gap .............. {snap['recent_gap']}")
    if snap["chart_patterns"]:
        print(f"  chart patterns ... {snap['chart_patterns']}")

    last = df.iloc[-1]
    print("\n  indicators:")
    for k in ("rsi_14", "macd_hist", "mfi_14", "momentum_10", "vol_surge", "bb_pct"):
        if k in df.columns and pd.notna(last.get(k)):
            print(f"    {k:<14} {float(last[k]):>10.3f}")

    hits = pat.detect_latest(df, within=args.within)
    print(f"\n  candlestick patterns in the last {args.within} bars: {len(hits)}")
    for h in hits:
        conf = "confirmed" if pat.confirmed(h, df) else (
            "awaiting confirmation" if h.needs_confirmation else "no confirmation needed")
        print(f"    {h.timestamp}  {h.name:<26} {h.direction:<8} "
              f"strength {h.strength:.2f}  ({conf})")
        for n in h.notes:
            print(f"        · {n}")

    sig = strat.generate(sym, df, cfg, args.equity or cfg.risk.starting_equity)
    print("\n  decision:")
    print("   ", (sig.rationale().replace("\n", "\n    ") if sig else "no directional bias"))
    return 0


def cmd_backtest(args, cfg: Config) -> int:
    symbols = args.symbols or cfg.watchlist
    print(f"Loading {len(symbols)} symbols...")
    frames = _load_frames(cfg, symbols, args.synthetic, getattr(args, 'bars', 3000))
    if not frames:
        print("\nNo data. Use --synthetic to test offline.")
        return 1

    if args.min_score is not None:
        cfg.signal.min_confluence_score = args.min_score
    if args.equity:
        cfg.risk.starting_equity = args.equity

    journal = Journal(cfg.journal_db) if args.record else None
    print("\nRunning backtest...\n")
    res = Backtester(cfg, journal).run(
        frames, warmup=args.warmup, record=args.record, verbose=args.verbose
    )
    print(res.summary())

    if args.walk_forward:
        print("\n\nWALK-FORWARD (consistency across time slices)")
        print("=" * 60)
        wf = walk_forward(cfg, frames, splits=cfg.learn.walk_forward_splits, warmup=args.warmup)
        print(wf.to_string(index=False) if not wf.empty else "  not enough history to split")

    if args.out:
        Path(args.out).write_text(res.summary())
        if not res.trades.empty:
            res.trades.to_csv(Path(args.out).with_suffix(".trades.csv"), index=False)
        print(f"\nwritten to {args.out}")
    return 0


def cmd_run(args, cfg: Config) -> int:
    if args.broker:
        cfg.execution.broker = args.broker
    if args.equity:
        cfg.risk.starting_equity = args.equity

    live = args.live
    if live and cfg.execution.broker == "paper":
        print("--live with the paper broker is still simulated. Nothing real happens.")
    if live and cfg.execution.broker != "paper":
        print("\n" + "!" * 66)
        print("LIVE MODE — this will route real orders.")
        print(f"  broker  : {cfg.execution.broker}")
        print(f"  equity  : ${cfg.risk.starting_equity:,.2f}")
        print(f"  risk    : {cfg.risk.risk_per_trade_pct:.2%} per trade, "
              f"{cfg.risk.max_daily_loss_pct:.0%} daily stop")
        print("!" * 66)
        if input("\nType LIVE to confirm: ").strip() != "LIVE":
            print("aborted.")
            return 1

    session = TradingSession(cfg, dry_run=not live)
    print(f"\nsession started — broker={session.broker.name} "
          f"dry_run={session.dry_run} equity=${session._equity():,.2f}")
    eq = session._equity()
    rem = session.pdt.remaining(eq)
    print(f"PDT: {'unrestricted' if rem < 0 else f'{rem} day trade(s) available'}")

    if args.loop:
        result = session.run_day(poll_seconds=args.interval)
        print("\n" + result["review"].report())
    else:
        result = session.run_once()
        print("\n" + json.dumps(
            {k: v for k, v in result.items() if k != "closed"}, indent=2, default=str
        ))
    return 0


def cmd_eod(args, cfg: Config) -> int:
    session = TradingSession(cfg, dry_run=not args.live)
    result = session.end_of_day()
    print(f"\nend of day {result['date']}")
    print(f"  equity ......... ${result['equity']:,.2f}")
    print(f"  realized P&L ... ${result['realized_pnl']:+,.2f}")
    print(f"  trades ......... {result['trades']}")
    print(f"  report ......... {result['report']}\n")
    print(result["review"].report())
    return 0


def cmd_report(args, cfg: Config) -> int:
    journal = Journal(cfg.journal_db)
    if args.kind == "daily":
        day = date.fromisoformat(args.date) if args.date else date.today()
        text = daily_report(journal, cfg, day)
        print(text)
        if args.save:
            print(f"\nsaved to {save_daily(journal, cfg, day)}")
    elif args.kind == "performance":
        print(performance_report(journal, cfg, limit=args.limit))
    elif args.kind == "html":
        out = html_report(journal, cfg, Path(args.out) if args.out else None)
        print(f"written to {out}")
    return 0


def cmd_review(args, cfg: Config) -> int:
    journal = Journal(cfg.journal_db)
    outcome = Learner(cfg, journal).review(scope=args.scope, persist=not args.dry_run)
    print(outcome.report())
    if outcome.changes and not args.dry_run:
        print(f"\nconfig updated → {cfg.save()}")
    return 0


def cmd_learn(args, cfg: Config) -> int:
    journal = Journal(cfg.journal_db)
    perf = evidence_performance(journal, lookback=args.lookback)
    if not perf:
        print("No attributable trades yet. Trade (or backtest with --record) first.")
        return 0
    print("EVIDENCE PERFORMANCE")
    print("=" * 62)
    for e in sorted(perf, key=lambda x: x.avg_r, reverse=True):
        print(f"  {e.source:<20} {e.trades:>4} trades  win {e.win_rate:>5.1%}  "
              f"avgR {e.avg_r:>+6.2f}  {e.verdict}")
    print("\ncurrent weights:")
    for k, v in cfg.signal.weights.items():
        print(f"  {k:<20} {v:.3f}")
    return 0


def cmd_sweep(args, cfg: Config) -> int:
    frames = _load_frames(cfg, args.symbols or cfg.watchlist[:3], args.synthetic,
                          getattr(args, 'bars', 3000))
    if not frames:
        return 1
    grid = {
        "signal.min_confluence_score": [2.0, 2.5, 3.0, 3.5],
        "risk.stop_atr_multiple": [1.0, 1.5, 2.0],
        "risk.target_atr_multiple": [2.0, 2.5, 3.0],
    }
    print("\nsweeping (this takes a while)...\n")
    df = sweep(cfg, frames, grid, warmup=args.warmup)
    print(df.head(20).to_string(index=False) if not df.empty else "no results")
    print(
        "\nNote: the top row is the best fit to THIS sample. Prefer settings that "
        "stay positive across many rows over the single highest number — that is "
        "the difference between an edge and a curve fit."
    )
    return 0


def cmd_risk(args, cfg: Config) -> int:
    journal = Journal(cfg.journal_db)
    stats = journal.stats(base_equity=cfg.risk.starting_equity)
    eq = args.equity or cfg.risk.starting_equity
    pdt = PDTTracker(cfg.pdt)
    pdt.load(journal.day_trades())

    print("\nRISK STATUS")
    print("=" * 62)
    print(f"  equity .................. ${eq:,.2f}")
    print(f"  risk per trade .......... {cfg.risk.risk_per_trade_pct:.2%} "
          f"(${eq * cfg.risk.risk_per_trade_pct:,.2f})")
    print(f"  max position ............ {cfg.risk.max_position_pct:.0%} "
          f"(${eq * cfg.risk.max_position_pct:,.2f})")
    print(f"  daily loss limit ........ {cfg.risk.max_daily_loss_pct:.0%} "
          f"(${eq * cfg.risk.max_daily_loss_pct:,.2f})")
    print(f"  drawdown halt ........... {cfg.risk.max_drawdown_halt_pct:.0%}")
    print(f"  max trades/day .......... {cfg.risk.max_daily_trades}")
    print(f"  max open positions ...... {cfg.risk.max_open_positions}")

    print("\n  PATTERN DAY TRADER")
    restricted = pdt.is_restricted(eq)
    print(f"    status ................ {'RESTRICTED' if restricted else 'unrestricted'}")
    if restricted:
        print(f"    used in 5-day window .. {pdt.count()}/{cfg.pdt.max_day_trades}")
        print(f"    available now ......... {pdt.remaining(eq)}")
        print(f"    shortfall to lift ..... ${cfg.pdt.equity_threshold - eq:,.2f}")

    if stats["trades"] >= 10:
        print(f"\n  MEASURED EDGE ({stats['trades']} trades)")
        print(f"    win rate .............. {stats['win_rate']:.1%}")
        print(f"    expectancy ............ {stats['expectancy']:.4%}")
        ruin = probability_of_ruin(stats["win_rate"],
                                   max(1, int(1 / max(cfg.risk.risk_per_trade_pct, 1e-6))))
        print(f"    probability of ruin ... {ruin:.4%}")
        if stats["avg_loss"] > 0:
            mc = monte_carlo_ruin(stats["win_rate"], stats["avg_win"] / stats["avg_loss"],
                                  1.0, cfg.risk.risk_per_trade_pct)
            print(f"    Monte Carlo P(-40%) ... {mc['prob_ruin']:.2%}")
            print(f"    median max drawdown ... {mc['median_max_drawdown']:.1%}")
    else:
        print(f"\n  MEASURED EDGE: only {stats['trades']} trades — not enough to measure.")
    return 0


def cmd_fidelity(args, cfg: Config) -> int:
    bridge = FidelityBridge(cfg.execution, Path(cfg.reports_dir) / "tickets")
    journal = Journal(cfg.journal_db)

    if args.action == "tickets":
        symbols = args.symbols or cfg.watchlist
        frames = _load_frames(cfg, symbols, args.synthetic, getattr(args, 'bars', 3000))
        equity = args.equity or cfg.risk.starting_equity
        signals = strat.scan({s: enrich(d) for s, d in frames.items()}, cfg, equity,
                             journal.stats(base_equity=equity))
        actionable = [s for s in signals if s.actionable][: args.limit]
        if not actionable:
            print("No actionable signals — nothing to enter.")
            return 0
        for sig in actionable:
            side = "buy" if sig.direction == "long" else "sell_short"
            offset = sig.entry * (cfg.execution.limit_offset_bps / 10_000)
            from .brokers import Order
            bridge.pending.append(Order(
                sig.symbol, side, sig.sizing.shares, "limit",
                limit_price=round(sig.entry + (offset if side == "buy" else -offset), 2),
                note=f"{sig.strategy}, score {sig.score:.2f}, "
                     f"stop {sig.stop:.2f}, target {sig.target:.2f}",
            ))
        paths = bridge.write_tickets()
        print(Path(paths["txt"]).read_text())
        print(f"\nalso written: {paths['csv']}  {paths['json']}")
        print("\nAfter you place these in Fidelity, remember to enter the matching "
              "stop-loss orders shown in each ticket's note.")

    elif args.action == "import":
        if not args.file:
            print("--file required (your Fidelity Activity CSV export)")
            return 1
        df = FidelityBridge.parse_activity(args.file)
        print(f"parsed {len(df)} execution rows")
        print(df.head(20).to_string(index=False))

    elif args.action == "positions":
        if not args.file:
            print("--file required (your Fidelity Positions CSV export)")
            return 1
        df = FidelityBridge.parse_positions(args.file)
        print(df.to_string(index=False))

    elif args.action == "reconcile":
        if not args.file:
            print("--file required (your Fidelity Activity CSV export)")
            return 1
        res = bridge.reconcile(args.file, journal)
        print(json.dumps(res, indent=2))
    return 0


def cmd_kb(args, cfg: Config) -> int:
    kb = KnowledgeBase(cfg.kb_db)
    if args.topics:
        from .knowledge import TOPIC_QUERIES
        print("known topics:")
        for t in sorted(TOPIC_QUERIES):
            print(f"  {t}")
        return 0
    if not args.query:
        s = kb.stats()
        print(f"knowledge base: {s['books']} books, {s['words']:,} words, "
              f"{s['passages']:,} passages")
        for slug, words in kb.books():
            print(f"  {slug.replace('_', ' ')}  ({words:,} words)")
        return 0
    print(explain(args.query, kb, limit=args.limit))
    return 0


def cmd_journal(args, cfg: Config) -> int:
    journal = Journal(cfg.journal_db)
    if args.what == "trades":
        df = journal.closed_trades(limit=args.limit)
        if df.empty:
            print("no closed trades yet")
            return 0
        cols = [c for c in ["entry_ts", "exit_ts", "symbol", "direction", "strategy",
                            "qty", "entry_price", "exit_price", "exit_reason",
                            "net_pnl", "r_multiple"] if c in df.columns]
        print(df[cols].to_string(index=False))
    elif args.what == "open":
        df = journal.open_trades()
        print(df.to_string(index=False) if not df.empty else "no open positions")
    elif args.what == "signals":
        with journal._con() as con:
            df = pd.read_sql_query(
                f"SELECT ts,symbol,direction,strategy,score,taken,rejected_reason "
                f"FROM signal ORDER BY ts DESC LIMIT {int(args.limit)}", con)
        print(df.to_string(index=False) if not df.empty else "no signals recorded")
    elif args.what == "events":
        df = journal.events(limit=args.limit)
        print(df.to_string(index=False) if not df.empty else "no events")
    elif args.what == "reviews":
        df = journal.reviews(limit=args.limit)
        if df.empty:
            print("no reviews yet")
            return 0
        for _, r in df.iterrows():
            print(f"\n{r['ts']}  [{r['scope']}]  {r['trades_reviewed']} trades")
            print(r["lessons"])
    return 0


def cmd_config(args, cfg: Config) -> int:
    if args.set:
        for pair in args.set:
            key, val = pair.split("=", 1)
            section, attr = key.split(".", 1)
            target = getattr(cfg, section)
            old = getattr(target, attr)
            new = type(old)(val) if not isinstance(old, bool) else val.lower() in ("1", "true", "yes")
            setattr(target, attr, new)
            print(f"  {key}: {old} → {new}")
        print(f"\nsaved to {cfg.save()}")
        return 0
    print(json.dumps(json.loads(json.dumps(cfg, default=lambda o: getattr(o, "__dict__", str(o)))), indent=2))
    return 0


def cmd_notify(args, cfg: Config) -> int:
    journal = Journal(cfg.journal_db)
    if args.action == "test":
        n = Notifier(args.channels or ["console"], dry_run=args.dry_run)
        print(f"channels: {[c.name for c in n.channels]}\n")
        print(json.dumps({k: str(v) for k, v in n.test().items()}, indent=2))
        return 0

    if args.action == "gate":
        verdict = AlertGate(cfg, journal).evaluate()
        print("\nALERT VALIDATION GATE")
        print("=" * 62)
        for k, v in verdict.checks.items():
            print(f"  {k:<20} {v}")
        print()
        if verdict.live_allowed:
            print("  LIVE-TRADABLE — the strategy has met the validation bar.")
        else:
            print("  PAPER MODE — alerts will be stamped 'do not place this order'.")
            print("  Blocking reasons:")
            for r in verdict.reasons:
                print(f"    - {r}")
        return 0
    return 0


def cmd_watch(args, cfg: Config) -> int:
    """Real-time loop: scan, alert, manage, repeat until the close."""
    import time as _t
    from datetime import time as _time_cls

    if args.equity:
        cfg.risk.starting_equity = args.equity

    notifier = Notifier(
        args.channels or ["console"],
        dry_run=args.dry_run,
        log_path=Path(cfg.reports_dir) / "alerts.jsonl",
    )
    session = TradingSession(cfg, dry_run=not args.live, notifier=notifier)

    print(f"\nwatching {len(cfg.watchlist)} symbols every {args.interval}s")
    print(f"  channels : {[c.name for c in notifier.channels]}")
    print(f"  broker   : {session.broker.name} (dry_run={session.dry_run})")
    print(f"  equity   : ${session._equity():,.2f}")

    rem = session.pdt.remaining(session._equity())
    print(f"  PDT      : {'unrestricted' if rem < 0 else f'{rem} day trade(s) left'}")

    if not session.gate.live_allowed:
        print("\n  " + "!" * 60)
        print("  ALERTS ARE IN PAPER MODE — the strategy is not validated:")
        for r in session.gate.reasons:
            print(f"    - {r}")
        print("  Signals will still arrive, stamped 'do not place this order'.")
        print("  " + "!" * 60)

    close_t = _parse_hhmm(cfg.session.market_close)
    cycles = 0
    try:
        while cycles < args.max_cycles:
            now = datetime.now()
            if now.time() >= close_t and not args.ignore_clock:
                print("\nmarket closed — running end of day")
                break
            session.run_once(now=now)
            cycles += 1
            if args.once:
                break
            _t.sleep(args.interval)
    except KeyboardInterrupt:
        print("\ninterrupted — positions left as-is; run `trady eod` to close out")
        return 130

    if not args.once:
        result = session.end_of_day()
        print(f"\nday done: ${result['realized_pnl']:+,.2f} over "
              f"{result['trades']} trades")
        print(result["review"].report())
    return 0


def _parse_hhmm(hhmm: str):
    from datetime import time as _t

    h, m = hhmm.split(":")
    return _t(int(h), int(m))


# =====================================================================
#  Parser
# =====================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trady",
        description="Automated day-trading agent built on five trading references.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python3 -m trady kb "pattern day trader rule"
  python3 -m trady analyze AAPL --synthetic
  python3 -m trady scan --synthetic
  python3 -m trady backtest --synthetic --record --walk-forward
  python3 -m trady risk
  python3 -m trady run                      # dry run, no orders
  python3 -m trady fidelity tickets --synthetic
  python3 -m trady report performance
""",
    )
    p.add_argument("--config", help="path to config.json")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="scan the watchlist for signals")
    s.add_argument("symbols", nargs="*")
    s.add_argument("--equity", type=float)
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--synthetic", action="store_true")
    s.add_argument("--record", action="store_true", help="write signals to the journal")
    s.add_argument("--bars", type=int, default=3000, help="synthetic bar count")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("analyze", help="deep analysis of one symbol")
    s.add_argument("symbol")
    s.add_argument("--equity", type=float)
    s.add_argument("--within", type=int, default=5)
    s.add_argument("--synthetic", action="store_true")
    s.add_argument("--bars", type=int, default=3000, help="synthetic bar count")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("backtest", help="backtest the strategy")
    s.add_argument("symbols", nargs="*")
    s.add_argument("--warmup", type=int, default=120)
    s.add_argument("--equity", type=float)
    s.add_argument("--min-score", type=float, dest="min_score")
    s.add_argument("--synthetic", action="store_true")
    s.add_argument("--record", action="store_true", help="write results to the journal")
    s.add_argument("--walk-forward", action="store_true", dest="walk_forward")
    s.add_argument("--verbose", action="store_true")
    s.add_argument("--out")
    s.add_argument("--bars", type=int, default=3000, help="synthetic bar count")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("run", help="run a trading session")
    s.add_argument("--live", action="store_true", help="actually route orders")
    s.add_argument("--loop", action="store_true", help="poll until the close")
    s.add_argument("--interval", type=int, default=300)
    s.add_argument("--broker", choices=["paper", "fidelity", "alpaca"])
    s.add_argument("--equity", type=float)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("eod", help="close out, report, and review")
    s.add_argument("--live", action="store_true")
    s.set_defaults(func=cmd_eod)

    s = sub.add_parser("report", help="reports")
    s.add_argument("kind", choices=["daily", "performance", "html"])
    s.add_argument("--date")
    s.add_argument("--limit", type=int)
    s.add_argument("--save", action="store_true")
    s.add_argument("--out")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("review", help="run the self-correction review")
    s.add_argument("--scope", default="manual", choices=["daily", "periodic", "manual"])
    s.add_argument("--dry-run", action="store_true", dest="dry_run")
    s.set_defaults(func=cmd_review)

    s = sub.add_parser("learn", help="show which evidence is actually paying")
    s.add_argument("--lookback", type=int, default=200)
    s.set_defaults(func=cmd_learn)

    s = sub.add_parser("sweep", help="parameter sweep with over-fitting warning")
    s.add_argument("symbols", nargs="*")
    s.add_argument("--warmup", type=int, default=120)
    s.add_argument("--synthetic", action="store_true")
    s.add_argument("--bars", type=int, default=3000, help="synthetic bar count")
    s.set_defaults(func=cmd_sweep)

    s = sub.add_parser("risk", help="current risk posture and PDT status")
    s.add_argument("--equity", type=float)
    s.set_defaults(func=cmd_risk)

    s = sub.add_parser("fidelity", help="Fidelity tickets and statement import")
    s.add_argument("action", choices=["tickets", "import", "positions", "reconcile"])
    s.add_argument("symbols", nargs="*")
    s.add_argument("--file", help="Fidelity CSV export")
    s.add_argument("--equity", type=float)
    s.add_argument("--limit", type=int, default=5)
    s.add_argument("--synthetic", action="store_true")
    s.add_argument("--bars", type=int, default=3000, help="synthetic bar count")
    s.set_defaults(func=cmd_fidelity)

    s = sub.add_parser("kb", help="search the book knowledge base")
    s.add_argument("query", nargs="?")
    s.add_argument("--limit", type=int, default=3)
    s.add_argument("--topics", action="store_true", help="list known topics")
    s.set_defaults(func=cmd_kb)

    s = sub.add_parser("journal", help="inspect the journal")
    s.add_argument("what", choices=["trades", "open", "signals", "events", "reviews"])
    s.add_argument("--limit", type=int, default=25)
    s.set_defaults(func=cmd_journal)

    s = sub.add_parser("watch", help="real-time loop with phone alerts")
    s.add_argument("--channels", nargs="+",
                   choices=["console", "ntfy", "pushover", "telegram", "email"])
    s.add_argument("--interval", type=int, default=300, help="seconds between scans")
    s.add_argument("--live", action="store_true", help="actually route orders")
    s.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="print alerts instead of sending them")
    s.add_argument("--once", action="store_true", help="single pass then exit")
    s.add_argument("--max-cycles", type=int, default=200, dest="max_cycles")
    s.add_argument("--ignore-clock", action="store_true", dest="ignore_clock")
    s.add_argument("--equity", type=float)
    s.set_defaults(func=cmd_watch)

    s = sub.add_parser("notify", help="test alerts / check the validation gate")
    s.add_argument("action", choices=["test", "gate"])
    s.add_argument("--channels", nargs="+",
                   choices=["console", "ntfy", "pushover", "telegram", "email"])
    s.add_argument("--dry-run", action="store_true", dest="dry_run")
    s.set_defaults(func=cmd_notify)

    s = sub.add_parser("config", help="show or set configuration")
    s.add_argument("--set", nargs="+", metavar="section.key=value")
    s.set_defaults(func=cmd_config)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    try:
        return args.func(args, cfg)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
