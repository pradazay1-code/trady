"""The self-correction loop.

After every trade and at the end of every day, the agent asks three questions
and writes the answers back into its own configuration:

1. **Which evidence actually paid?** Every signal stored the evidence that
   produced it. Joining that against realised R-multiples shows whether, say,
   "volume confirmation" or "bullish engulfing" genuinely preceded winners.
   Evidence weights move toward what worked.
2. **Which strategies are earning their place?** A strategy whose expectancy
   stays negative over a probation window gets suspended rather than left to
   bleed.
3. **Is the risk level still right?** Win rate and win/loss ratio feed the Kelly
   calculation, so position size tracks the measured edge instead of a guess.

Two guardrails, both straight out of the books:

* Adjustments are small and bounded (`weight_step`, floors and ceilings). The
  books warn that switching systems whenever things look bad is itself a losing
  pattern — markets cycle, and no system works all the time.
* Nothing adapts until there are enough trades to mean anything
  (`min_trades_before_adapt`). Fitting to five trades is curve-fitting.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from .config import Config
from .journal import Journal, compute_stats, stats_by
from .risk import kelly_fraction, monte_carlo_ruin, probability_of_ruin


@dataclass
class EvidencePerformance:
    source: str
    trades: int
    win_rate: float
    avg_r: float
    total_r: float
    verdict: str  # "helping" | "neutral" | "hurting" | "insufficient"


@dataclass
class ReviewOutcome:
    scope: str
    trades_reviewed: int
    metrics: dict
    evidence: list[EvidencePerformance]
    changes: dict            # param -> (old, new, reason)
    lessons: list[str]
    suspended_strategies: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            f"REVIEW ({self.scope}) — {self.trades_reviewed} trades",
            "=" * 62,
        ]
        m = self.metrics
        if m.get("trades"):
            lines += [
                f"  win rate ......... {m['win_rate']:.1%}",
                f"  expectancy ....... {m['expectancy']:.4%} per trade",
                f"  average R ........ {m['avg_r']:.3f}",
                f"  profit factor .... {m['profit_factor']}",
                f"  net P&L .......... ${m['net_pnl']:,.2f}",
                f"  max drawdown ..... {m['max_drawdown']:.1%}",
            ]
        if self.evidence:
            lines += ["", "  evidence performance:"]
            for e in sorted(self.evidence, key=lambda x: x.avg_r, reverse=True):
                lines.append(
                    f"    {e.source:<20} {e.trades:>4} trades  "
                    f"win {e.win_rate:>5.1%}  avgR {e.avg_r:>+6.2f}  → {e.verdict}"
                )
        if self.changes:
            lines += ["", "  adjustments made:"]
            for k, (old, new, why) in self.changes.items():
                lines.append(f"    {k}: {old} → {new}  ({why})")
        else:
            lines += ["", "  adjustments made: none"]
        if self.suspended_strategies:
            lines += ["", f"  suspended: {', '.join(self.suspended_strategies)}"]
        if self.lessons:
            lines += ["", "  lessons:"] + [f"    - {l}" for l in self.lessons]
        return "\n".join(lines)


# =====================================================================
#  Evidence attribution
# =====================================================================
def evidence_performance(
    journal: Journal, lookback: int = 200, min_trades: int = 8
) -> list[EvidencePerformance]:
    """Join stored signal evidence against realised trade outcomes."""
    with journal._con() as con:
        rows = con.execute(
            """SELECT s.evidence_json, t.r_multiple, t.net_pnl
               FROM trade t JOIN signal s ON t.signal_id = s.id
               WHERE t.open = 0 AND s.evidence_json IS NOT NULL
               ORDER BY t.exit_ts DESC LIMIT ?""",
            (int(lookback),),
        ).fetchall()

    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        try:
            ev = json.loads(r["evidence_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        rmult = r["r_multiple"]
        if rmult is None or not np.isfinite(rmult):
            continue
        # Only credit evidence that pointed the way the trade was taken.
        for e in ev:
            if e.get("contribution", 0) == 0:
                continue
            buckets[e["source"]].append((float(rmult), float(r["net_pnl"] or 0.0)))

    out: list[EvidencePerformance] = []
    for source, vals in buckets.items():
        rs = np.array([v[0] for v in vals], dtype=float)
        n = len(rs)
        win_rate = float((rs > 0).mean()) if n else 0.0
        avg_r = float(rs.mean()) if n else 0.0
        if n < min_trades:
            verdict = "insufficient"
        elif avg_r > 0.08:
            verdict = "helping"
        elif avg_r < -0.08:
            verdict = "hurting"
        else:
            verdict = "neutral"
        out.append(
            EvidencePerformance(source, n, round(win_rate, 4), round(avg_r, 4),
                                round(float(rs.sum()), 3), verdict)
        )
    return out


# =====================================================================
#  The review
# =====================================================================
class Learner:
    def __init__(self, cfg: Config, journal: Journal):
        self.cfg = cfg
        self.journal = journal

    # -----------------------------------------------------------------
    def review(self, scope: str = "daily", persist: bool = True) -> ReviewOutcome:
        lc = self.cfg.learn
        trades = self.journal.closed_trades(limit=lc.lookback_trades)
        n = len(trades)
        metrics = compute_stats(trades, base_equity=self.cfg.risk.starting_equity)
        changes: dict[str, tuple] = {}
        lessons: list[str] = []
        suspended: list[str] = []
        ev_perf = evidence_performance(self.journal, lookback=lc.lookback_trades)

        if not lc.enabled:
            lessons.append("learning disabled in config; review is read-only")
            return ReviewOutcome(scope, n, metrics, ev_perf, {}, lessons)

        if n < lc.min_trades_before_adapt:
            lessons.append(
                f"only {n} closed trades — need {lc.min_trades_before_adapt} before "
                "changing anything. Adapting on a small sample is curve-fitting."
            )
            outcome = ReviewOutcome(scope, n, metrics, ev_perf, {}, lessons)
            if persist:
                self._persist(outcome)
            return outcome

        # ---- 1. evidence weights ------------------------------------
        for e in ev_perf:
            if e.verdict == "insufficient" or e.source not in self.cfg.signal.weights:
                continue
            old = self.cfg.signal.weights[e.source]
            if e.verdict == "helping":
                new = min(lc.weight_ceiling, old + lc.weight_step)
                why = f"avg R {e.avg_r:+.2f} over {e.trades} trades"
            elif e.verdict == "hurting":
                new = max(lc.weight_floor, old - lc.weight_step)
                why = f"avg R {e.avg_r:+.2f} over {e.trades} trades"
            else:
                continue
            if abs(new - old) > 1e-9:
                self.cfg.signal.weights[e.source] = round(new, 4)
                changes[f"signal.weights.{e.source}"] = (round(old, 4), round(new, 4), why)

        # ---- 2. strategy probation ----------------------------------
        by_strat = stats_by(trades, "strategy", base_equity=self.cfg.risk.starting_equity)
        if not by_strat.empty:
            for _, row in by_strat.iterrows():
                name = row["strategy"]
                if row["trades"] < lc.strategy_probation_trades:
                    continue
                if row["expectancy"] <= lc.strategy_min_expectancy:
                    suspended.append(name)
                    lessons.append(
                        f"strategy '{name}' has negative expectancy "
                        f"({row['expectancy']:.4%}) over {int(row['trades'])} trades — "
                        "suspended pending re-test"
                    )
            best = by_strat.iloc[0]
            if best["net_pnl"] > 0:
                lessons.append(
                    f"'{best['strategy']}' is the strongest setup: "
                    f"{int(best['trades'])} trades, {best['win_rate']:.0%} win rate, "
                    f"${best['net_pnl']:,.2f} net"
                )

        # ---- 3. risk level from the measured edge -------------------
        wr, aw, al = metrics["win_rate"], metrics["avg_win"], metrics["avg_loss"]
        if wr > 0 and aw > 0 and al > 0:
            k = kelly_fraction(wr, aw, al)
            half_k = k * self.cfg.risk.kelly_fraction
            target = float(np.clip(half_k, 0.0025, self.cfg.risk.kelly_cap_pct))
            old = self.cfg.risk.risk_per_trade_pct
            # Move at most 25% of the way, so one good week cannot double risk.
            new = round(old + 0.25 * (target - old), 5)
            if abs(new - old) / max(old, 1e-9) > 0.02:
                self.cfg.risk.risk_per_trade_pct = new
                changes["risk.risk_per_trade_pct"] = (
                    old, new, f"half-Kelly on measured edge (W={wr:.2f}, R={aw/al:.2f})"
                )

            ruin = probability_of_ruin(wr, max(1, int(1 / max(new, 1e-6))))
            mc = monte_carlo_ruin(wr, aw / max(al, 1e-9), 1.0, new)
            metrics["probability_of_ruin"] = round(ruin, 5)
            metrics["monte_carlo"] = mc
            if mc["prob_ruin"] > 0.05:
                lessons.append(
                    f"Monte Carlo puts P(40% loss) at {mc['prob_ruin']:.1%} at the "
                    f"current {new:.2%} risk level — consider reducing it"
                )

        # ---- 4. behavioural checks ----------------------------------
        if metrics["trades"] >= 10:
            if metrics["max_consecutive_losses"] >= self.cfg.risk.max_consecutive_losses + 2:
                lessons.append(
                    f"{metrics['max_consecutive_losses']} losses in a row occurred — the "
                    "cool-off threshold may be too loose"
                )
            fee_drag = metrics["fees"] / max(abs(metrics["gross_pnl"]), 1e-9)
            if fee_drag > 0.15:
                lessons.append(
                    f"fees are {fee_drag:.0%} of gross P&L — this is the overtrading "
                    "signature the books warn about; fewer, larger trades keep more"
                )
            if not trades.empty and "exit_reason" in trades.columns:
                stops = (trades["exit_reason"] == "stop").mean()
                if stops > 0.6:
                    lessons.append(
                        f"{stops:.0%} of exits were stops — entries may be early or "
                        "stops too tight relative to ATR"
                    )
                eod = (trades["exit_reason"] == "eod_flat").mean()
                if eod > 0.4:
                    lessons.append(
                        f"{eod:.0%} of exits were forced at the close — targets may be "
                        "too far for an intraday holding period"
                    )

        if metrics["expectancy"] > 0:
            lessons.append(
                f"system expectancy is positive at {metrics['expectancy']:.4%} per trade "
                f"over {n} trades"
            )
        else:
            lessons.append(
                f"system expectancy is NEGATIVE at {metrics['expectancy']:.4%} over {n} "
                "trades — do not increase size; paper-trade until it turns"
            )

        outcome = ReviewOutcome(scope, n, metrics, ev_perf, changes, lessons, suspended)
        if persist:
            self._persist(outcome)
            self.cfg.save()
        return outcome

    # -----------------------------------------------------------------
    def _persist(self, outcome: ReviewOutcome) -> int:
        return self.journal.record_review(
            scope=outcome.scope,
            trades_reviewed=outcome.trades_reviewed,
            metrics=outcome.metrics,
            changes=outcome.changes,
            lessons="\n".join(outcome.lessons),
        )

    # -----------------------------------------------------------------
    def after_trade(self, trade_id: int) -> dict:
        """Lightweight per-trade check. Full re-weighting happens at review time."""
        with self.journal._con() as con:
            row = con.execute("SELECT * FROM trade WHERE id=?", (trade_id,)).fetchone()
        if row is None:
            return {}
        r = row["r_multiple"]
        notes: list[str] = []

        if r is not None and np.isfinite(r):
            if r <= -1.5:
                notes.append(
                    f"loss of {r:.2f}R exceeded the planned 1R — the stop did not hold "
                    "(gap or slippage); check whether the stop was realistic"
                )
            elif r >= 2.0:
                notes.append(f"{r:.2f}R winner — record what made this setup work")

        if row["exit_reason"] == "eod_flat" and (row["net_pnl"] or 0) > 0:
            notes.append("closed profitably at the bell — target may be set too far out")
        if row["mae"] is not None and row["planned_stop"] and row["entry_price"]:
            risk = abs(row["entry_price"] - row["planned_stop"])
            if risk > 0 and abs(row["mae"]) > 0.9 * risk and (row["net_pnl"] or 0) > 0:
                notes.append(
                    "this winner came within 10% of the stop first — the entry was early"
                )

        if notes:
            self.journal.log("info", f"trade {trade_id}: " + " | ".join(notes))
        return {"trade_id": trade_id, "r_multiple": r, "notes": notes}


# =====================================================================
#  Parameter sweep (offline, walk-forward guarded)
# =====================================================================
def sweep(
    cfg: Config,
    frames: dict[str, pd.DataFrame],
    grid: dict[str, list],
    warmup: int = 120,
) -> pd.DataFrame:
    """Grid-search parameters and report out-of-sample consistency.

    Reports every combination rather than only the winner, because the winner of
    a grid search on one sample is usually the best-fitted, not the best.
    """
    from itertools import product

    from .backtest import Backtester

    keys = list(grid)
    rows = []
    for combo in product(*[grid[k] for k in keys]):
        trial = Config.load()
        for k, v in zip(keys, combo):
            section, attr = k.split(".", 1)
            setattr(getattr(trial, section), attr, v)
        trial.risk.starting_equity = cfg.risk.starting_equity
        res = Backtester(trial).run(frames, warmup=warmup)
        rows.append({
            **dict(zip(keys, combo)),
            "trades": res.stats["trades"],
            "win_rate": res.stats["win_rate"],
            "expectancy": res.stats["expectancy"],
            "net_pnl": res.stats["net_pnl"],
            "max_drawdown": res.stats["max_drawdown"],
            "avg_r": res.stats["avg_r"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Prefer robust settings: positive expectancy with enough trades to trust.
    df["trustworthy"] = (df["trades"] >= 30) & (df["expectancy"] > 0)
    return df.sort_values(["trustworthy", "expectancy"], ascending=False).reset_index(drop=True)
