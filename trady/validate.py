"""Unattended validation: run the whole evidence pipeline and report a verdict.

Designed to run with nobody watching — from cron, a systemd timer, or GitHub
Actions. It fetches data, backtests, walk-forwards, sweeps the threshold, checks
consistency, and produces a single machine-readable verdict plus a human report.

The verdict is deliberately hard to pass. A strategy that only works in-sample,
or only in some periods, or only on a handful of trades, is reported as NOT
validated no matter how large the headline profit.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as datamod
from .backtest import Backtester, walk_forward
from .config import Config


@dataclass
class Criterion:
    name: str
    passed: bool
    detail: str
    value: float | None = None
    threshold: float | None = None


@dataclass
class ValidationReport:
    started: str
    finished: str
    provider: str
    symbols: list[str]
    bars_per_symbol: dict
    criteria: list[Criterion] = field(default_factory=list)
    full_sample: dict = field(default_factory=dict)
    walk_forward: list = field(default_factory=list)
    sweep: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    environment: dict = field(default_factory=dict)

    @property
    def validated(self) -> bool:
        return bool(self.criteria) and all(c.passed for c in self.criteria)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["validated"] = self.validated
        return d

    def report(self) -> str:
        lines = [
            "=" * 72,
            "UNATTENDED VALIDATION",
            "=" * 72,
            f"  started .......... {self.started}",
            f"  finished ......... {self.finished}",
            f"  provider ......... {self.provider}",
            f"  symbols .......... {', '.join(self.symbols)}",
            f"  bars ............. {self.bars_per_symbol}",
            "",
        ]

        if self.full_sample:
            s = self.full_sample
            lines += [
                "FULL SAMPLE",
                f"  trades ........... {s.get('trades', 0)}",
                f"  win rate ......... {s.get('win_rate', 0):.1%}",
                f"  expectancy ....... {s.get('expectancy', 0):+.4%}",
                f"  net P&L .......... ${s.get('net_pnl', 0):+,.2f}",
                f"  profit factor .... {s.get('profit_factor', 0)}",
                f"  max drawdown ..... {s.get('max_drawdown', 0):.2%}",
                "",
            ]

        if self.walk_forward:
            wf = pd.DataFrame(self.walk_forward)
            pos = int((wf["expectancy"] > 0).sum())
            lines += [
                "WALK-FORWARD",
                f"  slices ........... {len(wf)}",
                f"  positive ......... {pos}/{len(wf)}  ({pos / max(len(wf), 1):.0%})",
                f"  total trades ..... {int(wf['trades'].sum())}",
                f"  net across slices  ${wf['net_pnl'].sum():+,.2f}",
                "",
            ]

        if self.sweep:
            lines += ["THRESHOLD SWEEP", "-" * 72]
            sw = pd.DataFrame(self.sweep)
            lines.append("  " + sw.to_string(index=False).replace("\n", "\n  "))
            lines.append("")

        lines += ["CRITERIA", "-" * 72]
        for c in self.criteria:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name}")
            lines.append(f"         {c.detail}")

        lines += [
            "",
            "=" * 72,
            f"  VERDICT: {'VALIDATED' if self.validated else 'NOT VALIDATED'}",
            "=" * 72,
        ]
        if not self.validated:
            lines.append(
                "  Do not trade the signal engine with real money on this evidence."
            )
        if self.errors:
            lines += ["", "ERRORS"] + [f"  - {e}" for e in self.errors]
        return "\n".join(lines)


# =====================================================================
class Validator:
    """Runs every check and decides. Never raises: failures become criteria."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(
        self,
        symbols: list[str],
        *,
        provider: str = "yfinance",
        interval: str = "5m",
        days: int = 60,
        warmup: int = 120,
        splits: int = 4,
        sweep_values: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5),
        min_trades: int = 100,
        min_positive_slice_fraction: float = 0.6,
    ) -> ValidationReport:
        started = datetime.now().isoformat(timespec="seconds")
        rep = ValidationReport(
            started=started, finished="", provider=provider, symbols=symbols,
            bars_per_symbol={},
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
        )

        # ---- data ----
        frames: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = datamod.load(sym, provider=provider, interval=interval, days=days)
                if len(df) < warmup + 50:
                    rep.errors.append(f"{sym}: only {len(df)} bars, skipped")
                    continue
                frames[sym] = df
                rep.bars_per_symbol[sym] = len(df)
            except Exception as exc:
                rep.errors.append(f"{sym}: {type(exc).__name__}: {exc}")

        if not frames:
            rep.criteria.append(Criterion(
                "data available", False,
                "no symbol produced usable data — validation could not run",
            ))
            rep.finished = datetime.now().isoformat(timespec="seconds")
            return rep

        rep.criteria.append(Criterion(
            "data available", True,
            f"{len(frames)} symbol(s), {sum(rep.bars_per_symbol.values()):,} bars total",
        ))

        # ---- full sample ----
        try:
            res = Backtester(self.cfg).run(frames, warmup=warmup)
            rep.full_sample = res.stats
        except Exception as exc:
            rep.errors.append(f"backtest failed: {type(exc).__name__}: {exc}")
            rep.criteria.append(Criterion("backtest ran", False, str(exc)))
            rep.finished = datetime.now().isoformat(timespec="seconds")
            return rep

        trades = rep.full_sample.get("trades", 0)
        rep.criteria.append(Criterion(
            "sample size", trades >= min_trades,
            f"{trades} trades (need >= {min_trades} to distinguish edge from luck)",
            float(trades), float(min_trades),
        ))

        exp = rep.full_sample.get("expectancy", 0.0)
        rep.criteria.append(Criterion(
            "positive expectancy (full sample)", exp > 0,
            f"{exp:+.4%} per trade", exp, 0.0,
        ))

        pf = rep.full_sample.get("profit_factor", 0.0)
        pf_ok = isinstance(pf, (int, float)) and pf > 1.2
        rep.criteria.append(Criterion(
            "profit factor > 1.2", bool(pf_ok),
            f"profit factor {pf}", float(pf) if isinstance(pf, (int, float)) else None, 1.2,
        ))

        dd = rep.full_sample.get("max_drawdown", 1.0)
        rep.criteria.append(Criterion(
            "drawdown within halt threshold",
            dd < self.cfg.risk.max_drawdown_halt_pct,
            f"max drawdown {dd:.2%} vs halt at "
            f"{self.cfg.risk.max_drawdown_halt_pct:.0%}",
            dd, self.cfg.risk.max_drawdown_halt_pct,
        ))

        # ---- walk-forward: the one that matters ----
        try:
            wf = walk_forward(self.cfg, frames, splits=splits, warmup=warmup)
            rep.walk_forward = wf.to_dict("records") if not wf.empty else []
        except Exception as exc:
            rep.errors.append(f"walk-forward failed: {type(exc).__name__}: {exc}")
            wf = pd.DataFrame()

        if wf.empty:
            rep.criteria.append(Criterion(
                "walk-forward consistency", False,
                "not enough history to split into out-of-sample periods",
            ))
        else:
            pos = int((wf["expectancy"] > 0).sum())
            frac = pos / len(wf)
            rep.criteria.append(Criterion(
                "walk-forward consistency", frac >= min_positive_slice_fraction,
                f"{pos}/{len(wf)} slices positive ({frac:.0%}); "
                f"need >= {min_positive_slice_fraction:.0%}",
                frac, min_positive_slice_fraction,
            ))
            net = float(wf["net_pnl"].sum())
            rep.criteria.append(Criterion(
                "walk-forward net positive", net > 0,
                f"${net:+,.2f} across all out-of-sample slices", net, 0.0,
            ))

        # ---- threshold sweep ----
        for thr in sweep_values:
            try:
                trial = Config.load()
                trial.signal.min_confluence_score = thr
                trial.risk.starting_equity = self.cfg.risk.starting_equity
                r = Backtester(trial).run(frames, warmup=warmup)
                rep.sweep.append({
                    "threshold": thr, "trades": r.stats["trades"],
                    "win_rate": round(r.stats["win_rate"], 4),
                    "expectancy": round(r.stats["expectancy"], 6),
                    "net_pnl": round(r.stats["net_pnl"], 2),
                    "avg_r": round(r.stats["avg_r"], 3),
                })
            except Exception as exc:
                rep.errors.append(f"sweep {thr}: {type(exc).__name__}: {exc}")

        # A sweep whose trade count does not fall as the threshold rises is not
        # measuring the threshold — something else is gating entries. Catching
        # that automatically is what stopped a previous run being misread.
        if len(rep.sweep) >= 3:
            counts = [s["trades"] for s in rep.sweep]
            monotonic = all(a >= b for a, b in zip(counts, counts[1:]))
            rep.criteria.append(Criterion(
                "sweep is interpretable", monotonic,
                f"trade counts {counts} "
                + ("fall as the threshold rises, as they should"
                   if monotonic else
                   "are NOT monotonic — entries are gated by something other than "
                   "the threshold (likely position-slot starvation), so the sweep "
                   "cannot be read as a tuning curve"),
            ))

        rep.finished = datetime.now().isoformat(timespec="seconds")
        return rep


# =====================================================================
def run_and_save(
    cfg: Config,
    symbols: list[str],
    out_dir: Path,
    **kwargs,
) -> tuple[ValidationReport, Path, Path]:
    """Run validation and write both the human report and the JSON verdict."""
    rep = Validator(cfg).run(symbols, **kwargs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt = out_dir / f"validation_{stamp}.txt"
    txt.write_text(rep.report(), encoding="utf-8")

    js = out_dir / f"validation_{stamp}.json"
    js.write_text(json.dumps(rep.to_dict(), indent=2, default=str), encoding="utf-8")

    # Stable filenames so automation always knows where to look.
    (out_dir / "validation_latest.txt").write_text(rep.report(), encoding="utf-8")
    (out_dir / "validation_latest.json").write_text(
        json.dumps(rep.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    return rep, txt, js
