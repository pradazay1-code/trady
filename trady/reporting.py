"""Reports: the full breakdown of what the agent did and what it earned or lost.

Three levels:
  daily_report    — one session: every trade, every rejected signal, P&L, rule state
  performance_report — cumulative: equity curve, per-strategy and per-symbol edge
  html_report     — the same, as a self-contained page you can open in a browser
"""

from __future__ import annotations

import html
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .journal import Journal, compute_stats, stats_by


def _money(x: float) -> str:
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def _rule(width: int = 78) -> str:
    return "=" * width


# =====================================================================
#  Daily
# =====================================================================
def daily_report(journal: Journal, cfg: Config, day: date | None = None) -> str:
    day = day or date.today()
    ds = day.isoformat()

    with journal._con() as con:
        trades = pd.read_sql_query(
            "SELECT * FROM trade WHERE substr(exit_ts,1,10)=? AND open=0 ORDER BY exit_ts",
            con, params=(ds,),
        )
        opened = pd.read_sql_query(
            "SELECT * FROM trade WHERE substr(entry_ts,1,10)=? ORDER BY entry_ts",
            con, params=(ds,),
        )
        signals = pd.read_sql_query(
            "SELECT * FROM signal WHERE substr(ts,1,10)=? ORDER BY ts", con, params=(ds,)
        )
        marks = pd.read_sql_query(
            "SELECT * FROM equity_mark WHERE day=? ORDER BY ts", con, params=(ds,)
        )
        events = pd.read_sql_query(
            "SELECT * FROM event WHERE substr(ts,1,10)=? ORDER BY ts", con, params=(ds,)
        )

    stats = compute_stats(trades, base_equity=cfg.risk.starting_equity)
    lines = [
        _rule(),
        f"DAILY TRADING REPORT — {day:%A, %B %d, %Y}",
        _rule(),
        "",
    ]

    # ---- account ----
    if not marks.empty:
        start_eq = float(marks["equity"].iloc[0])
        end_eq = float(marks["equity"].iloc[-1])
        chg = end_eq - start_eq
        pct = chg / start_eq if start_eq else 0.0
        lines += [
            "ACCOUNT",
            f"  opening equity ...... {_money(start_eq)}",
            f"  closing equity ...... {_money(end_eq)}",
            f"  change .............. {_money(chg)}  ({pct:+.2%})",
            "",
        ]

    # ---- P&L ----
    lines += [
        "PROFIT & LOSS",
        f"  trades closed ....... {stats['trades']}",
        f"  winners ............. {stats['wins']}",
        f"  losers .............. {stats['losses']}",
        f"  win rate ............ {stats['win_rate']:.1%}",
        f"  gross P&L ........... {_money(stats['gross_pnl'])}",
        f"  fees & commissions .. {_money(stats['fees'])}",
        f"  NET P&L ............. {_money(stats['net_pnl'])}",
        f"  best trade .......... {_money(stats['best'])}",
        f"  worst trade ......... {_money(stats['worst'])}",
        f"  average R ........... {stats['avg_r']:+.3f}",
        f"  profit factor ....... {stats['profit_factor']}",
        "",
    ]

    # ---- trade log ----
    if not trades.empty:
        lines += ["TRADE LOG", "-" * 78]
        hdr = (f"  {'time':<6} {'sym':<6} {'dir':<5} {'strategy':<10} {'qty':>4} "
               f"{'entry':>9} {'exit':>9} {'why':<9} {'net':>10} {'R':>6}")
        lines += [hdr, "-" * 78]
        for _, t in trades.iterrows():
            lines.append(
                f"  {str(t['exit_ts'])[11:16]:<6} {t['symbol']:<6} {t['direction']:<5} "
                f"{str(t['strategy'])[:10]:<10} {int(t['qty']):>4} "
                f"{float(t['entry_price']):>9.2f} {float(t['exit_price']):>9.2f} "
                f"{str(t['exit_reason'])[:9]:<9} {_money(float(t['net_pnl'])):>10} "
                f"{(float(t['r_multiple']) if pd.notna(t['r_multiple']) else 0):>+6.2f}"
            )
        lines.append("")

    # ---- open positions ----
    still_open = journal.open_trades()
    if not still_open.empty:
        lines += ["OPEN POSITIONS (carried overnight)", "-" * 78]
        for _, t in still_open.iterrows():
            lines.append(
                f"  {t['symbol']:<6} {t['direction']:<5} {int(t['qty']):>4} @ "
                f"{float(t['entry_price']):.2f}   stop {t['planned_stop']}  "
                f"target {t['planned_target']}"
            )
        lines += [
            "  NOTE: a day-trading system should normally end the session flat.",
            "",
        ]

    # ---- decisions not taken ----
    if not signals.empty:
        taken = int(signals["taken"].sum())
        lines += [
            "SIGNALS CONSIDERED",
            f"  evaluated ........... {len(signals)}",
            f"  taken ............... {taken}",
            f"  rejected ............ {len(signals) - taken}",
        ]
        rej = signals[signals["taken"] == 0]["rejected_reason"].dropna()
        if not rej.empty:
            lines.append("  rejected because:")
            for reason, n in rej.value_counts().head(8).items():
                lines.append(f"    {n:>4}  {str(reason)[:66]}")
        lines.append("")

    # ---- rule state ----
    day_trades = int(trades["is_day_trade"].sum()) if not trades.empty else 0
    equity_now = float(marks["equity"].iloc[-1]) if not marks.empty else cfg.risk.starting_equity
    lines += ["RULE COMPLIANCE"]
    if equity_now < cfg.pdt.equity_threshold:
        used = len(journal.day_trades())
        lines += [
            f"  PDT status .......... RESTRICTED (equity {_money(equity_now)} < "
            f"${cfg.pdt.equity_threshold:,.0f})",
            f"  day trades today .... {day_trades}",
            f"  rolling 5-day count . {used} of {cfg.pdt.max_day_trades} allowed",
        ]
    else:
        lines += [
            f"  PDT status .......... unrestricted (equity ≥ "
            f"${cfg.pdt.equity_threshold:,.0f})",
            f"  day trades today .... {day_trades}",
        ]
    lines += [
        f"  daily loss limit .... {cfg.risk.max_daily_loss_pct:.0%} "
        f"({'BREACHED' if stats['net_pnl'] < -abs(cfg.risk.max_daily_loss_pct * equity_now) else 'ok'})",
        f"  trade cap ........... {stats['trades']}/{cfg.risk.max_daily_trades}",
        "",
    ]

    if not events.empty:
        lines += ["EVENTS", "-" * 78]
        for _, e in events.iterrows():
            lines.append(f"  {str(e['ts'])[11:19]}  [{e['kind']}] {str(e['detail'])[:60]}")
        lines.append("")

    lines += [_rule(), f"generated {datetime.now():%Y-%m-%d %H:%M:%S}", _rule()]
    return "\n".join(lines)


# =====================================================================
#  Cumulative
# =====================================================================
def performance_report(journal: Journal, cfg: Config, limit: int | None = None) -> str:
    trades = journal.closed_trades(limit)
    base = cfg.risk.starting_equity
    stats = compute_stats(trades, base_equity=base)
    curve = journal.equity_curve()

    lines = [
        _rule(),
        "CUMULATIVE PERFORMANCE",
        _rule(),
        "",
        "HEADLINE",
        f"  trades .............. {stats['trades']}",
        f"  win rate ............ {stats['win_rate']:.1%} "
        f"({stats['wins']}W / {stats['losses']}L)",
        f"  net P&L ............. {_money(stats['net_pnl'])}",
        f"  gross P&L ........... {_money(stats['gross_pnl'])}",
        f"  fees ................ {_money(stats['fees'])}",
        f"  expectancy .......... {stats['expectancy']:.4%} per trade",
        f"  average R ........... {stats['avg_r']:+.3f}",
        f"  profit factor ....... {stats['profit_factor']}",
        f"  max drawdown ........ {stats['max_drawdown']:.2%}",
        f"  Sharpe (annualised) . {stats['sharpe']}",
        f"  worst losing streak . {stats['max_consecutive_losses']}",
        f"  day trades .......... {stats['day_trades']}",
        "",
    ]

    if not curve.empty:
        eq = curve["equity"]
        lines += [
            "EQUITY",
            f"  starting ............ {_money(float(eq.iloc[0]))}",
            f"  current ............. {_money(float(eq.iloc[-1]))}",
            f"  peak ................ {_money(float(eq.max()))}",
            f"  trough .............. {_money(float(eq.min()))}",
            f"  total return ........ {(float(eq.iloc[-1]) / float(eq.iloc[0]) - 1):+.2%}",
            "",
        ]

    for col, title in (("strategy", "BY STRATEGY"), ("symbol", "BY SYMBOL"),
                       ("direction", "BY DIRECTION"), ("exit_reason", "BY EXIT REASON")):
        tbl = stats_by(trades, col, base_equity=base)
        if not tbl.empty:
            lines += [title, "-" * 78, tbl.to_string(index=False), ""]

    # Time-of-day edge.
    if not trades.empty and "entry_ts" in trades.columns:
        tmp = trades.copy()
        tmp["hour"] = pd.to_datetime(tmp["entry_ts"], errors="coerce").dt.hour
        by_hour = stats_by(tmp.dropna(subset=["hour"]), "hour", base_equity=base)
        if not by_hour.empty:
            lines += ["BY ENTRY HOUR", "-" * 78, by_hour.to_string(index=False), ""]

    revs = journal.reviews(limit=5)
    if not revs.empty:
        lines += ["RECENT REVIEWS", "-" * 78]
        for _, r in revs.iterrows():
            lines.append(f"  {str(r['ts'])[:16]}  [{r['scope']}]  "
                         f"{r['trades_reviewed']} trades")
            for l in str(r["lessons"] or "").split("\n"):
                if l.strip():
                    lines.append(f"      - {l.strip()[:70]}")
        lines.append("")

    lines += [_rule(), f"generated {datetime.now():%Y-%m-%d %H:%M:%S}", _rule()]
    return "\n".join(lines)


# =====================================================================
#  HTML
# =====================================================================
def html_report(journal: Journal, cfg: Config, out_path: Path | None = None) -> Path:
    """Self-contained HTML dashboard (inline SVG equity curve, no external assets)."""
    trades = journal.closed_trades()
    base = cfg.risk.starting_equity
    stats = compute_stats(trades, base_equity=base)
    curve = journal.equity_curve()

    # --- equity sparkline ---
    svg = "<p class='muted'>No equity marks recorded yet.</p>"
    if len(curve) > 1:
        eq = curve["equity"].astype(float).to_numpy()
        w, h, pad = 900, 260, 30
        lo, hi = float(eq.min()), float(eq.max())
        rngv = max(hi - lo, 1e-9)
        pts = [
            (pad + i * (w - 2 * pad) / max(len(eq) - 1, 1),
             h - pad - (v - lo) / rngv * (h - 2 * pad))
            for i, v in enumerate(eq)
        ]
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
        area = path + f" L{pts[-1][0]:.1f},{h - pad} L{pts[0][0]:.1f},{h - pad} Z"
        up = eq[-1] >= eq[0]
        stroke = "var(--pos)" if up else "var(--neg)"
        svg = f"""<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img"
     aria-label="Equity curve from {_money(float(eq[0]))} to {_money(float(eq[-1]))}">
  <path d="{area}" fill="{stroke}" opacity="0.12"/>
  <path d="{path}" fill="none" stroke="{stroke}" stroke-width="2"
        stroke-linejoin="round" stroke-linecap="round"/>
</svg>"""

    def tile(label: str, value: str, tone: str = "") -> str:
        return (f"<div class='tile'><div class='tile-label'>{html.escape(label)}</div>"
                f"<div class='tile-value {tone}'>{html.escape(value)}</div></div>")

    tone = "pos" if stats["net_pnl"] >= 0 else "neg"
    tiles = "".join([
        tile("Net P&L", _money(stats["net_pnl"]), tone),
        tile("Trades", str(stats["trades"])),
        tile("Win rate", f"{stats['win_rate']:.1%}"),
        tile("Expectancy", f"{stats['expectancy']:.3%}"),
        tile("Average R", f"{stats['avg_r']:+.2f}",
             "pos" if stats["avg_r"] >= 0 else "neg"),
        tile("Profit factor", str(stats["profit_factor"])),
        tile("Max drawdown", f"{stats['max_drawdown']:.1%}"),
        tile("Fees", _money(stats["fees"])),
    ])

    def table(df: pd.DataFrame, caption: str) -> str:
        if df is None or df.empty:
            return ""
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
        rows = ""
        for _, r in df.iterrows():
            cells = ""
            for c in df.columns:
                v = r[c]
                cls = ""
                if isinstance(v, (int, float, np.floating)) and c in (
                    "net_pnl", "expectancy", "avg_r", "r_multiple"
                ):
                    cls = " class='pos'" if float(v) >= 0 else " class='neg'"
                if isinstance(v, float):
                    v = f"{v:,.4g}"
                cells += f"<td{cls}>{html.escape(str(v))}</td>"
            rows += f"<tr>{cells}</tr>"
        return (f"<h2>{html.escape(caption)}</h2><div class='scroll'>"
                f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>")

    recent = trades.tail(40)[
        [c for c in ["exit_ts", "symbol", "direction", "strategy", "qty", "entry_price",
                     "exit_price", "exit_reason", "net_pnl", "r_multiple"]
         if c in trades.columns]
    ] if not trades.empty else pd.DataFrame()

    body = f"""<title>Trady Performance</title>
<style>
  :root {{
    --bg:#ffffff; --fg:#16181d; --muted:#5f6672; --line:#e3e6ec;
    --card:#f7f8fa; --pos:#0a7d4a; --neg:#c0392b; --accent:#2f6feb;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#0f1115; --fg:#e8eaed; --muted:#9aa2b1; --line:#262b34;
      --card:#171a21; --pos:#3ddc97; --neg:#ff6b6b; --accent:#6ea8ff;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#0f1115; --fg:#e8eaed; --muted:#9aa2b1; --line:#262b34;
    --card:#171a21; --pos:#3ddc97; --neg:#ff6b6b; --accent:#6ea8ff;
  }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--fg); margin:0; padding:32px 20px;
         font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
  .wrap {{ max-width:1000px; margin:0 auto; }}
  h1 {{ font-size:1.6rem; margin:0 0 4px; letter-spacing:-0.01em; }}
  h2 {{ font-size:1.05rem; margin:32px 0 10px; letter-spacing:-0.01em; }}
  .muted {{ color:var(--muted); font-size:0.88rem; }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
            gap:12px; margin:22px 0; }}
  .tile {{ background:var(--card); border:1px solid var(--line);
           border-radius:10px; padding:14px 16px; }}
  .tile-label {{ color:var(--muted); font-size:0.76rem; text-transform:uppercase;
                 letter-spacing:0.05em; }}
  .tile-value {{ font-size:1.35rem; font-weight:600; margin-top:5px;
                 font-variant-numeric:tabular-nums; }}
  .pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }}
  svg {{ width:100%; height:auto; background:var(--card);
         border:1px solid var(--line); border-radius:10px; }}
  .scroll {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; font-size:0.86rem;
           font-variant-numeric:tabular-nums; }}
  th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line);
           white-space:nowrap; }}
  th {{ color:var(--muted); font-weight:600; font-size:0.74rem;
        text-transform:uppercase; letter-spacing:0.04em; }}
  footer {{ margin-top:36px; padding-top:14px; border-top:1px solid var(--line); }}
</style>
<div class="wrap">
  <h1>Trady — Performance</h1>
  <p class="muted">Generated {datetime.now():%Y-%m-%d %H:%M:%S} ·
     starting equity {_money(base)} · broker {html.escape(cfg.execution.broker)}</p>
  <div class="tiles">{tiles}</div>
  <h2>Equity curve</h2>
  {svg}
  {table(stats_by(trades, 'strategy', base_equity=base), 'By strategy')}
  {table(stats_by(trades, 'symbol', base_equity=base), 'By symbol')}
  {table(stats_by(trades, 'exit_reason', base_equity=base), 'By exit reason')}
  {table(recent, 'Recent trades')}
  <footer class="muted">
    Past performance does not indicate future results. This report describes what
    this system did, not what it will do.
  </footer>
</div>"""

    out = Path(out_path or (Path(cfg.reports_dir) / "performance.html"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    return out


def save_daily(journal: Journal, cfg: Config, day: date | None = None) -> Path:
    day = day or date.today()
    out = Path(cfg.reports_dir) / f"daily_{day.isoformat()}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(daily_report(journal, cfg, day), encoding="utf-8")
    return out
