"""Trade journal — the agent's permanent memory.

Everything the system does is written here: signals it considered (including the
ones it rejected and why), orders, fills, closed trades with full P&L
attribution, daily equity marks, and the reviews the learning loop produces.

Two reasons this is the centre of the design rather than a log file:

1. The books are emphatic that a trading diary recording *why* each trade was
   taken is what separates a system that improves from one that just churns.
   Rejected signals matter as much as taken ones — they are the counterfactual.
2. It survives restarts. State lives in SQLite, so a session can be interrupted
   and resumed without losing history.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS signal (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    strategy TEXT,
    entry REAL, stop REAL, target REAL,
    score REAL, reward_risk REAL,
    shares INTEGER, risk_dollars REAL,
    sizing_method TEXT,
    taken INTEGER NOT NULL DEFAULT 0,
    rejected_reason TEXT,
    evidence_json TEXT,
    structure_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_signal_ts ON signal(ts);
CREATE INDEX IF NOT EXISTS ix_signal_symbol ON signal(symbol);

CREATE TABLE IF NOT EXISTS trade (
    id INTEGER PRIMARY KEY,
    signal_id INTEGER REFERENCES signal(id),
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    strategy TEXT,
    qty INTEGER NOT NULL,
    entry_ts TEXT NOT NULL,
    entry_price REAL NOT NULL,
    planned_stop REAL,
    planned_target REAL,
    exit_ts TEXT,
    exit_price REAL,
    exit_reason TEXT,
    gross_pnl REAL,
    fees REAL DEFAULT 0,
    net_pnl REAL,
    pnl_pct REAL,
    r_multiple REAL,
    mae REAL,            -- maximum adverse excursion
    mfe REAL,            -- maximum favourable excursion
    bars_held INTEGER,
    is_day_trade INTEGER DEFAULT 0,
    broker TEXT,
    notes TEXT,
    open INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_trade_open ON trade(open);
CREATE INDEX IF NOT EXISTS ix_trade_entry ON trade(entry_ts);

CREATE TABLE IF NOT EXISTS equity_mark (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    day TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL,
    positions_value REAL,
    realized_pnl_today REAL,
    unrealized_pnl REAL,
    open_positions INTEGER,
    trades_today INTEGER,
    note TEXT
);
CREATE INDEX IF NOT EXISTS ix_equity_day ON equity_mark(day);

CREATE TABLE IF NOT EXISTS review (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    scope TEXT NOT NULL,          -- 'daily' | 'periodic' | 'manual'
    trades_reviewed INTEGER,
    metrics_json TEXT,
    changes_json TEXT,            -- weight/param adjustments made
    lessons TEXT
);

CREATE TABLE IF NOT EXISTS param_history (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    review_id INTEGER REFERENCES review(id),
    param TEXT NOT NULL,
    old_value REAL,
    new_value REAL,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS event (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,           -- halt|resume|error|pdt_block|risk_block|info
    detail TEXT
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Journal:
    path: Path

    def __post_init__(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._con() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _con(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ---------------------------------------------------------- signals
    def record_signal(self, sig, taken: bool = False) -> int:
        d = sig.to_dict()
        with self._con() as con:
            cur = con.execute(
                """INSERT INTO signal
                   (ts, symbol, direction, strategy, entry, stop, target, score,
                    reward_risk, shares, risk_dollars, sizing_method, taken,
                    rejected_reason, evidence_json, structure_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    d["timestamp"], d["symbol"], d["direction"], d["strategy"],
                    d["entry"], d["stop"], d["target"], d["score"], d["reward_risk"],
                    d["shares"], d["risk_dollars"], d["sizing_method"],
                    int(taken), d["rejected_reason"],
                    json.dumps(d["evidence"]), json.dumps(d["structure"], default=str),
                ),
            )
            return int(cur.lastrowid)

    # ----------------------------------------------------------- trades
    def open_trade(
        self,
        symbol: str,
        direction: str,
        qty: int,
        entry_price: float,
        entry_ts,
        *,
        strategy: str = "",
        signal_id: int | None = None,
        planned_stop: float | None = None,
        planned_target: float | None = None,
        broker: str = "paper",
        notes: str = "",
    ) -> int:
        with self._con() as con:
            cur = con.execute(
                """INSERT INTO trade
                   (signal_id, symbol, direction, strategy, qty, entry_ts, entry_price,
                    planned_stop, planned_target, broker, notes, open)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,1)""",
                (signal_id, symbol, direction, strategy, qty, str(entry_ts),
                 float(entry_price), planned_stop, planned_target, broker, notes),
            )
            return int(cur.lastrowid)

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_ts,
        exit_reason: str,
        *,
        fees: float = 0.0,
        mae: float | None = None,
        mfe: float | None = None,
        bars_held: int | None = None,
    ) -> dict:
        with self._con() as con:
            row = con.execute("SELECT * FROM trade WHERE id=?", (trade_id,)).fetchone()
            if row is None:
                raise KeyError(f"no trade {trade_id}")

            qty, entry = row["qty"], row["entry_price"]
            sign = 1.0 if row["direction"] == "long" else -1.0
            gross = (float(exit_price) - float(entry)) * qty * sign
            net = gross - float(fees)
            pnl_pct = net / (entry * qty) if entry and qty else 0.0

            risk_per_share = (
                abs(entry - row["planned_stop"]) if row["planned_stop"] else 0.0
            )
            r_mult = (
                net / (risk_per_share * qty) if risk_per_share and qty else None
            )

            same_day = str(row["entry_ts"])[:10] == str(exit_ts)[:10]

            con.execute(
                """UPDATE trade SET exit_ts=?, exit_price=?, exit_reason=?,
                       gross_pnl=?, fees=?, net_pnl=?, pnl_pct=?, r_multiple=?,
                       mae=?, mfe=?, bars_held=?, is_day_trade=?, open=0
                   WHERE id=?""",
                (str(exit_ts), float(exit_price), exit_reason, gross, float(fees),
                 net, pnl_pct, r_mult, mae, mfe, bars_held, int(same_day), trade_id),
            )
        return {
            "trade_id": trade_id, "gross_pnl": gross, "fees": fees, "net_pnl": net,
            "pnl_pct": pnl_pct, "r_multiple": r_mult, "is_day_trade": same_day,
        }

    def open_trades(self) -> pd.DataFrame:
        with self._con() as con:
            return pd.read_sql_query("SELECT * FROM trade WHERE open=1", con)

    def closed_trades(self, limit: int | None = None) -> pd.DataFrame:
        q = "SELECT * FROM trade WHERE open=0 ORDER BY exit_ts DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        with self._con() as con:
            df = pd.read_sql_query(q, con)
        return df.iloc[::-1].reset_index(drop=True)

    def day_trades(self, since: date | None = None) -> list[tuple[str, date]]:
        """Same-day round trips, for the PDT tracker."""
        q = "SELECT symbol, substr(exit_ts,1,10) AS d FROM trade WHERE open=0 AND is_day_trade=1"
        params: tuple = ()
        if since:
            q += " AND substr(exit_ts,1,10) >= ?"
            params = (since.isoformat(),)
        with self._con() as con:
            rows = con.execute(q, params).fetchall()
        return [(r["symbol"], date.fromisoformat(r["d"])) for r in rows]

    # ----------------------------------------------------------- equity
    def mark_equity(
        self, equity: float, *, cash: float = 0.0, positions_value: float = 0.0,
        realized_pnl_today: float = 0.0, unrealized_pnl: float = 0.0,
        open_positions: int = 0, trades_today: int = 0, ts=None, note: str = "",
    ) -> None:
        ts = str(ts or _now())
        with self._con() as con:
            con.execute(
                """INSERT INTO equity_mark
                   (ts, day, equity, cash, positions_value, realized_pnl_today,
                    unrealized_pnl, open_positions, trades_today, note)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (ts, ts[:10], float(equity), cash, positions_value,
                 realized_pnl_today, unrealized_pnl, open_positions, trades_today, note),
            )

    def equity_curve(self) -> pd.DataFrame:
        with self._con() as con:
            df = pd.read_sql_query("SELECT * FROM equity_mark ORDER BY ts", con)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
        return df

    # ---------------------------------------------------------- reviews
    def record_review(
        self, scope: str, trades_reviewed: int, metrics: dict,
        changes: dict, lessons: str,
    ) -> int:
        with self._con() as con:
            cur = con.execute(
                """INSERT INTO review (ts, scope, trades_reviewed, metrics_json,
                                       changes_json, lessons)
                   VALUES (?,?,?,?,?,?)""",
                (_now(), scope, trades_reviewed,
                 json.dumps(metrics, default=str), json.dumps(changes, default=str), lessons),
            )
            rid = int(cur.lastrowid)
            for param, (old, new, reason) in changes.items():
                con.execute(
                    """INSERT INTO param_history (ts, review_id, param, old_value,
                                                  new_value, reason)
                       VALUES (?,?,?,?,?,?)""",
                    (_now(), rid, param, old, new, reason),
                )
            return rid

    def reviews(self, limit: int = 20) -> pd.DataFrame:
        with self._con() as con:
            return pd.read_sql_query(
                f"SELECT * FROM review ORDER BY ts DESC LIMIT {int(limit)}", con
            )

    def log(self, kind: str, detail: str) -> None:
        with self._con() as con:
            con.execute(
                "INSERT INTO event (ts, kind, detail) VALUES (?,?,?)",
                (_now(), kind, detail),
            )

    def events(self, limit: int = 50) -> pd.DataFrame:
        with self._con() as con:
            return pd.read_sql_query(
                f"SELECT * FROM event ORDER BY ts DESC LIMIT {int(limit)}", con
            )

    # -------------------------------------------------------- analytics
    def stats(self, limit: int | None = None, strategy: str | None = None,
              base_equity: float | None = None) -> dict:
        """Performance summary used both for reporting and for Kelly sizing."""
        df = self.closed_trades(limit)
        if strategy:
            df = df[df["strategy"] == strategy]
        return compute_stats(df, base_equity=base_equity)


# =====================================================================
#  Metrics
# =====================================================================
def compute_stats(df: pd.DataFrame, base_equity: float | None = None) -> dict:
    """Win rate, expectancy, R-multiples, drawdown, Sharpe, profit factor.

    `base_equity` anchors the drawdown calculation. Without it, drawdown would be
    measured against a cumulative-P&L series that can be negative, which produces
    meaningless percentages (a two-trade losing sequence is not an 89% drawdown).
    When omitted it is inferred from the largest position notional in the sample.
    """
    empty = {
        "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
        "avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0,
        "expectancy_r": 0.0, "profit_factor": 0.0, "net_pnl": 0.0,
        "gross_pnl": 0.0, "fees": 0.0, "best": 0.0, "worst": 0.0,
        "max_drawdown": 0.0, "sharpe": 0.0, "avg_r": 0.0,
        "max_consecutive_losses": 0, "day_trades": 0,
    }
    if df is None or df.empty:
        return empty

    net = df["net_pnl"].astype(float).fillna(0.0)
    pct = df["pnl_pct"].astype(float).fillna(0.0)
    wins, losses = net[net > 0], net[net <= 0]

    avg_win = float(pct[net > 0].mean()) if len(wins) else 0.0
    avg_loss = float(abs(pct[net <= 0].mean())) if len(losses) else 0.0
    win_rate = len(wins) / len(net) if len(net) else 0.0

    # Equity curve from the trade sequence, anchored on a positive base so the
    # drawdown ratio stays meaningful.
    if base_equity is None or base_equity <= 0:
        notional = (
            (df["entry_price"].astype(float) * df["qty"].astype(float)).max()
            if {"entry_price", "qty"} <= set(df.columns) else 0.0
        )
        base_equity = max(float(notional) * 10.0, abs(float(net.sum())) * 10.0, 1.0)
    curve = base_equity + net.cumsum()
    peak = curve.cummax()
    max_dd = float(((peak - curve) / peak.where(peak > 0, np.nan)).max())
    max_dd = 0.0 if not np.isfinite(max_dd) else max_dd

    # Consecutive losses.
    streak = best_streak = 0
    for v in net:
        if v <= 0:
            streak += 1
            best_streak = max(best_streak, streak)
        else:
            streak = 0

    r = df["r_multiple"].astype(float).dropna() if "r_multiple" in df.columns else pd.Series(dtype=float)
    sharpe = (
        float(pct.mean() / pct.std(ddof=1) * np.sqrt(252)) if len(pct) > 2 and pct.std(ddof=1) > 0 else 0.0
    )

    return {
        "trades": int(len(net)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 5),
        "avg_loss": round(avg_loss, 5),
        "expectancy": round(win_rate * avg_win - (1 - win_rate) * avg_loss, 6),
        "expectancy_r": round(float(r.mean()), 4) if len(r) else 0.0,
        "avg_r": round(float(r.mean()), 4) if len(r) else 0.0,
        "profit_factor": (
            round(float(wins.sum() / abs(losses.sum())), 3)
            if len(losses) and losses.sum() != 0 else float("inf") if len(wins) else 0.0
        ),
        "net_pnl": round(float(net.sum()), 2),
        "gross_pnl": round(float(df["gross_pnl"].astype(float).fillna(0).sum()), 2),
        "fees": round(float(df["fees"].astype(float).fillna(0).sum()), 2),
        "best": round(float(net.max()), 2),
        "worst": round(float(net.min()), 2),
        "max_drawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 3),
        "max_consecutive_losses": int(best_streak),
        "day_trades": int(df["is_day_trade"].fillna(0).sum()) if "is_day_trade" in df.columns else 0,
    }


def stats_by(df: pd.DataFrame, column: str, base_equity: float | None = None) -> pd.DataFrame:
    """Break performance down by strategy, symbol, direction, hour, ..."""
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame()
    rows = []
    for key, grp in df.groupby(column):
        s = compute_stats(grp, base_equity=base_equity)
        s[column] = key
        rows.append(s)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    cols = [column, "trades", "win_rate", "expectancy", "avg_r", "net_pnl",
            "profit_factor", "max_drawdown"]
    return out[[c for c in cols if c in out.columns]].sort_values(
        "net_pnl", ascending=False
    ).reset_index(drop=True)
