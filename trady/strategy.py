"""Signal generation: turn evidence into a decision, with its reasoning attached.

Design rule: a signal is never a bare direction. It carries every piece of
evidence that produced it, each with its weight and contribution, so that the
journal records *why* a trade was taken and the learning loop can later ask which
kinds of evidence actually paid.

Strategies implemented (all named in the books):
  reversal   — candlestick reversal at support/resistance ("East meets West")
  trend      — pull back inside an established trend, enter on resumption
  breakout   — close beyond a range, volume-confirmed
  range      — buy support / sell resistance inside a channel
  contrarian — fade a stretched move (explicitly the riskiest; fights the trend)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from . import patterns as pat
from . import structure as st
from .config import Config, SignalConfig
from .indicators import atr, enrich
from .risk import SizingResult, position_size, stop_and_target


@dataclass
class Evidence:
    """One reason to take (or avoid) a trade."""

    source: str        # matches a key in SignalConfig.weights
    detail: str
    direction: str     # "bullish" | "bearish"
    raw: float         # 0..1 strength before weighting
    weight: float

    @property
    def contribution(self) -> float:
        sign = 1.0 if self.direction == "bullish" else -1.0
        return sign * self.raw * self.weight

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "detail": self.detail,
            "direction": self.direction,
            "raw": round(self.raw, 3),
            "weight": round(self.weight, 3),
            "contribution": round(self.contribution, 3),
        }


@dataclass
class Signal:
    symbol: str
    timestamp: object
    direction: str            # "long" | "short"
    strategy: str
    entry: float
    stop: float
    target: float
    score: float              # absolute confluence score
    evidence: list[Evidence] = field(default_factory=list)
    structure: dict = field(default_factory=dict)
    sizing: SizingResult | None = None
    rejected_reason: str | None = None

    @property
    def actionable(self) -> bool:
        return self.rejected_reason is None and (self.sizing is None or self.sizing.ok)

    @property
    def reward_risk(self) -> float:
        r = abs(self.entry - self.stop)
        return abs(self.target - self.entry) / r if r > 0 else 0.0

    def rationale(self) -> str:
        """Plain-English explanation, written into the trade journal."""
        top = sorted(self.evidence, key=lambda e: abs(e.contribution), reverse=True)[:5]
        bullets = "\n".join(
            f"  - [{e.source}] {e.detail} ({e.contribution:+.2f})" for e in top
        )
        head = (
            f"{self.direction.upper()} {self.symbol} @ {self.entry:.2f} "
            f"via {self.strategy} (score {self.score:.2f}, R:R {self.reward_risk:.2f})"
        )
        tail = f"\n  stop {self.stop:.2f} / target {self.target:.2f}"
        if self.rejected_reason:
            tail += f"\n  REJECTED: {self.rejected_reason}"
        return f"{head}\n{bullets}{tail}"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": str(self.timestamp),
            "direction": self.direction,
            "strategy": self.strategy,
            "entry": round(self.entry, 4),
            "stop": round(self.stop, 4),
            "target": round(self.target, 4),
            "score": round(self.score, 3),
            "reward_risk": round(self.reward_risk, 3),
            "evidence": [e.to_dict() for e in self.evidence],
            "structure": self.structure,
            "shares": self.sizing.shares if self.sizing else 0,
            "risk_dollars": self.sizing.risk_dollars if self.sizing else 0.0,
            "sizing_method": self.sizing.method if self.sizing else None,
            "caps_applied": self.sizing.caps_applied if self.sizing else [],
            "rejected_reason": self.rejected_reason,
        }


# =====================================================================
#  Evidence collection
# =====================================================================
def _w(cfg: SignalConfig, key: str) -> float:
    return float(cfg.weights.get(key, 1.0))


def gather_evidence(
    df: pd.DataFrame, snap: dict, cfg: SignalConfig, lookback_bars: int = 3
) -> list[Evidence]:
    """Read the chart once and emit every piece of directional evidence."""
    ev: list[Evidence] = []
    price = snap["price"]

    # ---- candlestick reversal patterns ----
    for hit in pat.detect_latest(df, within=lookback_bars):
        conf = pat.confirmed(hit, df)
        raw = hit.strength * (1.0 if conf or not hit.needs_confirmation else 0.55)
        detail = hit.name.replace("_", " ")
        if hit.needs_confirmation:
            detail += " (confirmed)" if conf else " (awaiting confirmation)"
        ev.append(Evidence("candlestick", detail, hit.direction, raw, _w(cfg, "candlestick")))

    # ---- trend ----
    tdir, tstr, phase = snap["trend_direction"], snap["trend_strength"], snap["trend_phase"]
    if tdir != "sideways":
        ev.append(
            Evidence(
                "trend",
                f"{tdir}trend, {phase} phase, strength {tstr:.2f}",
                "bullish" if tdir == "up" else "bearish",
                tstr,
                _w(cfg, "trend"),
            )
        )

    # ---- support / resistance proximity ----
    if snap["support"] and snap["support_distance_pct"] is not None:
        d = snap["support_distance_pct"]
        if d < 0.01:
            ev.append(
                Evidence(
                    "support_resistance",
                    f"price {d:.2%} above support {snap['support']:.2f} "
                    f"(strength {snap['support_strength']:.2f})",
                    "bullish",
                    snap["support_strength"] * (1.0 - d / 0.01),
                    _w(cfg, "support_resistance"),
                )
            )
    if snap["resistance"] and snap["resistance_distance_pct"] is not None:
        d = snap["resistance_distance_pct"]
        if d < 0.01:
            ev.append(
                Evidence(
                    "support_resistance",
                    f"price {d:.2%} below resistance {snap['resistance']:.2f} "
                    f"(strength {snap['resistance_strength']:.2f})",
                    "bearish",
                    snap["resistance_strength"] * (1.0 - d / 0.01),
                    _w(cfg, "support_resistance"),
                )
            )

    # ---- momentum indicators ----
    last = df.iloc[-1]
    if "rsi_14" in df.columns and not pd.isna(last.get("rsi_14")):
        r = float(last["rsi_14"])
        if r < 30:
            ev.append(Evidence("momentum", f"RSI {r:.0f} oversold", "bullish",
                               min(1.0, (30 - r) / 15), _w(cfg, "momentum")))
        elif r > 70:
            ev.append(Evidence("momentum", f"RSI {r:.0f} overbought", "bearish",
                               min(1.0, (r - 70) / 15), _w(cfg, "momentum")))
    if "macd_hist" in df.columns and len(df) > 2:
        h, hp = float(last.get("macd_hist", 0) or 0), float(df["macd_hist"].iloc[-2] or 0)
        if h > 0 and hp <= 0:
            ev.append(Evidence("momentum", "MACD histogram crossed up", "bullish",
                               0.6, _w(cfg, "momentum")))
        elif h < 0 and hp >= 0:
            ev.append(Evidence("momentum", "MACD histogram crossed down", "bearish",
                               0.6, _w(cfg, "momentum")))
    if "mfi_14" in df.columns and not pd.isna(last.get("mfi_14")):
        m = float(last["mfi_14"])
        if m < 20:
            ev.append(Evidence("momentum", f"money-flow index {m:.0f} oversold", "bullish",
                               0.5, _w(cfg, "momentum")))
        elif m > 80:
            ev.append(Evidence("momentum", f"money-flow index {m:.0f} overbought", "bearish",
                               0.5, _w(cfg, "momentum")))

    # ---- volume ----
    if "vol_surge" in df.columns and not pd.isna(last.get("vol_surge")):
        vs = float(last["vol_surge"])
        if vs >= cfg.volume_surge_ratio:
            bar_dir = "bullish" if last["close"] >= last["open"] else "bearish"
            ev.append(
                Evidence("volume", f"volume {vs:.1f}x its 20-bar average", bar_dir,
                         min(1.0, (vs - 1.0) / 2.0), _w(cfg, "volume"))
            )
    if "obv" in df.columns and len(df) > 20:
        obv_now, obv_then = float(df["obv"].iloc[-1]), float(df["obv"].iloc[-20])
        px_now, px_then = float(df["close"].iloc[-1]), float(df["close"].iloc[-20])
        if obv_now > obv_then and px_now <= px_then:
            ev.append(Evidence("volume", "on-balance volume rising while price is not "
                               "— accumulation", "bullish", 0.55, _w(cfg, "volume")))
        elif obv_now < obv_then and px_now >= px_then:
            ev.append(Evidence("volume", "on-balance volume falling while price is not "
                               "— distribution", "bearish", 0.55, _w(cfg, "volume")))

    # ---- breakout ----
    bo = snap.get("recent_breakout")
    if bo:
        raw = 0.75 if bo["volume_confirmed"] else 0.3
        note = "volume-confirmed" if bo["volume_confirmed"] else "NOT volume-confirmed (false-breakout risk)"
        ev.append(
            Evidence("chart_pattern", f"{bo['direction']} breakout of {bo['level']:.2f}, {note}",
                     "bullish" if bo["direction"] == "up" else "bearish", raw,
                     _w(cfg, "chart_pattern"))
        )

    # ---- gap ----
    gp = snap.get("recent_gap")
    if gp and not gp["filled"]:
        ev.append(
            Evidence("gap", f"unfilled {gp['direction']} gap of {gp['size_pct']:.2%} "
                     f"{gp['bars_ago']} bars ago",
                     "bullish" if gp["direction"] == "up" else "bearish",
                     min(1.0, gp["size_pct"] / 0.02), _w(cfg, "gap"))
        )

    # ---- classic chart patterns ----
    for c in snap.get("chart_patterns", []):
        ev.append(
            Evidence("chart_pattern", c["name"].replace("_", " "), c["direction"],
                     c["strength"], _w(cfg, "chart_pattern"))
        )

    return ev


# =====================================================================
#  Strategies
# =====================================================================
def _score(ev: list[Evidence]) -> float:
    return float(sum(e.contribution for e in ev))


def _has(ev: list[Evidence], source: str, direction: str | None = None) -> bool:
    return any(
        e.source == source and (direction is None or e.direction == direction) for e in ev
    )


def classify_strategy(ev: list[Evidence], snap: dict, direction: str) -> str:
    """Name the setup, so per-strategy performance can be tracked separately."""
    want = "bullish" if direction == "long" else "bearish"
    bo = snap.get("recent_breakout")
    if bo and ((bo["direction"] == "up") == (direction == "long")) and bo["volume_confirmed"]:
        return "breakout"
    if _has(ev, "candlestick", want) and _has(ev, "support_resistance", want):
        return "reversal"
    trend_up = snap["trend_direction"] == "up"
    if snap["trend_direction"] != "sideways" and (trend_up == (direction == "long")):
        return "trend" if snap["trend_phase"] in ("continuation", "accumulation", "retracement") else "trend"
    if snap["trend_direction"] == "sideways":
        return "range"
    return "contrarian"


# =====================================================================
#  Public entry point
# =====================================================================
def generate(
    symbol: str,
    df: pd.DataFrame,
    cfg: Config,
    equity: float,
    stats: dict | None = None,
    now: datetime | None = None,
) -> Signal | None:
    """Analyse one symbol and return a Signal, or None when nothing qualifies.

    A Signal is still returned when it fails a *quality* filter (with
    `rejected_reason` set) so the journal can record near-misses; None means
    there was not even a directional bias worth recording.
    """
    if len(df) < 60:
        return None

    frame = df if "rsi_14" in df.columns else enrich(df)
    snap = st.snapshot(frame)
    scfg = cfg.signal
    price = snap["price"]

    # ---- liquidity / tradability filters ----
    reject = None
    if not (scfg.min_price <= price <= scfg.max_price):
        reject = f"price {price:.2f} outside tradable band " \
                 f"[{scfg.min_price}, {scfg.max_price}]"
    avg_vol = float(frame["volume"].tail(20).mean())
    if reject is None and avg_vol < scfg.min_avg_volume / 78:  # per-5m-bar proxy
        reject = f"average volume {avg_vol:,.0f}/bar too thin to trade"

    ev = gather_evidence(frame, snap, scfg)
    if not ev:
        return None

    score = _score(ev)
    direction = "long" if score > 0 else "short"
    want = "bullish" if direction == "long" else "bearish"

    # ---- quality gates ----
    if reject is None and abs(score) < scfg.min_confluence_score:
        reject = (
            f"confluence {abs(score):.2f} below the {scfg.min_confluence_score:.2f} "
            "minimum — not enough agreement to risk capital"
        )

    if reject is None and scfg.require_trend_alignment:
        tdir = snap["trend_direction"]
        aligned = (
            (direction == "long" and tdir == "up")
            or (direction == "short" and tdir == "down")
            or tdir == "sideways"
        )
        counter_ok = _has(ev, "candlestick", want) and _has(ev, "support_resistance", want)
        if not aligned and not counter_ok:
            reject = (
                f"{direction} against a {tdir}trend without a reversal pattern at a level "
                "— the market is right in the short term"
            )

    if reject is None and scfg.require_volume_confirmation and not _has(ev, "volume"):
        reject = "no volume confirmation — price tells what, volume tells how many"

    # ---- levels ----
    a = float(frame["atr_14"].iloc[-1]) if "atr_14" in frame.columns else float(atr(frame).iloc[-1])
    if not np.isfinite(a) or a <= 0:
        a = price * 0.01
    entry = price
    stop, target = stop_and_target(entry, a, direction, cfg.risk)

    # Prefer a structural stop just beyond the level that would invalidate the idea.
    if direction == "long" and snap["support"]:
        structural = snap["support"] * 0.999
        if structural < entry:
            stop = max(stop, structural) if structural > stop else stop
    elif direction == "short" and snap["resistance"]:
        structural = snap["resistance"] * 1.001
        if structural > entry:
            stop = min(stop, structural) if structural < stop else stop

    # Target the opposing level when it is closer than the ATR target.
    if direction == "long" and snap["resistance"]:
        if entry < snap["resistance"] < target:
            target = snap["resistance"] * 0.999
    elif direction == "short" and snap["support"]:
        if target < snap["support"] < entry:
            target = snap["support"] * 1.001

    rr = abs(target - entry) / max(abs(entry - stop), 1e-9)
    if reject is None and rr < cfg.risk.min_reward_risk:
        reject = (
            f"reward:risk {rr:.2f} below the {cfg.risk.min_reward_risk:.2f} minimum "
            "— the payoff does not justify the stop"
        )

    strategy = classify_strategy(ev, snap, direction)
    sizing = position_size(equity, entry, stop, target, cfg.risk, stats=stats)
    if reject is None and not sizing.ok:
        reject = f"position sizing returned 0 shares ({', '.join(sizing.caps_applied) or 'n/a'})"

    return Signal(
        symbol=symbol,
        timestamp=now or frame.index[-1],
        direction=direction,
        strategy=strategy,
        entry=round(entry, 4),
        stop=round(stop, 4),
        target=round(target, 4),
        score=abs(score),
        evidence=ev,
        structure=snap,
        sizing=sizing,
        rejected_reason=reject,
    )


def scan(
    frames: dict[str, pd.DataFrame],
    cfg: Config,
    equity: float,
    stats: dict | None = None,
) -> list[Signal]:
    """Rank actionable signals across a watchlist, best confluence first."""
    out: list[Signal] = []
    for sym, df in frames.items():
        try:
            sig = generate(sym, df, cfg, equity, stats=stats)
        except Exception as exc:  # one bad symbol must not kill the scan
            print(f"  ! {sym}: {type(exc).__name__}: {exc}")
            continue
        if sig is not None:
            out.append(sig)
    out.sort(key=lambda s: (s.actionable, s.score), reverse=True)
    return out
