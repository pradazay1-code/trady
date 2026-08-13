"""Market structure: trend, support/resistance, gaps, breakouts, chart patterns.

This is the "Western" half of the analysis the candlestick book argues for
blending with the "Eastern" half: a mediocre candlestick pattern at a strong
support level is worth more than a textbook pattern in open space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import atr, sma


# =====================================================================
#  Swings and trend
# =====================================================================
@dataclass
class Swing:
    index: int
    price: float
    kind: str  # "high" | "low"


def swings(df: pd.DataFrame, order: int = 3) -> list[Swing]:
    """Fractal pivots: a bar higher/lower than `order` bars on both sides."""
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    out: list[Swing] = []
    for i in range(order, len(df) - order):
        window = slice(i - order, i + order + 1)
        if highs[i] == highs[window].max() and (highs[window] == highs[i]).sum() == 1:
            out.append(Swing(i, float(highs[i]), "high"))
        if lows[i] == lows[window].min() and (lows[window] == lows[i]).sum() == 1:
            out.append(Swing(i, float(lows[i]), "low"))
    return sorted(out, key=lambda s: s.index)


@dataclass
class TrendState:
    direction: str      # "up" | "down" | "sideways"
    strength: float     # 0..1
    phase: str          # accumulation|continuation|consolidation|retracement|distribution|reversal
    slope_pct: float
    notes: list[str] = field(default_factory=list)


def trend(df: pd.DataFrame, fast: int = 20, slow: int = 50, lookback: int = 40) -> TrendState:
    """Classify trend direction, strength and phase.

    Phases follow the book's cycle: accumulation, main/continuation,
    consolidation/congestion, retracement, distribution, reversal.
    """
    if len(df) < max(slow, lookback) + 5:
        return TrendState("sideways", 0.0, "consolidation", 0.0, ["insufficient history"])

    close = df["close"].astype(float)
    ma_f = sma(close, fast)
    ma_s = sma(close, slow)
    recent = close.iloc[-lookback:]
    slope = float((recent.iloc[-1] - recent.iloc[0]) / max(recent.iloc[0], 1e-9))

    # Higher highs / lower lows over the recent swing structure.
    sw = [s for s in swings(df.iloc[-lookback * 2 :], order=3)]
    hi = [s.price for s in sw if s.kind == "high"][-3:]
    lo = [s.price for s in sw if s.kind == "low"][-3:]
    hh = len(hi) >= 2 and hi[-1] > hi[-2]
    hl = len(lo) >= 2 and lo[-1] > lo[-2]
    lh = len(hi) >= 2 and hi[-1] < hi[-2]
    ll = len(lo) >= 2 and lo[-1] < lo[-2]

    f, s_ = float(ma_f.iloc[-1]), float(ma_s.iloc[-1])
    above = f > s_
    notes: list[str] = []

    # Direction.
    if slope > 0.004 and (above or (hh and hl)):
        direction = "up"
    elif slope < -0.004 and (not above or (lh and ll)):
        direction = "down"
    else:
        direction = "sideways"

    # Strength: normalised slope, MA separation, structure agreement.
    sep = abs(f - s_) / max(s_, 1e-9)
    structure = (hh and hl) or (lh and ll)
    strength = float(
        np.clip(min(1.0, abs(slope) / 0.03) * 0.5 + min(1.0, sep / 0.02) * 0.3
                + (0.2 if structure else 0.0), 0.0, 1.0)
    )

    # Phase.
    band = float((close.iloc[-lookback:].max() - close.iloc[-lookback:].min())
                 / max(close.iloc[-1], 1e-9))
    vol_now = float(df["volume"].iloc[-10:].mean())
    vol_prev = float(df["volume"].iloc[-40:-10].mean()) if len(df) > 40 else vol_now
    vol_rising = vol_now > vol_prev * 1.1

    if direction == "sideways" and band < 0.02:
        phase = "consolidation"
        notes.append("tight range — scalper's market, poor for trend entries")
    elif direction == "up" and hh and hl:
        phase = "continuation" if not vol_rising else "accumulation"
    elif direction == "down" and lh and ll:
        phase = "continuation" if not vol_rising else "distribution"
    elif direction == "up" and lh:
        phase = "retracement"
        notes.append("pullback inside an uptrend")
    elif direction == "down" and hh:
        phase = "retracement"
        notes.append("bounce inside a downtrend")
    else:
        phase = "consolidation"

    if direction != "sideways" and structure is False:
        notes.append("slope and swing structure disagree — treat trend as weak")

    return TrendState(direction, strength, phase, slope, notes)


# =====================================================================
#  Support and resistance
# =====================================================================
@dataclass
class Level:
    price: float
    kind: str        # "support" | "resistance"
    touches: int
    strength: float  # 0..1
    last_index: int

    def distance_pct(self, price: float) -> float:
        return abs(price - self.price) / max(price, 1e-9)


def levels(
    df: pd.DataFrame, order: int = 3, tolerance: float = 0.004, max_levels: int = 8
) -> list[Level]:
    """Cluster swing pivots into horizontal levels.

    A level touched more often, and more recently, is stronger.
    """
    sw = swings(df, order=order)
    if not sw:
        return []
    last = len(df) - 1
    clusters: list[dict] = []

    for s in sw:
        placed = False
        for c in clusters:
            if abs(s.price - c["price"]) / max(c["price"], 1e-9) <= tolerance:
                n = c["touches"]
                c["price"] = (c["price"] * n + s.price) / (n + 1)
                c["touches"] = n + 1
                c["last_index"] = max(c["last_index"], s.index)
                c["kinds"].append(s.kind)
                placed = True
                break
        if not placed:
            clusters.append(
                {"price": s.price, "touches": 1, "last_index": s.index, "kinds": [s.kind]}
            )

    price_now = float(df["close"].iloc[-1])
    out: list[Level] = []
    for c in clusters:
        if c["touches"] < 2:
            continue
        recency = 1.0 - min(1.0, (last - c["last_index"]) / max(len(df), 1))
        strength = float(np.clip(0.35 * min(1.0, c["touches"] / 4.0) + 0.65 * recency, 0, 1))
        kind = "resistance" if c["price"] > price_now else "support"
        out.append(Level(round(c["price"], 4), kind, c["touches"], strength, c["last_index"]))

    out.sort(key=lambda l: l.strength, reverse=True)
    return out[:max_levels]


def nearest_level(lv: list[Level], price: float, kind: str | None = None) -> Level | None:
    cand = [l for l in lv if kind is None or l.kind == kind]
    return min(cand, key=lambda l: l.distance_pct(price)) if cand else None


# =====================================================================
#  Gaps
# =====================================================================
@dataclass
class Gap:
    index: int
    direction: str  # "up" | "down"
    size_pct: float
    from_price: float
    to_price: float
    filled: bool


def gaps(df: pd.DataFrame, min_pct: float = 0.003, lookback: int = 60) -> list[Gap]:
    """Opening gaps: a break between one bar's range and the next bar's range."""
    start = max(1, len(df) - lookback)
    out: list[Gap] = []
    highs, lows = df["high"].to_numpy(float), df["low"].to_numpy(float)
    for i in range(start, len(df)):
        prev_h, prev_l = highs[i - 1], lows[i - 1]
        if lows[i] > prev_h:
            size = (lows[i] - prev_h) / max(prev_h, 1e-9)
            if size >= min_pct:
                filled = bool((lows[i:] <= prev_h).any())
                out.append(Gap(i, "up", size, prev_h, float(lows[i]), filled))
        elif highs[i] < prev_l:
            size = (prev_l - highs[i]) / max(prev_l, 1e-9)
            if size >= min_pct:
                filled = bool((highs[i:] >= prev_l).any())
                out.append(Gap(i, "down", size, prev_l, float(highs[i]), filled))
    return out


# =====================================================================
#  Breakouts
# =====================================================================
@dataclass
class Breakout:
    index: int
    direction: str
    level: float
    volume_confirmed: bool
    strength: float


def breakouts(
    df: pd.DataFrame, lookback: int = 20, vol_ratio: float = 1.3, buffer: float = 0.001
) -> list[Breakout]:
    """Close beyond the prior `lookback`-bar range.

    Volume confirmation matters: the books warn a breakout on flat volume is
    more likely a false breakout that reverses and traps whoever chased it.
    """
    if len(df) < lookback + 5:
        return []
    close = df["close"].to_numpy(float)
    high, low = df["high"].to_numpy(float), df["low"].to_numpy(float)
    vol = df["volume"].to_numpy(float)
    out: list[Breakout] = []

    for i in range(lookback, len(df)):
        hh = high[i - lookback : i].max()
        ll = low[i - lookback : i].min()
        avg_v = vol[max(0, i - lookback) : i].mean()
        vconf = bool(avg_v > 0 and vol[i] / avg_v >= vol_ratio)
        if close[i] > hh * (1 + buffer):
            mag = (close[i] - hh) / max(hh, 1e-9)
            out.append(
                Breakout(i, "up", float(hh), vconf,
                         float(np.clip(0.4 + 30 * mag + (0.25 if vconf else -0.15), 0, 1)))
            )
        elif close[i] < ll * (1 - buffer):
            mag = (ll - close[i]) / max(ll, 1e-9)
            out.append(
                Breakout(i, "down", float(ll), vconf,
                         float(np.clip(0.4 + 30 * mag + (0.25 if vconf else -0.15), 0, 1)))
            )
    return out


# =====================================================================
#  Classic chart patterns
# =====================================================================
@dataclass
class ChartPattern:
    name: str
    direction: str
    index: int
    strength: float
    notes: list[str] = field(default_factory=list)


def chart_patterns(df: pd.DataFrame, lookback: int = 90) -> list[ChartPattern]:
    """Head-and-shoulders, double top/bottom, flags and pennants."""
    if len(df) < 40:
        return []
    window = df.iloc[-lookback:] if len(df) > lookback else df
    offset = len(df) - len(window)
    sw = swings(window, order=3)
    highs = [s for s in sw if s.kind == "high"]
    lows = [s for s in sw if s.kind == "low"]
    out: list[ChartPattern] = []

    # ---- head and shoulders (and inverse) ----
    if len(highs) >= 3:
        l, h, r = highs[-3], highs[-2], highs[-1]
        if h.price > l.price and h.price > r.price:
            sym = 1.0 - min(1.0, abs(l.price - r.price) / max(h.price, 1e-9) / 0.05)
            prominence = (h.price - max(l.price, r.price)) / max(h.price, 1e-9)
            if prominence > 0.01 and sym > 0.3:
                out.append(
                    ChartPattern(
                        "head_and_shoulders", "bearish", offset + r.index,
                        float(np.clip(0.45 + 0.3 * sym + 10 * prominence, 0, 1)),
                        ["one of the most bearish classic formations",
                         "neckline break confirms"],
                    )
                )
    if len(lows) >= 3:
        l, h, r = lows[-3], lows[-2], lows[-1]
        if h.price < l.price and h.price < r.price:
            sym = 1.0 - min(1.0, abs(l.price - r.price) / max(h.price, 1e-9) / 0.05)
            prominence = (min(l.price, r.price) - h.price) / max(h.price, 1e-9)
            if prominence > 0.01 and sym > 0.3:
                out.append(
                    ChartPattern(
                        "inverse_head_and_shoulders", "bullish", offset + r.index,
                        float(np.clip(0.45 + 0.3 * sym + 10 * prominence, 0, 1)),
                        ["inverted H&S at the end of a downtrend"],
                    )
                )

    # ---- double top / bottom ----
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if abs(a.price - b.price) / max(a.price, 1e-9) < 0.015 and b.index - a.index >= 5:
            out.append(
                ChartPattern("double_top", "bearish", offset + b.index, 0.55,
                             ["parallel peaks — 'M' shape; confirms below the middle"])
            )
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        if abs(a.price - b.price) / max(a.price, 1e-9) < 0.015 and b.index - a.index >= 5:
            out.append(
                ChartPattern("double_bottom", "bullish", offset + b.index, 0.55,
                             ["parallel bottoms — 'W' shape; confirms above the middle"])
            )

    # ---- flag / pennant (retracement on falling volume) ----
    if len(window) >= 30:
        recent = window.iloc[-12:]
        prior = window.iloc[-30:-12]
        prior_move = float(
            (prior["close"].iloc[-1] - prior["close"].iloc[0]) / max(prior["close"].iloc[0], 1e-9)
        )
        rng_recent = float((recent["high"].max() - recent["low"].min()) / max(recent["close"].iloc[-1], 1e-9))
        vol_falling = float(recent["volume"].mean()) < float(prior["volume"].mean()) * 0.85
        highs_r = recent["high"].to_numpy(float)
        lows_r = recent["low"].to_numpy(float)
        converging = (highs_r[:6].max() - lows_r[:6].min()) > (highs_r[6:].max() - lows_r[6:].min()) * 1.25

        if abs(prior_move) > 0.02 and rng_recent < 0.025 and vol_falling:
            name = "pennant" if converging else "flag"
            direction = "bullish" if prior_move > 0 else "bearish"
            out.append(
                ChartPattern(
                    name, direction, len(df) - 1, 0.5,
                    ["retracement mid-trend on falling volume",
                     "if volume is NOT falling this is more likely a reversal"],
                )
            )
    return out


# =====================================================================
#  Snapshot
# =====================================================================
def snapshot(df: pd.DataFrame) -> dict:
    """One structural read of the current bar, for signals and journalling."""
    price = float(df["close"].iloc[-1])
    t = trend(df)
    lv = levels(df)
    a = float(atr(df).iloc[-1]) if len(df) > 15 else price * 0.01
    sup = nearest_level(lv, price, "support")
    res = nearest_level(lv, price, "resistance")
    bo = breakouts(df)
    gp = gaps(df)
    cp = chart_patterns(df)

    return {
        "price": price,
        "atr": a,
        "atr_pct": a / max(price, 1e-9),
        "trend_direction": t.direction,
        "trend_strength": round(t.strength, 3),
        "trend_phase": t.phase,
        "trend_slope_pct": round(t.slope_pct, 4),
        "trend_notes": t.notes,
        "support": sup.price if sup else None,
        "support_strength": round(sup.strength, 3) if sup else None,
        "support_distance_pct": round(sup.distance_pct(price), 4) if sup else None,
        "resistance": res.price if res else None,
        "resistance_strength": round(res.strength, 3) if res else None,
        "resistance_distance_pct": round(res.distance_pct(price), 4) if res else None,
        "levels": [(l.price, l.kind, l.touches, round(l.strength, 2)) for l in lv],
        "recent_breakout": (
            {"direction": bo[-1].direction, "level": bo[-1].level,
             "volume_confirmed": bo[-1].volume_confirmed,
             "bars_ago": len(df) - 1 - bo[-1].index}
            if bo and len(df) - 1 - bo[-1].index <= 5 else None
        ),
        "recent_gap": (
            {"direction": gp[-1].direction, "size_pct": round(gp[-1].size_pct, 4),
             "filled": gp[-1].filled, "bars_ago": len(df) - 1 - gp[-1].index}
            if gp and len(df) - 1 - gp[-1].index <= 10 else None
        ),
        "chart_patterns": [
            {"name": c.name, "direction": c.direction, "strength": round(c.strength, 2)}
            for c in cp if len(df) - 1 - c.index <= 10
        ],
    }
