"""Japanese candlestick reversal patterns.

Criteria follow *Getting Started in Candlestick Charting* (Logan, 2008) chapter 3
literally, including the book's strength modifiers and its accepted variations.

Two rules from the book govern everything here:

1. "The reversal patterns all share one common trait: they must follow a
   directional move." A hammer shape in a sideways drift is not a hammer, so
   every detector is gated on `prior_move`.
2. Non-ideal patterns still carry information ("try not to be too rigid ... or
   you may miss many good trading opportunities"). So detectors accept the
   book's stated variations and report a graded `strength` rather than a bool.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- tunables
DOJI_BODY_RATIO = 0.05      # body <= 5% of range counts as a doji
NEAR_DOJI_RATIO = 0.10
SMALL_BODY_RATIO = 0.30     # "small real body" / spinning top
LONG_BODY_RATIO = 0.60      # "long real body"
SHADOW_MULTIPLE = 2.0       # long shadow >= 2x body (book's criterion)
TINY_SHADOW_RATIO = 0.25    # "no upper shadow or a very small one"
TREND_LOOKBACK = 5          # bars used to establish the preceding move
TREND_MIN_MOVE = 0.005      # 0.5% net move counts as directional


@dataclass
class PatternHit:
    """One detected pattern at one bar."""

    name: str
    direction: str            # "bullish" | "bearish"
    index: int                # positional index of the pattern's final bar
    timestamp: object
    strength: float           # 0..1, graded by the book's strength modifiers
    bars: int                 # candles composing the pattern
    needs_confirmation: bool
    confirm_above: float | None = None   # confirm long if price trades above
    confirm_below: float | None = None   # confirm short if price trades below
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "direction": self.direction,
            "index": int(self.index),
            "timestamp": str(self.timestamp),
            "strength": round(float(self.strength), 3),
            "bars": self.bars,
            "needs_confirmation": self.needs_confirmation,
            "confirm_above": self.confirm_above,
            "confirm_below": self.confirm_below,
            "notes": list(self.notes),
        }


# ------------------------------------------------------------- candle math
class Candle:
    """Geometry of a single candlestick line."""

    __slots__ = ("o", "h", "l", "c", "v", "rng", "body", "upper", "lower")

    def __init__(self, o: float, h: float, l: float, c: float, v: float = 0.0):
        self.o, self.h, self.l, self.c, self.v = o, h, l, c, v
        self.rng = max(h - l, 1e-12)
        self.body = abs(c - o)
        self.upper = h - max(o, c)
        self.lower = min(o, c) - l

    # colour ---------------------------------------------------------
    @property
    def white(self) -> bool:      # bullish / up close
        return self.c > self.o

    @property
    def black(self) -> bool:      # bearish / down close
        return self.c < self.o

    @property
    def body_top(self) -> float:
        return max(self.o, self.c)

    @property
    def body_bottom(self) -> float:
        return min(self.o, self.c)

    # shape ratios ---------------------------------------------------
    @property
    def body_ratio(self) -> float:
        return self.body / self.rng

    @property
    def is_doji(self) -> bool:
        return self.body_ratio <= DOJI_BODY_RATIO

    @property
    def is_near_doji(self) -> bool:
        return self.body_ratio <= NEAR_DOJI_RATIO

    @property
    def is_small_body(self) -> bool:
        return self.body_ratio <= SMALL_BODY_RATIO

    @property
    def is_long_body(self) -> bool:
        return self.body_ratio >= LONG_BODY_RATIO

    @property
    def is_spinning_top(self) -> bool:
        return self.is_small_body and self.upper > self.body and self.lower > self.body

    @property
    def is_marubozu(self) -> bool:
        return (
            self.body_ratio > 0.9
            and self.upper < 0.05 * self.rng
            and self.lower < 0.05 * self.rng
        )

    # umbrella shapes ------------------------------------------------
    @property
    def is_umbrella(self) -> bool:
        """Small body near the high, long lower shadow, little/no upper shadow.

        Shape shared by the hammer (bullish) and hanging man (bearish).
        """
        if self.body_ratio > SMALL_BODY_RATIO:
            return False
        if self.body <= 0:  # doji-shaped umbrellas handled as dragonfly doji
            return self.lower >= SHADOW_MULTIPLE * (self.rng * 0.1) and self.upper <= TINY_SHADOW_RATIO * self.rng
        return (
            self.lower >= SHADOW_MULTIPLE * self.body
            and self.upper <= TINY_SHADOW_RATIO * self.rng
        )

    @property
    def is_inverted_umbrella(self) -> bool:
        """Small body near the low, long upper shadow, little/no lower shadow.

        Shape shared by the shooting star (bearish) and inverted hammer (bullish).
        """
        if self.body_ratio > SMALL_BODY_RATIO:
            return False
        if self.body <= 0:
            return self.upper >= SHADOW_MULTIPLE * (self.rng * 0.1) and self.lower <= TINY_SHADOW_RATIO * self.rng
        return (
            self.upper >= SHADOW_MULTIPLE * self.body
            and self.lower <= TINY_SHADOW_RATIO * self.rng
        )


def _candles(df: pd.DataFrame) -> list[Candle]:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = (
        df["volume"].to_numpy(float)
        if "volume" in df.columns
        else np.zeros(len(df), dtype=float)
    )
    return [Candle(o[i], h[i], l[i], c[i], v[i]) for i in range(len(df))]


# ------------------------------------------------------------- context
def prior_move(closes: np.ndarray, i: int, lookback: int = TREND_LOOKBACK) -> float:
    """Net fractional move over the bars preceding index `i`.

    Positive = the pattern follows an advance (bearish reversals qualify),
    negative = follows a decline (bullish reversals qualify).
    """
    start = max(0, i - lookback)
    if i <= start:
        return 0.0
    ref = closes[start]
    if ref == 0:
        return 0.0
    return float((closes[i - 1] - ref) / ref)


def _after_advance(closes, i, lookback=TREND_LOOKBACK) -> bool:
    return prior_move(closes, i, lookback) >= TREND_MIN_MOVE


def _after_decline(closes, i, lookback=TREND_LOOKBACK) -> bool:
    return prior_move(closes, i, lookback) <= -TREND_MIN_MOVE


def _steepness(closes, i, lookback=TREND_LOOKBACK) -> float:
    """0..1 bonus for a long or steep preceding move (book strength modifier)."""
    return float(min(1.0, abs(prior_move(closes, i, lookback)) / 0.05))


def _vol_bonus(cands: list[Candle], i: int, period: int = 20) -> float:
    """0..1 bonus for heavy volume on the signal bar."""
    start = max(0, i - period)
    hist = [c.v for c in cands[start:i] if c.v > 0]
    if not hist or cands[i].v <= 0:
        return 0.0
    avg = float(np.mean(hist))
    if avg <= 0:
        return 0.0
    return float(np.clip((cands[i].v / avg - 1.0) / 1.0, 0.0, 1.0))


# =====================================================================
#  Single-line patterns
# =====================================================================
def _umbrella_patterns(df, cands, closes, i) -> list[PatternHit]:
    cur = cands[i]
    hits: list[PatternHit] = []
    ts = df.index[i]

    if cur.is_umbrella:
        # Strength: smaller body, shorter upper shadow, longer lower shadow.
        base = 0.45
        base += 0.20 * (1.0 - min(1.0, cur.body_ratio / SMALL_BODY_RATIO))
        base += 0.15 * min(1.0, cur.lower / max(cur.body, cur.rng * 0.05) / 4.0)
        base += 0.10 * (1.0 - min(1.0, cur.upper / (TINY_SHADOW_RATIO * cur.rng + 1e-12)))
        vb = _vol_bonus(cands, i)

        if _after_advance(closes, i):
            # Hanging man — book: wait for bearish confirmation.
            s = min(1.0, base + 0.15 * _steepness(closes, i) + 0.10 * vb)
            hits.append(
                PatternHit(
                    "hanging_man", "bearish", i, ts, s, 1,
                    needs_confirmation=True,
                    confirm_below=cur.body_bottom,
                    notes=["umbrella after advance", "confirm with close below real body"],
                )
            )
        elif _after_decline(closes, i):
            # Hammer — book: confirmation not strictly required.
            s = min(1.0, base + 0.15 * _steepness(closes, i) + 0.10 * vb + 0.05)
            hits.append(
                PatternHit(
                    "hammer", "bullish", i, ts, s, 1,
                    needs_confirmation=False,
                    confirm_above=cur.body_top,
                    notes=["umbrella after decline", "strong bullish reversal signal"],
                )
            )

    if cur.is_inverted_umbrella:
        base = 0.42
        base += 0.20 * (1.0 - min(1.0, cur.body_ratio / SMALL_BODY_RATIO))
        base += 0.15 * min(1.0, cur.upper / max(cur.body, cur.rng * 0.05) / 4.0)
        base += 0.10 * (1.0 - min(1.0, cur.lower / (TINY_SHADOW_RATIO * cur.rng + 1e-12)))
        vb = _vol_bonus(cands, i)

        if _after_advance(closes, i):
            s = min(1.0, base + 0.15 * _steepness(closes, i) + 0.10 * vb)
            notes = ["inverted umbrella after advance", "confirm with close below real body"]
            if i > 0 and cur.body_bottom > cands[i - 1].body_top:
                s = min(1.0, s + 0.08)
                notes.append("ideal: body gapped up from prior body")
            hits.append(
                PatternHit(
                    "shooting_star", "bearish", i, ts, s, 1,
                    needs_confirmation=True, confirm_below=cur.body_bottom, notes=notes,
                )
            )
        elif _after_decline(closes, i):
            # Book: a long upper shadow after a decline is ambiguous — confirm.
            s = min(1.0, base + 0.15 * _steepness(closes, i) + 0.10 * vb)
            hits.append(
                PatternHit(
                    "inverted_hammer", "bullish", i, ts, s, 1,
                    needs_confirmation=True,
                    confirm_above=cur.body_top,
                    notes=["inverted umbrella after decline", "confirm with close above real body"],
                )
            )
    return hits


def _doji_patterns(df, cands, closes, i) -> list[PatternHit]:
    cur = cands[i]
    if not cur.is_doji:
        return []
    ts = df.index[i]
    hits: list[PatternHit] = []

    # Distinctive doji by where the open/close sits in the range.
    upper_frac = cur.upper / cur.rng
    lower_frac = cur.lower / cur.rng
    if upper_frac > 0.6 and lower_frac < 0.1:
        variant, bias = "gravestone_doji", "bearish"
    elif lower_frac > 0.6 and upper_frac < 0.1:
        variant, bias = "dragonfly_doji", "bullish"
    elif upper_frac > 0.3 and lower_frac > 0.3:
        variant, bias = "long_legged_doji", None
    else:
        variant, bias = "doji", None

    vb = _vol_bonus(cands, i)
    steep = _steepness(closes, i)

    if _after_advance(closes, i):
        # Book: doji are more potent and reliable at tops than at bottoms.
        s = 0.50 + 0.20 * steep + 0.10 * vb
        if variant == "gravestone_doji":
            s += 0.12
        if bias == "bullish":
            s -= 0.10  # dragonfly at a top is a mixed message
        hits.append(
            PatternHit(
                f"northern_{variant}", "bearish", i, ts, float(np.clip(s, 0, 1)), 1,
                needs_confirmation=True, confirm_below=cur.l,
                notes=["doji after advance (northern)", "indecision; stronger at tops"],
            )
        )
    elif _after_decline(closes, i):
        # Book: doji lose some potency in a declining market — wait for more.
        s = 0.38 + 0.18 * steep + 0.10 * vb
        if variant == "dragonfly_doji":
            s += 0.12
        if bias == "bearish":
            s -= 0.10
        hits.append(
            PatternHit(
                f"southern_{variant}", "bullish", i, ts, float(np.clip(s, 0, 1)), 1,
                needs_confirmation=True, confirm_above=cur.h,
                notes=["doji after decline (southern)", "less potent than northern doji"],
            )
        )
    return hits


# =====================================================================
#  Two-line patterns
# =====================================================================
def _two_line_patterns(df, cands, closes, i) -> list[PatternHit]:
    if i < 1:
        return []
    prev, cur = cands[i - 1], cands[i]
    ts = df.index[i]
    hits: list[PatternHit] = []
    vb = _vol_bonus(cands, i)
    steep = _steepness(closes, i)

    # ---------------------------------------------------- dark cloud cover
    # 1st: strong white. 2nd: gaps up above prior high (ideal) or prior close
    # (accepted variation), closes deeply (>=50%) into the white body but does
    # not engulf it (that would be a bearish engulfing).
    if (
        _after_advance(closes, i)
        and prev.white
        and prev.body_ratio >= 0.45
        and cur.black
        and cur.o > prev.c
        and cur.c < prev.body_top
        and cur.c > prev.body_bottom  # not engulfing
    ):
        penetration = (prev.c - cur.c) / max(prev.body, 1e-12)
        if penetration >= 0.5:
            ideal = cur.o > prev.h
            s = 0.55 + 0.25 * min(1.0, (penetration - 0.5) / 0.5)
            s += 0.08 if ideal else 0.0
            s += 0.12 * vb + 0.10 * steep
            notes = ["strong bearish reversal after advance"]
            notes.append("ideal: gapped above prior high" if ideal else "variation: gapped above prior close only")
            if vb > 0.5:
                notes.append("heavy volume — possible blow-off top")
            hits.append(
                PatternHit(
                    "dark_cloud_cover", "bearish", i, ts, float(np.clip(s, 0, 1)), 2,
                    needs_confirmation=penetration < 0.5,
                    confirm_below=cur.l, notes=notes,
                )
            )

    # -------------------------------------------------------- piercing
    if (
        _after_decline(closes, i)
        and prev.black
        and prev.body_ratio >= 0.45
        and cur.white
        and cur.o < prev.c
        and cur.c > prev.body_bottom
        and cur.c < prev.body_top  # not engulfing
    ):
        penetration = (cur.c - prev.c) / max(prev.body, 1e-12)
        if penetration >= 0.5:
            ideal = cur.o < prev.l
            s = 0.55 + 0.25 * min(1.0, (penetration - 0.5) / 0.5)
            s += 0.08 if ideal else 0.0
            s += 0.12 * vb + 0.10 * steep
            notes = ["strong bullish reversal after decline"]
            notes.append("ideal: gapped below prior low" if ideal else "variation: gapped below prior close only")
            if vb > 0.5:
                notes.append("heavy volume — possible selling climax")
            hits.append(
                PatternHit(
                    "piercing_pattern", "bullish", i, ts, float(np.clip(s, 0, 1)), 2,
                    needs_confirmation=penetration < 0.5,
                    confirm_above=cur.h, notes=notes,
                )
            )

    # ------------------------------------------------------- engulfing
    # Second real body completely surrounds the first; opposite colours;
    # second body strictly longer. Prices may match at one end, not both.
    engulfs = (
        cur.body_top >= prev.body_top
        and cur.body_bottom <= prev.body_bottom
        and cur.body > prev.body
    )
    if engulfs and prev.body > 0:
        # Book strength modifiers.
        bonus = 0.0
        notes: list[str] = []
        if prev.is_small_body:
            bonus += 0.10
            notes.append("first body very small (spinning top/doji)")
        if cur.is_marubozu:
            bonus += 0.08
            notes.append("engulfing candle is shaven (marubozu)")
        if vb > 0.4:
            bonus += 0.10
            notes.append("heavy volume on engulfing session")
        size = min(1.0, (cur.body / prev.body - 1.0) / 2.0)

        if _after_advance(closes, i) and prev.white and cur.black:
            s = 0.60 + 0.15 * size + 0.15 * steep + bonus
            hits.append(
                PatternHit(
                    "bearish_engulfing", "bearish", i, ts, float(np.clip(s, 0, 1)), 2,
                    needs_confirmation=False, confirm_below=cur.l,
                    notes=["abrupt turnaround in sentiment", *notes],
                )
            )
        elif _after_decline(closes, i) and prev.black and cur.white:
            s = 0.60 + 0.15 * size + 0.15 * steep + bonus
            hits.append(
                PatternHit(
                    "bullish_engulfing", "bullish", i, ts, float(np.clip(s, 0, 1)), 2,
                    needs_confirmation=False, confirm_above=cur.h,
                    notes=["abrupt turnaround in sentiment", *notes],
                )
            )

    # ---------------------------------------------------------- harami
    # Small second body contained inside a long first body. Colours may match.
    inside = (
        cur.body_top <= prev.body_top
        and cur.body_bottom >= prev.body_bottom
        and prev.is_long_body
        and cur.body < prev.body * 0.5
    )
    if inside:
        cross = cur.is_doji
        s_base = 0.42 + 0.18 * (1.0 - cur.body_ratio / max(SMALL_BODY_RATIO, 1e-9))
        s_base += 0.12 * steep
        if cross:
            s_base += 0.15
        if prev.body_ratio > 0.75:
            s_base += 0.08  # "unusually long" first candle

        if _after_advance(closes, i) and prev.white:
            name = "bearish_harami_cross" if cross else "bearish_harami"
            notes = ["momentum loss after advance"]
            if cross:
                notes.append("harami cross — ignored at a long trader's peril")
            hits.append(
                PatternHit(
                    name, "bearish", i, ts, float(np.clip(s_base, 0, 1)), 2,
                    needs_confirmation=True, confirm_below=cur.body_bottom, notes=notes,
                )
            )
        elif _after_decline(closes, i) and prev.black:
            name = "bullish_harami_cross" if cross else "bullish_harami"
            notes = ["momentum loss after decline"]
            if cross:
                notes.append("harami cross — stronger at tops than bottoms")
            hits.append(
                PatternHit(
                    name, "bullish", i, ts, float(np.clip(s_base * 0.95, 0, 1)), 2,
                    needs_confirmation=True, confirm_above=cur.body_top, notes=notes,
                )
            )

    # ------------------------------------------------- western inside/outside
    if cur.h > prev.h and cur.l < prev.l:
        if _after_advance(closes, i) and cur.black:
            hits.append(
                PatternHit(
                    "outside_day", "bearish", i, ts, 0.40 + 0.15 * steep, 2,
                    needs_confirmation=True, confirm_below=cur.l,
                    notes=["western outside day after advance"],
                )
            )
        elif _after_decline(closes, i) and cur.white:
            hits.append(
                PatternHit(
                    "outside_day", "bullish", i, ts, 0.40 + 0.15 * steep, 2,
                    needs_confirmation=True, confirm_above=cur.h,
                    notes=["western outside day after decline"],
                )
            )
    return hits


# =====================================================================
#  Three-line patterns
# =====================================================================
def _three_line_patterns(df, cands, closes, i) -> list[PatternHit]:
    if i < 2:
        return []
    first, star, third = cands[i - 2], cands[i - 1], cands[i]
    ts = df.index[i]
    hits: list[PatternHit] = []
    vb = _vol_bonus(cands, i)
    steep = _steepness(closes, i - 2, TREND_LOOKBACK)

    # ---------------------------------------------------- evening star
    # long white, gap-up small body, black body intruding deeply into the white.
    if (
        _after_advance(closes, i - 1)
        and first.white
        and first.is_long_body
        and star.is_small_body
        and star.body_bottom > first.body_top          # gap between real bodies
        and third.black
        and third.c < first.body_top
    ):
        intrusion = (first.c - third.c) / max(first.body, 1e-12)
        if intrusion >= 0.3:
            doji_star = star.is_doji
            s = 0.62 + 0.20 * min(1.0, intrusion) + 0.12 * vb + 0.10 * steep
            if doji_star:
                s += 0.10
            notes = ["three-candle top reversal"]
            if doji_star:
                notes.append("star is a doji — evening doji star, stronger signal")
            if third.c < first.o:
                notes.append("variation: third candle erased the whole first body")
            hits.append(
                PatternHit(
                    "evening_doji_star" if doji_star else "evening_star",
                    "bearish", i, ts, float(np.clip(s, 0, 1)), 3,
                    needs_confirmation=False, confirm_below=third.l, notes=notes,
                )
            )

    # ---------------------------------------------------- morning star
    if (
        _after_decline(closes, i - 1)
        and first.black
        and first.is_long_body
        and star.is_small_body
        and star.body_top < first.body_bottom          # gap between real bodies
        and third.white
        and third.c > first.body_bottom
    ):
        intrusion = (third.c - first.c) / max(first.body, 1e-12)
        if intrusion >= 0.3:
            doji_star = star.is_doji
            s = 0.62 + 0.20 * min(1.0, intrusion) + 0.12 * vb + 0.10 * steep
            if doji_star:
                s += 0.10
            notes = ["three-candle bottom reversal"]
            if doji_star:
                notes.append("star is a doji — morning doji star, stronger signal")
            if third.c > first.o:
                notes.append("variation: third candle exceeded the whole first body")
            hits.append(
                PatternHit(
                    "morning_doji_star" if doji_star else "morning_star",
                    "bullish", i, ts, float(np.clip(s, 0, 1)), 3,
                    needs_confirmation=False, confirm_above=third.h, notes=notes,
                )
            )
    return hits


# =====================================================================
#  Public API
# =====================================================================
ALL_DETECTORS = (
    _umbrella_patterns,
    _doji_patterns,
    _two_line_patterns,
    _three_line_patterns,
)


def detect(df: pd.DataFrame, start: int | None = None) -> list[PatternHit]:
    """Scan an OHLCV frame and return every pattern hit, oldest first."""
    if len(df) < 4:
        return []
    cands = _candles(df)
    closes = df["close"].to_numpy(float)
    lo = max(3, start if start is not None else 3)

    hits: list[PatternHit] = []
    for i in range(lo, len(df)):
        for detector in ALL_DETECTORS:
            hits.extend(detector(df, cands, closes, i))
    return hits


def detect_latest(df: pd.DataFrame, within: int = 2) -> list[PatternHit]:
    """Patterns whose final bar is within the last `within` bars."""
    if len(df) < 4:
        return []
    cutoff = len(df) - within
    return [h for h in detect(df, start=max(3, cutoff - 3)) if h.index >= cutoff]


def confirmed(hit: PatternHit, df: pd.DataFrame) -> bool:
    """Has the book's stated confirmation occurred on a bar after the pattern?

    Confirmation = price trades beyond the pattern's real body in the signal's
    direction on a subsequent session.
    """
    nxt = hit.index + 1
    if nxt >= len(df):
        return False
    after = df.iloc[nxt:]
    if hit.direction == "bullish" and hit.confirm_above is not None:
        return bool((after["close"] > hit.confirm_above).any())
    if hit.direction == "bearish" and hit.confirm_below is not None:
        return bool((after["close"] < hit.confirm_below).any())
    return False


def summarize(hits: list[PatternHit]) -> pd.DataFrame:
    """Tabular view of hits, for reports and the CLI."""
    if not hits:
        return pd.DataFrame(
            columns=["timestamp", "name", "direction", "strength", "bars", "needs_confirmation"]
        )
    return pd.DataFrame([h.to_dict() for h in hits])[
        ["timestamp", "name", "direction", "strength", "bars", "needs_confirmation"]
    ]


def net_bias(hits: list[PatternHit], decay: float = 0.85) -> float:
    """Aggregate recent hits into one signed score. Positive = bullish."""
    if not hits:
        return 0.0
    newest = max(h.index for h in hits)
    total = 0.0
    for h in hits:
        age = newest - h.index
        w = decay ** age
        total += w * h.strength * (1.0 if h.direction == "bullish" else -1.0)
    return float(total)
