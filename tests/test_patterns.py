"""Candlestick pattern detection.

Each test hand-builds candles that satisfy (or deliberately violate) one
criterion from *Getting Started in Candlestick Charting* ch. 3, so a change in
the detector that drifts away from the book's definition fails here.

The most important tests are the negative ones: the book is explicit that a
reversal pattern must have something to reverse, so the same shape appearing in
a sideways drift must NOT be reported.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trady.patterns import (
    Candle,
    confirmed,
    detect,
    detect_latest,
    net_bias,
    prior_move,
    summarize,
)


# ---------------------------------------------------------------- helpers
def frame(bars: list[tuple], volumes: list[float] | None = None) -> pd.DataFrame:
    """Build an OHLCV frame from (open, high, low, close) tuples."""
    idx = pd.date_range("2024-01-02 09:30", periods=len(bars), freq="5min")
    df = pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = volumes if volumes else [1_000_000.0] * len(bars)
    df.index.name = "date"
    return df


def run_up(n: int = 8, start: float = 100.0, step: float = 0.6) -> list[tuple]:
    """A clean advance, so bearish reversals have something to reverse."""
    out = []
    for i in range(n):
        o = start + i * step
        c = o + step * 0.8
        out.append((o, c + 0.1, o - 0.1, c))
    return out


def run_down(n: int = 8, start: float = 100.0, step: float = 0.6) -> list[tuple]:
    out = []
    for i in range(n):
        o = start - i * step
        c = o - step * 0.8
        out.append((o, o + 0.1, c - 0.1, c))
    return out


def flat(n: int = 8, price: float = 100.0) -> list[tuple]:
    """Sideways drift — no directional move to reverse."""
    return [(price, price + 0.05, price - 0.05, price) for _ in range(n)]


def names(hits) -> set[str]:
    return {h.name for h in hits}


# =====================================================================
#  Candle geometry
# =====================================================================
class TestCandleGeometry:
    def test_colour(self):
        assert Candle(10, 12, 9, 11).white
        assert Candle(11, 12, 9, 10).black

    def test_body_and_shadows(self):
        c = Candle(10, 13, 9, 12)
        assert c.body == pytest.approx(2.0)
        assert c.upper == pytest.approx(1.0)
        assert c.lower == pytest.approx(1.0)
        assert c.rng == pytest.approx(4.0)

    def test_doji_detection(self):
        assert Candle(10, 11, 9, 10).is_doji
        assert not Candle(10, 11, 9, 10.9).is_doji

    def test_marubozu_has_no_shadows(self):
        assert Candle(10, 12, 10, 12).is_marubozu
        assert not Candle(10, 13, 9, 12).is_marubozu

    def test_umbrella_requires_long_lower_shadow(self):
        # body 0.2 at the top, lower shadow 2.0 -> 10x the body
        assert Candle(11.8, 12.0, 10.0, 12.0).is_umbrella
        # same body but a long UPPER shadow is not an umbrella
        assert not Candle(10.0, 12.0, 9.8, 10.2).is_umbrella

    def test_inverted_umbrella_requires_long_upper_shadow(self):
        assert Candle(10.0, 12.0, 9.9, 10.2).is_inverted_umbrella
        assert not Candle(11.8, 12.0, 10.0, 12.0).is_inverted_umbrella

    def test_spinning_top(self):
        assert Candle(10, 11.5, 8.5, 10.2).is_spinning_top


# =====================================================================
#  Context: prior move
# =====================================================================
class TestPriorMove:
    def test_positive_after_advance(self):
        df = frame(run_up())
        assert prior_move(df["close"].to_numpy(), len(df) - 1) > 0

    def test_negative_after_decline(self):
        df = frame(run_down())
        assert prior_move(df["close"].to_numpy(), len(df) - 1) < 0

    def test_near_zero_when_flat(self):
        df = frame(flat())
        assert abs(prior_move(df["close"].to_numpy(), len(df) - 1)) < 0.005


# =====================================================================
#  Single-line patterns
# =====================================================================
class TestUmbrellaPatterns:
    def test_hammer_after_decline(self):
        # Small body at the top, lower shadow ~3x the body, no upper shadow.
        bars = run_down() + [(95.4, 95.5, 93.5, 95.45)]
        hits = detect_latest(frame(bars), within=1)
        assert "hammer" in names(hits)
        h = next(x for x in hits if x.name == "hammer")
        assert h.direction == "bullish"
        # The book says a hammer does not require confirmation.
        assert h.needs_confirmation is False

    def test_hanging_man_after_advance(self):
        bars = run_up() + [(104.4, 104.5, 102.5, 104.45)]
        hits = detect_latest(frame(bars), within=1)
        assert "hanging_man" in names(hits)
        h = next(x for x in hits if x.name == "hanging_man")
        assert h.direction == "bearish"
        # The book says to wait for bearish confirmation.
        assert h.needs_confirmation is True

    def test_same_shape_ignored_without_a_directional_move(self):
        # THE key rule: "the pattern must have something to reverse".
        bars = flat() + [(100.0, 100.05, 98.0, 100.02)]
        hits = detect_latest(frame(bars), within=1)
        assert "hammer" not in names(hits)
        assert "hanging_man" not in names(hits)

    def test_shooting_star_after_advance(self):
        bars = run_up() + [(104.5, 106.5, 104.45, 104.55)]
        hits = detect_latest(frame(bars), within=1)
        assert "shooting_star" in names(hits)
        assert next(x for x in hits if x.name == "shooting_star").direction == "bearish"

    def test_inverted_hammer_after_decline(self):
        bars = run_down() + [(95.5, 97.5, 95.45, 95.55)]
        hits = detect_latest(frame(bars), within=1)
        assert "inverted_hammer" in names(hits)
        h = next(x for x in hits if x.name == "inverted_hammer")
        # The book: a long upper shadow after a decline is ambiguous — confirm.
        assert h.needs_confirmation is True

    def test_long_upper_shadow_disqualifies_a_hammer(self):
        bars = run_down() + [(95.4, 97.5, 93.5, 95.45)]
        assert "hammer" not in names(detect_latest(frame(bars), within=1))


class TestDoji:
    def test_northern_doji_is_bearish(self):
        bars = run_up() + [(104.5, 105.2, 103.8, 104.5)]
        hits = detect_latest(frame(bars), within=1)
        found = [h for h in hits if "northern" in h.name]
        assert found and found[0].direction == "bearish"

    def test_southern_doji_is_bullish(self):
        bars = run_down() + [(95.5, 96.2, 94.8, 95.5)]
        hits = detect_latest(frame(bars), within=1)
        found = [h for h in hits if "southern" in h.name]
        assert found and found[0].direction == "bullish"

    def test_northern_doji_scores_above_southern(self):
        # The book: doji are more potent and reliable at tops than at bottoms.
        up = detect_latest(frame(run_up() + [(104.5, 105.2, 103.8, 104.5)]), within=1)
        down = detect_latest(frame(run_down() + [(95.5, 96.2, 94.8, 95.5)]), within=1)
        n = next(h for h in up if "northern" in h.name)
        s = next(h for h in down if "southern" in h.name)
        assert n.strength > s.strength

    def test_gravestone_variant_named(self):
        bars = run_up() + [(104.5, 106.5, 104.48, 104.5)]
        assert any("gravestone" in n for n in names(detect_latest(frame(bars), within=1)))

    def test_dragonfly_variant_named(self):
        bars = run_down() + [(95.5, 95.52, 93.5, 95.5)]
        assert any("dragonfly" in n for n in names(detect_latest(frame(bars), within=1)))


# =====================================================================
#  Two-line patterns
# =====================================================================
class TestEngulfing:
    def test_bearish_engulfing(self):
        bars = run_up() + [(104.0, 104.6, 103.9, 104.5), (104.8, 104.9, 103.4, 103.5)]
        hits = detect_latest(frame(bars), within=1)
        assert "bearish_engulfing" in names(hits)

    def test_bullish_engulfing(self):
        bars = run_down() + [(96.0, 96.1, 95.4, 95.5), (95.2, 96.7, 95.1, 96.6)]
        hits = detect_latest(frame(bars), within=1)
        assert "bullish_engulfing" in names(hits)

    def test_requires_opposite_colours(self):
        # Two white candles cannot be a bearish engulfing however large.
        bars = run_up() + [(104.0, 104.6, 103.9, 104.5), (103.4, 105.2, 103.3, 105.1)]
        assert "bearish_engulfing" not in names(detect_latest(frame(bars), within=1))

    def test_requires_full_body_containment(self):
        # Second body does not cover the first -> not engulfing.
        bars = run_up() + [(104.0, 104.6, 103.9, 104.5), (104.4, 104.5, 104.1, 104.2)]
        assert "bearish_engulfing" not in names(detect_latest(frame(bars), within=1))

    def test_heavy_volume_raises_strength(self):
        bars = run_up() + [(104.0, 104.6, 103.9, 104.5), (104.8, 104.9, 103.4, 103.5)]
        vols = [1_000_000.0] * (len(bars) - 1)
        quiet = detect_latest(frame(bars, vols + [900_000.0]), within=1)
        loud = detect_latest(frame(bars, vols + [4_000_000.0]), within=1)
        q = next(h for h in quiet if h.name == "bearish_engulfing")
        l = next(h for h in loud if h.name == "bearish_engulfing")
        assert l.strength > q.strength


class TestPiercingAndDarkCloud:
    def test_dark_cloud_cover(self):
        # Long white, then a gap-up open closing well past the white midpoint.
        bars = run_up() + [(103.0, 105.1, 102.9, 105.0), (105.6, 105.7, 103.5, 103.6)]
        hits = detect_latest(frame(bars), within=1)
        assert "dark_cloud_cover" in names(hits)

    def test_piercing_pattern(self):
        bars = run_down() + [(97.0, 97.1, 94.9, 95.0), (94.4, 96.6, 94.3, 96.5)]
        hits = detect_latest(frame(bars), within=1)
        assert "piercing_pattern" in names(hits)

    def test_shallow_penetration_is_not_dark_cloud(self):
        # Closing less than halfway into the white body fails the criterion.
        bars = run_up() + [(103.0, 105.1, 102.9, 105.0), (105.6, 105.7, 104.7, 104.8)]
        assert "dark_cloud_cover" not in names(detect_latest(frame(bars), within=1))

    def test_full_engulf_is_engulfing_not_dark_cloud(self):
        # The book: if it engulfs the whole body it is an engulfing pattern.
        bars = run_up() + [(103.0, 105.1, 102.9, 105.0), (105.6, 105.7, 102.5, 102.6)]
        found = names(detect_latest(frame(bars), within=1))
        assert "bearish_engulfing" in found
        assert "dark_cloud_cover" not in found


class TestHarami:
    def test_bearish_harami(self):
        bars = run_up() + [(102.0, 105.2, 101.9, 105.0), (104.0, 104.4, 103.6, 103.8)]
        assert "bearish_harami" in names(detect_latest(frame(bars), within=1))

    def test_bullish_harami(self):
        bars = run_down() + [(98.0, 98.1, 94.8, 95.0), (96.0, 96.4, 95.6, 96.2)]
        assert "bullish_harami" in names(detect_latest(frame(bars), within=1))

    def test_harami_cross_when_second_is_a_doji(self):
        bars = run_up() + [(102.0, 105.2, 101.9, 105.0), (103.9, 104.3, 103.5, 103.9)]
        assert "bearish_harami_cross" in names(detect_latest(frame(bars), within=1))

    def test_second_body_must_be_inside_the_first(self):
        bars = run_up() + [(102.0, 105.2, 101.9, 105.0), (105.4, 105.8, 105.1, 105.6)]
        found = names(detect_latest(frame(bars), within=1))
        assert "bearish_harami" not in found and "bearish_harami_cross" not in found


# =====================================================================
#  Three-line patterns
# =====================================================================
class TestStars:
    def test_evening_star(self):
        # long white, gapped-up small body, black closing deep into the white
        bars = run_up() + [
            (101.0, 105.1, 100.9, 105.0),
            (105.8, 106.2, 105.6, 105.9),
            (105.0, 105.1, 102.2, 102.3),
        ]
        hits = detect_latest(frame(bars), within=1)
        assert any(n.startswith("evening") for n in names(hits))

    def test_morning_star(self):
        bars = run_down() + [
            (99.0, 99.1, 94.9, 95.0),
            (94.2, 94.4, 93.8, 94.1),
            (95.0, 97.8, 94.9, 97.7),
        ]
        hits = detect_latest(frame(bars), within=1)
        assert any(n.startswith("morning") for n in names(hits))

    def test_evening_doji_star_named_when_middle_is_a_doji(self):
        bars = run_up() + [
            (101.0, 105.1, 100.9, 105.0),
            (105.8, 106.2, 105.5, 105.8),
            (105.0, 105.1, 102.2, 102.3),
        ]
        assert "evening_doji_star" in names(detect_latest(frame(bars), within=1))

    def test_gap_between_bodies_is_required(self):
        # No gap between the first and second real bodies -> not a star.
        bars = run_up() + [
            (101.0, 105.1, 100.9, 105.0),
            (104.5, 104.9, 104.3, 104.7),
            (104.4, 104.5, 102.2, 102.3),
        ]
        assert not any(n.startswith("evening") for n in names(detect_latest(frame(bars), within=1)))

    def test_shallow_third_candle_fails(self):
        bars = run_up() + [
            (101.0, 105.1, 100.9, 105.0),
            (105.8, 106.2, 105.6, 105.9),
            (105.7, 105.8, 104.9, 105.0),
        ]
        assert not any(n.startswith("evening") for n in names(detect_latest(frame(bars), within=1)))


# =====================================================================
#  API surface
# =====================================================================
class TestDetectionApi:
    def test_short_frame_returns_nothing(self):
        assert detect(frame(run_up(3))) == []

    def test_detect_latest_windows_correctly(self):
        bars = run_down() + [(95.4, 95.5, 93.5, 95.45)] + run_up(4, start=96.0)
        assert "hammer" not in names(detect_latest(frame(bars), within=1))
        assert "hammer" in names(detect_latest(frame(bars), within=6))

    def test_confirmation_requires_a_later_bar(self):
        bars = run_up() + [(104.4, 104.5, 102.5, 104.45)]
        df = frame(bars)
        hit = next(h for h in detect_latest(df, within=1) if h.name == "hanging_man")
        assert confirmed(hit, df) is False  # nothing after it yet

        confirmed_df = frame(bars + [(104.0, 104.1, 101.0, 101.2)])
        hit2 = next(
            h for h in detect(confirmed_df) if h.name == "hanging_man"
        )
        assert confirmed(hit2, confirmed_df) is True

    def test_strength_is_bounded(self):
        for hit in detect(frame(run_up() + run_down(8, start=104.0))):
            assert 0.0 <= hit.strength <= 1.0

    def test_net_bias_sign_follows_direction(self):
        bull = detect_latest(frame(run_down() + [(95.4, 95.5, 93.5, 95.45)]), within=1)
        assert net_bias(bull) > 0
        bear = detect_latest(frame(run_up() + [(104.4, 104.5, 102.5, 104.45)]), within=1)
        assert net_bias(bear) < 0
        assert net_bias([]) == 0.0

    def test_summarize_shape(self):
        hits = detect(frame(run_up() + run_down(8, start=104.0)))
        out = summarize(hits)
        assert list(out.columns) == [
            "timestamp", "name", "direction", "strength", "bars", "needs_confirmation"
        ]
        assert len(out) == len(hits)

    def test_summarize_handles_empty(self):
        assert summarize([]).empty

    def test_to_dict_roundtrip(self):
        hits = detect_latest(frame(run_down() + [(95.4, 95.5, 93.5, 95.45)]), within=1)
        d = hits[0].to_dict()
        assert {"name", "direction", "strength", "bars", "notes"} <= set(d)
