"""Backtest engine and journal integrity.

The headline test is `TestNoLookAhead`: a backtest that can see the future
produces beautiful results and loses real money. Everything else in this project
is worthless if that property does not hold, so it is asserted directly rather
than assumed.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from trady import data as datamod
from trady.backtest import Backtester, _reason_bucket, walk_forward
from trady.config import Config
from trady.journal import Journal, compute_stats, stats_by


@pytest.fixture
def cfg(tmp_path) -> Config:
    c = Config()
    c.journal_db = tmp_path / "journal.sqlite"
    c.data_dir = tmp_path
    c.reports_dir = tmp_path / "reports"
    c.signal.min_confluence_score = 1.8
    c.risk.starting_equity = 30_000.0
    return c


@pytest.fixture
def journal(tmp_path) -> Journal:
    return Journal(tmp_path / "j.sqlite")


# =====================================================================
#  Look-ahead
# =====================================================================
class TestNoLookAhead:
    def test_future_bars_cannot_change_past_decisions(self, cfg):
        """Appending future data must not alter trades already taken.

        Run the same backtest twice: once on the first 1,200 bars, once on all
        2,000. Trades entered inside the shared prefix must be byte-identical.
        If the engine peeked ahead, the longer run would decide differently.
        """
        full = datamod.synthetic("AAA", bars=2000, seed=5)
        short = full.iloc[:1200]

        res_short = Backtester(cfg).run({"AAA": short}, warmup=120)
        res_full = Backtester(cfg).run({"AAA": full}, warmup=120)

        if res_short.trades.empty:
            pytest.skip("no trades generated in the short window")

        cutoff = short.index[-1]
        a = res_short.trades[res_short.trades["exit_ts"] <= cutoff]
        b = res_full.trades[res_full.trades["exit_ts"] <= cutoff]

        assert len(a) == len(b), "trade count changed when future data was appended"
        cols = ["symbol", "direction", "entry_ts", "entry_price", "exit_ts", "exit_price"]
        pd.testing.assert_frame_equal(
            a[cols].reset_index(drop=True), b[cols].reset_index(drop=True),
            check_dtype=False,
        )

    def test_entry_fills_after_the_signal_bar(self, cfg):
        """Entries must fill on the NEXT bar, never the signal bar's close."""
        df = datamod.synthetic("AAA", bars=1500, seed=9)
        res = Backtester(cfg).run({"AAA": df}, warmup=120)
        if res.trades.empty:
            pytest.skip("no trades generated")

        for _, t in res.trades.iterrows():
            bar = df.loc[t["entry_ts"]]
            # Fill is the bar's open (plus slippage), not its close.
            assert abs(t["entry_price"] - bar["open"]) < abs(bar["open"]) * 0.01

    def test_exit_never_precedes_entry(self, cfg):
        df = datamod.synthetic("AAA", bars=1500, seed=11)
        res = Backtester(cfg).run({"AAA": df}, warmup=120)
        if res.trades.empty:
            pytest.skip("no trades generated")
        # Same-bar round trips are legitimate — a position opened at a bar's
        # open can genuinely be stopped later in that same bar; OHLC just
        # cannot express the intrabar time. Going backwards is not.
        assert (res.trades["exit_ts"] >= res.trades["entry_ts"]).all()

    def test_stops_sit_outside_single_bar_noise(self, cfg):
        """Regression: stops were once placed between entry and the level.

        The structural-stop comparison was inverted, pulling the stop toward the
        entry instead of beyond the level that invalidates the idea. 64% of
        trades were then stopped out on their own entry bar by an ordinary wick,
        before the idea had been tested at all.
        """
        df = datamod.synthetic("AAA", bars=1500, seed=11)
        res = Backtester(cfg).run({"AAA": df}, warmup=120)
        if res.trades.empty:
            pytest.skip("no trades generated")
        same_bar = (res.trades["exit_ts"] == res.trades["entry_ts"]).mean()
        assert same_bar < 0.5, (
            f"{same_bar:.0%} of trades died on their entry bar — stops are "
            "inside single-bar noise"
        )

    def test_long_stop_is_below_support_not_between(self, cfg):
        """A long's stop must sit beyond support, never between entry and it."""
        from trady.indicators import enrich
        from trady.strategy import generate

        df = enrich(datamod.synthetic("AAA", bars=1200, seed=31))
        for end in range(400, len(df), 37):
            sig = generate("AAA", df.iloc[:end], cfg, 30_000.0)
            if sig is None:
                continue
            sup = sig.structure.get("support")
            res = sig.structure.get("resistance")
            if sig.direction == "long" and sup and sup < sig.entry:
                assert sig.stop <= sup * 1.0001, "long stop landed above support"
            if sig.direction == "short" and res and res > sig.entry:
                assert sig.stop >= res * 0.9999, "short stop landed below resistance"


# =====================================================================
#  Engine behaviour
# =====================================================================
class TestBacktestEngine:
    def test_empty_input_is_handled(self, cfg):
        res = Backtester(cfg).run({}, warmup=120)
        assert res.stats["trades"] == 0
        assert res.warnings

    def test_too_short_series_is_skipped(self, cfg):
        res = Backtester(cfg).run({"AAA": datamod.synthetic("AAA", bars=50)}, warmup=120)
        assert res.stats["trades"] == 0

    def test_deterministic_for_a_given_seed(self, cfg):
        df = datamod.synthetic("AAA", bars=1200, seed=3)
        a = Backtester(cfg).run({"AAA": df}, warmup=120)
        b = Backtester(cfg).run({"AAA": df}, warmup=120)
        assert a.stats["trades"] == b.stats["trades"]
        assert a.stats["net_pnl"] == b.stats["net_pnl"]

    def test_positions_respect_the_concurrency_cap(self, cfg):
        cfg.risk.max_open_positions = 2
        frames = {
            s: datamod.synthetic(s, bars=1500, seed=i)
            for i, s in enumerate(["AAA", "BBB", "CCC", "DDD"])
        }
        res = Backtester(cfg).run(frames, warmup=120)
        if res.trades.empty:
            pytest.skip("no trades generated")
        # Reconstruct concurrency from entry/exit stamps.
        events = [(t["entry_ts"], 1) for _, t in res.trades.iterrows()]
        events += [(t["exit_ts"], -1) for _, t in res.trades.iterrows()]
        events.sort(key=lambda e: (e[0], e[1]))
        live = peak = 0
        for _, delta in events:
            live += delta
            peak = max(peak, live)
        assert peak <= cfg.risk.max_open_positions

    def test_stops_bound_the_loss_per_trade(self, cfg):
        """No loss should greatly exceed the planned 1R, allowing for slippage."""
        df = datamod.synthetic("AAA", bars=2000, seed=13)
        res = Backtester(cfg).run({"AAA": df}, warmup=120)
        if res.trades.empty:
            pytest.skip("no trades generated")
        r = res.trades["r_multiple"].dropna()
        if len(r):
            assert r.min() > -3.0, "a loss ran far past its stop"

    def test_day_trades_are_flagged(self, cfg):
        df = datamod.synthetic("AAA", bars=2000, seed=17)
        res = Backtester(cfg).run({"AAA": df}, warmup=120)
        if res.trades.empty:
            pytest.skip("no trades generated")
        same_day = (
            pd.to_datetime(res.trades["entry_ts"]).dt.date
            == pd.to_datetime(res.trades["exit_ts"]).dt.date
        )
        assert (res.trades["is_day_trade"].astype(bool) == same_day).all()

    def test_cooloff_resets_between_sessions(self, cfg):
        """Regression: a 3-loss day used to freeze the agent permanently."""
        cfg.risk.max_consecutive_losses = 2
        df = datamod.synthetic("AAA", bars=3000, seed=23)
        res = Backtester(cfg).run({"AAA": df}, warmup=120)
        if res.trades.empty:
            pytest.skip("no trades generated")
        days = pd.to_datetime(res.trades["entry_ts"]).dt.date.nunique()
        assert days > 1, "trading stopped after the first bad session — cool-off deadlock"

    def test_summary_renders(self, cfg):
        res = Backtester(cfg).run(
            {"AAA": datamod.synthetic("AAA", bars=1200, seed=2)}, warmup=120
        )
        text = res.summary()
        assert "BACKTEST RESULT" in text and "win rate" in text

    def test_rejections_are_bucketed(self, cfg):
        res = Backtester(cfg).run(
            {"AAA": datamod.synthetic("AAA", bars=1200, seed=2)}, warmup=120
        )
        assert isinstance(res.rejections, dict)
        assert all(isinstance(k, str) and isinstance(v, int)
                   for k, v in res.rejections.items())

    def test_reason_bucket_collapses_variants(self):
        assert _reason_bucket("confluence 1.2 below the 3.00 minimum") == \
            "confluence below minimum"
        assert _reason_bucket("reward:risk 0.9 below the 1.50 minimum") == \
            "reward:risk too low"


class TestWalkForward:
    def test_produces_one_row_per_slice(self, cfg):
        df = datamod.synthetic("AAA", bars=2400, seed=7)
        out = walk_forward(cfg, {"AAA": df}, splits=3, warmup=120)
        if out.empty:
            pytest.skip("not enough history to split")
        assert set(out["split"]) <= {1, 2, 3}
        assert {"symbol", "split", "trades", "expectancy"} <= set(out.columns)


# =====================================================================
#  Journal
# =====================================================================
class TestJournal:
    def test_schema_is_created(self, journal):
        with journal._con() as con:
            tables = {
                r[0] for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert {"signal", "trade", "equity_mark", "review", "event"} <= tables

    def test_long_pnl_maths(self, journal):
        tid = journal.open_trade("AAPL", "long", 100, 50.0, "2026-08-13 10:00:00",
                                 planned_stop=49.0, planned_target=53.0)
        res = journal.close_trade(tid, 52.0, "2026-08-13 14:00:00", "target", fees=1.5)
        assert res["gross_pnl"] == pytest.approx(200.0)
        assert res["net_pnl"] == pytest.approx(198.5)
        # risk was $1/share x 100 = $100 -> 1.985R
        assert res["r_multiple"] == pytest.approx(1.985, abs=1e-3)
        assert res["is_day_trade"] is True

    def test_short_pnl_maths(self, journal):
        tid = journal.open_trade("AAPL", "short", 50, 100.0, "2026-08-13 10:00:00",
                                 planned_stop=102.0)
        res = journal.close_trade(tid, 96.0, "2026-08-13 15:00:00", "target", fees=2.0)
        assert res["gross_pnl"] == pytest.approx(200.0)
        assert res["net_pnl"] == pytest.approx(198.0)

    def test_losing_trade_is_negative(self, journal):
        tid = journal.open_trade("AAPL", "long", 10, 100.0, "2026-08-13 10:00:00",
                                 planned_stop=98.0)
        res = journal.close_trade(tid, 98.0, "2026-08-13 11:00:00", "stop", fees=0.5)
        assert res["net_pnl"] < 0
        assert res["r_multiple"] == pytest.approx(-1.025, abs=1e-3)

    def test_overnight_hold_is_not_a_day_trade(self, journal):
        tid = journal.open_trade("AAPL", "long", 10, 100.0, "2026-08-13 15:00:00")
        res = journal.close_trade(tid, 101.0, "2026-08-14 10:00:00", "target")
        assert res["is_day_trade"] is False

    def test_open_then_closed_moves_between_views(self, journal):
        tid = journal.open_trade("AAPL", "long", 10, 100.0, "2026-08-13 10:00:00")
        assert len(journal.open_trades()) == 1
        assert journal.closed_trades().empty
        journal.close_trade(tid, 101.0, "2026-08-13 11:00:00", "target")
        assert journal.open_trades().empty
        assert len(journal.closed_trades()) == 1

    def test_closing_unknown_trade_raises(self, journal):
        with pytest.raises(KeyError):
            journal.close_trade(999, 100.0, "2026-08-13 10:00:00", "target")

    def test_day_trades_feed_the_pdt_tracker(self, journal):
        for i in range(3):
            tid = journal.open_trade("AAPL", "long", 10, 100.0, f"2026-08-1{i+1} 10:00:00")
            journal.close_trade(tid, 101.0, f"2026-08-1{i+1} 11:00:00", "target")
        out = journal.day_trades()
        assert len(out) == 3
        assert all(isinstance(d, date) for _, d in out)

    def test_events_and_reviews_persist(self, journal):
        journal.log("halt", "daily loss limit")
        assert len(journal.events()) == 1
        rid = journal.record_review(
            "daily", 10, {"win_rate": 0.5}, {"risk.x": (1.0, 0.9, "why")}, "lesson"
        )
        assert rid > 0
        assert len(journal.reviews()) == 1
        with journal._con() as con:
            assert con.execute("SELECT count(*) FROM param_history").fetchone()[0] == 1

    def test_equity_curve_ordering(self, journal):
        for i, eq in enumerate([100.0, 110.0, 105.0]):
            journal.mark_equity(eq, ts=f"2026-08-13 1{i}:00:00")
        curve = journal.equity_curve()
        assert list(curve["equity"]) == [100.0, 110.0, 105.0]
        assert curve["ts"].is_monotonic_increasing


# =====================================================================
#  Statistics
# =====================================================================
class TestStats:
    def _trades(self, pnls, base_price=100.0):
        return pd.DataFrame({
            "net_pnl": pnls,
            "gross_pnl": pnls,
            "fees": [0.0] * len(pnls),
            "pnl_pct": [p / 1000.0 for p in pnls],
            "r_multiple": [p / 100.0 for p in pnls],
            "is_day_trade": [1] * len(pnls),
            "entry_price": [base_price] * len(pnls),
            "qty": [10] * len(pnls),
        })

    def test_empty_is_zeroed_not_nan(self):
        s = compute_stats(pd.DataFrame())
        assert s["trades"] == 0 and s["win_rate"] == 0.0
        assert not np.isnan(s["expectancy"])

    def test_win_rate_and_counts(self):
        s = compute_stats(self._trades([100, -50, 100, -50, 100]))
        assert s["trades"] == 5 and s["wins"] == 3 and s["losses"] == 2
        assert s["win_rate"] == pytest.approx(0.6)

    def test_profit_factor(self):
        s = compute_stats(self._trades([100, 100, -50, -50]))
        assert s["profit_factor"] == pytest.approx(2.0)

    def test_drawdown_uses_a_positive_base(self):
        """Regression: two small losses once reported an 89.5% drawdown."""
        s = compute_stats(self._trades([-7.5, -6.7]), base_equity=25_000.0)
        assert s["max_drawdown"] < 0.01

    def test_drawdown_inferred_base_stays_sane(self):
        s = compute_stats(self._trades([-7.5, -6.7]))
        assert 0.0 <= s["max_drawdown"] < 1.0

    def test_drawdown_measured_from_a_peak(self):
        s = compute_stats(self._trades([1000, -500]), base_equity=10_000.0)
        # peak 11,000 -> trough 10,500
        assert s["max_drawdown"] == pytest.approx(500 / 11_000, abs=1e-4)

    def test_consecutive_loss_streak(self):
        s = compute_stats(self._trades([-1, -1, -1, 5, -1, -1]))
        assert s["max_consecutive_losses"] == 3

    def test_stats_by_groups(self):
        df = self._trades([100, -50, 100, -50])
        df["strategy"] = ["a", "a", "b", "b"]
        out = stats_by(df, "strategy", base_equity=10_000.0)
        assert set(out["strategy"]) == {"a", "b"}
        assert len(out) == 2

    def test_stats_by_missing_column(self):
        assert stats_by(self._trades([1]), "nope").empty


# =====================================================================
#  Data normalisation
# =====================================================================
class TestData:
    def test_synthetic_is_deterministic(self):
        a = datamod.synthetic("T", bars=300, seed=1)
        b = datamod.synthetic("T", bars=300, seed=1)
        pd.testing.assert_frame_equal(a, b)

    def test_synthetic_bars_are_structurally_valid(self):
        df = datamod.synthetic("T", bars=800, seed=4)
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
        assert (df[["open", "high", "low", "close"]] > 0).all().all()

    def test_session_calendar_has_no_weekends_or_after_hours(self):
        df = datamod.synthetic("T", bars=800, seed=6)
        assert df.index.weekday.max() < 5
        assert df.index.time.min() >= pd.Timestamp("09:30").time()
        assert df.index.time.max() <= pd.Timestamp("16:00").time()

    def test_session_index_bar_count_per_day(self):
        idx = datamod.session_index(78 * 3, interval_minutes=5)
        assert idx.normalize().nunique() == 3

    def test_normalise_drops_impossible_bars(self):
        raw = pd.DataFrame(
            {
                "Open": [10.0, 10.0], "High": [11.0, 9.0],   # second high < low
                "Low": [9.0, 10.0], "Close": [10.5, 9.5],
                "Volume": [1000.0, 1000.0],
            },
            index=pd.to_datetime(["2024-01-02 09:30", "2024-01-02 09:35"]),
        )
        out = datamod._normalise(raw, "T")
        assert len(out) == 1

    def test_normalise_lowercases_and_orders(self):
        raw = pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5],
             "Volume": [10.0], "Adj Close": [1.5]},
            index=pd.to_datetime(["2024-01-02"]),
        )
        out = datamod._normalise(raw, "T")
        assert list(out.columns) == ["open", "high", "low", "close", "volume"]

    def test_normalise_rejects_missing_columns(self):
        raw = pd.DataFrame({"Open": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
        with pytest.raises(datamod.DataError):
            datamod._normalise(raw, "T")

    def test_normalise_rejects_empty(self):
        with pytest.raises(datamod.DataError):
            datamod._normalise(pd.DataFrame(), "T")

    def test_cache_roundtrip(self, tmp_path):
        cache = datamod.BarCache(tmp_path / "bars.sqlite")
        df = datamod.synthetic("T", bars=200, seed=8)
        assert cache.put("T", "5m", df) == 200
        back = cache.get("T", "5m")
        assert len(back) == 200
        pd.testing.assert_series_equal(
            back["close"].round(6), df["close"].round(6), check_names=False
        )

    def test_cache_miss_raises(self, tmp_path):
        cache = datamod.BarCache(tmp_path / "bars.sqlite")
        with pytest.raises(datamod.DataError):
            cache.get("NOPE", "5m")

    def test_load_falls_back_to_synthetic_when_asked(self, tmp_path):
        df = datamod.load("NOPE", provider="unknown-provider",
                          fallback_synthetic=True)
        assert len(df) > 0

    def test_load_raises_without_fallback(self):
        with pytest.raises(datamod.DataError):
            datamod.load("NOPE", provider="unknown-provider")
