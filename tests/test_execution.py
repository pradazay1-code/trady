"""Brokers, the Fidelity bridge, strategy gates, and the learning loop.

The Fidelity parsing tests matter because they run against text shaped like a
real export — disclaimer preamble, currency symbols, a legal footer — rather
than a clean CSV. That preamble is exactly what breaks naive parsers.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from trady import data as datamod
from trady.brokers import (
    AlpacaBroker,
    FidelityBridge,
    Order,
    PaperBroker,
    Position,
    estimate_fees,
    make_broker,
)
from trady.config import Config, ExecutionConfig, RiskConfig
from trady.indicators import atr, enrich, momentum, on_balance_volume, rsi
from trady.journal import Journal
from trady.learn import Learner, evidence_performance
from trady.risk import stop_and_target
from trady.strategy import Evidence, generate, scan


# =====================================================================
#  Stop noise floor
# =====================================================================
class TestStopNoiseFloor:
    def test_floor_widens_a_stop_inside_the_noise(self):
        cfg = RiskConfig(stop_atr_multiple=1.5, stop_noise_floor_mult=1.25)
        # ATR 0.1 -> 0.15 stop, but bars routinely span 1.0
        tight, _ = stop_and_target(100.0, 0.1, "long", cfg, noise_floor=0.0)
        floored, _ = stop_and_target(100.0, 0.1, "long", cfg, noise_floor=1.0)
        assert 100.0 - tight == pytest.approx(0.15, abs=1e-9)
        assert 100.0 - floored == pytest.approx(1.25, abs=1e-9)

    def test_floor_does_not_narrow_an_already_wide_stop(self):
        cfg = RiskConfig(stop_atr_multiple=1.5, stop_noise_floor_mult=1.25)
        wide, _ = stop_and_target(100.0, 2.0, "long", cfg, noise_floor=0.1)
        assert 100.0 - wide == pytest.approx(3.0, abs=1e-9)

    def test_reward_risk_ratio_survives_the_floor(self):
        cfg = RiskConfig(stop_atr_multiple=1.5, target_atr_multiple=3.0)
        stop, target = stop_and_target(100.0, 0.1, "long", cfg, noise_floor=1.0)
        assert (target - 100.0) / (100.0 - stop) == pytest.approx(2.0, abs=1e-6)

    def test_short_side_floors_symmetrically(self):
        cfg = RiskConfig(stop_atr_multiple=1.5, stop_noise_floor_mult=1.25)
        stop, target = stop_and_target(100.0, 0.1, "short", cfg, noise_floor=1.0)
        assert stop - 100.0 == pytest.approx(1.25, abs=1e-9)
        assert target < 100.0


# =====================================================================
#  Fees
# =====================================================================
class TestFees:
    def test_buys_pay_no_regulatory_fees(self):
        cfg = ExecutionConfig()
        assert estimate_fees(cfg, "buy", 100, 50.0) == cfg.commission_per_trade

    def test_sells_pay_sec_and_taf(self):
        cfg = ExecutionConfig()
        fee = estimate_fees(cfg, "sell", 100, 50.0)
        assert fee > cfg.commission_per_trade

    def test_taf_is_capped(self):
        cfg = ExecutionConfig()
        huge = estimate_fees(cfg, "sell", 10_000_000, 1.0)
        assert huge < 10_000_000 * cfg.taf_fee_per_share


# =====================================================================
#  Paper broker
# =====================================================================
class TestPaperBroker:
    def setup_method(self):
        self.broker = PaperBroker(ExecutionConfig(), 25_000.0)
        self.now = datetime(2026, 8, 13, 11, 0)

    def test_buy_reduces_cash_and_creates_position(self):
        fill = self.broker.submit(Order("AAPL", "buy", 100, "market"), 50.0, self.now)
        assert fill is not None
        assert self.broker.cash < 25_000.0
        assert self.broker.positions()["AAPL"].qty == 100

    def test_round_trip_profit_lands_in_cash(self):
        self.broker.submit(Order("AAPL", "buy", 100, "market"), 50.0, self.now)
        self.broker.submit(Order("AAPL", "sell", 100, "market"), 55.0, self.now)
        assert not self.broker.positions()
        assert self.broker.cash > 25_000.0

    def test_short_round_trip(self):
        self.broker.submit(Order("AAPL", "sell_short", 100, "market"), 50.0, self.now)
        assert self.broker.positions()["AAPL"].direction == "short"
        self.broker.submit(Order("AAPL", "buy_to_cover", 100, "market"), 45.0, self.now)
        assert not self.broker.positions()
        assert self.broker.cash > 25_000.0

    def test_unreachable_buy_limit_does_not_fill(self):
        order = Order("AAPL", "buy", 100, "limit", limit_price=45.0)
        assert self.broker.submit(order, 50.0, self.now) is None

    def test_reachable_buy_limit_fills(self):
        order = Order("AAPL", "buy", 100, "limit", limit_price=55.0)
        assert self.broker.submit(order, 50.0, self.now) is not None

    def test_slippage_is_adverse_on_both_sides(self):
        buy = self.broker.submit(Order("A", "buy", 10, "market"), 100.0, self.now)
        sell = self.broker.submit(Order("B", "sell_short", 10, "market"), 100.0, self.now)
        assert buy.price >= 100.0
        assert sell.price <= 100.0

    def test_averaging_up_blends_the_entry(self):
        self.broker.submit(Order("AAPL", "buy", 100, "market"), 50.0, self.now)
        self.broker.submit(Order("AAPL", "buy", 100, "market"), 60.0, self.now)
        pos = self.broker.positions()["AAPL"]
        assert pos.qty == 200
        assert 50.0 < pos.entry_price < 60.0

    def test_equity_tracks_marked_prices(self):
        self.broker.submit(Order("AAPL", "buy", 100, "market"), 50.0, self.now)
        before = self.broker.equity
        self.broker.mark({"AAPL": 60.0})
        assert self.broker.equity > before


class TestPositionExcursion:
    def test_long_tracks_favourable_and_adverse(self):
        p = Position("AAPL", "long", 10, 100.0, datetime.now())
        p.update_excursion(105.0, 98.0)
        assert p.mfe == pytest.approx(5.0)
        assert p.mae == pytest.approx(-2.0)

    def test_short_is_mirrored(self):
        p = Position("AAPL", "short", 10, 100.0, datetime.now())
        p.update_excursion(103.0, 95.0)
        assert p.mfe == pytest.approx(5.0)
        assert p.mae == pytest.approx(-3.0)

    def test_unrealized_sign(self):
        long_ = Position("A", "long", 10, 100.0, datetime.now())
        short = Position("A", "short", 10, 100.0, datetime.now())
        assert long_.unrealized(110.0) == pytest.approx(100.0)
        assert short.unrealized(110.0) == pytest.approx(-100.0)


# =====================================================================
#  Fidelity bridge
# =====================================================================
ACTIVITY_CSV = """Brokerage

Run Date,Action,Symbol,Description,Type,Quantity,Price ($),Commission ($),Fees ($),Amount ($)
08/12/2026,YOU BOUGHT,AAPL,APPLE INC,Cash,100,214.35,0.00,0.00,-21435.00
08/12/2026,YOU SOLD,AAPL,APPLE INC,Cash,-100,216.10,0.00,0.02,21609.98
08/12/2026,DIVIDEND RECEIVED,MSFT,MICROSOFT CORP,Cash,,,,,12.50
08/13/2026,YOU BOUGHT,NVDA,NVIDIA CORP,Margin,50,131.20,0.00,0.00,-6560.00

"The data and information in this spreadsheet is provided to you solely for your use"
"and is not for distribution."
"""

POSITIONS_CSV = """Account Number,Account Name,Symbol,Description,Quantity,Last Price,Current Value,Average Cost Basis
X12345678,INDIVIDUAL,AAPL,APPLE INC,100,$214.35,$21435.00,$210.00
X12345678,INDIVIDUAL,NVDA,NVIDIA CORP,50,$131.20,$6560.00,$128.40
X12345678,INDIVIDUAL,Pending Activity,,,,$0.00,

"Brokerage services provided by Fidelity Brokerage Services LLC."
"""


class TestFidelityParsing:
    def test_activity_skips_preamble_and_footer(self, tmp_path):
        p = tmp_path / "activity.csv"
        p.write_text(ACTIVITY_CSV)
        df = FidelityBridge.parse_activity(p)
        # Three executions; the dividend row is not one.
        assert len(df) == 3
        assert set(df["symbol"]) == {"AAPL", "NVDA"}

    def test_activity_strips_currency_formatting(self, tmp_path):
        p = tmp_path / "activity.csv"
        p.write_text(ACTIVITY_CSV)
        df = FidelityBridge.parse_activity(p)
        assert df["price"].dtype.kind == "f"
        assert df["price"].iloc[0] == pytest.approx(214.35)

    def test_activity_parses_dates(self, tmp_path):
        p = tmp_path / "activity.csv"
        p.write_text(ACTIVITY_CSV)
        df = FidelityBridge.parse_activity(p)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_activity_without_header_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("nothing useful here\nor here\n")
        with pytest.raises(ValueError):
            FidelityBridge.parse_activity(p)

    def test_positions_parsed_and_pending_dropped(self, tmp_path):
        p = tmp_path / "positions.csv"
        p.write_text(POSITIONS_CSV)
        df = FidelityBridge.parse_positions(p)
        assert set(df["symbol"]) == {"AAPL", "NVDA"}
        assert df["quantity"].sum() == 150

    def test_sync_positions_builds_state(self, tmp_path):
        p = tmp_path / "positions.csv"
        p.write_text(POSITIONS_CSV)
        bridge = FidelityBridge(ExecutionConfig(), tmp_path)
        pos = bridge.sync_positions(p)
        assert pos["AAPL"].qty == 100
        assert pos["AAPL"].direction == "long"
        assert bridge.equity == pytest.approx(21435.0 + 6560.0)

    def test_reconcile_reports_counts(self, tmp_path):
        act = tmp_path / "activity.csv"
        act.write_text(ACTIVITY_CSV)
        journal = Journal(tmp_path / "j.sqlite")
        tid = journal.open_trade("AAPL", "long", 100, 214.35, "2026-08-12 10:00:00")
        journal.close_trade(tid, 216.10, "2026-08-12 14:00:00", "target")
        out = FidelityBridge(ExecutionConfig(), tmp_path).reconcile(act, journal)
        assert out["activity_rows"] == 3
        assert out["matched"] >= 1


class TestFidelityTickets:
    def test_submit_queues_rather_than_executing(self, tmp_path):
        bridge = FidelityBridge(ExecutionConfig(), tmp_path)
        fill = bridge.submit(
            Order("AAPL", "buy", 10, "limit", limit_price=100.0), 100.0, datetime.now()
        )
        # A human places this order — nothing is filled programmatically.
        assert fill is None
        assert len(bridge.pending) == 1

    def test_ticket_files_written(self, tmp_path):
        bridge = FidelityBridge(ExecutionConfig(), tmp_path)
        bridge.pending.append(
            Order("AAPL", "buy", 37, "limit", limit_price=214.35, note="reversal")
        )
        paths = bridge.write_tickets(datetime(2026, 8, 13, 10, 15))
        text = Path(paths["txt"]).read_text()
        assert "AAPL" in text and "214.35" in text and "37" in text
        assert Path(paths["csv"]).exists() and Path(paths["json"]).exists()

    def test_order_describe_covers_types(self):
        assert "LIMIT" in Order("A", "buy", 1, "limit", limit_price=10.0).describe()
        assert "MARKET" in Order("A", "buy", 1, "market").describe()
        assert "STOP" in Order("A", "sell", 1, "stop", stop_price=9.0).describe()


class TestBrokerFactory:
    def test_paper_is_the_default(self):
        assert isinstance(make_broker(Config()), PaperBroker)

    def test_fidelity_selectable(self, tmp_path):
        cfg = Config()
        cfg.execution.broker = "fidelity"
        cfg.reports_dir = tmp_path
        assert isinstance(make_broker(cfg), FidelityBridge)

    def test_unknown_broker_raises(self):
        cfg = Config()
        cfg.execution.broker = "nope"
        with pytest.raises(ValueError):
            make_broker(cfg)

    def test_alpaca_requires_credentials(self, monkeypatch):
        monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
        monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
        with pytest.raises(RuntimeError):
            AlpacaBroker(ExecutionConfig())


# =====================================================================
#  Indicators
# =====================================================================
class TestIndicators:
    def setup_method(self):
        self.df = datamod.synthetic("T", bars=400, seed=2)

    def test_rsi_bounded(self):
        r = rsi(self.df["close"], 14).dropna()
        assert r.between(0, 100).all()

    def test_atr_positive(self):
        a = atr(self.df, 14).dropna()
        assert (a > 0).all()

    def test_momentum_is_100_when_flat(self):
        flat = pd.Series([10.0] * 30)
        assert momentum(flat, 10).dropna().iloc[-1] == pytest.approx(100.0)

    def test_obv_follows_direction(self):
        df = pd.DataFrame({
            "open": [10, 10, 10], "high": [11, 11, 11], "low": [9, 9, 9],
            "close": [10.0, 11.0, 10.0], "volume": [100.0, 100.0, 100.0],
        })
        obv = on_balance_volume(df)
        assert obv.iloc[1] > obv.iloc[0]
        assert obv.iloc[2] < obv.iloc[1]

    def test_enrich_adds_expected_columns(self):
        out = enrich(self.df)
        for col in ("rsi_14", "atr_14", "macd", "vwap", "obv", "mfi_14", "pivot"):
            assert col in out.columns

    def test_enrich_does_not_mutate_input(self):
        before = self.df.copy()
        enrich(self.df)
        pd.testing.assert_frame_equal(self.df, before)


# =====================================================================
#  Strategy gates
# =====================================================================
class TestStrategyGates:
    def setup_method(self):
        self.cfg = Config()
        self.df = enrich(datamod.synthetic("T", bars=800, seed=3))

    def test_short_history_returns_none(self):
        assert generate("T", self.df.iloc[:30], self.cfg, 25_000.0) is None

    def test_rejected_signals_are_still_returned_for_the_record(self):
        self.cfg.signal.min_confluence_score = 99.0
        sig = generate("T", self.df, self.cfg, 25_000.0)
        if sig is not None:
            assert sig.rejected_reason and not sig.actionable

    def test_price_band_filter(self):
        self.cfg.signal.min_price = 1e9
        sig = generate("T", self.df, self.cfg, 25_000.0)
        if sig is not None:
            assert "outside tradable band" in sig.rejected_reason

    def test_evidence_contribution_signs(self):
        bull = Evidence("trend", "up", "bullish", 0.8, 1.0)
        bear = Evidence("trend", "down", "bearish", 0.8, 1.0)
        assert bull.contribution > 0 > bear.contribution

    def test_rationale_mentions_the_symbol_and_levels(self):
        sig = generate("T", self.df, self.cfg, 25_000.0)
        if sig is None:
            pytest.skip("no signal")
        text = sig.rationale()
        assert "T" in text and "stop" in text and "target" in text

    def test_scan_survives_a_broken_frame(self):
        frames = {"GOOD": self.df, "BAD": pd.DataFrame({"close": [1, 2, 3]})}
        out = scan(frames, self.cfg, 25_000.0)
        assert all(s.symbol != "BAD" for s in out)

    def test_scan_orders_actionable_first(self):
        self.cfg.signal.min_confluence_score = 0.1
        frames = {
            s: enrich(datamod.synthetic(s, bars=800, seed=i))
            for i, s in enumerate(["AAA", "BBB", "CCC"])
        }
        out = scan(frames, self.cfg, 25_000.0)
        flags = [s.actionable for s in out]
        assert flags == sorted(flags, reverse=True)


# =====================================================================
#  Learning loop
# =====================================================================
class TestLearner:
    def _journal_with(self, tmp_path, n: int, r_each: float) -> Journal:
        j = Journal(tmp_path / "j.sqlite")
        for i in range(n):
            with j._con() as con:
                con.execute(
                    "INSERT INTO signal (ts,symbol,direction,strategy,score,taken,"
                    "evidence_json) VALUES (?,?,?,?,?,1,?)",
                    (f"2026-08-{(i % 27) + 1:02d} 10:00:00", "AAA", "long", "trend",
                     3.0, '[{"source":"trend","contribution":0.8}]'),
                )
                sid = con.execute("SELECT last_insert_rowid()").fetchone()[0]
            tid = j.open_trade("AAA", "long", 10, 100.0,
                               f"2026-08-{(i % 27) + 1:02d} 10:00:00",
                               strategy="trend", signal_id=sid, planned_stop=99.0)
            exit_px = 100.0 + r_each * 1.0
            j.close_trade(tid, exit_px, f"2026-08-{(i % 27) + 1:02d} 11:00:00",
                          "target" if r_each > 0 else "stop")
        return j

    def test_no_adaptation_on_a_thin_sample(self, tmp_path):
        cfg = Config()
        j = self._journal_with(tmp_path, 5, 1.0)
        out = Learner(cfg, j).review(persist=False)
        assert out.changes == {}
        assert any("curve-fitting" in l or "need" in l for l in out.lessons)

    def test_winning_evidence_gains_weight(self, tmp_path):
        cfg = Config()
        before = cfg.signal.weights["trend"]
        j = self._journal_with(tmp_path, 40, 2.0)
        out = Learner(cfg, j).review(persist=False)
        assert cfg.signal.weights["trend"] >= before
        assert any(e.source == "trend" and e.verdict == "helping" for e in out.evidence)

    def test_losing_evidence_loses_weight(self, tmp_path):
        cfg = Config()
        before = cfg.signal.weights["trend"]
        j = self._journal_with(tmp_path, 40, -1.0)
        Learner(cfg, j).review(persist=False)
        assert cfg.signal.weights["trend"] < before

    def test_weights_stay_within_bounds(self, tmp_path):
        cfg = Config()
        j = self._journal_with(tmp_path, 40, 3.0)
        learner = Learner(cfg, j)
        for _ in range(200):
            learner.review(persist=False)
        for w in cfg.signal.weights.values():
            assert cfg.learn.weight_floor <= w <= cfg.learn.weight_ceiling

    def test_negative_expectancy_never_raises_risk(self, tmp_path):
        cfg = Config()
        before = cfg.risk.risk_per_trade_pct
        j = self._journal_with(tmp_path, 40, -1.0)
        Learner(cfg, j).review(persist=False)
        assert cfg.risk.risk_per_trade_pct <= before

    def test_risk_never_exceeds_the_gann_cap(self, tmp_path):
        cfg = Config()
        j = self._journal_with(tmp_path, 60, 3.0)
        learner = Learner(cfg, j)
        for _ in range(100):
            learner.review(persist=False)
        assert cfg.risk.risk_per_trade_pct <= cfg.risk.kelly_cap_pct

    def test_review_is_persisted(self, tmp_path):
        cfg = Config()
        j = self._journal_with(tmp_path, 25, 1.0)
        Learner(cfg, j).review(scope="daily", persist=True)
        assert len(j.reviews()) == 1

    def test_evidence_performance_shape(self, tmp_path):
        j = self._journal_with(tmp_path, 30, 1.0)
        perf = evidence_performance(j)
        assert perf and all(0.0 <= p.win_rate <= 1.0 for p in perf)

    def test_report_renders(self, tmp_path):
        cfg = Config()
        j = self._journal_with(tmp_path, 30, 1.0)
        text = Learner(cfg, j).review(persist=False).report()
        assert "REVIEW" in text and "win rate" in text


# =====================================================================
#  Bundled real market data
# =====================================================================
class TestBundledData:
    def test_loads_real_symbols(self):
        df = datamod.from_bundled("AAPL")
        assert len(df) > 1000
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_bars_are_structurally_valid(self):
        df = datamod.from_bundled("MSFT")
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["volume"] > 0).all()

    def test_index_is_sorted_and_unique(self):
        df = datamod.from_bundled("IBM")
        assert df.index.is_monotonic_increasing
        assert df.index.is_unique

    def test_unknown_symbol_lists_alternatives(self):
        with pytest.raises(datamod.DataError, match="available"):
            datamod.from_bundled("NOTREAL")

    def test_reachable_through_load(self):
        df = datamod.load("GOOG", provider="bundled")
        assert len(df) > 500

    def test_bars_stamped_inside_the_session(self):
        # Daily bars carry no intraday time; stamped mid-session so the
        # session-clock gate sees a valid trading hour.
        df = datamod.from_bundled("AAPL")
        assert df.index.hour.min() == df.index.hour.max() == 11
