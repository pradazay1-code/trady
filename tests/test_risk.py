"""Risk layer: sizing formulas and the permission gate.

These are the tests that matter most. A bug in pattern detection costs a bad
trade; a bug here costs the account or a 90-day regulatory restriction.

The formula tests pin behaviour to the books' own worked examples, so a future
refactor that silently changes the maths fails loudly.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from trady.config import Config, PDTConfig, RiskConfig
from trady.risk import (
    AccountState,
    PDTTracker,
    RiskManager,
    expectancy,
    fixed_fractional_shares,
    fixed_ratio_units,
    kelly_fraction,
    monte_carlo_ruin,
    optimal_f_shares,
    position_size,
    probability_of_ruin,
    stop_and_target,
    trail_stop,
)

TRADING_TIME = datetime(2026, 8, 13, 11, 0)  # Thursday, inside the entry window


# =====================================================================
#  Book worked examples  (Day Trading For Dummies, 3rd ed., ch. 6)
# =====================================================================
class TestBookFormulas:
    def test_expectancy_matches_book(self):
        # 40% lose 1%, 60% win 1.5% -> 0.5% per trade
        assert expectancy(0.6, 0.015, 0.01) == pytest.approx(0.005, abs=1e-9)

    def test_expectancy_negative_edge(self):
        assert expectancy(0.3, 0.01, 0.02) < 0

    def test_probability_of_ruin_matches_book(self):
        # 60% win rate, account split into 10 units -> ~1.7%
        assert probability_of_ruin(0.6, 10) == pytest.approx(0.0173, abs=5e-4)

    def test_probability_of_ruin_certain_without_edge(self):
        assert probability_of_ruin(0.5, 10) == 1.0
        assert probability_of_ruin(0.4, 10) == 1.0

    def test_probability_of_ruin_falls_with_more_units(self):
        assert probability_of_ruin(0.6, 20) < probability_of_ruin(0.6, 5)

    def test_kelly_matches_book(self):
        # W=0.60, R=1.5/1.0 -> 33.3%
        assert kelly_fraction(0.6, 0.015, 0.010) == pytest.approx(0.3333, abs=1e-3)

    def test_kelly_is_zero_without_edge(self):
        # A losing system must never produce a positive bet size.
        assert kelly_fraction(0.3, 0.01, 0.02) == 0.0
        assert kelly_fraction(0.5, 0.01, 0.01) == 0.0

    def test_fixed_ratio_matches_book(self):
        # P=$10,000 profit, delta=$3,500 -> 2.94 contracts
        assert fixed_ratio_units(10_000, 3_500) == pytest.approx(2.94, abs=0.01)

    def test_optimal_f_matches_book(self):
        # equity 25k, F=0.30, worst loss 40%, price $25 -> 750 shares
        assert optimal_f_shares(25_000, 0.30, 0.40, 25) == 750

    def test_fixed_fractional(self):
        # 10% of $20,000 risking $3,500/unit -> 0 whole units (0.57 truncated)
        assert fixed_fractional_shares(20_000, 0.10, 3_500) == 0
        assert fixed_fractional_shares(25_000, 0.01, 1.0) == 250

    def test_degenerate_inputs_do_not_explode(self):
        assert fixed_fractional_shares(25_000, 0.01, 0) == 0
        assert optimal_f_shares(25_000, 0.3, 0, 25) == 0
        assert fixed_ratio_units(-100, 3500) == pytest.approx(1.0)


# =====================================================================
#  Position sizing
# =====================================================================
class TestPositionSize:
    def setup_method(self):
        self.cfg = RiskConfig()

    def test_respects_gann_ten_percent_cap(self):
        r = position_size(25_000, 100.0, 99.9, 102.0, self.cfg)
        # A 10c stop would otherwise buy thousands of shares.
        assert r.notional <= 25_000 * self.cfg.max_position_pct + 100
        assert "gann_10pct_position_cap" in r.caps_applied

    def test_risk_per_trade_respected_when_not_capped(self):
        r = position_size(100_000, 50.0, 48.0, 56.0, self.cfg)
        assert r.risk_pct_of_equity <= self.cfg.risk_per_trade_pct * 1.001

    def test_clamps_absurdly_wide_stop(self):
        # A 50% stop must be clamped to max_stop_pct.
        r = position_size(25_000, 100.0, 50.0, 200.0, self.cfg)
        assert "stop_clamped_to_max_stop_pct" in r.caps_applied
        assert abs(100.0 - r.stop_price) / 100.0 == pytest.approx(
            self.cfg.max_stop_pct, abs=1e-6
        )

    def test_zero_risk_returns_no_position(self):
        r = position_size(25_000, 100.0, 100.0, 105.0, self.cfg)
        assert r.shares == 0
        assert not r.ok

    def test_kelly_used_once_there_is_data(self):
        stats = {"win_rate": 0.6, "avg_win": 0.015, "avg_loss": 0.01, "trades": 50}
        r = position_size(25_000, 100.0, 98.0, 104.0, self.cfg, stats=stats)
        assert "kelly" in r.method

    def test_kelly_ignored_on_thin_sample(self):
        stats = {"win_rate": 0.9, "avg_win": 0.05, "avg_loss": 0.01, "trades": 3}
        r = position_size(25_000, 100.0, 98.0, 104.0, self.cfg, stats=stats)
        assert r.method == "fixed_fractional"

    def test_kelly_never_exceeds_gann_cap(self):
        # A wildly favourable sample must still not size above 10%.
        stats = {"win_rate": 0.95, "avg_win": 0.10, "avg_loss": 0.005, "trades": 200}
        r = position_size(25_000, 100.0, 98.0, 130.0, self.cfg, stats=stats)
        assert r.notional <= 25_000 * self.cfg.max_position_pct + 100

    def test_losing_system_falls_back_not_up(self):
        stats = {"win_rate": 0.2, "avg_win": 0.01, "avg_loss": 0.02, "trades": 60}
        r = position_size(25_000, 100.0, 98.0, 104.0, self.cfg, stats=stats)
        assert "kelly_nonpositive_fell_back_to_fixed_fractional" in r.caps_applied

    def test_reward_risk_computed(self):
        r = position_size(25_000, 100.0, 98.0, 104.0, self.cfg)
        assert r.reward_risk == pytest.approx(2.0, abs=1e-6)

    def test_short_side_sizing(self):
        r = position_size(25_000, 100.0, 102.0, 94.0, self.cfg)
        assert r.shares > 0
        assert r.reward_risk == pytest.approx(3.0, abs=1e-6)


class TestStopsAndTargets:
    def test_long_stop_below_target_above(self):
        stop, target = stop_and_target(100.0, 2.0, "long", RiskConfig())
        assert stop < 100.0 < target

    def test_short_stop_above_target_below(self):
        stop, target = stop_and_target(100.0, 2.0, "short", RiskConfig())
        assert target < 100.0 < stop

    def test_trailing_stop_only_ratchets_favourably(self):
        cfg = RiskConfig()
        assert trail_stop(95.0, 110.0, 2.0, "long", cfg) > 95.0   # raised
        assert trail_stop(95.0, 96.0, 2.0, "long", cfg) == 95.0   # never lowered
        assert trail_stop(105.0, 90.0, 2.0, "short", cfg) < 105.0  # lowered
        assert trail_stop(105.0, 104.0, 2.0, "short", cfg) == 105.0

    def test_trailing_disabled_is_a_noop(self):
        cfg = RiskConfig(use_trailing_stop=False)
        assert trail_stop(95.0, 200.0, 2.0, "long", cfg) == 95.0


# =====================================================================
#  Pattern day trader guard  (FINRA NASD 2520)
# =====================================================================
class TestPDT:
    def setup_method(self):
        self.cfg = PDTConfig()
        self.pdt = PDTTracker(self.cfg)
        self.today = date(2026, 8, 13)  # Thursday

    def test_unrestricted_at_or_above_threshold(self):
        assert not self.pdt.is_restricted(25_000)
        assert not self.pdt.is_restricted(30_000)
        assert self.pdt.remaining(30_000, self.today) == -1
        assert self.pdt.can_day_trade(30_000, self.today)[0]

    def test_restricted_below_threshold(self):
        assert self.pdt.is_restricted(24_999)

    def test_reserve_holds_one_back(self):
        # 3 allowed, 1 reserved for an emergency exit -> 2 usable.
        assert self.pdt.remaining(10_000, self.today) == 2

    def test_budget_exhausts_and_then_blocks(self):
        for _ in range(2):
            assert self.pdt.can_day_trade(10_000, self.today)[0]
            self.pdt.record("AAPL", self.today)
        allowed, why = self.pdt.can_day_trade(10_000, self.today)
        assert not allowed
        assert "PDT budget exhausted" in why
        assert self.pdt.remaining(10_000, self.today) == 0

    def test_rolling_window_expires_old_trades(self):
        old = self.today - timedelta(days=30)
        for _ in range(3):
            self.pdt.record("AAPL", old)
        assert self.pdt.count(self.today) == 0
        assert self.pdt.can_day_trade(10_000, self.today)[0]

    def test_window_spans_five_business_days_not_calendar(self):
        # Thu 13th back five business days reaches Fri the 7th, so a trade on
        # the 10th (Mon) is inside the window and one on the 6th is not.
        self.pdt.record("A", date(2026, 8, 10))
        self.pdt.record("B", date(2026, 8, 6))
        assert self.pdt.count(self.today) == 1

    def test_equity_crossing_threshold_changes_regime(self):
        for _ in range(3):
            self.pdt.record("AAPL", self.today)
        assert not self.pdt.can_day_trade(24_000, self.today)[0]
        assert self.pdt.can_day_trade(26_000, self.today)[0]

    def test_disabled_config_never_restricts(self):
        pdt = PDTTracker(PDTConfig(enabled=False))
        for _ in range(10):
            pdt.record("AAPL", self.today)
        assert pdt.can_day_trade(1_000, self.today)[0]

    def test_load_replaces_state(self):
        self.pdt.record("AAPL", self.today)
        self.pdt.load([("MSFT", self.today), ("NVDA", self.today)])
        assert self.pdt.count(self.today) == 2


# =====================================================================
#  The permission gate
# =====================================================================
def _state(**kw) -> AccountState:
    base = dict(
        equity=30_000.0, starting_equity_today=30_000.0, peak_equity=30_000.0,
        open_positions=0, gross_exposure=0.0, realized_pnl_today=0.0,
        trades_today=0, consecutive_losses=0,
    )
    base.update(kw)
    return AccountState(**base)


class TestRiskManager:
    def setup_method(self):
        self.cfg = Config()
        self.rm = RiskManager(self.cfg)

    def test_clean_state_allows(self):
        d = self.rm.check(_state(), now=TRADING_TIME)
        assert d.allowed and bool(d) is True

    def test_halt_blocks_everything(self):
        d = self.rm.check(_state(halted=True, halt_reason="drawdown"), now=TRADING_TIME)
        assert not d.allowed and "halted" in d.reason

    def test_daily_loss_limit_blocks(self):
        # -2% of 30,000 = -600
        d = self.rm.check(_state(realized_pnl_today=-700.0), now=TRADING_TIME)
        assert not d.allowed and "daily loss limit" in d.reason

    def test_just_inside_daily_loss_limit_allows(self):
        d = self.rm.check(_state(realized_pnl_today=-500.0), now=TRADING_TIME)
        assert d.allowed

    def test_drawdown_halt_blocks(self):
        d = self.rm.check(
            _state(equity=26_000.0, peak_equity=30_000.0), now=TRADING_TIME
        )
        assert not d.allowed and "drawdown" in d.reason

    def test_trade_cap_blocks(self):
        d = self.rm.check(
            _state(trades_today=self.cfg.risk.max_daily_trades), now=TRADING_TIME
        )
        assert not d.allowed and "daily trade cap" in d.reason

    def test_consecutive_losses_block(self):
        d = self.rm.check(
            _state(consecutive_losses=self.cfg.risk.max_consecutive_losses),
            now=TRADING_TIME,
        )
        assert not d.allowed and "losses in a row" in d.reason

    def test_position_count_blocks(self):
        d = self.rm.check(
            _state(open_positions=self.cfg.risk.max_open_positions), now=TRADING_TIME
        )
        assert not d.allowed and "positions" in d.reason

    def test_gross_exposure_blocks(self):
        d = self.rm.check(
            _state(gross_exposure=29_000.0), now=TRADING_TIME, new_notional=5_000.0
        )
        assert not d.allowed and "exposure" in d.reason

    # -- session clock ------------------------------------------------
    @pytest.mark.parametrize(
        "hh,mm,expect_ok,fragment",
        [
            (9, 45, False, "opening window"),   # gap-and-crap window
            (10, 30, True, ""),
            (12, 30, False, "chop"),            # lunch
            (14, 0, True, ""),
            (15, 45, False, "too late"),
            (17, 0, False, "closed"),
            (8, 0, False, "closed"),
        ],
    )
    def test_session_windows(self, hh, mm, expect_ok, fragment):
        d = self.rm.check(_state(), now=datetime(2026, 8, 13, hh, mm))
        assert d.allowed is expect_ok
        if fragment:
            assert fragment in d.reason

    # -- PDT integration ----------------------------------------------
    def test_pdt_blocks_same_day_entry_when_exhausted(self):
        cfg = Config()
        cfg.risk.max_drawdown_halt_pct = 1.0
        pdt = PDTTracker(cfg.pdt)
        for _ in range(3):
            pdt.record("AAPL", TRADING_TIME.date())
        rm = RiskManager(cfg, pdt)
        d = rm.check(
            _state(equity=10_000.0, starting_equity_today=10_000.0,
                   peak_equity=10_000.0),
            now=TRADING_TIME, intends_same_day_exit=True,
        )
        assert not d.allowed and "PDT" in d.reason

    def test_pdt_ignored_for_swing_entry(self):
        cfg = Config()
        pdt = PDTTracker(cfg.pdt)
        for _ in range(3):
            pdt.record("AAPL", TRADING_TIME.date())
        rm = RiskManager(cfg, pdt)
        d = rm.check(
            _state(equity=10_000.0, starting_equity_today=10_000.0,
                   peak_equity=10_000.0),
            now=TRADING_TIME, intends_same_day_exit=False,
        )
        assert d.allowed

    def test_checks_dict_is_populated_for_the_journal(self):
        d = self.rm.check(_state(), now=TRADING_TIME)
        for key in ("session_window", "drawdown", "daily_pnl_pct", "trades_today"):
            assert key in d.checks

    def test_should_halt_reports_reason(self):
        halt, why = self.rm.should_halt(
            _state(equity=26_000.0, peak_equity=30_000.0)
        )
        assert halt and "drawdown" in why
        halt, _ = self.rm.should_halt(_state())
        assert not halt


class TestMonteCarlo:
    def test_positive_edge_beats_negative_edge(self):
        good = monte_carlo_ruin(0.6, 2.0, 1.0, 0.01)
        bad = monte_carlo_ruin(0.35, 1.0, 1.0, 0.01)
        assert good["median_ending_multiple"] > bad["median_ending_multiple"]
        assert good["prob_ruin"] <= bad["prob_ruin"]

    def test_more_risk_means_more_ruin(self):
        low = monte_carlo_ruin(0.45, 1.2, 1.0, 0.005)
        high = monte_carlo_ruin(0.45, 1.2, 1.0, 0.05)
        assert high["prob_ruin"] >= low["prob_ruin"]

    def test_deterministic_for_a_given_seed(self):
        a = monte_carlo_ruin(0.55, 1.5, 1.0, 0.01, seed=3)
        b = monte_carlo_ruin(0.55, 1.5, 1.0, 0.01, seed=3)
        assert a == b


# =====================================================================
#  Sector concentration
# =====================================================================
class TestSectorConcentration:
    def setup_method(self):
        self.cfg = Config()
        self.rm = RiskManager(self.cfg)

    def test_blocks_a_third_correlated_position(self):
        d = self.rm.check(
            _state(open_symbols=["AAPL", "MSFT"]), now=TRADING_TIME, symbol="NVDA"
        )
        assert not d.allowed
        assert "tech" in d.reason

    def test_allows_a_second_in_the_same_sector(self):
        d = self.rm.check(
            _state(open_symbols=["AAPL"]), now=TRADING_TIME, symbol="MSFT"
        )
        assert d.allowed

    def test_allows_a_different_sector(self):
        d = self.rm.check(
            _state(open_symbols=["AAPL", "MSFT"]), now=TRADING_TIME, symbol="XOM"
        )
        assert d.allowed

    def test_unknown_symbols_are_unconstrained(self):
        # An incomplete sector map must never silently block trades.
        d = self.rm.check(
            _state(open_symbols=["ZZZZ", "YYYY"]), now=TRADING_TIME, symbol="WWWW"
        )
        assert d.allowed

    def test_limit_is_configurable(self):
        self.cfg.risk.max_sector_concentration = 1
        d = self.rm.check(
            _state(open_symbols=["AAPL"]), now=TRADING_TIME, symbol="MSFT"
        )
        assert not d.allowed

    def test_sector_appears_in_checks(self):
        d = self.rm.check(
            _state(open_symbols=["AAPL"]), now=TRADING_TIME, symbol="MSFT"
        )
        assert "tech" in d.checks.get("sector", "")
