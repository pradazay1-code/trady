"""Options pricing, Greeks, cost analysis and sizing.

Black-Scholes is pinned to textbook reference values and to put-call parity, so
a refactor that breaks the maths fails loudly rather than quietly mispricing
every contract.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from trady.config import RiskConfig
from trady.options import (
    CONTRACT_MULTIPLIER,
    OptionContract,
    black_scholes,
    fidelity_option_order,
    implied_volatility,
    round_trip_cost_pct,
    select_contract,
    size_option_position,
)


def _contract(**kw) -> OptionContract:
    base = dict(
        underlying="AAPL", strike=100.0,
        expiry=date.today() + timedelta(days=30), kind="call",
        bid=4.90, ask=5.10, volume=500, open_interest=2000, spot=100.0,
    )
    base.update(kw)
    return OptionContract(**base)


# =====================================================================
#  Black-Scholes
# =====================================================================
class TestBlackScholes:
    def test_matches_textbook_reference(self):
        # S=100 K=100 T=1y vol=20% r=5% -> call 10.4506
        price, _ = black_scholes(100, 100, 365, 0.20, 0.05, "call")
        assert price == pytest.approx(10.4506, abs=1e-3)

    def test_put_reference(self):
        price, _ = black_scholes(100, 100, 365, 0.20, 0.05, "put")
        assert price == pytest.approx(5.5735, abs=1e-3)

    def test_put_call_parity(self):
        c, _ = black_scholes(100, 95, 200, 0.25, 0.04, "call")
        p, _ = black_scholes(100, 95, 200, 0.25, 0.04, "put")
        expected = 100 - 95 * math.exp(-0.04 * 200 / 365)
        assert (c - p) == pytest.approx(expected, abs=1e-4)

    def test_at_expiry_is_intrinsic(self):
        itm, _ = black_scholes(110, 100, 0, 0.30, 0.04, "call")
        otm, _ = black_scholes(90, 100, 0, 0.30, 0.04, "call")
        assert itm == pytest.approx(10.0)
        assert otm == pytest.approx(0.0)

    def test_price_never_negative(self):
        for spot in (1, 50, 100, 500):
            for strike in (1, 50, 100, 500):
                p, _ = black_scholes(spot, strike, 30, 0.3, 0.04, "call")
                assert p >= 0.0

    def test_more_time_costs_more(self):
        short, _ = black_scholes(100, 100, 7, 0.25, 0.04, "call")
        long_, _ = black_scholes(100, 100, 90, 0.25, 0.04, "call")
        assert long_ > short

    def test_more_volatility_costs_more(self):
        low, _ = black_scholes(100, 100, 30, 0.15, 0.04, "call")
        high, _ = black_scholes(100, 100, 30, 0.45, 0.04, "call")
        assert high > low

    def test_call_delta_bounded_0_1(self):
        for spot in (60, 100, 160):
            _, g = black_scholes(spot, 100, 30, 0.3, 0.04, "call")
            assert 0.0 <= g.delta <= 1.0

    def test_put_delta_bounded_minus1_0(self):
        for spot in (60, 100, 160):
            _, g = black_scholes(spot, 100, 30, 0.3, 0.04, "put")
            assert -1.0 <= g.delta <= 0.0

    def test_long_option_theta_is_negative(self):
        # The books' "wasting asset": time works against the holder.
        _, g = black_scholes(100, 100, 30, 0.3, 0.04, "call")
        assert g.theta < 0

    def test_theta_accelerates_near_expiry(self):
        _, far = black_scholes(100, 100, 90, 0.3, 0.04, "call")
        _, near = black_scholes(100, 100, 3, 0.3, 0.04, "call")
        # "Current-month options decay at faster rates."
        assert abs(near.theta) > abs(far.theta)

    def test_gamma_peaks_near_the_money(self):
        _, atm = black_scholes(100, 100, 30, 0.3, 0.04, "call")
        _, otm = black_scholes(70, 100, 30, 0.3, 0.04, "call")
        assert atm.gamma > otm.gamma

    def test_bad_kind_raises(self):
        with pytest.raises(ValueError):
            black_scholes(100, 100, 30, 0.3, 0.04, "banana")


class TestImpliedVolatility:
    def test_round_trips(self):
        price, _ = black_scholes(100, 100, 180, 0.3456, 0.04, "call")
        assert implied_volatility(price, 100, 100, 180, 0.04, "call") == pytest.approx(
            0.3456, abs=1e-3
        )

    def test_round_trips_for_puts(self):
        price, _ = black_scholes(100, 105, 90, 0.28, 0.04, "put")
        assert implied_volatility(price, 100, 105, 90, 0.04, "put") == pytest.approx(
            0.28, abs=1e-3
        )

    def test_intrinsic_only_price_gives_zero(self):
        assert implied_volatility(10.0, 110, 100, 30, 0.04, "call") == 0.0

    def test_expired_gives_zero(self):
        assert implied_volatility(5.0, 100, 100, 0, 0.04, "call") == 0.0


# =====================================================================
#  Contract
# =====================================================================
class TestOptionContract:
    def test_mid_and_spread(self):
        c = _contract(bid=4.90, ask=5.10)
        assert c.mid == pytest.approx(5.0)
        assert c.spread == pytest.approx(0.20)
        assert c.spread_pct == pytest.approx(0.04)

    def test_occ_symbol_format(self):
        c = _contract(underlying="AAPL", strike=215.0,
                      expiry=date(2026, 8, 21), kind="call")
        assert c.occ_symbol == "AAPL  260821C00215000"

    def test_occ_symbol_for_put(self):
        c = _contract(underlying="SPY", strike=500.0,
                      expiry=date(2026, 1, 16), kind="put")
        assert c.occ_symbol == "SPY   260116P00500000"

    def test_intrinsic_and_extrinsic_split(self):
        c = _contract(strike=100.0, spot=110.0, bid=12.0, ask=12.2)
        assert c.intrinsic == pytest.approx(10.0)
        assert c.extrinsic == pytest.approx(c.mid - 10.0)

    def test_put_intrinsic(self):
        c = _contract(kind="put", strike=100.0, spot=90.0)
        assert c.intrinsic == pytest.approx(10.0)

    def test_moneyness(self):
        assert _contract(strike=100, spot=110).moneyness == "ITM"
        assert _contract(strike=100, spot=90).moneyness == "OTM"
        assert _contract(strike=100, spot=100).moneyness == "ATM"

    def test_put_moneyness_is_mirrored(self):
        assert _contract(kind="put", strike=100, spot=90).moneyness == "ITM"
        assert _contract(kind="put", strike=100, spot=110).moneyness == "OTM"

    def test_days_to_expiry_never_negative(self):
        c = _contract(expiry=date.today() - timedelta(days=5))
        assert c.days_to_expiry() == 0

    def test_greeks_available_from_quote(self):
        g = _contract().greeks()
        assert 0.0 < g.delta < 1.0
        assert g.theta < 0


# =====================================================================
#  Cost analysis — the number that decides options trades
# =====================================================================
class TestCostAnalysis:
    def test_tight_liquid_contract_is_acceptable(self):
        c = _contract(bid=4.95, ask=5.05, expiry=date.today() + timedelta(days=45))
        a = round_trip_cost_pct(c, hold_days=1.0)
        assert a.verdict in ("acceptable", "expensive")
        assert a.spread_cost_pct < 0.03

    def test_wide_spread_short_dated_is_prohibitive(self):
        c = _contract(bid=1.05, ask=1.20, strike=110, spot=100,
                      expiry=date.today() + timedelta(days=3))
        a = round_trip_cost_pct(c, hold_days=1.0)
        assert a.verdict == "prohibitive"
        assert a.total_hold_cost_pct > 0.15

    def test_theta_dominates_near_expiry(self):
        near = round_trip_cost_pct(
            _contract(expiry=date.today() + timedelta(days=2)), hold_days=1.0)
        far = round_trip_cost_pct(
            _contract(expiry=date.today() + timedelta(days=60)), hold_days=1.0)
        assert near.theta_cost_pct_per_day > far.theta_cost_pct_per_day

    def test_longer_hold_costs_more(self):
        c = _contract()
        assert (round_trip_cost_pct(c, hold_days=5).total_hold_cost_pct
                > round_trip_cost_pct(c, hold_days=1).total_hold_cost_pct)

    def test_breakeven_move_is_reported(self):
        a = round_trip_cost_pct(_contract(), hold_days=1.0)
        assert a.breakeven_underlying_move_pct > 0

    def test_thin_liquidity_is_flagged(self):
        a = round_trip_cost_pct(_contract(open_interest=10, volume=2))
        joined = " ".join(a.notes)
        assert "open interest" in joined

    def test_expiring_contract_is_flagged(self):
        a = round_trip_cost_pct(_contract(expiry=date.today() + timedelta(days=1)))
        assert any("theta is brutal" in n for n in a.notes)

    def test_unpriceable_contract_degrades_safely(self):
        a = round_trip_cost_pct(_contract(bid=0, ask=0, last=0))
        assert a.verdict == "unpriceable"


# =====================================================================
#  Sizing
# =====================================================================
class TestOptionSizing:
    def setup_method(self):
        self.cfg = RiskConfig()

    def test_max_loss_equals_premium_paid(self):
        # "The most an option holder can lose is the amount paid."
        s = size_option_position(30_000, _contract(), self.cfg)
        assert s.max_loss == pytest.approx(s.total_premium)

    def test_respects_gann_position_cap(self):
        s = size_option_position(30_000, _contract(), self.cfg)
        assert s.total_premium <= 30_000 * self.cfg.max_position_pct + 1

    def test_expensive_premium_sizes_to_zero(self):
        pricey = _contract(bid=400.0, ask=402.0)
        s = size_option_position(5_000, pricey, self.cfg)
        assert s.contracts == 0
        assert not s.ok

    def test_contracts_are_whole_numbers(self):
        s = size_option_position(30_000, _contract(), self.cfg)
        assert isinstance(s.contracts, int)

    def test_stop_and_target_derive_from_premium(self):
        s = size_option_position(30_000, _contract(), self.cfg,
                                 stop_pct_of_premium=0.5,
                                 target_pct_of_premium=1.0)
        assert s.stop_premium == pytest.approx(s.premium_per_contract * 0.5)
        assert s.target_premium == pytest.approx(s.premium_per_contract * 2.0)
        assert s.reward_risk == pytest.approx(2.0)

    def test_bigger_account_buys_more(self):
        small = size_option_position(10_000, _contract(), self.cfg)
        big = size_option_position(100_000, _contract(), self.cfg)
        assert big.contracts >= small.contracts

    def test_unpriceable_contract_sizes_to_zero(self):
        s = size_option_position(30_000, _contract(bid=0, ask=0, last=0), self.cfg)
        assert s.contracts == 0

    def test_risk_stays_within_budget(self):
        s = size_option_position(30_000, _contract(), self.cfg)
        if s.ok:
            assert s.risk_pct_of_equity <= self.cfg.max_daily_loss_pct + 1e-6


# =====================================================================
#  Contract selection
# =====================================================================
class TestSelectContract:
    def _chain(self):
        today = date.today()
        out = []
        for dte in (1, 14, 30, 120):
            for strike in (90, 95, 100, 105, 110):
                for kind in ("call", "put"):
                    price = max(0.5, 5 - abs(100 - strike) * 0.3)
                    out.append(OptionContract(
                        "AAPL", float(strike), today + timedelta(days=dte), kind,
                        bid=price - 0.05, ask=price + 0.05,
                        volume=500, open_interest=2000, spot=100.0,
                    ))
        return out

    def test_picks_a_call_for_long(self):
        c, _ = select_contract(self._chain(), "long")
        assert c is not None and c.kind == "call"

    def test_picks_a_put_for_short(self):
        c, _ = select_contract(self._chain(), "short")
        assert c is not None and c.kind == "put"

    def test_respects_the_dte_window(self):
        c, _ = select_contract(self._chain(), "long", min_dte=7, max_dte=45)
        assert 7 <= c.days_to_expiry() <= 45

    def test_rejects_illiquid_contracts(self):
        chain = [
            OptionContract("AAPL", 100.0, date.today() + timedelta(days=30), "call",
                           bid=4.9, ask=5.1, volume=1, open_interest=5, spot=100.0)
        ]
        c, rejected = select_contract(chain, "long")
        assert c is None
        assert any("OI" in r for r in rejected)

    def test_rejects_wide_spreads(self):
        chain = [
            OptionContract("AAPL", 100.0, date.today() + timedelta(days=30), "call",
                           bid=1.0, ask=3.0, volume=500, open_interest=2000, spot=100.0)
        ]
        c, rejected = select_contract(chain, "long")
        assert c is None
        assert any("spread" in r for r in rejected)

    def test_explains_every_rejection(self):
        _, rejected = select_contract(self._chain(), "long")
        assert rejected  # the 1-DTE and 120-DTE contracts must be explained

    def test_empty_chain_returns_none(self):
        c, _ = select_contract([], "long")
        assert c is None


class TestFidelityOptionOrder:
    def test_ticket_fields(self):
        c = _contract()
        o = fidelity_option_order(c, 4, "buy_to_open", 5.10)
        assert o["quantity"] == 4
        assert o["order_type"] == "limit"
        assert o["approval_level_required"] == 2
        assert o["notional"] == pytest.approx(4 * 5.10 * CONTRACT_MULTIPLIER)

    def test_carries_the_occ_symbol(self):
        o = fidelity_option_order(_contract(), 1, "buy_to_open", 5.0)
        assert len(o["occ_symbol"]) == 21
