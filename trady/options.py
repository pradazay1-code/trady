"""Options: pricing, Greeks, sizing, and the costs that decide the outcome.

Options change the arithmetic of day trading in three ways that matter more than
direction:

1. **Theta.** An option is, in the books' phrase, a *wasting asset* — "as the
   option moves closer to its date of expiration, the value of the option
   declines." Current-month contracts decay fastest. On a stock you can be
   early and still be right. On a long option, being early is a loss even when
   the direction is right.
2. **The spread.** A liquid stock quotes a spread near a penny on a $200 name —
   about 0.005%. A liquid option quotes $0.05 on a $2.00 contract — 2.5%, five
   hundred times worse. Round trip, you pay it twice before direction matters.
3. **Bounded loss, magnified odds.** The most a long option can lose is the
   premium, which makes sizing simpler and caps disasters. It also means a
   100% loss is an ordinary outcome, not a tail event.

`round_trip_cost_pct` puts 1–3 into a single number: how far the underlying must
move before you break even. Look at it before every options trade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime

CONTRACT_MULTIPLIER = 100  # one equity option contract covers 100 shares


# =====================================================================
#  Normal distribution helpers (no scipy dependency)
# =====================================================================
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# =====================================================================
#  Black-Scholes
# =====================================================================
@dataclass
class Greeks:
    delta: float   # price change per $1 of underlying
    gamma: float   # delta change per $1 of underlying
    theta: float   # price change per calendar day (negative for long options)
    vega: float    # price change per 1 volatility point
    rho: float     # price change per 1% of rates

    def to_dict(self) -> dict:
        return {
            "delta": round(self.delta, 4), "gamma": round(self.gamma, 5),
            "theta": round(self.theta, 4), "vega": round(self.vega, 4),
            "rho": round(self.rho, 4),
        }


def black_scholes(
    spot: float,
    strike: float,
    days_to_expiry: float,
    volatility: float,
    rate: float = 0.04,
    kind: str = "call",
    dividend_yield: float = 0.0,
) -> tuple[float, Greeks]:
    """Theoretical price and Greeks. `volatility` is annualised (0.30 = 30%).

    Theta is returned per *calendar day*, which is how it is actually
    experienced, rather than per year.
    """
    kind = kind.lower()
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")

    t = max(days_to_expiry, 0.0) / 365.0
    # At expiry (or with no volatility) the option is worth its intrinsic value.
    if t <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
        intrinsic = max(0.0, spot - strike) if kind == "call" else max(0.0, strike - spot)
        return intrinsic, Greeks(
            delta=(1.0 if intrinsic > 0 else 0.0) * (1 if kind == "call" else -1),
            gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
        )

    sqrt_t = math.sqrt(t)
    d1 = (
        math.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility ** 2) * t
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    disc_r = math.exp(-rate * t)
    disc_q = math.exp(-dividend_yield * t)

    if kind == "call":
        price = spot * disc_q * _norm_cdf(d1) - strike * disc_r * _norm_cdf(d2)
        delta = disc_q * _norm_cdf(d1)
        theta_year = (
            -(spot * _norm_pdf(d1) * volatility * disc_q) / (2 * sqrt_t)
            - rate * strike * disc_r * _norm_cdf(d2)
            + dividend_yield * spot * disc_q * _norm_cdf(d1)
        )
        rho = strike * t * disc_r * _norm_cdf(d2) / 100.0
    else:
        price = strike * disc_r * _norm_cdf(-d2) - spot * disc_q * _norm_cdf(-d1)
        delta = -disc_q * _norm_cdf(-d1)
        theta_year = (
            -(spot * _norm_pdf(d1) * volatility * disc_q) / (2 * sqrt_t)
            + rate * strike * disc_r * _norm_cdf(-d2)
            - dividend_yield * spot * disc_q * _norm_cdf(-d1)
        )
        rho = -strike * t * disc_r * _norm_cdf(-d2) / 100.0

    gamma = disc_q * _norm_pdf(d1) / (spot * volatility * sqrt_t)
    vega = spot * disc_q * _norm_pdf(d1) * sqrt_t / 100.0

    return max(price, 0.0), Greeks(delta, gamma, theta_year / 365.0, vega, rho)


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    days_to_expiry: float,
    rate: float = 0.04,
    kind: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Back out volatility from a quoted price by bisection.

    Bisection rather than Newton-Raphson: vega collapses toward zero for deep
    in- or out-of-the-money contracts, and Newton diverges there. Bisection is
    slower and always converges.
    """
    intrinsic = (max(0.0, spot - strike) if kind == "call"
                 else max(0.0, strike - spot))
    if market_price <= intrinsic + 1e-9 or days_to_expiry <= 0:
        return 0.0

    lo, hi = 1e-4, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price, _ = black_scholes(spot, strike, days_to_expiry, mid, rate, kind)
        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# =====================================================================
#  Contracts
# =====================================================================
@dataclass
class OptionContract:
    """One listed contract, with the quote as it actually trades."""

    underlying: str
    strike: float
    expiry: date
    kind: str                    # "call" | "put"
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_vol: float = 0.0
    spot: float = 0.0

    # -- quote ---------------------------------------------------------
    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last or self.ask or self.bid

    @property
    def spread(self) -> float:
        return max(0.0, self.ask - self.bid)

    @property
    def spread_pct(self) -> float:
        m = self.mid
        return self.spread / m if m > 0 else 1.0

    # -- contract terms ------------------------------------------------
    def days_to_expiry(self, asof: date | None = None) -> int:
        return max(0, (self.expiry - (asof or date.today())).days)

    @property
    def occ_symbol(self) -> str:
        """OCC option symbol, e.g. AAPL  260821C00215000."""
        root = self.underlying.upper().ljust(6)
        cp = "C" if self.kind == "call" else "P"
        strike_int = int(round(self.strike * 1000))
        return f"{root}{self.expiry:%y%m%d}{cp}{strike_int:08d}"

    def describe(self) -> str:
        return (f"{self.underlying} {self.expiry:%b %d} "
                f"${self.strike:g} {self.kind.upper()}")

    # -- economics -----------------------------------------------------
    @property
    def intrinsic(self) -> float:
        if self.kind == "call":
            return max(0.0, self.spot - self.strike)
        return max(0.0, self.strike - self.spot)

    @property
    def extrinsic(self) -> float:
        """Time value — the part theta eats."""
        return max(0.0, self.mid - self.intrinsic)

    @property
    def moneyness(self) -> str:
        if self.spot <= 0:
            return "unknown"
        diff = (self.spot - self.strike) / self.spot
        if self.kind == "put":
            diff = -diff
        if diff > 0.02:
            return "ITM"
        if diff < -0.02:
            return "OTM"
        return "ATM"

    def greeks(self, rate: float = 0.04, asof: date | None = None) -> Greeks:
        vol = self.implied_vol or implied_volatility(
            self.mid, self.spot, self.strike, self.days_to_expiry(asof), rate, self.kind
        )
        _, g = black_scholes(
            self.spot, self.strike, self.days_to_expiry(asof), vol, rate, self.kind
        )
        return g


# =====================================================================
#  The number that decides options trades
# =====================================================================
@dataclass
class OptionCostAnalysis:
    spread_cost_pct: float          # round-trip spread, as % of premium
    theta_cost_pct_per_day: float   # daily decay, as % of premium
    commission_cost_pct: float
    total_hold_cost_pct: float      # all-in for the intended holding period
    breakeven_underlying_move_pct: float
    verdict: str
    notes: list[str] = field(default_factory=list)


def round_trip_cost_pct(
    contract: OptionContract,
    hold_days: float = 1.0,
    commission_per_contract: float = 0.65,
    contracts: int = 1,
    rate: float = 0.04,
) -> OptionCostAnalysis:
    """How far must the underlying move before this trade breaks even?

    This is the single most useful number in options day trading, and the one
    most often skipped. It combines the spread paid twice, theta over the
    holding period, and commissions, then converts the total into the underlying
    move required to cover it via delta.
    """
    premium = contract.mid
    notes: list[str] = []
    if premium <= 0:
        return OptionCostAnalysis(1.0, 1.0, 1.0, 1.0, 1.0, "unpriceable",
                                  ["no usable quote"])

    # Spread is paid on the way in and again on the way out.
    spread_cost = contract.spread_pct
    if contract.bid <= 0 or contract.ask <= 0:
        spread_cost = 0.05
        notes.append("no two-sided quote — spread cost assumed at 5%")

    g = contract.greeks(rate)
    theta_per_day_pct = abs(g.theta) / premium if premium > 0 else 0.0

    commission_pct = (
        (commission_per_contract * 2 * contracts)
        / (premium * CONTRACT_MULTIPLIER * contracts)
    )

    total = spread_cost + theta_per_day_pct * hold_days + commission_pct

    # Convert the cost into the underlying move needed to cover it.
    delta = abs(g.delta)
    if delta > 1e-6 and contract.spot > 0:
        premium_needed = total * premium
        underlying_move = premium_needed / delta
        breakeven_pct = underlying_move / contract.spot
    else:
        breakeven_pct = 1.0
        notes.append("delta near zero — this contract barely tracks the underlying")

    dte = contract.days_to_expiry()
    if dte <= 2:
        notes.append(
            f"{dte} day(s) to expiry — theta is brutal here and gamma risk is extreme"
        )
    if contract.open_interest < 100:
        notes.append(
            f"open interest {contract.open_interest} — you may not get out at a fair price"
        )
    if contract.volume < 50:
        notes.append(f"volume {contract.volume} today — thin")
    if spread_cost > 0.10:
        notes.append(
            f"spread is {spread_cost:.0%} of premium — you start {spread_cost:.0%} down"
        )
    if contract.moneyness == "OTM" and dte <= 7:
        notes.append(
            "far OTM with little time left: the most likely outcome is a 100% loss"
        )

    if total < 0.06:
        verdict = "acceptable"
    elif total < 0.15:
        verdict = "expensive"
    else:
        verdict = "prohibitive"

    return OptionCostAnalysis(
        spread_cost_pct=round(spread_cost, 4),
        theta_cost_pct_per_day=round(theta_per_day_pct, 4),
        commission_cost_pct=round(commission_pct, 4),
        total_hold_cost_pct=round(total, 4),
        breakeven_underlying_move_pct=round(breakeven_pct, 5),
        verdict=verdict,
        notes=notes,
    )


# =====================================================================
#  Sizing
# =====================================================================
@dataclass
class OptionSizing:
    contracts: int
    premium_per_contract: float
    total_premium: float
    max_loss: float
    risk_pct_of_equity: float
    target_premium: float
    stop_premium: float
    reward_risk: float
    caps_applied: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.contracts > 0


def size_option_position(
    equity: float,
    contract: OptionContract,
    risk_cfg,
    stop_pct_of_premium: float = 0.50,
    target_pct_of_premium: float = 1.00,
) -> OptionSizing:
    """Size a long option position.

    A long option's maximum loss is the premium — the books are explicit: "the
    most an option holder can lose is the amount paid for the option contract."
    That bounded loss is the one genuine advantage here, so sizing works
    backwards from it rather than from a stop that may never fill.

    The stop is expressed as a percentage of premium (default: exit at -50%),
    because an option's price is far more volatile than its underlying and a
    price-based stop on the *underlying* does not map cleanly onto the contract.
    """
    caps: list[str] = []
    premium = contract.mid
    if premium <= 0 or equity <= 0:
        return OptionSizing(0, 0, 0, 0, 0, 0, 0, 0, ["no usable quote"])

    cost_per_contract = premium * CONTRACT_MULTIPLIER
    risk_per_contract = cost_per_contract * stop_pct_of_premium

    # Fixed-fractional on the amount actually at risk.
    risk_budget = equity * risk_cfg.risk_per_trade_pct
    contracts = int(risk_budget / risk_per_contract) if risk_per_contract > 0 else 0

    # Gann's 10%: premium outlay is capital committed, so it is capped too.
    max_by_position = int((equity * risk_cfg.max_position_pct) / cost_per_contract)
    if contracts > max_by_position:
        contracts = max_by_position
        caps.append("gann_10pct_position_cap")

    # Never let total premium at risk exceed the daily loss limit in one trade.
    max_by_daily = int(
        (equity * risk_cfg.max_daily_loss_pct) / max(risk_per_contract, 1e-9)
    )
    if contracts > max_by_daily:
        contracts = max_by_daily
        caps.append("daily_loss_limit_cap")

    if contracts < 1:
        caps.append("premium_too_large_for_risk_budget")
        contracts = 0

    total_premium = contracts * cost_per_contract
    max_loss = total_premium  # a long option cannot lose more than the premium
    stop_premium = premium * (1.0 - stop_pct_of_premium)
    target_premium = premium * (1.0 + target_pct_of_premium)

    return OptionSizing(
        contracts=contracts,
        premium_per_contract=round(premium, 4),
        total_premium=round(total_premium, 2),
        max_loss=round(max_loss, 2),
        risk_pct_of_equity=round(
            contracts * risk_per_contract / equity, 5) if equity else 0.0,
        target_premium=round(target_premium, 4),
        stop_premium=round(stop_premium, 4),
        reward_risk=round(target_pct_of_premium / stop_pct_of_premium, 3),
        caps_applied=caps,
    )


# =====================================================================
#  Contract selection
# =====================================================================
def select_contract(
    chain: list[OptionContract],
    direction: str,
    *,
    min_dte: int = 7,
    max_dte: int = 45,
    target_delta: float = 0.45,
    min_open_interest: int = 250,
    min_volume: int = 25,
    max_spread_pct: float = 0.10,
    asof: date | None = None,
) -> tuple[OptionContract | None, list[str]]:
    """Pick the contract to trade from a chain, and say why the rest were cut.

    Defaults encode three lessons:

    * **7-45 days to expiry.** Nearer contracts decay fastest ("prices of
      current-month options decay at faster rates"), and 0-2 DTE turns a
      directional trade into a coin flip with a 100% downside.
    * **~0.45 delta.** Near the money, so the contract actually tracks the
      underlying. Cheap far-OTM contracts look like leverage and behave like
      lottery tickets.
    * **Liquidity floors.** A contract you cannot exit at a fair price is a
      position you do not control.
    """
    kind = "call" if direction in ("long", "bullish", "call") else "put"
    rejected: list[str] = []
    candidates: list[tuple[float, OptionContract]] = []

    for c in chain:
        if c.kind != kind:
            continue
        dte = c.days_to_expiry(asof)
        if not (min_dte <= dte <= max_dte):
            rejected.append(f"{c.describe()}: {dte} DTE outside {min_dte}-{max_dte}")
            continue
        if c.open_interest < min_open_interest:
            rejected.append(f"{c.describe()}: OI {c.open_interest} < {min_open_interest}")
            continue
        if c.volume < min_volume:
            rejected.append(f"{c.describe()}: volume {c.volume} < {min_volume}")
            continue
        if c.spread_pct > max_spread_pct:
            rejected.append(
                f"{c.describe()}: spread {c.spread_pct:.1%} > {max_spread_pct:.0%}")
            continue
        if c.mid <= 0:
            rejected.append(f"{c.describe()}: no usable quote")
            continue

        delta = abs(c.greeks().delta)
        candidates.append((abs(delta - target_delta), c))

    if not candidates:
        return None, rejected
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1], rejected


# =====================================================================
#  Fidelity specifics
# =====================================================================
APPROVAL_LEVELS = {
    1: "covered calls, cash-secured puts",
    2: "level 1 + long calls and long puts (what this system uses)",
    3: "level 2 + spreads (debit and credit)",
    4: "level 3 + uncovered/naked writing — unlimited risk",
}


def fidelity_option_order(
    contract: OptionContract,
    contracts: int,
    action: str,
    limit_price: float,
) -> dict:
    """Fields as Fidelity's options ticket asks for them."""
    return {
        "symbol": contract.underlying.upper(),
        "occ_symbol": contract.occ_symbol,
        "description": contract.describe(),
        "action": action,          # buy_to_open | sell_to_close | ...
        "quantity": contracts,
        "order_type": "limit",
        "limit_price": round(limit_price, 2),
        "tif": "day",
        "approval_level_required": 2,
        "notional": round(contracts * limit_price * CONTRACT_MULTIPLIER, 2),
    }
