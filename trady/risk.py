"""Money management and hard risk guards.

Two separable concerns:

* **Sizing** (`position_size`) — how many shares, from the book formulas:
  fixed-fractional, Gann's 10% cap, Kelly / half-Kelly, fixed-ratio, Optimal F.
* **Permission** (`RiskManager.check`) — whether the trade may happen at all:
  FINRA pattern-day-trader budget, daily loss limit, drawdown halt, position
  count, exposure, consecutive-loss cool-off, session clock.

The permission layer is deliberately blunt: it returns a hard allow/deny with a
reason string. Nothing downstream may override a denial — the point of a rule
you can talk yourself out of is that you will.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

import numpy as np

from .config import Config, PDTConfig, RiskConfig


# =====================================================================
#  Sizing formulas
# =====================================================================
def expectancy(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Per-trade expected return.

    % losing x loss + % winning x gain. `avg_loss` is passed as a positive
    magnitude. (Day Trading For Dummies, ch. 6, "Finding your expected return")
    """
    return win_rate * avg_win - (1.0 - win_rate) * abs(avg_loss)


def probability_of_ruin(win_rate: float, units: int) -> float:
    """R = ((1 - A) / (1 + A)) ** c, with A = advantage, c = units of capital.

    Advantage A = win% - loss%. `units` is how many equal parts the account is
    divided into. (Day Trading For Dummies, ch. 6, "Determining your probability
    of ruin")
    """
    advantage = win_rate - (1.0 - win_rate)
    if advantage <= 0:
        return 1.0
    if advantage >= 1:
        return 0.0
    return float(((1.0 - advantage) / (1.0 + advantage)) ** max(1, units))


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Kelly% = W - (1 - W) / R, with R = avg win / avg loss.

    (Day Trading For Dummies, ch. 6, "Finding the ideal percentage")
    Returns 0 when the edge is negative — do not size into a losing system.
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    r = avg_win / abs(avg_loss)
    k = win_rate - (1.0 - win_rate) / r
    return float(max(0.0, k))


def fixed_fractional_shares(
    equity: float, fraction: float, risk_per_share: float
) -> int:
    """N = (f x equity) / |trade risk|.

    (Day Trading For Dummies, ch. 6, "Limiting portions: Fixed fractional")
    """
    if risk_per_share <= 0:
        return 0
    return int((fraction * equity) / abs(risk_per_share))


def fixed_ratio_units(accumulated_profit: float, delta: float) -> float:
    """N = 0.5 x (sqrt(8P/delta + 1) + 1).

    (Day Trading For Dummies, ch. 6, "Protecting profits: Fixed ratio")
    """
    if delta <= 0:
        return 1.0
    p = max(0.0, accumulated_profit)
    return float(0.5 * (math.sqrt(8.0 * p / delta + 1.0) + 1.0))


def optimal_f_shares(equity: float, f: float, worst_loss_pct: float, price: float) -> int:
    """N = (equity x F) / worst_loss_pct / price.

    (Day Trading For Dummies, ch. 6, "Considering past performance: Optimal F")
    """
    if worst_loss_pct <= 0 or price <= 0:
        return 0
    return int((equity * f) / worst_loss_pct / price)


def monte_carlo_ruin(
    win_rate: float,
    avg_win_r: float,
    avg_loss_r: float,
    risk_pct: float,
    trades: int = 250,
    sims: int = 2000,
    ruin_threshold: float = 0.6,
    seed: int = 7,
) -> dict:
    """Simulate equity paths to estimate ruin odds and drawdown at a risk level.

    Returns median/percentile ending multiples and P(equity < threshold).
    """
    rng = np.random.default_rng(seed)
    wins = rng.random((sims, trades)) < win_rate
    r = np.where(wins, avg_win_r, -abs(avg_loss_r))
    paths = np.cumprod(1.0 + r * risk_pct, axis=1)
    running_max = np.maximum.accumulate(paths, axis=1)
    max_dd = float(np.median((1.0 - paths / running_max).max(axis=1)))
    ending = paths[:, -1]
    return {
        "median_ending_multiple": float(np.median(ending)),
        "p05_ending_multiple": float(np.percentile(ending, 5)),
        "p95_ending_multiple": float(np.percentile(ending, 95)),
        "prob_ruin": float((paths.min(axis=1) < ruin_threshold).mean()),
        "median_max_drawdown": max_dd,
        "risk_pct": risk_pct,
    }


@dataclass
class SizingResult:
    shares: int
    notional: float
    risk_dollars: float
    risk_pct_of_equity: float
    method: str
    stop_price: float
    target_price: float
    reward_risk: float
    caps_applied: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.shares > 0


def position_size(
    equity: float,
    entry: float,
    stop: float,
    target: float,
    cfg: RiskConfig,
    stats: dict | None = None,
    method: str = "auto",
) -> SizingResult:
    """Decide share count under every applicable cap.

    `auto` uses half-Kelly once there is a measured edge, otherwise
    fixed-fractional. Every cap that binds is recorded in `caps_applied` so the
    journal can explain why a trade was the size it was.
    """
    caps: list[str] = []
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0 or entry <= 0 or equity <= 0:
        return SizingResult(0, 0, 0, 0, method, stop, target, 0.0, ["invalid_inputs"])

    # Never risk more than max_stop_pct of the entry price on one share.
    if risk_per_share / entry > cfg.max_stop_pct:
        risk_per_share = entry * cfg.max_stop_pct
        stop = entry - risk_per_share if target > entry else entry + risk_per_share
        caps.append("stop_clamped_to_max_stop_pct")

    reward_risk = abs(target - entry) / risk_per_share

    fraction = cfg.risk_per_trade_pct
    used = "fixed_fractional"
    if method in ("auto", "kelly") and stats:
        wr = stats.get("win_rate", 0.0)
        aw = stats.get("avg_win", 0.0)
        al = stats.get("avg_loss", 0.0)
        n = stats.get("trades", 0)
        if n >= 20 and wr > 0 and aw > 0 and al > 0:
            k = kelly_fraction(wr, aw, al) * cfg.kelly_fraction
            if k > 0:
                fraction = min(k, cfg.kelly_cap_pct)
                used = f"half_kelly({cfg.kelly_fraction})"
                if k > cfg.kelly_cap_pct:
                    caps.append("kelly_capped_at_gann_10pct")
            else:
                caps.append("kelly_nonpositive_fell_back_to_fixed_fractional")

    shares = fixed_fractional_shares(equity, fraction, risk_per_share)

    # Gann's rule: no single trade exceeds 10% of the account.
    max_by_position = int((equity * cfg.max_position_pct) / entry)
    if shares > max_by_position:
        shares = max_by_position
        caps.append("gann_10pct_position_cap")

    if shares * entry < cfg.min_position_value:
        shares = 0
        caps.append("below_min_position_value")

    return SizingResult(
        shares=max(0, shares),
        notional=round(max(0, shares) * entry, 2),
        risk_dollars=round(max(0, shares) * risk_per_share, 2),
        risk_pct_of_equity=round(max(0, shares) * risk_per_share / equity, 5),
        method=used,
        stop_price=round(stop, 4),
        target_price=round(target, 4),
        reward_risk=round(reward_risk, 3),
        caps_applied=caps,
    )


def stop_and_target(
    entry: float,
    atr_value: float,
    direction: str,
    cfg: RiskConfig,
    noise_floor: float = 0.0,
) -> tuple[float, float]:
    """ATR-based stop and target, floored so noise cannot take the stop out.

    `noise_floor` is the recent median true range. A stop nearer than that is
    inside one bar's ordinary excursion and will be hit on the entry bar itself,
    regardless of whether the idea was right. The target keeps its distance in
    the same units so the reward:risk ratio stays honest after the floor binds.
    """
    if atr_value <= 0:
        atr_value = entry * 0.01

    stop_dist = cfg.stop_atr_multiple * atr_value
    floor = cfg.stop_noise_floor_mult * max(noise_floor, 0.0)
    if floor > stop_dist:
        stop_dist = floor

    # Scale the target off the same distance, preserving the intended ratio.
    ratio = cfg.target_atr_multiple / max(cfg.stop_atr_multiple, 1e-9)
    target_dist = stop_dist * ratio

    if direction == "long":
        return entry - stop_dist, entry + target_dist
    return entry + stop_dist, entry - target_dist


def trail_stop(
    current_stop: float, price: float, atr_value: float, direction: str, cfg: RiskConfig
) -> float:
    """Ratchet a stop in the favourable direction only."""
    if not cfg.use_trailing_stop:
        return current_stop
    if direction == "long":
        return max(current_stop, price - cfg.trail_atr_multiple * atr_value)
    return min(current_stop, price + cfg.trail_atr_multiple * atr_value)


# =====================================================================
#  Pattern-day-trader budget
# =====================================================================
@dataclass
class DayTrade:
    """A round trip opened and closed on the same session."""

    symbol: str
    trade_date: date


class PDTTracker:
    """Enforces FINRA NASD Rule 2520.

    A day trade is buying and selling the same security on the same day in a
    margin account. Four or more inside five rolling business days makes the
    account a pattern day trader, which then requires >= $25,000 equity at the
    start of each trading day.

    Below the threshold this tracker is the thing standing between the account
    and a 90-day restriction, so `remaining` is deliberately pessimistic: it
    can hold one day trade back in reserve for an emergency exit.
    """

    def __init__(self, cfg: PDTConfig):
        self.cfg = cfg
        self._trades: list[DayTrade] = []

    # -- state --------------------------------------------------------
    def record(self, symbol: str, when: date | datetime) -> None:
        d = when.date() if isinstance(when, datetime) else when
        self._trades.append(DayTrade(symbol, d))

    def load(self, trades: list[tuple[str, date]]) -> None:
        self._trades = [DayTrade(s, d) for s, d in trades]

    @staticmethod
    def _business_days_back(anchor: date, n: int) -> date:
        """Start of an n-business-day rolling window ending at `anchor`."""
        d, counted = anchor, 1
        while counted < n:
            d -= timedelta(days=1)
            if d.weekday() < 5:
                counted += 1
        return d

    def window_trades(self, anchor: date | None = None) -> list[DayTrade]:
        anchor = anchor or date.today()
        start = self._business_days_back(anchor, self.cfg.rolling_business_days)
        return [t for t in self._trades if start <= t.trade_date <= anchor]

    def count(self, anchor: date | None = None) -> int:
        return len(self.window_trades(anchor))

    # -- policy -------------------------------------------------------
    def is_restricted(self, equity: float) -> bool:
        """True when the $25k rule actually binds this account."""
        return self.cfg.enabled and equity < self.cfg.equity_threshold

    def remaining(self, equity: float, anchor: date | None = None) -> int:
        """Day trades still available today. Unlimited (-1) at/above $25k."""
        if not self.is_restricted(equity):
            return -1
        budget = self.cfg.max_day_trades
        if self.cfg.reserve_last_day_trade:
            budget -= 1  # keep one for an emergency exit
        return max(0, budget - self.count(anchor))

    def can_day_trade(self, equity: float, anchor: date | None = None) -> tuple[bool, str]:
        if not self.is_restricted(equity):
            return True, "equity at or above $25,000 — PDT limit does not bind"
        left = self.remaining(equity, anchor)
        used = self.count(anchor)
        if left > 0:
            return True, (
                f"{used}/{self.cfg.max_day_trades} day trades used in the rolling "
                f"{self.cfg.rolling_business_days}-business-day window; {left} usable"
            )
        return False, (
            f"PDT budget exhausted: {used}/{self.cfg.max_day_trades} day trades used in "
            f"the rolling {self.cfg.rolling_business_days}-business-day window and equity "
            f"${equity:,.0f} is below the ${self.cfg.equity_threshold:,.0f} threshold. "
            "Opening a position that must close today risks a 90-day restriction."
        )


# =====================================================================
#  Permission layer
# =====================================================================
@dataclass
class RiskDecision:
    allowed: bool
    reason: str
    checks: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.allowed


@dataclass
class AccountState:
    equity: float
    starting_equity_today: float
    peak_equity: float
    open_positions: int = 0
    gross_exposure: float = 0.0
    realized_pnl_today: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    halted: bool = False
    halt_reason: str = ""
    open_symbols: list[str] = field(default_factory=list)


class RiskManager:
    """Single gate every order passes through."""

    def __init__(self, cfg: Config, pdt: PDTTracker | None = None):
        self.cfg = cfg
        self.risk = cfg.risk
        self.pdt = pdt or PDTTracker(cfg.pdt)

    # -- session clock ------------------------------------------------
    def _in_entry_window(self, now: datetime) -> tuple[bool, str]:
        s = self.cfg.session
        t = now.time()

        def _p(hhmm: str) -> time:
            h, m = hhmm.split(":")
            return time(int(h), int(m))

        if t < _p(s.market_open) or t >= _p(s.market_close):
            return False, "market is closed"
        if t < _p(s.no_entry_before):
            return False, (
                f"inside the opening window before {s.no_entry_before} — "
                "avoiding the opening gap trap"
            )
        if t >= _p(s.no_entry_after):
            return False, (
                f"after {s.no_entry_after} — too late to open; day trades must be "
                "closed before the bell"
            )
        if s.avoid_lunch_chop and _p(s.lunch_start) <= t < _p(s.lunch_end):
            return False, "midday chop window — liquidity thins and ranges compress"
        return True, "within entry window"

    # -- main gate ----------------------------------------------------
    def check(
        self,
        state: AccountState,
        *,
        now: datetime | None = None,
        intends_same_day_exit: bool = True,
        new_notional: float = 0.0,
        symbol: str = "",
    ) -> RiskDecision:
        now = now or datetime.now()
        checks: dict = {}

        if state.halted:
            return RiskDecision(False, f"trading halted: {state.halt_reason}", checks)

        # Session window.
        ok, why = self._in_entry_window(now)
        checks["session_window"] = why
        if not ok:
            return RiskDecision(False, why, checks)

        # Drawdown halt (peak to trough).
        dd = 0.0 if state.peak_equity <= 0 else 1.0 - state.equity / state.peak_equity
        checks["drawdown"] = round(dd, 4)
        if dd >= self.risk.max_drawdown_halt_pct:
            return RiskDecision(
                False,
                f"drawdown {dd:.1%} has reached the {self.risk.max_drawdown_halt_pct:.0%} "
                "halt threshold — stop and review before trading again",
                checks,
            )

        # Daily loss limit.
        day_pnl_pct = (
            state.realized_pnl_today / state.starting_equity_today
            if state.starting_equity_today > 0 else 0.0
        )
        checks["daily_pnl_pct"] = round(day_pnl_pct, 4)
        if day_pnl_pct <= -self.risk.max_daily_loss_pct:
            return RiskDecision(
                False,
                f"daily loss limit hit ({day_pnl_pct:.2%} vs limit "
                f"-{self.risk.max_daily_loss_pct:.0%}) — done for the day",
                checks,
            )

        # Overtrading.
        checks["trades_today"] = state.trades_today
        if state.trades_today >= self.risk.max_daily_trades:
            return RiskDecision(
                False,
                f"daily trade cap reached ({state.trades_today}/"
                f"{self.risk.max_daily_trades}) — overtrading erodes returns via costs",
                checks,
            )

        # Consecutive losses cool-off.
        checks["consecutive_losses"] = state.consecutive_losses
        if state.consecutive_losses >= self.risk.max_consecutive_losses:
            return RiskDecision(
                False,
                f"{state.consecutive_losses} losses in a row — cooling off; "
                "re-evaluate the setup before the next entry",
                checks,
            )

        # Concurrent positions.
        checks["open_positions"] = state.open_positions
        if state.open_positions >= self.risk.max_open_positions:
            return RiskDecision(
                False,
                f"already holding {state.open_positions}/"
                f"{self.risk.max_open_positions} positions",
                checks,
            )

        # Sector concentration. Three tech longs are one tech position at 3x
        # size — they lose together on the same headline.
        sector = self.cfg.sectors.get(symbol.upper()) if symbol else None
        if sector:
            held = sum(
                1 for s in state.open_symbols
                if self.cfg.sectors.get(s.upper()) == sector
            )
            checks["sector"] = f"{sector}: {held} open"
            if held >= self.risk.max_sector_concentration:
                return RiskDecision(
                    False,
                    f"already holding {held} {sector} position(s), at the "
                    f"{self.risk.max_sector_concentration} limit — correlated "
                    "positions concentrate risk rather than spread it",
                    checks,
                )

        # Gross exposure.
        exposure = (state.gross_exposure + new_notional) / max(state.equity, 1e-9)
        checks["gross_exposure"] = round(exposure, 3)
        if exposure > self.risk.max_gross_exposure_pct:
            return RiskDecision(
                False,
                f"gross exposure {exposure:.0%} would exceed the "
                f"{self.risk.max_gross_exposure_pct:.0%} cap",
                checks,
            )

        # PDT budget — only relevant if this position is meant to close today.
        if intends_same_day_exit:
            can, why_pdt = self.pdt.can_day_trade(state.equity, now.date())
            checks["pdt"] = why_pdt
            checks["pdt_remaining"] = self.pdt.remaining(state.equity, now.date())
            if not can and self.cfg.pdt.block_when_exhausted:
                return RiskDecision(False, why_pdt, checks)

        return RiskDecision(True, "all risk checks passed", checks)

    # -- post-trade ---------------------------------------------------
    def should_halt(self, state: AccountState) -> tuple[bool, str]:
        dd = 0.0 if state.peak_equity <= 0 else 1.0 - state.equity / state.peak_equity
        if dd >= self.risk.max_drawdown_halt_pct:
            return True, f"peak-to-trough drawdown {dd:.1%}"
        day_pnl_pct = (
            state.realized_pnl_today / state.starting_equity_today
            if state.starting_equity_today > 0 else 0.0
        )
        if day_pnl_pct <= -self.risk.max_daily_loss_pct:
            return True, f"daily loss {day_pnl_pct:.2%}"
        return False, ""
