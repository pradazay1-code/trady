"""Central configuration — every hard rule the agent obeys lives here.

Values are sourced from the five reference books (see knowledge/rulebook.yaml for
the citations). Nothing in the trading path invents its own limits: it reads them
from a Config instance so that one file governs the agent's whole risk posture.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RiskConfig:
    """Money management. Defaults are deliberately conservative."""

    # --- Account ---------------------------------------------------------
    starting_equity: float = 25_000.0
    account_type: str = "margin"  # "margin" | "cash"

    # --- Per-trade sizing ------------------------------------------------
    # Gann's rule: never more than 10% of the account in one trade.
    max_position_pct: float = 0.10
    # Fixed-fractional: fraction of equity risked (not deployed) per trade.
    risk_per_trade_pct: float = 0.0075  # 0.75% of equity at risk
    # Kelly is scaled down; the books warn raw Kelly is too aggressive.
    kelly_fraction: float = 0.5  # "half-Kelly"
    kelly_cap_pct: float = 0.10  # never exceed Gann's 10% regardless of Kelly
    min_position_value: float = 100.0

    # --- Stops and targets ----------------------------------------------
    stop_atr_multiple: float = 1.5
    target_atr_multiple: float = 2.5
    min_reward_risk: float = 1.5  # reject setups below this R:R
    max_stop_pct: float = 0.05  # never risk >5% adverse move per share
    use_trailing_stop: bool = True
    trail_atr_multiple: float = 2.0
    breakeven_at_r: float = 1.0  # move stop to breakeven after +1R

    # --- Daily circuit breakers -----------------------------------------
    max_daily_loss_pct: float = 0.02  # stop trading for the day at -2%
    max_daily_trades: int = 8  # overtrading is a named book mistake
    max_open_positions: int = 4
    max_consecutive_losses: int = 3  # cool-off trigger
    max_sector_concentration: int = 2

    # --- Portfolio guards ------------------------------------------------
    max_gross_exposure_pct: float = 1.0  # 1.0 = no leverage by default
    max_drawdown_halt_pct: float = 0.10  # halt all trading at -10% peak-to-trough

    # --- Profit handling -------------------------------------------------
    withdraw_profit_pct: float = 0.10  # sweep 10% of quarterly profit out


@dataclass
class PDTConfig:
    """FINRA pattern-day-trader rule (NASD 2520) enforcement.

    Day trade = buying and selling the same security on the same day in a
    margin account. 4+ such trades in 5 rolling business days makes you a
    pattern day trader, which requires >= $25,000 equity at the start of the
    trading day. Under $25k this guard is what keeps the account legal.
    """

    enabled: bool = True
    equity_threshold: float = 25_000.0
    max_day_trades: int = 3  # in the rolling window, when under threshold
    rolling_business_days: int = 5
    # If equity is under the threshold, refuse to open a position that would
    # have to be closed same-day once the budget is spent.
    block_when_exhausted: bool = True
    reserve_last_day_trade: bool = True  # keep 1 in reserve for emergency exits


@dataclass
class SignalConfig:
    """How raw pattern/indicator evidence becomes a trade decision."""

    min_confluence_score: float = 3.0  # weighted evidence needed to act
    require_trend_alignment: bool = True
    require_volume_confirmation: bool = True
    volume_surge_ratio: float = 1.3  # vs. 20-bar average
    confirmation_bars: int = 1  # wait N bars for pattern confirmation

    # Evidence weights. Tuned by learn.py from realised outcomes.
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "candlestick": 1.0,
            "trend": 1.0,
            "support_resistance": 1.2,
            "volume": 0.8,
            "momentum": 0.9,
            "gap": 0.7,
            "chart_pattern": 1.0,
            "market_regime": 0.6,
        }
    )

    # Liquidity filters — the books warn against thin issues.
    min_price: float = 5.0
    max_price: float = 1_000.0
    min_avg_volume: int = 1_000_000
    max_spread_pct: float = 0.005


@dataclass
class SessionConfig:
    """Trading-day clock, US/Eastern."""

    market_open: str = "09:30"
    market_close: str = "16:00"
    # "Gap and crap": many traders avoid the first 30 minutes.
    no_entry_before: str = "10:00"
    # Day traders close out; leave room to exit cleanly.
    no_entry_after: str = "15:30"
    force_flat_at: str = "15:55"
    avoid_lunch_chop: bool = True
    lunch_start: str = "11:45"
    lunch_end: str = "13:15"


@dataclass
class ExecutionConfig:
    broker: str = "paper"  # paper | fidelity | alpaca
    commission_per_trade: float = 0.0  # Fidelity: $0 online equity commissions
    sec_fee_rate: float = 0.0000278  # sell-side, per dollar
    taf_fee_per_share: float = 0.000166  # FINRA TAF, sell-side
    taf_cap: float = 8.30
    slippage_bps: float = 3.0  # modelled adverse fill, basis points
    default_order_type: str = "limit"
    limit_offset_bps: float = 5.0
    require_confirmation: bool = True  # human ack before any live order


@dataclass
class LearnConfig:
    """Self-correction loop."""

    enabled: bool = True
    min_trades_before_adapt: int = 20
    lookback_trades: int = 100
    weight_step: float = 0.05  # per-review nudge
    weight_floor: float = 0.2
    weight_ceiling: float = 2.5
    # Disable a strategy whose expectancy stays negative over this many trades.
    strategy_probation_trades: int = 25
    strategy_min_expectancy: float = 0.0
    walk_forward_splits: int = 4


@dataclass
class Config:
    risk: RiskConfig = field(default_factory=RiskConfig)
    pdt: PDTConfig = field(default_factory=PDTConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    learn: LearnConfig = field(default_factory=LearnConfig)

    watchlist: list[str] = field(
        default_factory=lambda: [
            "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD",
            "TSLA", "AMZN", "META", "GOOGL",
        ]
    )
    data_provider: str = "yfinance"  # yfinance | stooq | csv | synthetic
    bar_interval: str = "5m"
    history_days: int = 60
    timezone: str = "America/New_York"

    data_dir: Path = field(default=REPO_ROOT / "data")
    journal_db: Path = field(default=REPO_ROOT / "data" / "journal.sqlite")
    kb_db: Path = field(default=REPO_ROOT / "knowledge" / "kb.sqlite")
    reports_dir: Path = field(default=REPO_ROOT / "reports")

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load config.json if present, else defaults. Unknown keys are ignored."""
        cfg = cls()
        path = Path(path) if path else REPO_ROOT / "config.json"
        if not path.exists():
            return cfg
        raw = json.loads(path.read_text())
        sections = {f.name: f for f in fields(cls)}
        for key, val in raw.items():
            if key not in sections:
                continue
            current = getattr(cfg, key)
            if hasattr(current, "__dataclass_fields__") and isinstance(val, dict):
                for k, v in val.items():
                    if hasattr(current, k):
                        setattr(current, k, v)
            elif key.endswith("_dir") or key.endswith("_db"):
                setattr(cfg, key, Path(val))
            else:
                setattr(cfg, key, val)
        return cfg

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else REPO_ROOT / "config.json"
        data = asdict(self)
        for k, v in list(data.items()):
            if isinstance(v, Path):
                data[k] = str(v)
        path.write_text(json.dumps(data, indent=2, default=str))
        return path

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.reports_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
