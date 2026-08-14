"""Phone notifications: trade alerts pushed to your handset.

Providers, in the order most people should try them:

  ntfy      free, no account. Install the ntfy app, pick a topic name, done.
            Topics are public to anyone who guesses the name, so use something
            unguessable — the alert contains your position sizes.
  pushover  $5 one-time, most reliable delivery, proper priority levels.
  telegram  free, needs a bot token from @BotFather.
  email     any SMTP account; also reaches SMS via carrier gateways.
  console   prints instead of sending. The default, and what `--dry-run` forces.

Every alert carries the whole trade: symbol, side, quantity, entry, stop,
target, reward:risk, and the reasoning. A notification that says only "BUY AAPL"
is worse than useless — it invites you to enter a position with no exit plan.

Nothing here decides anything. It formats and delivers what the strategy already
decided, and it refuses to deliver live alerts from an unvalidated strategy (see
`AlertGate`).
"""

from __future__ import annotations

import json
import os
import smtplib
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from enum import IntEnum
from pathlib import Path


class Priority(IntEnum):
    LOW = -1        # informational: end-of-day summary
    NORMAL = 0      # a new signal
    HIGH = 1        # act now: entry window closing, stop approaching
    URGENT = 2      # bypass quiet hours: stop hit, halt triggered


# =====================================================================
#  Alert payload
# =====================================================================
@dataclass
class Alert:
    """One notification. Formatting lives here so every channel agrees."""

    title: str
    body: str
    priority: Priority = Priority.NORMAL
    tags: list[str] = field(default_factory=list)
    kind: str = "info"          # entry | exit | risk | summary | info
    symbol: str = ""

    def as_text(self) -> str:
        return f"{self.title}\n{self.body}"


def entry_alert(sig, equity: float, pdt_remaining: int, live: bool) -> Alert:
    """Full entry instruction — everything needed to place the order."""
    side = "BUY" if sig.direction == "long" else "SELL SHORT"
    qty = sig.sizing.shares if sig.sizing else 0
    risk = sig.sizing.risk_dollars if sig.sizing else 0.0

    lines = [
        f"{side} {qty} {sig.symbol} @ {sig.entry:.2f}",
        "",
        f"Stop      {sig.stop:.2f}   (risk ${risk:,.0f})",
        f"Target    {sig.target:.2f}   (R:R {sig.reward_risk:.2f})",
        f"Notional  ${sig.sizing.notional:,.0f}" if sig.sizing else "",
        "",
        f"Setup: {sig.strategy}  ·  confluence {sig.score:.2f}",
        "Why:",
    ]
    for e in sorted(sig.evidence, key=lambda x: abs(x.contribution), reverse=True)[:4]:
        lines.append(f"  • {e.detail}")

    if pdt_remaining >= 0:
        lines += ["", f"PDT: {pdt_remaining} day trade(s) left this window"]
    if not live:
        lines += ["", "PAPER MODE — do not place this order"]

    return Alert(
        title=f"{'📈' if sig.direction == 'long' else '📉'} {side} {sig.symbol}",
        body="\n".join(l for l in lines if l != "" or True),
        priority=Priority.HIGH,
        tags=["chart_with_upwards_trend" if sig.direction == "long" else "chart_with_downwards_trend"],
        kind="entry",
        symbol=sig.symbol,
    )


def exit_alert(symbol: str, direction: str, qty: int, price: float,
               reason: str, net_pnl: float, r_multiple: float | None) -> Alert:
    side = "SELL" if direction == "long" else "BUY TO COVER"
    won = net_pnl >= 0
    r = f"{r_multiple:+.2f}R" if r_multiple is not None else "—"
    return Alert(
        title=f"{'✅' if won else '🛑'} {side} {qty} {symbol} @ {price:.2f}",
        body="\n".join([
            f"Closed: {reason}",
            f"P&L: ${net_pnl:+,.2f}  ({r})",
        ]),
        priority=Priority.HIGH if reason == "stop" else Priority.NORMAL,
        tags=["white_check_mark" if won else "octagonal_sign"],
        kind="exit",
        symbol=symbol,
    )


def risk_alert(reason: str, detail: str = "") -> Alert:
    return Alert(
        title=f"⚠️ Trading halted",
        body=f"{reason}\n{detail}".strip(),
        priority=Priority.URGENT,
        tags=["warning"],
        kind="risk",
    )


def summary_alert(stats: dict, equity: float, day_pnl: float) -> Alert:
    return Alert(
        title=f"📊 Session close: ${day_pnl:+,.2f}",
        body="\n".join([
            f"Equity    ${equity:,.2f}",
            f"Trades    {stats.get('trades', 0)}",
            f"Win rate  {stats.get('win_rate', 0):.0%}",
            f"Avg R     {stats.get('avg_r', 0):+.2f}",
        ]),
        priority=Priority.LOW,
        tags=["bar_chart"],
        kind="summary",
    )


# =====================================================================
#  Channels
# =====================================================================
class Channel(ABC):
    name = "abstract"

    @abstractmethod
    def send(self, alert: Alert) -> bool: ...


class ConsoleChannel(Channel):
    """Prints. The default, so a misconfiguration never silently drops alerts."""

    name = "console"

    def send(self, alert: Alert) -> bool:
        bar = "─" * 52
        print(f"\n{bar}\n{alert.title}\n{bar}\n{alert.body}\n{bar}")
        return True


class NtfyChannel(Channel):
    """https://ntfy.sh — free, no account. Set TRADY_NTFY_TOPIC."""

    name = "ntfy"

    def __init__(self, topic: str | None = None, server: str = "https://ntfy.sh"):
        self.topic = topic or os.environ.get("TRADY_NTFY_TOPIC", "")
        self.server = server.rstrip("/")
        if not self.topic:
            raise RuntimeError(
                "ntfy needs a topic: set TRADY_NTFY_TOPIC to something unguessable "
                "(alerts contain your position sizes and anyone who knows the "
                "topic name can read them)"
            )

    def send(self, alert: Alert) -> bool:
        req = urllib.request.Request(
            f"{self.server}/{self.topic}",
            data=alert.body.encode("utf-8"),
            headers={
                "Title": alert.title.encode("ascii", "ignore").decode(),
                "Priority": str(int(alert.priority) + 3),  # ntfy uses 1..5
                "Tags": ",".join(alert.tags) or "chart",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300


class PushoverChannel(Channel):
    """https://pushover.net — set TRADY_PUSHOVER_TOKEN and TRADY_PUSHOVER_USER."""

    name = "pushover"

    def __init__(self, token: str | None = None, user: str | None = None):
        self.token = token or os.environ.get("TRADY_PUSHOVER_TOKEN", "")
        self.user = user or os.environ.get("TRADY_PUSHOVER_USER", "")
        if not (self.token and self.user):
            raise RuntimeError(
                "pushover needs TRADY_PUSHOVER_TOKEN and TRADY_PUSHOVER_USER"
            )

    def send(self, alert: Alert) -> bool:
        payload = urllib.parse.urlencode({
            "token": self.token,
            "user": self.user,
            "title": alert.title,
            "message": alert.body,
            "priority": int(alert.priority),
        }).encode()
        req = urllib.request.Request(
            "https://api.pushover.net/1/messages.json", data=payload, method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300


class TelegramChannel(Channel):
    """Set TRADY_TELEGRAM_TOKEN (from @BotFather) and TRADY_TELEGRAM_CHAT."""

    name = "telegram"

    def __init__(self, token: str | None = None, chat_id: str | None = None):
        self.token = token or os.environ.get("TRADY_TELEGRAM_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TRADY_TELEGRAM_CHAT", "")
        if not (self.token and self.chat_id):
            raise RuntimeError(
                "telegram needs TRADY_TELEGRAM_TOKEN and TRADY_TELEGRAM_CHAT"
            )

    def send(self, alert: Alert) -> bool:
        payload = json.dumps({
            "chat_id": self.chat_id,
            "text": f"*{alert.title}*\n```\n{alert.body}\n```",
            "parse_mode": "Markdown",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{self.token}/sendMessage",
            data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300


class EmailChannel(Channel):
    """SMTP. Also reaches SMS via carrier gateways (e.g. 5551234567@vtext.com).

    Env: TRADY_SMTP_HOST, TRADY_SMTP_PORT, TRADY_SMTP_USER,
         TRADY_SMTP_PASS, TRADY_ALERT_TO
    """

    name = "email"

    def __init__(self, to: str | None = None):
        self.host = os.environ.get("TRADY_SMTP_HOST", "smtp.gmail.com")
        self.port = int(os.environ.get("TRADY_SMTP_PORT", "587"))
        self.user = os.environ.get("TRADY_SMTP_USER", "")
        self.password = os.environ.get("TRADY_SMTP_PASS", "")
        self.to = to or os.environ.get("TRADY_ALERT_TO", "")
        if not (self.user and self.password and self.to):
            raise RuntimeError(
                "email needs TRADY_SMTP_USER, TRADY_SMTP_PASS and TRADY_ALERT_TO"
            )

    def send(self, alert: Alert) -> bool:
        msg = EmailMessage()
        msg["Subject"] = alert.title
        msg["From"] = self.user
        msg["To"] = self.to
        msg.set_content(alert.body)
        with smtplib.SMTP(self.host, self.port, timeout=20) as s:
            s.starttls()
            s.login(self.user, self.password)
            s.send_message(msg)
        return True


CHANNELS = {
    "console": ConsoleChannel,
    "ntfy": NtfyChannel,
    "pushover": PushoverChannel,
    "telegram": TelegramChannel,
    "email": EmailChannel,
}


# =====================================================================
#  Validation gate
# =====================================================================
@dataclass
class GateVerdict:
    live_allowed: bool
    reasons: list[str]
    checks: dict


class AlertGate:
    """Decides whether alerts may be labelled as live-tradable.

    An alert that says "BUY 37 AAPL" reads as advice to commit money. It should
    only carry that weight once the strategy has demonstrated an edge on real
    data — otherwise the notification is just an untested program's opinion,
    delivered with the authority of a push notification.

    Until the checks pass, alerts still fire; they are simply stamped PAPER MODE
    so the difference between "tested" and "not tested" cannot be missed.
    """

    def __init__(self, cfg, journal):
        self.cfg = cfg
        self.journal = journal

    def evaluate(self) -> GateVerdict:
        """Assess validation. An explicit override is honoured but recorded.

        The account is the owner's and so is the risk decision. This gate exists
        to make sure that decision is informed rather than accidental — not to
        take it away. `execution.override_validation_gate` turns live labelling
        back on, and the verdict still reports every check that was failing when
        it was overridden.
        """
        verdict = self._assess()
        if getattr(self.cfg.execution, "override_validation_gate", False):
            return GateVerdict(
                True,
                [],
                {**verdict.checks,
                 "OVERRIDDEN": True,
                 "overridden_despite": verdict.reasons},
            )
        return verdict

    def _assess(self) -> GateVerdict:
        from .journal import compute_stats

        reasons: list[str] = []
        checks: dict = {}

        trades = self.journal.closed_trades()
        stats = compute_stats(trades, base_equity=self.cfg.risk.starting_equity)
        checks["trades"] = stats["trades"]
        checks["expectancy"] = stats["expectancy"]
        checks["win_rate"] = stats["win_rate"]

        min_trades = 50
        if stats["trades"] < min_trades:
            reasons.append(
                f"only {stats['trades']} closed trades on record — need at least "
                f"{min_trades} before treating the signal as validated"
            )
        if stats["trades"] >= min_trades and stats["expectancy"] <= 0:
            reasons.append(
                f"measured expectancy is {stats['expectancy']:.4%} per trade — "
                "the strategy loses money on its own record"
            )

        real = trades[~trades["broker"].fillna("").str.contains("synthetic|backtest",
                                                               case=False)] \
            if not trades.empty and "broker" in trades.columns else trades
        checks["real_data_trades"] = int(len(real))
        if len(real) < min_trades:
            reasons.append(
                f"only {len(real)} trades came from real market data — synthetic "
                "bars are a random walk and prove nothing about edge"
            )

        return GateVerdict(not reasons, reasons, checks)


# =====================================================================
#  Notifier
# =====================================================================
class Notifier:
    """Fans an alert out to configured channels, with de-duplication."""

    def __init__(
        self,
        channels: list[str] | None = None,
        dry_run: bool = False,
        quiet_kinds: tuple[str, ...] = (),
        log_path: Path | None = None,
    ):
        self.dry_run = dry_run
        self.quiet_kinds = quiet_kinds
        self.log_path = Path(log_path) if log_path else None
        self._sent: set[str] = set()
        self.channels: list[Channel] = []

        names = channels or ["console"]
        if dry_run:
            names = ["console"]
        for name in names:
            cls = CHANNELS.get(name)
            if cls is None:
                print(f"  notify: unknown channel {name!r}, skipping")
                continue
            try:
                self.channels.append(cls())
            except Exception as exc:
                print(f"  notify: {name} unavailable — {exc}")
        if not self.channels:
            self.channels = [ConsoleChannel()]

    # ------------------------------------------------------------------
    def send(self, alert: Alert, dedupe_key: str | None = None) -> dict:
        """Deliver to every channel. One failing channel never blocks the rest."""
        if alert.kind in self.quiet_kinds:
            return {"skipped": "quiet"}
        if dedupe_key:
            if dedupe_key in self._sent:
                return {"skipped": "duplicate"}
            self._sent.add(dedupe_key)

        results: dict[str, object] = {}
        for ch in self.channels:
            try:
                results[ch.name] = ch.send(alert)
            except urllib.error.URLError as exc:
                results[ch.name] = f"network error: {exc.reason}"
            except Exception as exc:  # a channel must never kill the session
                results[ch.name] = f"{type(exc).__name__}: {exc}"

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as fh:
                fh.write(json.dumps({
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "kind": alert.kind, "symbol": alert.symbol,
                    "title": alert.title, "results": {k: str(v) for k, v in results.items()},
                }) + "\n")
        return results

    def test(self) -> dict:
        return self.send(Alert(
            title="✅ Trady alerts working",
            body="If you can read this on your phone, notifications are configured.",
            priority=Priority.NORMAL, tags=["white_check_mark"], kind="info",
        ))
