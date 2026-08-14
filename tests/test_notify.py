"""Alerts and the validation gate.

The gate tests matter most: they are what stops an unvalidated strategy from
sending notifications that read like trading advice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from trady import data as datamod
from trady.config import Config
from trady.indicators import enrich
from trady.journal import Journal
from trady.notify import (
    Alert,
    AlertGate,
    ConsoleChannel,
    Notifier,
    Priority,
    entry_alert,
    exit_alert,
    risk_alert,
    summary_alert,
)
from trady.strategy import generate


@pytest.fixture
def journal(tmp_path) -> Journal:
    return Journal(tmp_path / "j.sqlite")


@pytest.fixture
def cfg(tmp_path) -> Config:
    c = Config()
    c.journal_db = tmp_path / "j.sqlite"
    c.reports_dir = tmp_path / "reports"
    c.data_dir = tmp_path
    return c


def _signal(cfg):
    df = enrich(datamod.synthetic("AAPL", bars=1200, seed=7))
    cfg.signal.min_confluence_score = 0.1
    return generate("AAPL", df, cfg, 25_000.0)


# =====================================================================
#  Alert formatting
# =====================================================================
class TestAlertContent:
    def test_entry_alert_carries_the_whole_trade(self, cfg):
        sig = _signal(cfg)
        if sig is None or sig.sizing is None:
            pytest.skip("no signal")
        a = entry_alert(sig, 25_000.0, pdt_remaining=2, live=True)
        # An alert without an exit plan invites an unbounded position.
        assert "Stop" in a.body
        assert "Target" in a.body
        assert str(sig.sizing.shares) in a.body
        assert "AAPL" in a.title

    def test_entry_alert_explains_itself(self, cfg):
        sig = _signal(cfg)
        if sig is None:
            pytest.skip("no signal")
        a = entry_alert(sig, 25_000.0, 2, live=True)
        assert "Why:" in a.body
        assert sig.strategy in a.body

    def test_paper_mode_is_stamped_when_not_live(self, cfg):
        sig = _signal(cfg)
        if sig is None:
            pytest.skip("no signal")
        paper = entry_alert(sig, 25_000.0, 2, live=False)
        live = entry_alert(sig, 25_000.0, 2, live=True)
        assert "PAPER MODE" in paper.body
        assert "PAPER MODE" not in live.body

    def test_pdt_budget_surfaces(self, cfg):
        sig = _signal(cfg)
        if sig is None:
            pytest.skip("no signal")
        restricted = entry_alert(sig, 10_000.0, pdt_remaining=1, live=True)
        assert "PDT" in restricted.body
        unrestricted = entry_alert(sig, 40_000.0, pdt_remaining=-1, live=True)
        assert "PDT" not in unrestricted.body

    def test_direction_shows_in_the_title(self, cfg):
        sig = _signal(cfg)
        if sig is None:
            pytest.skip("no signal")
        a = entry_alert(sig, 25_000.0, 2, live=True)
        assert ("BUY" in a.title) == (sig.direction == "long")

    def test_exit_alert_reports_pnl_and_r(self):
        a = exit_alert("AAPL", "long", 37, 216.10, "target", 64.75, 2.15)
        assert "+64.75" in a.body and "+2.15R" in a.body
        assert "SELL" in a.title

    def test_exit_alert_for_a_short_says_cover(self):
        a = exit_alert("AAPL", "short", 10, 96.0, "target", 40.0, 1.5)
        assert "COVER" in a.title

    def test_stop_exit_is_high_priority(self):
        stopped = exit_alert("A", "long", 1, 1.0, "stop", -10.0, -1.0)
        target = exit_alert("A", "long", 1, 1.0, "target", 10.0, 2.0)
        assert stopped.priority > target.priority

    def test_risk_alert_is_urgent(self):
        a = risk_alert("daily loss limit hit")
        assert a.priority == Priority.URGENT
        assert a.kind == "risk"

    def test_summary_alert_fields(self):
        a = summary_alert({"trades": 4, "win_rate": 0.5, "avg_r": 0.31}, 25_412.0, 412.0)
        assert "Equity" in a.body and "Win rate" in a.body


# =====================================================================
#  Delivery
# =====================================================================
class TestNotifier:
    def test_defaults_to_console(self):
        n = Notifier()
        assert [c.name for c in n.channels] == ["console"]

    def test_dry_run_forces_console(self):
        n = Notifier(["ntfy", "pushover"], dry_run=True)
        assert [c.name for c in n.channels] == ["console"]

    def test_unconfigured_channel_degrades_to_console(self, monkeypatch):
        for var in ("TRADY_NTFY_TOPIC", "TRADY_PUSHOVER_TOKEN", "TRADY_PUSHOVER_USER"):
            monkeypatch.delenv(var, raising=False)
        n = Notifier(["ntfy"])
        # Never silently drop alerts because a channel was misconfigured.
        assert [c.name for c in n.channels] == ["console"]

    def test_unknown_channel_is_skipped(self):
        n = Notifier(["not-a-channel"])
        assert [c.name for c in n.channels] == ["console"]

    def test_dedupe_suppresses_repeats(self, capsys):
        n = Notifier(["console"])
        a = Alert("t", "b")
        assert n.send(a, dedupe_key="k") != {"skipped": "duplicate"}
        assert n.send(a, dedupe_key="k") == {"skipped": "duplicate"}

    def test_quiet_kinds_are_suppressed(self):
        n = Notifier(["console"], quiet_kinds=("summary",))
        assert n.send(Alert("t", "b", kind="summary")) == {"skipped": "quiet"}
        assert n.send(Alert("t", "b", kind="entry")) != {"skipped": "quiet"}

    def test_a_failing_channel_does_not_block_others(self):
        class Boom(ConsoleChannel):
            name = "boom"

            def send(self, alert):
                raise RuntimeError("channel down")

        n = Notifier(["console"])
        n.channels.insert(0, Boom())
        results = n.send(Alert("t", "b"))
        assert "channel down" in str(results["boom"])
        assert results["console"] is True

    def test_alerts_are_logged_to_disk(self, tmp_path):
        log = tmp_path / "alerts.jsonl"
        n = Notifier(["console"], log_path=log)
        n.send(Alert("title", "body", kind="entry", symbol="AAPL"))
        rows = [json.loads(l) for l in log.read_text().splitlines()]
        assert rows[0]["symbol"] == "AAPL" and rows[0]["kind"] == "entry"

    def test_test_helper_sends(self):
        assert Notifier(["console"]).test()["console"] is True


# =====================================================================
#  Validation gate
# =====================================================================
class TestAlertGate:
    def _add_trades(self, journal, n, pnl_each, broker="paper"):
        for i in range(n):
            tid = journal.open_trade(
                "AAA", "long", 10, 100.0, f"2026-08-{(i % 27) + 1:02d} 10:00:00",
                planned_stop=99.0, broker=broker,
            )
            journal.close_trade(
                tid, 100.0 + pnl_each / 10.0,
                f"2026-08-{(i % 27) + 1:02d} 11:00:00",
                "target" if pnl_each > 0 else "stop",
            )

    def test_empty_journal_is_paper_mode(self, cfg, journal):
        v = AlertGate(cfg, journal).evaluate()
        assert not v.live_allowed
        assert any("closed trades" in r for r in v.reasons)

    def test_thin_sample_stays_paper(self, cfg, journal):
        self._add_trades(journal, 10, 5.0)
        v = AlertGate(cfg, journal).evaluate()
        assert not v.live_allowed

    def test_synthetic_trades_never_validate(self, cfg, journal):
        # Plenty of winning trades, but all from a random walk.
        self._add_trades(journal, 80, 5.0, broker="paper(backtest)")
        v = AlertGate(cfg, journal).evaluate()
        assert not v.live_allowed
        assert any("real market data" in r for r in v.reasons)

    def test_losing_strategy_never_validates(self, cfg, journal):
        self._add_trades(journal, 80, -5.0, broker="fidelity")
        v = AlertGate(cfg, journal).evaluate()
        assert not v.live_allowed
        assert any("expectancy" in r for r in v.reasons)

    def test_validated_strategy_opens_the_gate(self, cfg, journal):
        self._add_trades(journal, 80, 5.0, broker="fidelity")
        v = AlertGate(cfg, journal).evaluate()
        assert v.live_allowed
        assert v.reasons == []

    def test_checks_are_reported(self, cfg, journal):
        self._add_trades(journal, 20, 5.0)
        v = AlertGate(cfg, journal).evaluate()
        for key in ("trades", "expectancy", "win_rate", "real_data_trades"):
            assert key in v.checks


# =====================================================================
#  Session integration
# =====================================================================
class TestSessionAlerts:
    def test_session_computes_a_gate_on_construction(self, cfg):
        from trady.session import TradingSession

        s = TradingSession(cfg, dry_run=True, notifier=Notifier(["console"]))
        # A brand-new journal can never be live-tradable.
        assert s.gate.live_allowed is False

    def test_session_works_without_a_notifier(self, cfg):
        from trady.session import TradingSession

        s = TradingSession(cfg, dry_run=True)
        assert s.notifier is None


# =====================================================================
#  Gate override
# =====================================================================
class TestGateOverride:
    def test_override_opens_the_gate(self, cfg, journal):
        assert AlertGate(cfg, journal).evaluate().live_allowed is False
        cfg.execution.override_validation_gate = True
        assert AlertGate(cfg, journal).evaluate().live_allowed is True

    def test_override_still_records_what_was_failing(self, cfg, journal):
        cfg.execution.override_validation_gate = True
        v = AlertGate(cfg, journal).evaluate()
        # The choice stays visible: reasons move into the checks, not away.
        assert v.checks["OVERRIDDEN"] is True
        assert len(v.checks["overridden_despite"]) >= 1

    def test_override_defaults_off(self):
        from trady.config import Config as _C

        assert _C().execution.override_validation_gate is False
