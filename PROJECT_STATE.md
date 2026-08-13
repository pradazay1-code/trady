# Project state

Session-to-session memory. Read this first when resuming; update it before
stopping. Nothing important should live only in a chat transcript.

**Last updated:** 2026-08-13 (session 1)
**Branch:** `claude/affectionate-wozniak-n301u3`

---

## Where things stand

Session 1 built the whole system end to end. It runs, it is tested offline, and
nothing has touched a real account.

### Done

- **Knowledge base** — 5 EPUBs extracted to text (594,104 words), indexed into
  SQLite FTS5 (2,093 passages), searchable via `python3 -m trady kb`.
  Distilled into `knowledge/rulebook.yaml` with per-rule citations.
- **Indicators** (`indicators.py`) — book formulas, not generic library versions:
  ATR (exact true-range definition), OBV, force index, MFI, accumulation/
  distribution, pivots, momentum, plus RSI/MACD/Bollinger/VWAP.
- **Candlestick patterns** (`patterns.py`) — 19 types to the candlestick book's
  literal criteria, including its accepted variations, strength modifiers, and
  per-pattern confirmation rules. Gated on a preceding directional move.
- **Structure** (`structure.py`) — trend direction/strength/phase, clustered
  support/resistance, gaps, breakouts with volume confirmation, head-and-
  shoulders, double top/bottom, flags/pennants.
- **Strategy** (`strategy.py`) — evidence accumulation → weighted confluence →
  quality gates → sizing. Every signal carries its reasoning.
- **Risk** (`risk.py`) — all five book sizing formulas (verified against the
  books' own worked examples), plus the permission gate: PDT budget, daily loss
  limit, drawdown halt, trade cap, cool-off, session clock.
- **Brokers** (`brokers.py`) — paper simulator, Fidelity bridge (tickets out,
  CSV reconciliation in), Alpaca adapter.
- **Backtest** (`backtest.py`) — event-driven, next-bar fills, pessimistic
  intrabar exits, walk-forward splitting.
- **Journal** (`journal.py`) — SQLite; signals (taken and rejected), trades with
  MAE/MFE/R-multiple, equity marks, reviews, events.
- **Learning** (`learn.py`) — evidence attribution against realised R, bounded
  weight adjustment, strategy probation, half-Kelly risk recalibration.
- **Reporting** (`reporting.py`) — daily, cumulative, self-contained HTML.
- **Session + CLI** — `run`, `eod`, 13 commands total.

### Verified

- All five book sizing formulas reproduce the books' worked examples exactly
  (expectancy 0.005, P(ruin) 1.7%, Kelly 33.3%, fixed-ratio 2.94, Optimal F 750).
- PDT guard: blocks the 3rd day trade at $10k equity, unrestricted at $30k.
- 45-trade backtest on a realistic session calendar; session-clock rules,
  cool-off, and PDT all observed firing.
- Learning loop produced real adjustments and correct diagnoses.
- Fidelity ticket generation (txt/csv/json) verified.

### Bugs found and fixed

1. **Drawdown divided by a negative peak** — two small losses reported as an
   89.5% drawdown. Now anchored on a positive equity base.
2. **Cool-off deadlock** — after 3 consecutive losses the agent could never
   trade again, since no trade could produce the win that clears the counter.
   Now resets at the start of each session (`_roll_day`).
3. **Synthetic bars had no session boundaries** — the session-clock rules and
   the PDT rolling window were silently never exercised. Now generated on a real
   weekday 09:30–16:00 calendar.

---

## Known limitations

- **No real market data has been through this.** The sandbox blocks market-data
  hosts, so everything was validated on synthetic bars. Synthetic data is a
  random walk: it proves the plumbing works and proves nothing about edge.
- **No demonstrated edge.** On synthetic data the strategy loses roughly what
  fees and spread cost, which is the honest result for a random walk. Whether it
  has an edge on real data is untested and is the whole open question.
- **Fidelity execution is manual by design** — see README.
- **Sector concentration limit** is in config but not enforced (needs a sector
  map).
- **News and fundamentals** are not wired in. `FAFD` material is indexed but
  only informs config filters, not live signals.
- **Short selling** assumes borrow is available; no locate check.

---

## Next session — suggested order

1. **Real data first.** Run `python3 -m trady backtest AAPL MSFT NVDA --record`
   with `yfinance` reachable. Everything below is meaningless until this is done.
2. **Walk-forward** the result. If expectancy is not positive across most
   slices, do not proceed to paper trading — fix the strategy instead.
3. **Tune the confluence threshold** against real data. The 3.0 default was set
   from reasoning, not measurement; the sweep command exists for this.
4. **Paper trade** for several weeks: `python3 -m trady run --loop` with
   `broker=paper`, then `eod` each day and read the review.
5. Add a sector map to enforce `max_sector_concentration`.
6. Consider a pre-market gap scanner (the books rate opening gaps highly, but
   the current session clock deliberately sits out the first 30 minutes).

---

## Persistent state

| What | Where | Survives restart |
|---|---|---|
| Trades, signals, equity, reviews | `data/journal.sqlite` | yes |
| Bar cache | `data/bars.sqlite` | yes |
| Tuned parameters | `config.json` | yes (written by `review`) |
| Book knowledge | `knowledge/kb.sqlite` | yes |
| Reports | `reports/` | yes |

`config.json` is written by the learning loop, so it is the live state of the
agent's tuning. `git diff config.json` shows what it has changed about itself.

---

## Decisions made, and why

- **Paper is the default broker.** Live requires typing `LIVE` at a prompt.
- **No browser automation of Fidelity.** Violates their terms and risks the
  account. Tickets plus CSV reconciliation gets most of the benefit safely.
- **Half-Kelly, capped at Gann's 10%.** The book itself warns raw Kelly is too
  aggressive above ~20%.
- **One reserve day trade under $25k.** So a forced exit never costs a 90-day
  restriction.
- **Rejected signals are journalled.** They are the counterfactual; without them
  the learning loop can only see what was taken.
- **Bounded weight adjustment.** The books name system-hopping as its own
  failure mode.
