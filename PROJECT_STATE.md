# Project state

Session-to-session memory. Read this first when resuming; update it before
stopping. Nothing important should live only in a chat transcript.

**Last updated:** 2026-08-13 (session 2)
**Branch:** `claude/affectionate-wozniak-n301u3`

---

## Where things stand

Session 1 built the system. Session 2 tested it properly and found two real
bugs, one of which was materially damaging. Still nothing has touched a real
account.

### Session 2 — what changed

**Added a test suite: 211 tests, all passing.** Weighted toward the code where a
bug costs money rather than a bad trade. Written after the fact, which is why it
immediately found things.

**Bug: structural stops were inverted (the important one).**
The stop-placement logic took the *tighter* of the ATR stop and the nearby
support/resistance level, parking the stop between the entry and the level. A
long was therefore stopped out by ordinary noise before its idea — "support
holds" — had been tested at all. 64% of trades were stopped on their own entry
bar. Fixed to place the stop *beyond* the invalidating level. Same-bar stop-outs
fell to 0% on two of three test seeds, and on a three-symbol backtest average R
went from -0.86 to -0.11 and profit factor from 0.51 to 0.81.

**Added a stop noise floor.** ATR is an *average* range, so a 1.5x ATR stop still
sits inside any above-average bar, and volatility clusters. Stops are now floored
at 1.25x the recent median true range (`risk.stop_noise_floor_mult`).

**Implemented sector concentration.** `max_sector_concentration` was in config
but enforced nowhere — a rule that silently does nothing is worse than no rule.
Now enforced in the risk gate via a config sector map. Symbols absent from the
map are unconstrained, so an incomplete map never silently blocks trades.

### Done (cumulative)

- **Knowledge base** — 5 EPUBs extracted (594,104 words), FTS5-indexed (2,093
  passages), distilled into `knowledge/rulebook.yaml` with citations.
- **Indicators** — book formulas, not generic library versions.
- **Candlestick patterns** — 19 types to the candlestick book's literal criteria,
  gated on a preceding directional move.
- **Structure** — trend phase, support/resistance, gaps, breakouts, chart patterns.
- **Strategy** — evidence -> weighted confluence -> gates -> sizing, with the
  reasoning attached to every signal.
- **Risk** — five book sizing formulas plus the permission gate: PDT, daily loss,
  drawdown halt, trade cap, cool-off, session clock, sector concentration.
- **Brokers** — paper simulator, Fidelity bridge, Alpaca adapter.
- **Backtest** — event-driven, next-bar fills, walk-forward.
- **Journal** — SQLite; signals taken and rejected, trades with MAE/MFE/R.
- **Learning** — evidence attribution, bounded weight adjustment, probation.
- **Reporting** — daily, cumulative, HTML.
- **CLI** — 13 commands.

### Verified

- All five sizing formulas reproduce the books' worked examples exactly.
- PDT: blocks the third day trade at $10k, unrestricted at $30k, rolling window
  counts business days not calendar days.
- **No look-ahead**: running the same backtest on 1,200 vs 2,000 bars produces
  byte-identical trades inside the shared prefix.
- Entries fill on the next bar's open, never the signal bar's close.
- Pattern detectors reject the same shapes when there is no preceding move.
- Fidelity CSV parsing survives disclaimer preambles, currency symbols and legal
  footers.
- Learning loop keeps weights and risk inside their bounds under 200 iterations
  and never raises risk on negative expectancy.

### Bugs found and fixed (all sessions)

1. Drawdown divided by a negative peak — two small losses reported as 89.5%.
2. Cool-off deadlock — 3 consecutive losses froze the agent permanently.
3. Synthetic bars had no session boundaries, so clock rules were never exercised.
4. **Structural stops inverted** — stops placed inside the noise, 64% of trades
   stopped on their entry bar.

---

## Known limitations

- **No real market data has been through this.** The sandbox blocks market-data
  hosts; everything was validated on synthetic bars. Synthetic data is a random
  walk — it proves the plumbing, and proves nothing about edge.
- **No demonstrated edge.** On synthetic data the strategy loses roughly what
  fees and spread cost, which is the honest result for random data.
- **Fidelity execution is manual by design** — see README.
- Sector map covers ~30 common symbols; anything else is unconstrained.
- News and fundamentals are indexed but not wired into live signals.
- Short selling assumes borrow is available; no locate check.

---

## Next session — suggested order

1. **Real data.** `python3 -m trady backtest AAPL MSFT NVDA --record` with
   yfinance reachable. Everything below is meaningless until this is done.
2. **Walk-forward it.** If expectancy is not positive across most slices, fix the
   strategy — do not proceed to paper trading.
3. **Re-tune the confluence threshold** against real data. The 3.0 default came
   from reasoning, not measurement. `python3 -m trady sweep` exists for this.
4. **Paper trade** for weeks: `run --loop` with `broker=paper`, then `eod` daily.
5. Widen the sector map, or source it from an API.
6. Consider a pre-market gap scanner.

---

## Persistent state

| What | Where | Survives restart |
|---|---|---|
| Trades, signals, equity, reviews | `data/journal.sqlite` | yes |
| Bar cache | `data/bars.sqlite` | yes |
| Tuned parameters | `config.json` | yes (written by `review`) |
| Book knowledge | `knowledge/kb.sqlite` | yes (rebuild: `tools/build_kb.py`) |
| Reports | `reports/` | yes |

`config.json` is written by the learning loop, so `git diff config.json` shows
what the agent has changed about itself.

---

## Decisions made, and why

- **Paper is the default broker.** Live requires typing `LIVE` at a prompt.
- **No browser automation of Fidelity.** Violates their terms and risks the
  account. Tickets plus CSV reconciliation gets most of the benefit safely.
- **Half-Kelly, capped at Gann's 10%.** The book warns raw Kelly is too
  aggressive above ~20%.
- **One reserve day trade under $25k.** So a forced exit never costs 90 days.
- **Rejected signals are journalled.** They are the counterfactual.
- **Bounded weight adjustment.** The books name system-hopping as a failure mode.
- **Stops go beyond the level, never inside it.** A stop between entry and the
  invalidating level tests nothing except whether the market wiggles.
