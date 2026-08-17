# Trady

An automated day-trading agent built on five trading references. It analyses the
market, decides, sizes positions, enforces its own risk rules, records every
decision, and rewrites its own parameters from what actually worked.

---

## Read this first

Three things are true, and the system is designed around them rather than
pretending otherwise.

**1. Fidelity has no retail trading API.** There is no supported way for a
program to place an order in a retail Fidelity account. Anything claiming
otherwise is driving the website with a headless browser, which violates
Fidelity's terms of use and risks your account being locked. This system will
not do that to an account with real money in it.

What it does instead is a semi-automated loop:

- **Out** — it produces exact order tickets (symbol, action, quantity, order
  type, limit, stop) that take seconds to enter in Fidelity. All the analysis,
  sizing and risk control is automatic; you place the order.
- **In** — it reads Fidelity's own CSV exports and reconciles them against its
  journal, so your real fills and real P&L flow back in.

If you want the manual click removed too, `AlpacaBroker` talks to a broker that
has an official API and supports full automation. Run the strategy there and
keep Fidelity for longer-term holdings. That is a real decision with real
tradeoffs; the code supports either.

**2. The strategy has been tested on real data and no edge was found.**
Walk-forward across 16 time slices of real AAPL/MSFT/IBM/GOOG daily bars
(2001–2013): 7 positive, 9 negative, **net −$793.92**. The full details, and the
caveats that cut both ways, are in **[VALIDATION.md](VALIDATION.md)**. Trade the
signal engine with real money only after a walk-forward on *intraday* data shows
positive expectancy in most slices.

What is validated is the risk machinery — sizing, stops, PDT tracking, bracket
orders. `trady ticket` applies all of it to a trade idea you pick yourself, with
no dependence on the signal engine. That part is usable today.

**3. Nobody can guarantee consistent profit, including this.** The books this
system is built from say so directly: *"Most day traders lose money... some
research shows that 80 percent of day traders wash out in the first year."* and
*"Anyone with a surefire system has already made a fortune and retired."*
What this system can do is enforce discipline perfectly — position sizing, stops,
loss limits, and an honest record — which is the part humans reliably fail at.
Whether that produces profit depends on whether the strategy has a real edge,
which only out-of-sample testing and paper trading will tell you.

**4. If your account is under $25,000, the PDT rule governs everything.** FINRA
allows 3 day trades per rolling 5 business days below that threshold; the fourth
gets you restricted to cash-only for 90 days. The agent enforces this as a hard
block and keeps one trade in reserve for emergency exits. See
`knowledge/rulebook.yaml`.

---

## Quick start

```bash
pip install pandas numpy lxml beautifulsoup4 pyyaml
pip install yfinance                 # market data

python3 tools/extract_epub.py <epub_dir> knowledge/extracted/   # once
python3 tools/build_kb.py                                        # once

python3 -m trady kb "pattern day trader rule"   # ask the books
python3 -m trady analyze AAPL                   # analyse one symbol
python3 -m trady scan                           # scan the watchlist
python3 -m trady backtest --record              # test, write to journal
python3 -m trady review                         # self-correct from results
python3 -m trady report performance             # full breakdown
python3 -m trady risk                           # risk + PDT status
```

Everything works offline with `--synthetic`, which generates realistic bars on a
real trading calendar. **Synthetic data is for testing plumbing, never for
judging profitability** — it is a random walk, so any "profit" on it is noise.

---

## Runs without you

```bash
python3 -m trady validate                  # full evidence pipeline, exit 0=pass 2=fail
./scripts/schedule.sh install              # cron: validate 02:00, watch 09:55, eod 16:05
```

Two GitHub Actions workflows do it in the cloud, on a schedule, with nobody
watching:

| workflow | when | what |
|---|---|---|
| `.github/workflows/validate.yml` | 02:00 UTC weekdays | real **intraday** validation, uploads the report, pushes the verdict to your phone, **fails the build if the strategy does not validate** |
| `.github/workflows/tests.yml` | every push | the full test suite |

GitHub's runners have network access, which the build sandbox does not — so the
intraday test that could not run locally runs there. A red badge on `validate`
means: do not trade this.

Set the `TRADY_NTFY_TOPIC` repository secret to get the nightly verdict pushed
to your phone.

`validate` is deliberately hard to pass. Alongside expectancy and drawdown it
checks that the **threshold sweep is interpretable** — if trade counts do not
fall as the threshold rises, entries are being gated by something other than the
threshold and the sweep cannot be read as a tuning curve. That check exists
because misreading exactly that pattern once made a losing strategy look
tunable.

---

## Options

```bash
python3 -m trady option AAPL --strike 220 --dte 3 --bid 1.05 --ask 1.20 \
        --volume 800 --open-interest 1500 --spot 215
```

Black-Scholes pricing and Greeks (verified against textbook references and
put-call parity), contract selection, and sizing where max loss is the premium —
"the most an option holder can lose is the amount paid for the option contract."

The output that matters is the **true round-trip cost**:

```
  TRUE COST OF A 1-DAY HOLD
    spread (x2) .... 13.3%
    theta .......... 35.7% per day
    commission ..... 1.2%
    TOTAL .......... 50.2% of premium  [PROHIBITIVE]
    the underlying must move +1.01% just to break even
    ! far OTM with little time left: the most likely outcome is a 100% loss
```

That is a real 3-DTE AAPL call. Half the premium is gone in costs before
direction matters. Options day trading is not stock day trading with more
leverage — theta is a headwind measured in hours, and spreads are often 500x a
stock's. The books call an option a *wasting asset*; this command quantifies
exactly how fast it wastes.

Contract selection defaults to 7–45 DTE and ~0.45 delta with liquidity floors,
because cheap far-OTM contracts look like leverage and behave like lottery
tickets.

---

## Your own trade ideas, risk-managed (works today)

The signal engine is unproven. The risk machinery is not — it is verified
against the books' own worked examples. `ticket` bridges the two: you choose the
symbol and side, it does the maths and writes the Fidelity bracket.

```bash
python3 -m trady ticket AAPL --equity 30000 --record
python3 -m trady ticket NVDA --short --entry 131.20 --atr 2.4
python3 -m trady close 1 --price 445.20 --reason target
```

It sizes the position (half-Kelly capped by Gann's 10%), places the stop outside
single-bar noise, checks your PDT budget, writes a three-leg OTOCO ticket, and
journals the trade so your real P&L accumulates into the same reports.

---

## Phone alerts

```bash
export TRADY_NTFY_TOPIC=trady-9f3k2m8x1q      # long and unguessable
python3 -m trady notify test --channels ntfy  # confirm it reaches your phone
python3 -m trady watch --channels ntfy        # live loop, alerts as they fire
python3 -m trady notify gate                  # am I validated for live yet?
```

Channels: `ntfy` (free, no account), `pushover`, `telegram`, `email` (also
reaches SMS via carrier gateways), `console`.

Every alert carries the whole trade — symbol, side, quantity, entry, stop,
target, reward:risk, and the reasoning:

```
📈 BUY AAPL

BUY 37 AAPL @ 214.35

Stop      211.90   (risk $91)
Target    219.20   (R:R 2.15)

Setup: reversal  ·  confluence 3.42
Why:
  • bullish engulfing (confirmed)
  • price 0.4% above support 213.10 (strength 0.99)
  • volume 2.1x its 20-bar average

PDT: 2 day trade(s) left this window
```

### The validation gate

Alerts are stamped **PAPER MODE — do not place this order** until the strategy
has earned the right to be traded. Three conditions, all required:

- at least 50 closed trades on record
- positive measured expectancy over those trades
- at least 50 of them from **real** market data, not synthetic bars

A push notification carries the authority of advice. Until those checks pass,
this system's output is an untested program's opinion, and the stamp says so.
`notify gate` shows exactly what is still blocking.

See **[FIDELITY_PLAYBOOK.md](FIDELITY_PLAYBOOK.md)** for turning an alert into a
Fidelity order, including bracket (OTOCO) orders.

---

## The daily loop

```bash
python3 -m trady run                    # dry run: decides, places nothing
python3 -m trady run --loop             # poll until the close
python3 -m trady fidelity tickets       # order tickets to enter in Fidelity
python3 -m trady eod                    # close out, report, review
```

Then, after the session, feed your real results back:

```bash
# Fidelity: Accounts > Activity & Orders > Download
python3 -m trady fidelity import --file ~/Downloads/Accounts_History.csv
python3 -m trady fidelity reconcile --file ~/Downloads/Accounts_History.csv
```

---

## How a decision gets made

```
bars ──> indicators ──> structure ──> evidence ──> confluence ──> gates ──> size ──> order
              │             │             │            │            │         │
         RSI, ATR,    trend, S/R,   each piece    weighted     PDT, loss   half-Kelly
         MACD, OBV,   gaps, chart   scored and     sum         limit,      capped by
         MFI, VWAP    patterns      weighted                   session     Gann 10%
                           │                                   clock
                    candlestick patterns
                    (19 types, book criteria)
```

Nothing trades on one signal. A candlestick pattern must be corroborated by
structure and volume before it clears the confluence threshold — the book's
"East meets West" principle: a mediocre pattern at a strong support level beats
a textbook pattern in open space.

Every signal carries its evidence, so the journal records *why*:

```
LONG AAPL @ 214.35 via reversal (score 3.42, R:R 2.15)
  - [trend] uptrend, retracement phase, strength 0.80 (+0.80)
  - [candlestick] bullish engulfing (+0.73)
  - [support_resistance] price 0.4% above support 213.10 (strength 0.99) (+0.28)
  - [volume] volume 2.1x its 20-bar average (+0.59)
  stop 211.90 / target 219.20
```

---

## How it corrects itself

After every trade and every day, the agent asks three questions and writes the
answers into its own config:

1. **Which evidence actually paid?** Stored signal evidence is joined against
   realised R-multiples. Evidence types that preceded winners gain weight;
   those that preceded losers lose it.
2. **Which strategies earn their place?** A strategy with negative expectancy
   over 25+ trades gets suspended instead of left to bleed.
3. **Is the risk level right?** Measured win rate and win/loss ratio feed
   half-Kelly, so size tracks the real edge instead of a guess.

Real output from a run on synthetic data:

```
  adjustments made:
    signal.weights.candlestick: 1.0 → 0.95  (avg R -0.50 over 39 trades)
    signal.weights.volume: 0.8 → 0.75       (avg R -0.39 over 54 trades)
    risk.risk_per_trade_pct: 0.0075 → 0.00625  (half-Kelly on measured edge)

  lessons:
    - 'breakout' is the strongest setup: 8 trades, 50% win rate, $69.83 net
    - 78% of exits were stops — entries may be early or stops too tight
    - system expectancy is NEGATIVE — do not increase size; paper-trade
```

Adjustments are small and bounded on purpose. The books name system-hopping as
its own losing pattern: *"No system works all the time; if one did, everyone
would use it."* Nothing adapts until there are at least 20 trades, because
fitting to five trades is curve-fitting.

---

## What's enforced

| Rule | Default | Source |
|---|---|---|
| PDT day-trade budget under $25k | 3 per 5 business days, 1 reserved | FINRA 2520 |
| Max position size | 10% of equity | Gann's rule |
| Risk per trade | 0.75% of equity | fixed-fractional |
| Position sizing | half-Kelly, capped at 10% | Kelly Criterion |
| Minimum reward:risk | 1.5 | candlestick book |
| Daily loss limit | -2%, stop for the day | DTFD ch.21 |
| Drawdown halt | -10% peak-to-trough | — |
| Max trades/day | 8 | overtrading |
| Consecutive losses | 3, cool off till next session | tilt control |
| Sector concentration | 2 positions per sector | correlated risk |
| No entries before | 10:00 | the "gap and crap" |
| No entries after | 15:30 | needs time to work |
| Force flat | 15:55 | day traders close out |
| Live alerts | require 50+ real profitable trades | validation gate |

Full citations in `knowledge/rulebook.yaml`.

---

## Tests

```bash
pip install pytest
python3 -m pytest tests/ -q          # 298 tests, ~11 min
python3 -m pytest tests/test_risk.py -q   # risk only, <1s
```

The suite is weighted toward the parts where a bug costs money rather than a bad
trade:

- **`test_risk.py`** — all five sizing formulas pinned to the books' worked
  examples, the PDT budget including the rolling-business-day window, every
  risk gate, sector concentration.
- **`test_patterns.py`** — each pattern's book criteria, including the negative
  cases: the same candle shape in a sideways drift must NOT be reported,
  because a reversal needs something to reverse.
- **`test_backtest.py`** — the look-ahead test is the important one. It runs the
  same backtest on 1,200 bars and on 2,000, then asserts trades inside the
  shared prefix are identical. If the engine could see the future, appending
  data would change the past.
- **`test_execution.py`** — Fidelity CSV parsing against text shaped like a real
  export (disclaimer preamble, currency symbols, legal footer), broker fills and
  slippage, and the learning loop's bounds.
- **`test_notify.py`** — alert content (every alert must carry a stop), and the
  validation gate: a winning strategy on synthetic data must still fail to
  validate.
- **`test_options.py`** — Black-Scholes pinned to textbook values and put-call
  parity, theta negative and accelerating into expiry, max loss equal to premium,
  and illiquid contracts rejected with a stated reason.

## Layout

```
trady/
  config.py       every rule as a value — one file governs risk posture
  data.py         yfinance / stooq / alpaca / csv / synthetic
  indicators.py   book formulas: ATR, OBV, force index, MFI, A/D, pivots
  patterns.py     19 candlestick patterns to the book's exact criteria
  structure.py    trend, support/resistance, gaps, breakouts, chart patterns
  strategy.py     evidence -> confluence -> signal, with reasoning attached
  risk.py         sizing formulas + the permission gate (PDT, limits, clock)
  brokers.py      paper / Fidelity bridge / Alpaca
  backtest.py     event-driven, next-bar fills, walk-forward
  journal.py      SQLite: signals, trades, equity, reviews, events
  learn.py        evidence attribution -> weight and risk adjustment
  reporting.py    daily, cumulative, and HTML reports
  session.py      the trading day
  notify.py       phone alerts + the live-trading validation gate
  options.py      Black-Scholes, Greeks, true round-trip cost, contract choice
  validate.py     unattended validation pipeline and verdict
  cli.py          command line

knowledge/
  extracted/      full text of all five books (594k words)
  kb.sqlite       FTS5 index, 2,093 searchable passages
  rulebook.yaml   every rule with its citation
```

## Books

| Book | Used for |
|---|---|
| Day Trading For Dummies, 3rd ed. (2014) | PDT rule, money management, indicators, strategies, evaluation |
| Day Trading For Dummies (2008) | corroboration, arbitrage, short selling |
| Getting Started in Candlestick Charting (2008) | all 19 candlestick patterns, confirmation rules, East-meets-West |
| Trading For Dummies, 2nd ed. (2009) | chart patterns, technical analysis |
| Fundamental Analysis For Dummies (2009) | context filters, blending FA with TA |

The full text is indexed and searchable — the agent cites the source rather than
guessing:

```bash
python3 -m trady kb "when is a hammer confirmed"
python3 -m trady kb --topics
```

---

## Status

Working, 298 tests passing, verified end to end offline. Before risking money:

1. Backtest on **real** data for your symbols — synthetic proves nothing about edge.
2. Walk-forward validate: `python3 -m trady backtest --walk-forward`.
3. Paper trade for weeks. The books recommend months.
4. Only then consider live, small, with the PDT guard on.

Past performance does not indicate future results, and a backtest is not
performance.
