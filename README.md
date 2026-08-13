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

**2. Nobody can guarantee consistent profit, including this.** The books this
system is built from say so directly: *"Most day traders lose money... some
research shows that 80 percent of day traders wash out in the first year."* and
*"Anyone with a surefire system has already made a fortune and retired."*
What this system can do is enforce discipline perfectly — position sizing, stops,
loss limits, and an honest record — which is the part humans reliably fail at.
Whether that produces profit depends on whether the strategy has a real edge,
which only out-of-sample testing and paper trading will tell you.

**3. If your account is under $25,000, the PDT rule governs everything.** FINRA
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
| No entries before | 10:00 | the "gap and crap" |
| No entries after | 15:30 | needs time to work |
| Force flat | 15:55 | day traders close out |

Full citations in `knowledge/rulebook.yaml`.

---

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

Working and tested end-to-end offline. Before risking money:

1. Backtest on **real** data for your symbols — synthetic proves nothing about edge.
2. Walk-forward validate: `python3 -m trady backtest --walk-forward`.
3. Paper trade for weeks. The books recommend months.
4. Only then consider live, small, with the PDT guard on.

Past performance does not indicate future results, and a backtest is not
performance.
