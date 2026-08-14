# Validation results

What happened when the strategy was tested against real market data.

**Date:** 2026-08-14
**Data:** real daily OHLCV, AAPL / MSFT / IBM / GOOG, ~2001–2013 (3,000 bars
each; 2,148 for GOOG), bundled offline via `bokeh_sampledata`. Real price
action including the dot-com unwind and the 2008 crash.

---

## Headline

**No edge was demonstrated.** Walk-forward across 16 time slices: 7 positive,
9 negative, **net −$793.92**. That is a coin flip that loses to costs.

The strategy should not be traded with real money on this evidence.

---

## What was run

### 1. Full-sample backtest, default settings

| | |
|---|---|
| trades | 10 (over 12 years, 4 symbols) |
| win rate | 20% |
| net P&L | −$179.91 |
| expectancy | −0.60% per trade |

Ten trades in twelve years is far too few to measure anything. 4,350 signals
were rejected for insufficient confluence — the default threshold of 3.0 is
extremely restrictive on daily bars.

### 2. Threshold sweep

| threshold | trades | win rate | expectancy | net P&L |
|---|---|---|---|---|
| 0.75 | 2 | 0% | −6.18% | −$364.54 |
| 1.00 | 2 | 0% | −6.18% | −$364.54 |
| 1.50 | 7 | 43% | +1.45% | +$335.79 |
| 2.00 | 6 | 50% | +2.26% | +$438.99 |
| 2.50 | 32 | 34% | +0.82% | +$747.83 |
| 3.00 | 10 | 20% | −0.60% | −$179.91 |

At a glance this looks like a tunable edge peaking around 2.0–2.5. It isn't,
for two reasons.

**The trade counts are non-monotonic.** 2, 2, 7, 6, 32, 10. A *lower* threshold
should admit *more* trades, always. It doesn't, which means something other than
the threshold governs how many trades happen.

That something is position-slot starvation: average holding period is **9.1
days** on daily bars, and `max_open_positions` is 4. With four symbols, the
slots fill early and stay full, so most signals are blocked regardless of
threshold. The apparent "tuning curve" is an artefact of which trades happened
to grab a slot first.

**The samples are tiny.** The two best-looking rows are 7 and 6 trades. At that
size a single lucky trade dominates the statistic entirely.

### 3. Walk-forward — the decisive test

Threshold 2.5 (the only setting with a non-trivial sample), 4 time slices per
symbol:

| slices positive | 7 / 16 |
|---|---|
| total trades | 50 |
| **net across all slices** | **−$793.92** |

The +$747 from the single full-sample run does not survive being split into
sequential out-of-sample periods. It becomes −$794.

That is the textbook signature of curve-fitting, and the books name it:
over-optimisation produces a model that describes the past and predicts nothing.

---

## Honest limits of this test

These caveats cut both ways — the result is not proof the strategy is worthless,
only that no edge has been shown.

- **Daily bars, not intraday.** The strategy is built for 5-minute day trading.
  On daily bars the session-clock rules never engage and holding periods stretch
  to days, so this tests the *signal logic* rather than the intraday system.
- **Data ends in 2013.** It says nothing about current market microstructure.
- **Four symbols.** A wider universe would give more independent samples.
- **Position-slot starvation confounds the sweep**, as above.

A proper test needs intraday bars on a live feed. That was not available in this
environment — network policy blocks market-data hosts — and `bokeh_sampledata`
was the only route to genuine market prices.

---

## What this does *not* invalidate

The risk machinery is independent of whether the signal predicts anything, and
it is verified:

- All five position-sizing formulas reproduce the books' worked examples exactly
- PDT budget tracking against FINRA NASD 2520, business-day rolling window
- Stop placement outside single-bar noise
- Daily loss limit, drawdown halt, trade cap, cool-off, sector concentration
- Bracket (OTOCO) order generation
- Journal P&L, R-multiples, day-trade flagging

That is why `trady ticket` exists: it applies all of the above to a trade idea
*you* choose, with no dependence on the unproven signal engine.

---

## What would change the verdict

1. **Intraday data.** Run the same walk-forward on 5-minute bars from a live
   feed across 20+ symbols. That is the test this system was designed for.
2. **Fix slot starvation** for any daily-bar test — raise `max_open_positions`
   or add a maximum holding period so the sweep measures the threshold rather
   than queueing.
3. **More symbols.** 4 symbols × 4 slices is 16 samples. That is not enough to
   separate skill from luck.

Until a walk-forward on intraday data shows positive expectancy in *most*
slices, this strategy has not earned real money.

---

## Reproducing

```bash
pip install bokeh_sampledata
python3 -m trady backtest AAPL MSFT IBM GOOG --real --record
python3 -m trady backtest AAPL MSFT IBM GOOG --real --walk-forward
```
