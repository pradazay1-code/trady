# Fidelity day-trading playbook

How to take a Trady alert and turn it into a live order, and the routine around
it. Written for Fidelity specifically.

Fidelity changes its interface periodically. Menu names here were accurate when
written — if a label has moved, the *concept* still applies; find its new name
rather than guessing.

---

## 0. Before your first trade

### Account type decides almost everything

| | Cash account | Margin account |
|---|---|---|
| Short selling | not allowed | allowed |
| Day trades under $25k | limited by settlement | 3 per 5 business days |
| Day trades at $25k+ | still settlement-limited | unlimited |
| Buying power | your cash | up to 4:1 intraday if PDT |
| Main hazard | good-faith violations | margin calls, PDT restriction |

**In a cash account**, proceeds take until settlement (T+1) to become tradable.
Buying with unsettled proceeds and then selling before settlement is a
*good-faith violation*; three in twelve months and Fidelity restricts you to
settled cash for 90 days. This trips people who think they've dodged PDT by
using a cash account — you haven't dodged it, you've traded one constraint for
another.

**In a margin account under $25,000**, you get 3 day trades per rolling 5
business days. The fourth flags you as a pattern day trader, and since you're
under the threshold, Fidelity restricts the account to closing transactions for
90 days. Trady enforces this for you and holds one trade back in reserve — so
your working budget is 2, not 3.

To enable margin: **Accounts & Trade → Account Features → Brokerage & Trading →
Margin**. Read the agreement; you can lose more than you deposit.

### Get Active Trader Pro

The website is fine for one-off orders. It is not fine for day trading, mostly
because bracket orders live in Active Trader Pro (ATP) and brackets are what
make this whole system work.

**Accounts & Trade → Trading Platforms → Active Trader Pro** → download and
install. It's free.

Set it up once:
- Add a **Level II** window for your watchlist symbols
- Add a **Time & Sales** window
- Add **charts** with 1-minute and 5-minute views
- Enable **Directed Trading** if you want to choose your routing

---

## 1. The single most important thing: bracket orders

A day trade has three prices — entry, stop, target. If you place only the entry
and plan to "watch it," you have no system. You have an intention. When price
moves against you, intention is exactly what fails.

Fidelity supports **conditional orders** in ATP, and the one you want is
**OTOCO** — One Triggers a One-Cancels-the-Other.

```
        ┌─────────────────────┐
        │  ENTRY (limit buy)  │
        └──────────┬──────────┘
                   │ fills, which triggers:
        ┌──────────┴──────────┐
        │                     │
  ┌─────▼─────┐        ┌──────▼──────┐
  │   STOP    │  ◄──►  │   TARGET    │
  │ (sell stop)│  one   │ (sell limit)│
  └───────────┘ cancels └─────────────┘
                the other
```

One submission. If the entry fills, the stop and target both go live. If either
one fills, the other cancels automatically. You cannot forget your stop, and you
cannot talk yourself out of it at 2pm.

### Placing an OTOCO in Active Trader Pro

1. **Trade → Directed Trade & Conditional** (or the **Conditional** tab in the
   trade ticket)
2. Order strategy: **OTOCO**
3. **First order** (the trigger):
   - Action: `Buy` (or `Sell Short`)
   - Symbol, Quantity: from the alert
   - Order type: `Limit`
   - Limit price: the alert's entry
   - Time in force: `Day`
4. **Second order** (target):
   - Action: `Sell` (or `Buy to Cover`)
   - Same quantity
   - Order type: `Limit`, price = the alert's target
   - TIF: `Day`
5. **Third order** (stop):
   - Action: `Sell` (or `Buy to Cover`)
   - Same quantity
   - Order type: `Stop Loss` (or `Stop Limit`)
   - Stop price = the alert's stop
   - TIF: `Day`
6. **Preview** → check symbol, quantity, all three prices → **Place order**

**Stop Loss vs Stop Limit:** a Stop Loss becomes a market order when triggered —
it gets you out, at whatever the next price is. A Stop Limit becomes a limit
order and may not fill at all if price gaps through it, leaving you holding a
loser you thought you'd exited. For day-trading stops, use **Stop Loss**. Being
out at a bad price beats being trapped.

---

## 2. Reading a Trady alert

```
📈 BUY AAPL

BUY 37 AAPL @ 214.35

Stop      211.90   (risk $91)
Target    219.20   (R:R 2.15)
Notional  $7,931

Setup: reversal  ·  confluence 3.42
Why:
  • bullish engulfing (confirmed)
  • price 0.4% above support 213.10 (strength 0.99)
  • uptrend, retracement phase, strength 0.80
  • volume 2.1x its 20-bar average

PDT: 2 day trade(s) left this window
```

Every field maps to a box on the ticket:

| Alert | ATP field |
|---|---|
| `BUY 37 AAPL` | Action = Buy, Quantity = 37, Symbol = AAPL |
| `@ 214.35` | Order type = Limit, Limit price = 214.35 |
| `Stop 211.90` | Third leg: Stop Loss, Stop price = 211.90 |
| `Target 219.20` | Second leg: Limit, price = 219.20 |
| `risk $91` | what you lose if the stop hits — sanity-check it |
| `PDT: 2` | day trades left; at 0, do not open anything closing today |

**If the alert says `PAPER MODE — do not place this order`, do not place the
order.** That stamp means the strategy has not cleared validation. It is not a
formality.

---

## 3. The trading day, step by step

### Night before
```bash
python3 -m trady report performance     # how the system is actually doing
python3 -m trady risk                   # PDT budget, risk posture
```
If `review` suspended a strategy or expectancy went negative, do not trade the
next day. Fix it first.

### 09:00 — pre-market
```bash
python3 -m trady scan                   # what's setting up
```
Open ATP. Check your PDT budget. Decide your maximum number of trades today
before you see a single price — that number is much harder to inflate in
advance than in the moment.

### 09:30–10:00 — do nothing
The opening 30 minutes is the "gap and crap" window. A gap up invites everyone
to sell into it; whoever buys the gap gets run over. Trady refuses entries here
and so should you.

Use the time to watch: which symbols have real volume today, where did they open
relative to yesterday's close, which are trending versus chopping.

### 10:00–15:30 — the session
```bash
python3 -m trady watch --channels ntfy --interval 300
```
Alerts arrive on your phone. For each one:

1. **Check the stamp.** PAPER MODE → log it, don't trade it.
2. **Check your PDT budget.** At 0, stop.
3. **Look at the chart** before placing. The alert is the system's opinion; you
   are the last check. If the symbol just gapped on news the system doesn't
   know about, skip it.
4. **Place the OTOCO** exactly as specified. Don't round the quantity up. Don't
   widen the stop because it "looks close."
5. **Leave it alone.** The bracket handles both exits.

You will be tempted to move a stop that's about to be hit. That single habit
destroys more day-trading accounts than any bad entry.

### 15:30 — no new entries
A day trade needs room to work and time to exit. After 15:30, neither exists.

### 15:55 — flat
Close anything still open. Overnight gap risk is not day trading — it's an
unhedged bet on tomorrow's news made by someone who didn't intend to make it.

### 16:00+ — the part that compounds
```bash
python3 -m trady eod                    # close out, report, self-review
```
Then feed reality back in:
1. Fidelity → **Accounts & Trade → Portfolio → Activity & Orders → Download**
2. ```bash
   python3 -m trady fidelity import --file ~/Downloads/Accounts_History.csv
   python3 -m trady fidelity reconcile --file ~/Downloads/Accounts_History.csv
   ```

This is what makes the system improve rather than just run. Your real fills
carry real slippage, and slippage is invisible in a simulation.

---

## 4. Order types you'll actually use

| Type | What it does | When |
|---|---|---|
| **Limit** | Buy at your price or better, or don't buy | Every entry. Always. |
| **Stop Loss** | Becomes a market order at the stop price | Every exit-on-loss |
| **Stop Limit** | Becomes a *limit* order at the stop | Rarely — can fail to fill |
| **Trailing Stop Loss** | Stop follows price by $ or % | Letting a winner run |
| **Market** | Fill now at any price | Emergencies only |

**Never use a market order to enter.** In a fast market you can fill dollars
away from where you looked. The limit order is the difference between the price
you planned and the price you got.

Time in force: use **Day** for everything intraday. GTC orders survive
overnight, which is exactly what a day trader does not want.

---

## 5. Costs

- **$0 commission** on online US stock and ETF trades
- **Regulatory fees** on sells only: SEC fee ~$27.80 per $1M of principal, FINRA
  TAF ~$0.000166/share (capped ~$8.30). Small, but they compound across many
  trades — which is one concrete reason overtrading loses.
- **Margin interest** if you hold on borrowed money overnight. Day traders who
  actually close flat never pay this.
- **Short selling** requires locatable borrow; hard-to-borrow names carry fees.

Trady models the commission, SEC fee and TAF in its P&L, so its numbers are net,
not gross.

---

## 6. Where people actually lose the money

From the books this system is built on, and worth reading twice:

1. **Trading without a plan.** "Failing to plan is planning to fail."
2. **Ignoring cash management.** No single trade can be allowed to end you.
3. **No stops.** "If a trade isn't working, get out."
4. **Overtrading.** More trades ≠ more profit; fees and spread scale with count.
5. **Holding losers.** Hope is not a position-management strategy.
6. **Chasing.** Late entries buy the top and sell the bottom.
7. **System-hopping.** Switching after every bad week means you never learn any
   system's actual behaviour.
8. **Undercapitalisation.** Under $25k the PDT rule alone will strangle you.
9. **Getting emotional.** "The market doesn't know your position; even if it
   did, it wouldn't care."
10. **Unrealistic expectations.** ~80% of day traders wash out in the first year.

Trady mechanically enforces 1–6 and 8. It cannot enforce 7, 9 or 10 — those are
yours.

---

## 7. Realistic sequencing

You asked to start today. Here is what today can honestly be:

**Today** — set up alerts, confirm they reach your phone, read this document,
and start the validation backtest on real data.

**Week 1** — backtest and walk-forward on real bars for your symbols. If
expectancy isn't positive across most time slices, the strategy needs work and
no amount of notification plumbing changes that.

**Weeks 2–6** — paper trade. Alerts arrive, you place them in ATP's paper
environment or simply log them, and the journal builds a real track record.
The books recommend months, not weeks, and they're right.

**After 50+ real trades with positive expectancy** — the validation gate opens
on its own, alerts stop carrying the PAPER MODE stamp, and you can size up from
small.

Anyone offering a shorter path is selling something. From the source material:
*"Anyone with a surefire system has already made a fortune and retired to a
private island."*

---

## Quick reference

```bash
python3 -m trady notify test --channels ntfy   # confirm phone alerts
python3 -m trady notify gate                   # am I validated yet?
python3 -m trady watch --channels ntfy         # live session with alerts
python3 -m trady risk                          # PDT + risk posture
python3 -m trady eod                           # close out, report, review
python3 -m trady fidelity tickets              # printable order tickets
python3 -m trady kb "when is a hammer confirmed"
```

Setting up ntfy (easiest, free, no account):
1. Install **ntfy** from the App Store / Play Store
2. Subscribe to a topic — make it long and unguessable, e.g. `trady-9f3k2m8x1q`
3. `export TRADY_NTFY_TOPIC=trady-9f3k2m8x1q`
4. `python3 -m trady notify test --channels ntfy`

Anyone who knows the topic name can read your alerts, and those alerts contain
your position sizes. Treat the topic name like a password.

---

## 8. Options at Fidelity

### Approval levels

Options need separate approval: **Accounts & Trade → Account Features → Trading
Restrictions → Options → Apply**.

| Level | Allows |
|---|---|
| 1 | covered calls, cash-secured puts |
| 2 | **+ long calls and long puts — what this system uses** |
| 3 | + spreads |
| 4 | + naked writing (unlimited risk) |

Level 2 is enough. Do not apply for level 4 to day trade; naked writing has
unbounded loss and nothing here is built for it.

### Before every options trade, run the cost check

```bash
python3 -m trady option AAPL --strike 220 --dte 3 --bid 1.05 --ask 1.20 \
        --volume 800 --open-interest 1500 --spot 215
```

Get bid, ask, volume and open interest from the Fidelity option chain
(**Research → Options → Option Chain**, or the ATP chain window).

The output tells you the one thing that decides the trade:

```
TOTAL .......... 50.2% of premium  [PROHIBITIVE]
the underlying must move +1.01% just to break even
```

Half the premium gone in costs, on a one-day hold, before direction matters.
That contract is not a trade; it is a fee with a lottery ticket attached.

### Why options day trading is harder than stock day trading

| | Stock | Option |
|---|---|---|
| spread on a liquid name | ~0.005% | 2–15% |
| time decay | none | 5–40% **per day** near expiry |
| max loss | your stop | 100% of premium, routinely |
| being early | survivable | often fatal |

The books call an option a **wasting asset**: *"as the option moves closer to
its date of expiration, the value of the option declines"*, and *"current-month
options decay at faster rates than longer-dated options."*

The one real advantage: *"the most an option holder can lose is the amount paid
for the option contract."* Loss is bounded. That is why sizing here works
backwards from the premium.

### Placing the order

1. **Trade → Options** (or the ATP options ticket)
2. Action: **Buy to Open**
3. Contract: pick the exact expiry and strike from the ticket output
4. Quantity: contracts, not shares — each covers 100 shares
5. Order type: **Limit**, always. Never market an option; the spread will eat you
6. TIF: **Day**

Then immediately place a **Sell to Close limit** at your target.

**On stops:** a stop order on a thin option can fill far from your price,
because the "market" may be one wide quote. For options, prefer watching the
position and closing manually, or set the stop against the *underlying's* price
rather than the contract's.

### Contract selection defaults, and why

- **7–45 days to expiry.** Nearer decays fastest; 0–2 DTE is a coin flip with
  100% downside.
- **~0.45 delta** (near the money). The contract actually tracks the underlying.
  Far-OTM contracts look like cheap leverage and behave like lottery tickets.
- **Open interest ≥ 250, volume ≥ 25, spread ≤ 10%.** A contract you cannot exit
  at a fair price is a position you do not control.

### PDT applies to options too

An option round trip opened and closed the same day is a day trade, same as
stock. Under $25,000 it comes out of the same budget of 3 per rolling 5 business
days.
