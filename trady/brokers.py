"""Broker adapters.

    Broker (abstract)
      +- PaperBroker    full simulator: fills, slippage, fees, stops, EOD flat
      +- FidelityBridge order tickets + statement reconciliation (see below)
      +- AlpacaBroker   real REST API, paper or live, for true automation

Fidelity, honestly
------------------
Fidelity does not publish a retail trading API. There is no supported way for a
program to place an order in a retail Fidelity account. Anything claiming
otherwise is driving the website with a headless browser, which violates
Fidelity's terms of use and risks the account being locked — that is not a thing
this system will do to a real account holding real money.

So `FidelityBridge` gives you the two halves that *are* safe and legitimate:

* **Out** — it produces an exact order ticket (symbol, action, quantity, order
  type, limit, stop, time-in-force) that you enter into Fidelity's ticket in a
  few seconds. The agent still does all the analysis, sizing and risk control.
* **In** — it reads Fidelity's own CSV exports (Accounts → Activity & Orders →
  Download, and the Positions download) and reconciles them against the
  journal, so realised P&L, fills and slippage all come from your real account
  rather than from a simulation.

That is a semi-automated loop: fully automatic decisions, one manual click to
execute, and automatic reconciliation afterwards. If you want the click removed
too, `AlpacaBroker` is a broker with an official API that supports it; you would
run the strategy there and keep Fidelity for longer-term holdings.
"""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import ExecutionConfig


# =====================================================================
#  Value types
# =====================================================================
@dataclass
class Order:
    symbol: str
    side: str               # buy | sell | sell_short | buy_to_cover
    qty: int
    order_type: str = "limit"   # market | limit | stop | stop_limit
    limit_price: float | None = None
    stop_price: float | None = None
    tif: str = "day"            # day | gtc
    note: str = ""
    client_id: str = ""

    def describe(self) -> str:
        bits = [f"{self.side.upper().replace('_', ' ')} {self.qty} {self.symbol}"]
        if self.order_type == "limit":
            bits.append(f"LIMIT {self.limit_price:.2f}")
        elif self.order_type == "market":
            bits.append("MARKET")
        elif self.order_type == "stop":
            bits.append(f"STOP {self.stop_price:.2f}")
        elif self.order_type == "stop_limit":
            bits.append(f"STOP {self.stop_price:.2f} LIMIT {self.limit_price:.2f}")
        bits.append(self.tif.upper())
        return "  ".join(bits)


@dataclass
class Fill:
    order: Order
    price: float
    qty: int
    ts: datetime
    fees: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    symbol: str
    direction: str          # long | short
    qty: int
    entry_price: float
    entry_ts: datetime
    stop: float | None = None
    target: float | None = None
    trade_id: int | None = None
    strategy: str = ""
    mae: float = 0.0        # worst adverse excursion seen, in price
    mfe: float = 0.0        # best favourable excursion seen, in price
    bars_held: int = 0

    def unrealized(self, price: float) -> float:
        sign = 1.0 if self.direction == "long" else -1.0
        return (price - self.entry_price) * self.qty * sign

    def update_excursion(self, high: float, low: float) -> None:
        if self.direction == "long":
            self.mfe = max(self.mfe, high - self.entry_price)
            self.mae = min(self.mae, low - self.entry_price)
        else:
            self.mfe = max(self.mfe, self.entry_price - low)
            self.mae = min(self.mae, self.entry_price - high)


# =====================================================================
#  Fees
# =====================================================================
def estimate_fees(cfg: ExecutionConfig, side: str, qty: int, price: float) -> float:
    """Commission plus the regulatory fees that apply on the sell side only."""
    fees = cfg.commission_per_trade
    if side in ("sell", "sell_short"):
        principal = qty * price
        fees += principal * cfg.sec_fee_rate
        fees += min(qty * cfg.taf_fee_per_share, cfg.taf_cap)
    return round(fees, 4)


# =====================================================================
#  Abstract broker
# =====================================================================
class Broker(ABC):
    name = "abstract"

    @abstractmethod
    def submit(self, order: Order, ref_price: float, ts: datetime) -> Fill | None: ...

    @abstractmethod
    def positions(self) -> dict[str, Position]: ...

    @property
    @abstractmethod
    def equity(self) -> float: ...


# =====================================================================
#  Paper broker
# =====================================================================
class PaperBroker(Broker):
    """Simulator with modelled slippage and real fee arithmetic.

    Deliberately pessimistic: fills cross the spread against you by
    `slippage_bps`, and stop orders fill at the stop price plus further
    slippage, because in a fast market a stop is not a guaranteed price.
    """

    name = "paper"

    def __init__(self, cfg: ExecutionConfig, starting_cash: float = 25_000.0):
        self.cfg = cfg
        self.cash = float(starting_cash)
        self.starting_cash = float(starting_cash)
        self._positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.realized_pnl = 0.0
        self._last_prices: dict[str, float] = {}

    # -- broker api ---------------------------------------------------
    def submit(self, order: Order, ref_price: float, ts: datetime) -> Fill | None:
        slip = ref_price * (self.cfg.slippage_bps / 10_000.0)

        if order.order_type == "market":
            px = ref_price + slip if order.side in ("buy", "buy_to_cover") else ref_price - slip
        elif order.order_type == "limit":
            lim = order.limit_price if order.limit_price is not None else ref_price
            if order.side in ("buy", "buy_to_cover"):
                if ref_price > lim:
                    return None      # limit not reachable
                px = min(lim, ref_price + slip)
            else:
                if ref_price < lim:
                    return None
                px = max(lim, ref_price - slip)
        else:  # stop / stop_limit — assume triggered, fill with extra slippage
            base = order.stop_price if order.stop_price is not None else ref_price
            px = base - slip if order.side in ("sell", "sell_short") else base + slip

        fees = estimate_fees(self.cfg, order.side, order.qty, px)
        fill = Fill(order, round(px, 4), order.qty, ts, fees, abs(px - ref_price))
        self._apply(fill)
        self.fills.append(fill)
        self._last_prices[order.symbol] = px
        return fill

    def _apply(self, fill: Fill) -> None:
        o = fill.order
        sym, qty, px = o.symbol, fill.qty, fill.price
        pos = self._positions.get(sym)

        if o.side == "buy":
            self.cash -= qty * px + fill.fees
            if pos and pos.direction == "long":
                total = pos.qty + qty
                pos.entry_price = (pos.entry_price * pos.qty + px * qty) / total
                pos.qty = total
            else:
                self._positions[sym] = Position(sym, "long", qty, px, fill.ts)

        elif o.side == "sell":
            self.cash += qty * px - fill.fees
            if pos:
                self.realized_pnl += (px - pos.entry_price) * min(qty, pos.qty)
                pos.qty -= qty
                if pos.qty <= 0:
                    self._positions.pop(sym, None)

        elif o.side == "sell_short":
            self.cash += qty * px - fill.fees
            self._positions[sym] = Position(sym, "short", qty, px, fill.ts)

        elif o.side == "buy_to_cover":
            self.cash -= qty * px + fill.fees
            if pos:
                self.realized_pnl += (pos.entry_price - px) * min(qty, pos.qty)
                pos.qty -= qty
                if pos.qty <= 0:
                    self._positions.pop(sym, None)

    def positions(self) -> dict[str, Position]:
        return self._positions

    def mark(self, prices: dict[str, float]) -> None:
        self._last_prices.update(prices)

    @property
    def positions_value(self) -> float:
        total = 0.0
        for sym, p in self._positions.items():
            px = self._last_prices.get(sym, p.entry_price)
            total += p.qty * px if p.direction == "long" else p.unrealized(px)
        return total

    @property
    def equity(self) -> float:
        return self.cash + self.positions_value


# =====================================================================
#  Fidelity bridge
# =====================================================================
class FidelityBridge(Broker):
    """Order tickets out, statement reconciliation in.

    Never places an order by itself. `submit` writes the ticket to the pending
    queue and returns None; you enter it in Fidelity, then call
    `import_activity` to bring the real fills back in.
    """

    name = "fidelity"

    # Fidelity's Activity/Orders CSV varies by export; map the common headers.
    ACTIVITY_ALIASES = {
        "run date": "date", "date": "date", "trade date": "date",
        "settlement date": "settle_date",
        "action": "action", "description": "description",
        "symbol": "symbol", "security description": "description",
        "quantity": "quantity", "price": "price", "price ($)": "price",
        "amount": "amount", "amount ($)": "amount",
        "commission": "commission", "commission ($)": "commission",
        "fees": "fees", "fees ($)": "fees",
        "account": "account", "type": "type",
    }

    def __init__(self, cfg: ExecutionConfig, out_dir: Path, starting_equity: float = 0.0):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._positions: dict[str, Position] = {}
        self._equity = float(starting_equity)
        self.pending: list[Order] = []

    # -- outbound -----------------------------------------------------
    def submit(self, order: Order, ref_price: float, ts: datetime) -> Fill | None:
        self.pending.append(order)
        self.write_tickets(ts)
        return None  # a human places this order

    def write_tickets(self, ts: datetime | None = None) -> dict[str, Path]:
        """Emit the pending queue as a human ticket sheet, CSV and JSON."""
        ts = ts or datetime.now()
        stamp = ts.strftime("%Y%m%d_%H%M%S")
        txt = self.out_dir / f"fidelity_tickets_{stamp}.txt"
        csv_p = self.out_dir / f"fidelity_tickets_{stamp}.csv"
        json_p = self.out_dir / f"fidelity_tickets_{stamp}.json"

        lines = [
            "FIDELITY ORDER TICKETS",
            f"generated {ts:%Y-%m-%d %H:%M:%S}",
            "",
            "Enter each of these in Fidelity: Trade > Stocks/ETFs.",
            "Check the symbol and quantity against the ticket before submitting.",
            "=" * 68,
            "",
        ]
        for i, o in enumerate(self.pending, 1):
            lines += [
                f"[{i}] {o.describe()}",
                f"    Symbol .......... {o.symbol}",
                f"    Action .......... {o.side.replace('_', ' ').title()}",
                f"    Quantity ........ {o.qty}",
                f"    Order type ...... {o.order_type.replace('_', ' ').title()}",
            ]
            if o.limit_price is not None:
                lines.append(f"    Limit price ..... {o.limit_price:.2f}")
            if o.stop_price is not None:
                lines.append(f"    Stop price ...... {o.stop_price:.2f}")
            lines += [f"    Time in force ... {o.tif.upper()}"]
            if o.note:
                lines.append(f"    Why ............. {o.note}")
            lines.append("")

        txt.write_text("\n".join(lines))

        with csv_p.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["symbol", "action", "quantity", "order_type",
                        "limit_price", "stop_price", "tif", "note"])
            for o in self.pending:
                w.writerow([o.symbol, o.side, o.qty, o.order_type,
                            o.limit_price, o.stop_price, o.tif, o.note])

        json_p.write_text(json.dumps([asdict(o) for o in self.pending], indent=2))
        return {"txt": txt, "csv": csv_p, "json": json_p}

    def clear_pending(self) -> None:
        self.pending.clear()

    # -- inbound ------------------------------------------------------
    @classmethod
    def parse_activity(cls, path: str | Path) -> pd.DataFrame:
        """Parse a Fidelity Accounts→Activity CSV into normalised executions.

        Fidelity prefixes the file with disclaimer lines and suffixes it with a
        legal footer, so the header row is located rather than assumed.
        """
        raw = Path(path).read_text(errors="replace").splitlines()
        header_idx = None
        for i, line in enumerate(raw[:40]):
            low = line.lower()
            if "symbol" in low and ("quantity" in low or "action" in low):
                header_idx = i
                break
        if header_idx is None:
            raise ValueError(f"could not find a header row in {path}")

        body = []
        for line in raw[header_idx:]:
            if not line.strip():
                if body:
                    break
                continue
            # Footer disclaimers have no commas / are prose.
            if len(body) > 1 and line.count(",") < 2:
                break
            body.append(line)

        df = pd.read_csv(pd.io.common.StringIO("\n".join(body)))
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.rename(
            columns={c: cls.ACTIVITY_ALIASES[c] for c in df.columns if c in cls.ACTIVITY_ALIASES}
        )

        for col in ("quantity", "price", "amount", "commission", "fees"):
            if col in df.columns:
                df[col] = (
                    df[col].astype(str)
                    .str.replace(r"[$,()]", "", regex=True)
                    .str.strip()
                    .replace({"": None, "nan": None, "--": None})
                    .astype(float)
                )
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

        # Keep executions; drop dividends, transfers, interest.
        if "action" in df.columns:
            act = df["action"].astype(str).str.upper()
            df = df[act.str.contains("BOUGHT|SOLD|BUY|SELL", na=False)]
            df["side"] = act.where(act.str.contains("BOUGHT|BUY"), "sell")
            df["side"] = df["side"].where(~act.str.contains("BOUGHT|BUY"), "buy")
        return df.dropna(subset=["symbol"]).reset_index(drop=True)

    @classmethod
    def parse_positions(cls, path: str | Path) -> pd.DataFrame:
        """Parse a Fidelity Positions CSV into symbol/qty/cost/value."""
        raw = Path(path).read_text(errors="replace").splitlines()
        header_idx = next(
            (i for i, l in enumerate(raw[:30])
             if "symbol" in l.lower() and "quantity" in l.lower()), None
        )
        if header_idx is None:
            raise ValueError(f"could not find a header row in {path}")
        body = [l for l in raw[header_idx:] if l.strip() and l.count(",") >= 2]
        df = pd.read_csv(pd.io.common.StringIO("\n".join(body)))
        df.columns = [str(c).strip().lower() for c in df.columns]
        ren = {}
        for c in df.columns:
            if c.startswith("symbol"):
                ren[c] = "symbol"
            elif c.startswith("quantity"):
                ren[c] = "quantity"
            elif "current value" in c:
                ren[c] = "market_value"
            elif "average cost" in c or "cost basis per" in c:
                ren[c] = "avg_cost"
            elif c.startswith("last price"):
                ren[c] = "last_price"
        df = df.rename(columns=ren)
        for col in ("quantity", "market_value", "avg_cost", "last_price"):
            if col in df.columns:
                df[col] = (
                    df[col].astype(str).str.replace(r"[$,%]", "", regex=True)
                    .replace({"": None, "nan": None, "--": None, "n/a": None})
                    .astype(float)
                )
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
            df = df[~df["symbol"].isin(["", "NAN", "PENDING ACTIVITY"])]
        return df.reset_index(drop=True)

    def reconcile(self, activity_csv: str | Path, journal) -> dict:
        """Compare real Fidelity executions against journalled trades."""
        acts = self.parse_activity(activity_csv)
        journalled = journal.closed_trades()
        matched, unmatched = [], []

        for _, row in acts.iterrows():
            sym, qty = row.get("symbol"), abs(float(row.get("quantity") or 0))
            price = float(row.get("price") or 0)
            hit = journalled[
                (journalled["symbol"] == sym) & (journalled["qty"] == int(qty))
            ] if not journalled.empty else pd.DataFrame()
            (matched if not hit.empty else unmatched).append(
                {"symbol": sym, "qty": qty, "price": price,
                 "date": str(row.get("date"))[:10]}
            )

        return {
            "activity_rows": int(len(acts)),
            "matched": len(matched),
            "unmatched": len(unmatched),
            "unmatched_detail": unmatched[:25],
            "realized_from_fidelity": round(
                float(acts["amount"].sum()) if "amount" in acts.columns else 0.0, 2
            ),
        }

    # -- state --------------------------------------------------------
    def sync_positions(self, positions_csv: str | Path) -> dict[str, Position]:
        df = self.parse_positions(positions_csv)
        now = datetime.now()
        self._positions = {}
        for _, r in df.iterrows():
            q = float(r.get("quantity") or 0)
            if q == 0:
                continue
            self._positions[r["symbol"]] = Position(
                symbol=r["symbol"],
                direction="long" if q > 0 else "short",
                qty=int(abs(q)),
                entry_price=float(r.get("avg_cost") or r.get("last_price") or 0),
                entry_ts=now,
            )
        if "market_value" in df.columns:
            self._equity = float(df["market_value"].fillna(0).sum())
        return self._positions

    def positions(self) -> dict[str, Position]:
        return self._positions

    @property
    def equity(self) -> float:
        return self._equity


# =====================================================================
#  Alpaca (true automation path)
# =====================================================================
class AlpacaBroker(Broker):
    """Official REST API. Set APCA_API_KEY_ID / APCA_API_SECRET_KEY.

    Defaults to the paper endpoint; live requires passing paper=False
    explicitly, so no configuration slip can send an order to a live account.
    """

    name = "alpaca"

    def __init__(self, cfg: ExecutionConfig, paper: bool = True):
        import os

        self.cfg = cfg
        self.key = os.environ.get("APCA_API_KEY_ID")
        self.secret = os.environ.get("APCA_API_SECRET_KEY")
        if not (self.key and self.secret):
            raise RuntimeError("APCA_API_KEY_ID / APCA_API_SECRET_KEY not set")
        self.base = (
            "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        )
        self.paper = paper

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        import urllib.request

        req = urllib.request.Request(
            f"{self.base}{path}",
            method=method,
            data=json.dumps(body).encode() if body else None,
            headers={
                "APCA-API-KEY-ID": self.key,
                "APCA-API-SECRET-KEY": self.secret,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def submit(self, order: Order, ref_price: float, ts: datetime) -> Fill | None:
        side = "buy" if order.side in ("buy", "buy_to_cover") else "sell"
        body = {
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": side,
            "type": order.order_type.replace("_", "-"),
            "time_in_force": "day" if order.tif == "day" else "gtc",
        }
        if order.limit_price is not None:
            body["limit_price"] = f"{order.limit_price:.2f}"
        if order.stop_price is not None:
            body["stop_price"] = f"{order.stop_price:.2f}"
        res = self._call("POST", "/v2/orders", body)
        px = float(res.get("filled_avg_price") or order.limit_price or ref_price)
        return Fill(order, px, int(float(res.get("filled_qty") or order.qty)), ts)

    def positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for p in self._call("GET", "/v2/positions"):
            q = int(float(p["qty"]))
            out[p["symbol"]] = Position(
                p["symbol"], "long" if q > 0 else "short", abs(q),
                float(p["avg_entry_price"]), datetime.now(),
            )
        return out

    @property
    def equity(self) -> float:
        return float(self._call("GET", "/v2/account")["equity"])


def make_broker(cfg, starting_equity: float = 25_000.0) -> Broker:
    """Build the broker named in config. Paper is the default for a reason."""
    kind = cfg.execution.broker
    if kind == "paper":
        return PaperBroker(cfg.execution, starting_equity)
    if kind == "fidelity":
        return FidelityBridge(cfg.execution, Path(cfg.reports_dir) / "tickets", starting_equity)
    if kind == "alpaca":
        return AlpacaBroker(cfg.execution, paper=True)
    raise ValueError(f"unknown broker {kind!r}")
