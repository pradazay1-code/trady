"""Market data access.

Providers are tried in order and every one returns the same normalised frame:
a DatetimeIndex plus float columns open/high/low/close/volume.

`synthetic` exists so the whole pipeline — patterns, risk, backtest, journal,
reports — can be exercised with no network access. It is a geometric random walk
with volatility clustering and deliberately injected candlestick reversal setups,
so it is useful for testing plumbing and NEVER for judging profitability.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    pass


def _normalise(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Lowercase columns, coerce dtypes, sort, drop dupes and bad bars."""
    if df is None or df.empty:
        raise DataError(f"no data returned for {symbol}")
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            c[0] if isinstance(c, tuple) else c for c in out.columns
        ]
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]

    alias = {"adj_close": "adjclose", "vol": "volume", "last": "close"}
    out = out.rename(columns={k: v for k, v in alias.items() if k in out.columns})

    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise DataError(f"{symbol}: missing columns {missing}")

    out = out[REQUIRED].astype(float)
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # Drop structurally impossible bars rather than trading off them.
    bad = (
        (out["high"] < out["low"])
        | (out["high"] < out["open"]) | (out["high"] < out["close"])
        | (out["low"] > out["open"]) | (out["low"] > out["close"])
        | (out[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    if bad.any():
        out = out[~bad]
    return out.dropna()


# ------------------------------------------------------------- providers
def from_yfinance(symbol: str, interval: str = "5m", days: int = 60) -> pd.DataFrame:
    import yfinance as yf

    # Yahoo caps intraday history: 60d for <1h bars, 730d for 1h.
    cap = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730, "1h": 730}
    period_days = min(days, cap.get(interval, days))
    raw = yf.download(
        symbol,
        period=f"{period_days}d",
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    return _normalise(raw, symbol)


def from_stooq(symbol: str, interval: str = "d", days: int = 365) -> pd.DataFrame:
    """Free daily bars, no API key. Intraday is not available from stooq."""
    import io
    import urllib.request

    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = pd.read_csv(io.BytesIO(resp.read()))
    raw.columns = [c.lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date")
    return _normalise(raw, symbol).tail(days)


def from_csv(path: str | Path, symbol: str = "CSV") -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next(
        (c for c in df.columns if str(c).lower() in
         {"date", "datetime", "timestamp", "time"}), df.columns[0]
    )
    df[tcol] = pd.to_datetime(df[tcol])
    return _normalise(df.set_index(tcol), symbol)


def from_alpaca(symbol: str, interval: str = "5Min", days: int = 60) -> pd.DataFrame:
    """Alpaca market data (needs APCA_API_KEY_ID / APCA_API_SECRET_KEY)."""
    import os
    import urllib.request
    import json as _json

    key = os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("APCA_API_SECRET_KEY")
    if not (key and sec):
        raise DataError("alpaca credentials not set")
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=days)
    url = (
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        f"?timeframe={interval}&start={start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f"&end={end.strftime('%Y-%m-%dT%H:%M:%SZ')}&limit=10000&feed=iex"
    )
    req = urllib.request.Request(
        url, headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = _json.loads(resp.read())
    bars = payload.get("bars") or []
    if not bars:
        raise DataError(f"alpaca returned no bars for {symbol}")
    df = pd.DataFrame(bars).rename(
        columns={"t": "date", "o": "open", "h": "high",
                 "l": "low", "c": "close", "v": "volume"}
    )
    df["date"] = pd.to_datetime(df["date"])
    return _normalise(df.set_index("date"), symbol)


# ------------------------------------------------------------- synthetic
def session_index(
    bars: int, interval_minutes: int = 5, start: str = "2024-01-02"
) -> pd.DatetimeIndex:
    """Timestamps on a real US-equity calendar: weekdays, 09:30–16:00 only.

    Without genuine session boundaries the session-clock rules (no entry before
    10:00, force flat at 15:55) and the PDT rolling-day window are never
    exercised, so a backtest on continuous bars silently skips the very
    constraints that matter most live.
    """
    per_day = int((6 * 60 + 30) / interval_minutes)  # 09:30 -> 16:00
    stamps: list[pd.Timestamp] = []
    day = pd.Timestamp(start)
    while len(stamps) < bars:
        if day.weekday() < 5:
            open_ = day + pd.Timedelta(hours=9, minutes=30)
            stamps.extend(
                open_ + pd.Timedelta(minutes=interval_minutes) * k
                for k in range(min(per_day, bars - len(stamps)))
            )
        day += pd.Timedelta(days=1)
    return pd.DatetimeIndex(stamps[:bars], name="date")


def synthetic(
    symbol: str = "TEST",
    bars: int = 2000,
    start_price: float = 100.0,
    interval_minutes: int = 5,
    seed: int | None = 42,
    inject_patterns: bool = True,
) -> pd.DataFrame:
    """Deterministic offline OHLCV for testing the pipeline.

    Regime-switching drift plus GARCH-ish volatility clustering, so trends,
    ranges and reversals all appear. Optionally injects textbook candlestick
    setups so the detectors have something to find.
    """
    rng = np.random.default_rng(seed)

    # Volatility clustering.
    vol = np.zeros(bars)
    vol[0] = 0.0015
    for t in range(1, bars):
        vol[t] = np.sqrt(
            0.000001 + 0.85 * vol[t - 1] ** 2 + 0.10 * (rng.normal() * vol[t - 1]) ** 2
        )
    vol = np.clip(vol, 0.0004, 0.010)

    # Regime-switching drift.
    drift = np.zeros(bars)
    t = 0
    while t < bars:
        span = int(rng.integers(40, 220))
        mu = rng.choice([0.00018, -0.00016, 0.0], p=[0.4, 0.35, 0.25])
        drift[t : t + span] = mu
        t += span

    ret = drift + rng.normal(0, 1, bars) * vol
    close = start_price * np.exp(np.cumsum(ret))

    # Build OHLC around the close path.
    open_ = np.empty(bars)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.0004, bars - 1))
    span = np.abs(rng.normal(0, 1, bars)) * vol * close * 1.6
    high = np.maximum(open_, close) + span * rng.uniform(0.2, 1.0, bars)
    low = np.minimum(open_, close) - span * rng.uniform(0.2, 1.0, bars)

    base_vol = 900_000
    volume = base_vol * (1 + 2.5 * (vol / vol.mean() - 1) + rng.normal(0, 0.25, bars))
    volume = np.clip(volume, 50_000, None)

    if inject_patterns and bars > 200:
        for anchor in range(120, bars - 10, 137):
            r = close[anchor] * 0.02
            if (anchor // 137) % 2 == 0:
                # Bullish hammer after a pushed-down run.
                for k in range(anchor - 5, anchor):
                    close[k] = close[anchor - 6] * (1 - 0.004 * (k - anchor + 6))
                    open_[k] = close[k] * 1.002
                    high[k] = open_[k] * 1.001
                    low[k] = close[k] * 0.997
                open_[anchor] = close[anchor - 1] * 0.999
                close[anchor] = open_[anchor] * 1.0015
                low[anchor] = open_[anchor] - r
                high[anchor] = max(open_[anchor], close[anchor]) * 1.0005
                volume[anchor] *= 2.2
            else:
                # Bearish engulfing after a pushed-up run.
                for k in range(anchor - 5, anchor):
                    close[k] = close[anchor - 6] * (1 + 0.004 * (k - anchor + 6))
                    open_[k] = close[k] * 0.998
                    low[k] = open_[k] * 0.999
                    high[k] = close[k] * 1.003
                open_[anchor - 1] = close[anchor - 2]
                close[anchor - 1] = open_[anchor - 1] * 1.004
                high[anchor - 1] = close[anchor - 1] * 1.001
                low[anchor - 1] = open_[anchor - 1] * 0.999
                open_[anchor] = close[anchor - 1] * 1.002
                close[anchor] = open_[anchor - 1] * 0.994
                high[anchor] = open_[anchor] * 1.001
                low[anchor] = close[anchor] * 0.999
                volume[anchor] *= 2.4

    idx = session_index(bars, interval_minutes)
    df = pd.DataFrame(
        {
            "open": open_, "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close, "volume": volume.round(),
        },
        index=idx,
    )
    df.index.name = "date"
    return _normalise(df, symbol)


# ----------------------------------------------------------------- cache
@dataclass
class BarCache:
    """SQLite-backed bar cache so repeated runs don't refetch."""

    path: Path

    def __post_init__(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS bars (
                       symbol TEXT, interval TEXT, ts TEXT,
                       open REAL, high REAL, low REAL, close REAL, volume REAL,
                       PRIMARY KEY (symbol, interval, ts))"""
            )

    def put(self, symbol: str, interval: str, df: pd.DataFrame) -> int:
        rows = [
            (symbol, interval, ts.isoformat(), r.open, r.high, r.low, r.close, r.volume)
            for ts, r in df.iterrows()
        ]
        with sqlite3.connect(self.path) as con:
            con.executemany(
                "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?)", rows
            )
        return len(rows)

    def get(self, symbol: str, interval: str) -> pd.DataFrame:
        with sqlite3.connect(self.path) as con:
            df = pd.read_sql_query(
                "SELECT ts, open, high, low, close, volume FROM bars "
                "WHERE symbol=? AND interval=? ORDER BY ts",
                con, params=(symbol, interval),
            )
        if df.empty:
            raise DataError(f"cache miss for {symbol} {interval}")
        df["ts"] = pd.to_datetime(df["ts"])
        return _normalise(df.set_index("ts"), symbol)


# ------------------------------------------------------------------- api
def load(
    symbol: str,
    provider: str = "yfinance",
    interval: str = "5m",
    days: int = 60,
    cache: BarCache | None = None,
    fallback_synthetic: bool = False,
) -> pd.DataFrame:
    """Fetch bars, caching on success and optionally falling back offline."""
    try:
        if provider == "yfinance":
            df = from_yfinance(symbol, interval, days)
        elif provider == "stooq":
            df = from_stooq(symbol, days=days)
        elif provider == "alpaca":
            df = from_alpaca(symbol, interval, days)
        elif provider == "synthetic":
            df = synthetic(symbol, bars=max(400, days * 78))
        elif provider.startswith("csv:"):
            df = from_csv(provider.split(":", 1)[1], symbol)
        else:
            raise DataError(f"unknown provider {provider!r}")
        if cache is not None and provider != "synthetic":
            cache.put(symbol, interval, df)
        return df
    except Exception as exc:
        if cache is not None:
            try:
                return cache.get(symbol, interval)
            except DataError:
                pass
        if fallback_synthetic:
            return synthetic(symbol, bars=max(400, days * 78))
        raise DataError(f"{symbol}: {exc}") from exc
