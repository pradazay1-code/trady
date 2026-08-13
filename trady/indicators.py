"""Technical indicators.

Every function here implements the definition given in the reference books, not a
generic library version — where a book states an explicit formula (force index,
on-balance volume, money-flow index, accumulation/distribution, pivot points,
momentum, average true range) that formula is what is coded, with the citation in
the docstring so the behaviour is auditable against the source.

All functions take/return pandas objects and never mutate their input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Required OHLCV column names (lowercase).
OHLCV = ("open", "high", "low", "close", "volume")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        raise KeyError(f"missing column {name!r}; have {list(df.columns)}")
    return df[name].astype(float)


# ---------------------------------------------------------------- averages
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    w = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def vwap(df: pd.DataFrame, reset_daily: bool = True) -> pd.Series:
    """Volume-weighted average price, reset each session for intraday bars."""
    typical = (_col(df, "high") + _col(df, "low") + _col(df, "close")) / 3.0
    vol = _col(df, "volume")
    pv = typical * vol
    if reset_daily and isinstance(df.index, pd.DatetimeIndex):
        day = df.index.normalize()
        return pv.groupby(day).cumsum() / vol.groupby(day).cumsum().replace(0, np.nan)
    return pv.cumsum() / vol.cumsum().replace(0, np.nan)


# ---------------------------------------------------------------- momentum
def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """Book definition: today's close / close N bars ago * 100.

    100 = unchanged, >100 rising, <100 falling.
    (Day Trading For Dummies, ch. 7, "Monitoring momentum")
    """
    return series / series.shift(period) * 100.0


def rate_of_change(series: pd.Series, period: int = 10) -> pd.Series:
    return series.pct_change(period) * 100.0


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's relative strength index."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(100.0).where(avg_loss.notna(), np.nan)


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    line = ema(series, fast) - ema(series, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    high, low, close = _col(df, "high"), _col(df, "low"), _col(df, "close")
    hh = high.rolling(k).max()
    ll = low.rolling(k).min()
    pct_k = (close - ll) / (hh - ll).replace(0, np.nan) * 100.0
    return pd.DataFrame({"k": pct_k, "d": pct_k.rolling(d).mean()})


# ------------------------------------------------------------- volatility
def true_range(df: pd.DataFrame) -> pd.Series:
    """Greatest of: high-low, |high - prev close|, |low - prev close|.

    (Day Trading For Dummies, ch. 8, "Average true range")
    """
    high, low, close = _col(df, "high"), _col(df, "low"), _col(df, "close")
    prev = close.shift(1)
    return pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average true range over `period` bars (book uses 14)."""
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger(series: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, period)
    sd = series.rolling(period, min_periods=period).std(ddof=0)
    upper, lower = mid + std * sd, mid - std * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    pos = (series - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "width": width, "pct_b": pos}
    )


def historical_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """Annualised stdev of log returns."""
    logret = np.log(series / series.shift(1))
    return logret.rolling(period, min_periods=period).std(ddof=0) * np.sqrt(252)


def volatility_ratio(series: pd.Series, short: int = 20, long: int = 90) -> pd.Series:
    """Recent volatility vs. longer-run volatility. >1 means unusually volatile."""
    return historical_volatility(series, short) / historical_volatility(
        series, long
    ).replace(0, np.nan)


def beta(series: pd.Series, benchmark: pd.Series, period: int = 60) -> pd.Series:
    """Rolling covariance of returns vs. benchmark returns, over variance."""
    a = series.pct_change()
    b = benchmark.pct_change().reindex(a.index)
    cov = a.rolling(period).cov(b)
    var = b.rolling(period).var()
    return cov / var.replace(0, np.nan)


# ----------------------------------------------------------------- volume
def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    """Running total: add volume on up closes, subtract on down, flat if equal.

    (Day Trading For Dummies, ch. 8, "On-balance volume")
    """
    close, vol = _col(df, "close"), _col(df, "volume")
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * vol).cumsum()


def force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """volume x (today's moving average - yesterday's moving average).

    (Day Trading For Dummies, ch. 8, "Force index")
    """
    ma = sma(_col(df, "close"), period)
    return _col(df, "volume") * (ma - ma.shift(1))


def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    """((close - low) - (high - close)) / (high - low) x volume, cumulative.

    (Day Trading For Dummies, ch. 8, "Accumulation/distribution index")
    """
    high, low, close, vol = (
        _col(df, "high"), _col(df, "low"), _col(df, "close"), _col(df, "volume")
    )
    rng = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / rng
    return (clv.fillna(0.0) * vol).cumsum()


def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """MFI = 100 - 100 / (1 + money ratio).

    Money ratio = positive money flow / negative money flow over the period.
    >80 overbought, <20 oversold. (Day Trading For Dummies, ch. 8)
    """
    typical = (_col(df, "high") + _col(df, "low") + _col(df, "close")) / 3.0
    flow = typical * _col(df, "volume")
    up = flow.where(typical > typical.shift(1), 0.0)
    down = flow.where(typical < typical.shift(1), 0.0)
    pos = up.rolling(period, min_periods=period).sum()
    neg = down.rolling(period, min_periods=period).sum()
    ratio = pos / neg.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + ratio))


def volume_surge(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current volume as a multiple of its own N-bar average."""
    vol = _col(df, "volume")
    return vol / vol.rolling(period, min_periods=period).mean().replace(0, np.nan)


# ----------------------------------------------------------------- levels
def pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """Classic floor-trader pivots from the prior bar's high/low/close.

    Pivot = average of high, low and close. (Day Trading For Dummies, ch. 7)
    """
    high, low, close = _col(df, "high"), _col(df, "low"), _col(df, "close")
    p = ((high + low + close) / 3.0).shift(1)
    h, l = high.shift(1), low.shift(1)
    return pd.DataFrame(
        {
            "pivot": p,
            "r1": 2 * p - l,
            "s1": 2 * p - h,
            "r2": p + (h - l),
            "s2": p - (h - l),
            "r3": h + 2 * (p - l),
            "s3": l - 2 * (h - p),
        }
    )


def fib_retracements(high: float, low: float) -> dict[str, float]:
    """Fibonacci levels. 0.618 is the golden ratio the books single out; 0.50 is
    the Gann half-retracement many traders use even when rejecting the rest."""
    span = high - low
    return {
        "0.0": high,
        "0.236": high - 0.236 * span,
        "0.382": high - 0.382 * span,
        "0.500": high - 0.500 * span,
        "0.618": high - 0.618 * span,
        "0.786": high - 0.786 * span,
        "1.0": low,
    }


# ------------------------------------------------------------------ bundle
def enrich(df: pd.DataFrame, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Attach the standard indicator set used by the strategy layer."""
    out = df.copy()
    close = _col(out, "close")

    for p in (5, 9, 10, 20, 50, 200):
        out[f"sma_{p}"] = sma(close, p)
    for p in (9, 21):
        out[f"ema_{p}"] = ema(close, p)

    out["vwap"] = vwap(out)
    out["rsi_14"] = rsi(close, 14)
    out["momentum_10"] = momentum(close, 10)
    out["atr_14"] = atr(out, 14)
    out["atr_pct"] = out["atr_14"] / close
    out["hv_20"] = historical_volatility(close, 20)

    m = macd(close)
    out["macd"], out["macd_signal"], out["macd_hist"] = m["macd"], m["signal"], m["hist"]

    bb = bollinger(close)
    out["bb_upper"], out["bb_lower"] = bb["upper"], bb["lower"]
    out["bb_width"], out["bb_pct"] = bb["width"], bb["pct_b"]

    out["obv"] = on_balance_volume(out)
    out["force_index"] = force_index(out)
    out["ad_line"] = accumulation_distribution(out)
    out["mfi_14"] = money_flow_index(out)
    out["vol_surge"] = volume_surge(out)

    piv = pivot_points(out)
    for c in piv.columns:
        out[c] = piv[c]

    if benchmark is not None:
        out["beta_60"] = beta(close, benchmark)

    return out
