"""Causal indicators.

Every function here is strictly backward-looking. Two conventions are enforced
rather than trusted:

* No ``center=True`` anywhere. A centred rolling window is half made of the future.
* Anything describing a *prior* window (Donchian channels, breakout levels) is
  shifted so the current bar is excluded. A Donchian high computed with
  ``high.rolling(n).max()`` includes the current bar, so price is at the channel
  top on every breakout bar and the strategy appears to catch every move.

The value at index ``t`` is always computable from bars ``<= t``. Whether that is
*safe* depends on the engine filling at ``t+1`` open, which it does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(x)


def sma(series: pd.Series, window: int) -> pd.Series:
    return _as_series(series).rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return _as_series(series).ewm(span=span, adjust=False, min_periods=span).mean()


def wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing, the one used by the original RSI and ATR definitions.

    Equivalent to an EMA with alpha = 1/period rather than 2/(period+1). Using the
    wrong one shifts RSI(2) thresholds enough to change which trades fire.
    """
    return _as_series(series).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI. Returns values in [0, 100], NaN until the window fills."""
    s = _as_series(series).astype("float64")
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_ema(gain, period)
    avg_loss = wilder_ema(loss, period)
    # A flat window means no losses: RSI is 100 by definition, not a divide-by-zero.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)
    return out.where(avg_gain.notna() & avg_loss.notna())


def true_range(bars: pd.DataFrame) -> pd.Series:
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average true range in USD per ounce. The unit matters: every risk number in
    the report is derived from this and quoted in dollars."""
    return wilder_ema(true_range(bars), period)


def donchian(bars: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
    """Highest high and lowest low over the ``window`` bars *before* this one.

    The shift is the whole point. Without it, ``close > channel_high`` can never be
    true and ``close >= channel_high`` is true on every new high, which backtests
    as a breakout system that never misses.
    """
    upper = bars["high"].rolling(window, min_periods=window).max().shift(1)
    lower = bars["low"].rolling(window, min_periods=window).min().shift(1)
    return upper, lower


def rolling_return_vol(close: pd.Series, window: int) -> pd.Series:
    """Standard deviation of log returns over a trailing window."""
    log_ret = np.log(_as_series(close).astype("float64")).diff()
    return log_ret.rolling(window, min_periods=max(2, window // 2)).std()


def rate_of_change(series: pd.Series, periods: int) -> pd.Series:
    """Change over N periods, in the units of the input. Used for the macro filter,
    where a *falling real yield* means the N-day change is negative."""
    return _as_series(series).diff(periods)


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """+1 where fast crosses above slow, -1 where it crosses below, 0 otherwise."""
    above = (fast > slow).astype("float64")
    above = above.where(fast.notna() & slow.notna())
    change = above.diff()
    return change.fillna(0.0)
