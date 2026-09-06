"""Resampling 1-minute source bars upward.

Two rules, both of which exist because breaking either produces a backtest that
looks better than reality:

1. **Left-closed, left-labelled bins.** A bar stamped 10:00 on a 15-minute chart
   covers [10:00, 10:15). This is the universal OHLC convention, and it means the
   label is the *start* of the bar: the data is not complete until 10:15. The
   engine is what makes this safe. A signal computed from the bar stamped 10:00
   is filled at the open of the bar stamped 10:15, never inside the bar itself.
   The label convention and the fill rule have to be read together; either one
   alone is a lookahead bug.

2. **No bin ever spans a session break.** Bars are grouped by (session, bin) so a
   bucket straddling the Friday close cannot glue Friday last minute to Sunday
   first minute and report the weekend gap as an intraday range.

Empty bins are dropped rather than forward-filled. An invented bar is a fill the
backtest can trade against and the market never offered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.base import OHLCV, normalise_bars
from src.data.sessions import SessionCalendar


def is_fixed_frequency(rule: str) -> bool:
    try:
        offset = pd.tseries.frequencies.to_offset(rule)
    except ValueError:
        return False
    return isinstance(offset, pd.tseries.offsets.Tick)


def resample_bars(
    bars: pd.DataFrame,
    rule: str,
    calendar: SessionCalendar | None = None,
    *,
    min_source_bars: int = 1,
) -> pd.DataFrame:
    """Aggregate bars up to ``rule``.

    Returns the OHLCV frame plus ``n_source_bars``, the count of source bars that
    went into each output bar. A thin bar (one minute of data standing in for a
    whole hour) is a data-quality signal worth keeping rather than discarding.
    """
    bars = normalise_bars(bars)
    if not is_fixed_frequency(rule):
        raise ValueError(
            f"resample rule {rule!r} is not a fixed frequency; calendar-aware rules "
            "cannot honour the left-closed bin contract used here"
        )
    if bars.empty:
        out = bars.copy()
        out["n_source_bars"] = pd.Series(dtype="int64")
        return out

    calendar = calendar or SessionCalendar()

    # .floor() is left-closed and left-labelled by construction: every timestamp in
    # [10:00, 10:15) maps to 10:00. Using it instead of DataFrame.resample removes
    # any doubt about how closed=/label=/origin interact.
    bin_start = bars.index.floor(rule)
    session = calendar.session_id(bars.index)

    session_key = pd.Series(session, index=bars.index, name="session")
    bin_key = pd.Series(bin_start, index=bars.index, name="bin")
    grouped = bars.groupby([session_key, bin_key], sort=True)
    aggregation = dict(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        n_source_bars=("close", "size"),
    )
    if "spread" in bars.columns:
        # ``spread`` is quoted at the bar open, and the aggregate bar opens where
        # its first source bar opens, so "first" is the only aggregation that
        # keeps the column meaning what its name says. Averaging it here would
        # quietly turn a fill-time cost into a period average.
        aggregation["spread"] = ("spread", "first")
    out = grouped.agg(**aggregation)
    out.index = pd.DatetimeIndex(
        out.index.get_level_values("bin"), tz="UTC", name=bars.index.name
    )
    out = out.sort_index(kind="stable")

    if out.index.has_duplicates:
        # Only possible when one time bin is split across two sessions, meaning the
        # bin is wider than the daily break. Merging the halves would fabricate a
        # bar spanning the break, so refuse loudly instead.
        dupes = out.index[out.index.duplicated()].unique()
        raise ValueError(
            f"resampling to {rule!r} produced {len(dupes)} bins spanning a session "
            "break; choose a resolution that divides the trading day"
        )

    if min_source_bars > 1:
        out = out[out["n_source_bars"] >= min_source_bars]

    out["n_source_bars"] = out["n_source_bars"].astype("int64")
    return out[OHLCV + ["n_source_bars"]]


def assert_no_invented_bars(source: pd.DataFrame, resampled: pd.DataFrame) -> None:
    """Every output bar must be backed by source bars and lie inside the source
    span. Cheap enough to run on every load."""
    if resampled.empty:
        return
    if int(resampled["n_source_bars"].min()) < 1:
        raise AssertionError("resampled frame contains a bar with no source data")
    total = int(resampled["n_source_bars"].sum())
    if total != len(source):
        raise AssertionError(
            f"resampling accounted for {total} source bars but was given {len(source)}"
        )
    if resampled.index.max() > source.index.max():
        raise AssertionError("resampled frame extends past the source data")


def bars_per_year(index: pd.DatetimeIndex) -> float:
    """Empirical bars per year, measured from the data rather than assumed.

    The ~23-hour gold day and the weekend gap make any hardcoded annualisation
    factor wrong, and a factor that is off by 20 percent moves every Sharpe ratio
    in the report by 10 percent.
    """
    if len(index) < 2:
        return float("nan")
    span_years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 3600)
    if span_years <= 0:
        return float("nan")
    return float(len(index) / span_years)


def infer_bar_seconds(index: pd.DatetimeIndex) -> float:
    """Modal spacing between consecutive bars, in seconds."""
    if len(index) < 3:
        return float("nan")
    deltas = (index[1:] - index[:-1]).total_seconds().to_numpy()
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return float("nan")
    values, counts = np.unique(deltas, return_counts=True)
    return float(values[int(np.argmax(counts))])
