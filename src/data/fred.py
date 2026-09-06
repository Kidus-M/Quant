"""FRED daily macro series, for use as regime filters.

The series pulled by default:

* ``DFII10``   10-year TIPS real yield. The strongest single macro driver of gold.
* ``DTWEXBGS`` Broad trade-weighted dollar index.
* ``DGS10``    Nominal 10-year Treasury yield.
* ``T10YIE``   10-year breakeven inflation.

The join onto intraday bars is where the lookahead risk lives, and it gets its own
function with its own test. Two rules:

1. **Lag before joining.** A value stamped with date *d* was not on the wire at
   00:00 on day *d*. It is treated as available no earlier than *d + lag_days*.
2. **Forward-fill only.** ``merge_asof`` with ``direction="backward"`` makes it
   structurally impossible to pick up a value from the future, in a way that
   ``reindex(...).ffill()`` on a carelessly built index does not.
"""
from __future__ import annotations

import io
import logging
import os

import pandas as pd

from src.config import Config, load_dotenv
from src.data.cache import SeriesCache

log = logging.getLogger(__name__)

API_URL = "https://api.stlouisfed.org/fred/series/observations"
CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

DEFAULT_SERIES = ("DFII10", "DTWEXBGS", "DGS10", "T10YIE")


class FredClient:
    """Fetches daily series, caching each to parquet.

    An API key is read from ``FRED_API_KEY`` (env or the gitignored .env). Without
    one the public CSV endpoint is used, which needs no credentials but is less
    stable; both paths return the same shape.
    """

    def __init__(self, cache_dir: str = "data/cache/fred", api_key: str | None = None, session=None):
        load_dotenv()
        self.cache = SeriesCache(cache_dir)
        self.api_key = api_key or os.environ.get("FRED_API_KEY") or None
        self._session = session

    def _require_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def get(self, series_id: str, *, refresh: bool = False) -> pd.Series:
        if not refresh:
            cached = self.cache.read(series_id)
            if cached is not None and len(cached):
                log.info("fred: %s served from cache (%d observations)", series_id, len(cached))
                return cached
        series = self._download(series_id)
        self.cache.write(series_id, series)
        return series

    def _download(self, series_id: str) -> pd.Series:
        session = self._require_session()
        if self.api_key:
            resp = session.get(
                API_URL,
                params={
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                },
                timeout=30,
            )
            resp.raise_for_status()
            observations = resp.json()["observations"]
            frame = pd.DataFrame(observations)
            values = pd.to_numeric(frame["value"], errors="coerce")
            index = pd.DatetimeIndex(pd.to_datetime(frame["date"]), name="date")
        else:
            log.warning(
                "fred: no FRED_API_KEY set, falling back to the public CSV endpoint for %s",
                series_id,
            )
            resp = session.get(CSV_URL, params={"id": series_id}, timeout=30)
            resp.raise_for_status()
            frame = pd.read_csv(io.StringIO(resp.text))
            date_col, value_col = frame.columns[0], frame.columns[1]
            values = pd.to_numeric(frame[value_col], errors="coerce")
            index = pd.DatetimeIndex(pd.to_datetime(frame[date_col]), name="date")

        # FRED writes "." for holidays and missing prints. Those become NaN and are
        # dropped rather than filled here; filling belongs in the join, where the
        # publication lag is applied.
        series = pd.Series(values.to_numpy(), index=index, name=series_id).dropna()
        series = series[~series.index.duplicated(keep="last")].sort_index()
        return series

    def get_many(self, series_ids=DEFAULT_SERIES, *, refresh: bool = False) -> pd.DataFrame:
        frames = {s: self.get(s, refresh=refresh) for s in series_ids}
        return pd.concat(frames, axis=1, sort=True).sort_index()


def available_at(series: pd.Series | pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """Re-stamp a daily series with the earliest UTC instant it could have been used.

    A value for date *d* becomes usable at midnight UTC on *d + lag_days*. With the
    default lag of one day that is comfortably after the US afternoon publication
    time for these series.
    """
    if lag_days < 1:
        raise ValueError(
            "publication_lag_days must be at least 1. A daily macro print is not "
            "available to an intraday strategy at 00:00 on the day it describes."
        )
    frame = series.to_frame() if isinstance(series, pd.Series) else series.copy()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    frame = frame.set_axis(index.normalize() + pd.Timedelta(days=lag_days), axis=0)
    frame.index.name = "available_at"
    return frame.sort_index()


def join_macro_to_bars(
    bars: pd.DataFrame,
    macro: pd.Series | pd.DataFrame,
    *,
    lag_days: int = 1,
    prefix: str = "",
) -> pd.DataFrame:
    """Forward-fill lagged daily macro values onto an intraday bar index.

    Returns a frame indexed exactly like ``bars``. Values before the first
    publication are NaN, deliberately: a strategy that needs the macro filter
    should be flat there rather than trading on a backfilled guess.
    """
    stamped = available_at(macro, lag_days)
    left = pd.DataFrame(index=pd.DatetimeIndex(bars.index)).reset_index()
    time_col = left.columns[0]
    right = stamped.reset_index()

    merged = pd.merge_asof(
        left.sort_values(time_col),
        right.sort_values("available_at"),
        left_on=time_col,
        right_on="available_at",
        direction="backward",   # never looks forward. This is the whole point.
        allow_exact_matches=True,
    )
    merged = merged.set_index(time_col).drop(columns=["available_at"])
    merged.index.name = bars.index.name
    if prefix:
        merged.columns = [f"{prefix}{c}" for c in merged.columns]
    return merged


def load_macro(cfg: Config, bars: pd.DataFrame) -> pd.DataFrame:
    """Config-driven macro load. Returns an empty frame when macro is disabled."""
    if not bool(cfg.get("macro.enabled", False)):
        return pd.DataFrame(index=bars.index)
    client = FredClient(cache_dir=cfg.get("macro.cache_dir", "data/cache/fred"))
    series_ids = list(cfg.get("macro.series", DEFAULT_SERIES))
    lag_days = int(cfg.get("macro.publication_lag_days", 1))
    frame = client.get_many(series_ids)
    return join_macro_to_bars(bars, frame, lag_days=lag_days)
