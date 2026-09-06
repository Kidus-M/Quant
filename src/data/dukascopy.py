"""Dukascopy tick fetcher, aggregated locally to 1-minute bars.

Why Dukascopy, and why ticks rather than their pre-built candles:

* It is free, needs no account, and has the deepest intraday history of the three
  candidates, which is what a walk-forward study needs.
* Fetching **ticks** and building the minute bars here means the base resolution
  in the cache really is source data. It also gives the measured bid/ask spread
  per minute, which turns the most important assumption in the cost model into
  something that can be checked against the tape instead of asserted.

File layout, one LZMA-compressed file per hour::

    https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM-1:02d}/{DD:02d}/{HH:02d}h_ticks.bi5

Each record is 20 bytes big-endian: millisecond offset into the hour (uint32),
ask and bid as integers scaled by the instrument point value (uint32), then ask
and bid volume (float32). An empty or missing file means no ticks in that hour,
which is normal over the weekend and the daily break.

The point scaling is instrument-specific and getting it wrong scales every price
by a power of ten, so ``fetch`` sanity-checks the resulting prices against a
plausible range and refuses rather than caching nonsense.
"""
from __future__ import annotations

import logging
import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.base import (
    PLAUSIBLE_PRICE_RANGE,
    BarAdapter,
    empty_bars,
    normalise_bars,
    validate_price_range,
)

log = logging.getLogger(__name__)

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
_RECORD = struct.Struct(">3I2f")

# Decimal places used by Dukascopy for the integer price encoding.
POINT_DIGITS = {"XAUUSD": 3, "XAGUSD": 3, "EURUSD": 5, "USDJPY": 3, "GBPUSD": 5}

# Kept as a module alias: the plausible-range guard is shared with every other
# adapter that parses a numeric wire format.
PLAUSIBLE_RANGE = PLAUSIBLE_PRICE_RANGE


@dataclass
class _HourResult:
    hour: pd.Timestamp
    ticks: np.ndarray | None
    error: str | None = None


class DukascopyAdapter(BarAdapter):
    name = "dukascopy"
    native_resolution = "1min"
    is_synthetic = False

    def __init__(
        self,
        *,
        max_workers: int = 8,
        timeout: float = 30.0,
        retries: int = 3,
        retry_backoff: float = 1.5,
        session=None,
    ):
        self.max_workers = max_workers
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff
        self._session = session

    # ------------------------------------------------------------------ #
    def _require_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "xauusd-backtester/0.1 (research)"})
        return self._session

    @staticmethod
    def url_for(symbol: str, hour: pd.Timestamp) -> str:
        # Dukascopy months are zero-based. This is the single most common bug when
        # writing this fetcher, so it gets its own line and a comment.
        return (
            f"{BASE_URL}/{symbol.upper()}/{hour.year:04d}/{hour.month - 1:02d}/"
            f"{hour.day:02d}/{hour.hour:02d}h_ticks.bi5"
        )

    def _fetch_hour(self, symbol: str, hour: pd.Timestamp) -> _HourResult:
        url = self.url_for(symbol, hour)
        session = self._require_session()
        for attempt in range(self.retries):
            try:
                resp = session.get(url, timeout=self.timeout)
            except Exception as exc:  # network flake
                if attempt == self.retries - 1:
                    return _HourResult(hour, None, f"{type(exc).__name__}: {exc}")
                time.sleep(self.retry_backoff**attempt)
                continue
            if resp.status_code == 404:
                # Market closed for this hour. Not an error.
                return _HourResult(hour, np.empty((0, 5), dtype=np.float64))
            if resp.status_code != 200:
                if attempt == self.retries - 1:
                    return _HourResult(hour, None, f"HTTP {resp.status_code}")
                time.sleep(self.retry_backoff**attempt)
                continue
            if not resp.content:
                return _HourResult(hour, np.empty((0, 5), dtype=np.float64))
            try:
                return _HourResult(hour, decode_bi5(resp.content))
            except lzma.LZMAError as exc:
                return _HourResult(hour, None, f"lzma: {exc}")
        return _HourResult(hour, None, "exhausted retries")

    # ------------------------------------------------------------------ #
    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        symbol = symbol.upper()
        hours = pd.date_range(
            pd.Timestamp(start).tz_convert("UTC").floor("h"),
            pd.Timestamp(end).tz_convert("UTC").ceil("h"),
            freq="h",
            tz="UTC",
        )
        if len(hours) == 0:
            return empty_bars()

        log.info("dukascopy: fetching %d hourly tick files for %s", len(hours), symbol)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(lambda h: self._fetch_hour(symbol, h), hours))

        failures = [r for r in results if r.error]
        if failures:
            # Missing hours silently become missing bars, which silently becomes a
            # backtest over a different period than the one requested.
            sample = ", ".join(f"{r.hour}: {r.error}" for r in failures[:5])
            raise RuntimeError(
                f"dukascopy: {len(failures)} of {len(hours)} hourly files failed to "
                f"download ({sample}). Refusing to return a partial series."
            )

        frames = [
            _ticks_to_frame(r.ticks, r.hour, symbol)
            for r in results
            if r.ticks is not None and len(r.ticks)
        ]
        if not frames:
            return empty_bars()
        ticks = pd.concat(frames).sort_index()

        _validate_prices(ticks["mid"], symbol)
        bars = ticks_to_minute_bars(ticks)
        bars = bars[(bars.index >= start) & (bars.index <= end)]
        return normalise_bars(bars)

    def fetch_ticks(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Raw ticks with bid, ask and mid. Used to measure the real spread."""
        symbol = symbol.upper()
        hours = pd.date_range(
            pd.Timestamp(start).tz_convert("UTC").floor("h"),
            pd.Timestamp(end).tz_convert("UTC").ceil("h"),
            freq="h",
            tz="UTC",
        )
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(lambda h: self._fetch_hour(symbol, h), hours))
        frames = [
            _ticks_to_frame(r.ticks, r.hour, symbol)
            for r in results
            if r.ticks is not None and len(r.ticks)
        ]
        if not frames:
            return pd.DataFrame(columns=["bid", "ask", "mid", "spread", "volume"])
        return pd.concat(frames).sort_index()


def decode_bi5(payload: bytes) -> np.ndarray:
    """Decompress and unpack one hourly tick file into an (n, 5) float array.

    Columns: millisecond offset, ask, bid, ask volume, bid volume. Prices are the
    raw integers; scaling happens in ``_ticks_to_frame`` where the instrument is
    known.
    """
    raw = lzma.decompress(payload)
    count, remainder = divmod(len(raw), _RECORD.size)
    if remainder:
        raise ValueError(
            f"tick payload of {len(raw)} bytes is not a multiple of the "
            f"{_RECORD.size}-byte record size; the file format has changed"
        )
    out = np.empty((count, 5), dtype=np.float64)
    for i, values in enumerate(_RECORD.iter_unpack(raw)):
        out[i] = values
    return out


def _ticks_to_frame(ticks: np.ndarray, hour: pd.Timestamp, symbol: str) -> pd.DataFrame:
    scale = 10.0 ** POINT_DIGITS.get(symbol.upper(), 5)
    stamps = hour + pd.to_timedelta(ticks[:, 0], unit="ms")
    ask = ticks[:, 1] / scale
    bid = ticks[:, 2] / scale
    frame = pd.DataFrame(
        {
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2.0,
            "spread": ask - bid,
            "volume": ticks[:, 3] + ticks[:, 4],
        },
        index=pd.DatetimeIndex(stamps, name="timestamp"),
    )
    return frame


def _validate_prices(mid: pd.Series, symbol: str) -> None:
    validate_price_range(mid, symbol, source="dukascopy")


def ticks_to_minute_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ticks into left-closed, left-labelled 1-minute bars.

    Also carries ``mean_spread`` so the assumed spread in the cost model can be
    compared against what the tape actually offered.
    """
    minute = ticks.index.floor("min")
    grouped = ticks.groupby(minute)
    bars = grouped.agg(
        open=("mid", "first"),
        high=("mid", "max"),
        low=("mid", "min"),
        close=("mid", "last"),
        volume=("volume", "sum"),
        mean_spread=("spread", "mean"),
        max_spread=("spread", "max"),
        n_ticks=("mid", "size"),
    )
    bars.index.name = "timestamp"
    return bars
