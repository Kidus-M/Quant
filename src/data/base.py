"""The adapter interface every data source implements.

Strategy code never imports an adapter. It receives a bar frame in a fixed shape,
so swapping Dukascopy for a CSV dump touches exactly one line of config.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]
INDEX_NAME = "timestamp"

# Columns an adapter may supply in addition to OHLCV, which survive the cache and
# the resampler. ``spread`` is the measured ask-minus-bid at the bar open, in
# quote currency per unit. It exists so the single most load-bearing assumption in
# the cost model can be checked against the tape rather than asserted. Anything
# not listed here is dropped at the cache boundary.
MEASURED = ["spread"]


@dataclass(frozen=True)
class DataProvenance:
    """Where the bars came from.

    Carried into every report so that a synthetic-data run can never be mistaken
    for evidence about the real market.
    """

    adapter: str
    symbol: str
    native_resolution: str
    is_synthetic: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        base = f"{self.symbol} {self.native_resolution} via {self.adapter}"
        if self.is_synthetic:
            base += "  **SYNTHETIC DATA - results are not evidence about the market**"
        return base


class BarAdapter(ABC):
    """Fetches raw bars at the native resolution of the source.

    Contract for ``fetch``:
      * returns a DataFrame indexed by a tz-aware UTC DatetimeIndex named ``timestamp``
      * columns exactly ``open, high, low, close, volume``, all float64
      * sorted ascending, with no obligation to be free of duplicates or bad
        prints. That is the job of the quality layer, and hiding bad ticks inside
        an adapter is how data problems become invisible.
    """

    name: str = "base"
    native_resolution: str = "1min"
    is_synthetic: bool = False

    @abstractmethod
    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        ...

    @property
    def cache_namespace(self) -> str:
        """Directory the cache stores this adapter bars under.

        Any adapter setting that changes the *data* must appear here. The cache is
        keyed on symbol, resolution and this namespace, so an adapter that returns
        different bars for different settings and reports a constant namespace
        will serve one seed results for every other seed.
        """
        return self.name

    def provenance(self, symbol: str) -> DataProvenance:
        return DataProvenance(
            adapter=self.name,
            symbol=symbol,
            native_resolution=self.native_resolution,
            is_synthetic=self.is_synthetic,
        )


def normalise_bars(df: pd.DataFrame, *, sort: bool = True) -> pd.DataFrame:
    """Coerce a raw frame into the canonical bar shape, or raise.

    Deliberately strict. A timezone-naive index is rejected outright rather than
    localised with a guess, because guessing the timezone of gold data is a good
    way to shift every session by hours and never notice.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("adapter must return a DataFrame")
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"bar index must be a DatetimeIndex, got {type(df.index).__name__}")
    if df.index.tz is None:
        raise ValueError("bar index must be timezone-aware UTC; refusing to guess a timezone")
    if str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")
    df.index.name = INDEX_NAME

    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"bar frame missing required columns: {missing}")
    keep = OHLCV + [c for c in df.columns if c not in OHLCV]
    df = df[keep]
    df[OHLCV] = df[OHLCV].astype("float64")

    if sort:
        df = df.sort_index(kind="stable")
    return df


# A price outside this band means the decoding or scaling is wrong, not that the
# market moved. Shared by every adapter that parses a numeric wire format, because
# a silent factor-of-ten error produces a backtest that looks entirely plausible.
PLAUSIBLE_PRICE_RANGE = {
    "XAUUSD": (100.0, 20000.0),
    "XAGUSD": (2.0, 500.0),
    "EURUSD": (0.5, 2.0),
    "GBPUSD": (0.5, 3.0),
    "USDJPY": (50.0, 400.0),
}


def validate_price_range(prices: pd.Series, symbol: str, *, source: str = "adapter") -> None:
    """Raise if decoded prices are implausible for the instrument."""
    lo, hi = PLAUSIBLE_PRICE_RANGE.get(symbol.upper(), (1e-6, 1e9))
    observed_lo, observed_hi = float(prices.min()), float(prices.max())
    if not (lo <= observed_lo and observed_hi <= hi):
        raise ValueError(
            f"{source}: decoded {symbol} prices span {observed_lo:.4f}-{observed_hi:.4f}, "
            f"outside the plausible range {lo}-{hi}. The price scaling is probably "
            "wrong for this instrument; refusing to cache the result."
        )


def empty_bars() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name=INDEX_NAME)
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in OHLCV}, index=idx)
