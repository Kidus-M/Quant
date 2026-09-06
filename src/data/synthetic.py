"""Synthetic 1-minute XAU/USD bars.

This exists so the engine, the tests and the whole report pipeline can run with no
network and no credentials. It is **not** a data source for research. Anything it
produces is flagged ``is_synthetic=True``, that flag rides through the loader into
``DataProvenance``, and the report banner says so at the top. A strategy that looks
profitable on this has demonstrated nothing except that the code runs.

The generator is deliberately close to a driftless random walk with realistic
intraday volatility seasonality, fat tails and occasional jumps, because the useful
property of a null-edge series is that any strategy showing an edge on it is
revealing a bug in the backtester.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from src.data.base import BarAdapter, normalise_bars
from src.data.sessions import SessionCalendar

# Gold realised vol sits around 15% annualised; 252 trading days of ~23 hours.
_ANNUAL_VOL = 0.15
_MINUTES_PER_YEAR = 252 * 23 * 60


def _stable_seed(symbol: str, seed: int) -> int:
    """Process-independent seed derived from the symbol and the configured seed."""
    return (zlib.crc32(symbol.upper().encode("utf-8")) ^ (int(seed) * 2_654_435_761)) % (2**32)


class SyntheticAdapter(BarAdapter):
    name = "synthetic"
    native_resolution = "1min"
    is_synthetic = True

    def __init__(
        self,
        *,
        seed: int = 7,
        start_price: float = 1800.0,
        annual_vol: float = _ANNUAL_VOL,
        annual_drift: float = 0.0,
        calendar: SessionCalendar | None = None,
        jump_probability: float = 2e-5,
        jump_scale: float = 0.0025,
    ):
        self.seed = seed
        self.start_price = start_price
        self.annual_vol = annual_vol
        self.annual_drift = annual_drift
        self.calendar = calendar or SessionCalendar()
        self.jump_probability = jump_probability
        self.jump_scale = jump_scale

    @property
    def cache_namespace(self) -> str:
        # Every setting that changes the generated series belongs in the cache
        # path. Without this, changing the seed silently reuses the old bars.
        return (
            f"synthetic-seed{self.seed}-p{self.start_price:g}"
            f"-vol{self.annual_vol:g}-drift{self.annual_drift:g}"
        )

    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        start = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tz else pd.Timestamp(start).tz_localize("UTC")
        end = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tz else pd.Timestamp(end).tz_localize("UTC")

        minutes = pd.date_range(start, end, freq="1min", tz="UTC", name="timestamp")
        minutes = minutes[self.calendar.is_open(minutes)]
        n = len(minutes)
        if n == 0:
            from src.data.base import empty_bars

            return empty_bars()

        # Seed from the symbol too, so two instruments are not the same series.
        # crc32, not hash(): Python string hashing is randomised per process, so
        # seeding from it makes the "deterministic" generator produce different
        # bars on every run, and no result is reproducible.
        rng = np.random.default_rng(_stable_seed(symbol, self.seed))

        sigma = self.annual_vol / np.sqrt(_MINUTES_PER_YEAR)
        mu = self.annual_drift / _MINUTES_PER_YEAR

        seasonality = self._intraday_seasonality(minutes)
        # Student-t innovations give the fat tails that make stop placement and
        # slippage assumptions bite the way they do on real gold.
        shocks = rng.standard_t(df=4, size=n) / np.sqrt(4 / 2)
        returns = mu + sigma * seasonality * shocks

        jumps = rng.random(n) < self.jump_probability
        returns[jumps] += rng.normal(0.0, self.jump_scale, size=int(jumps.sum()))

        # Session gaps. These are applied between the previous close and this bar
        # open, not inside the bar, so that open != previous close across the
        # weekend and the daily break. That distinction matters: a series where
        # every bar opens exactly at the prior close makes the fill-at-next-open
        # rule free, and hides precisely the risk it exists to model.
        session = self.calendar.session_id(minutes)
        first_of_session = np.empty(n, dtype=bool)
        first_of_session[0] = True
        first_of_session[1:] = session[1:] != session[:-1]
        gaps = np.zeros(n, dtype="float64")
        gaps[first_of_session] = rng.normal(0.0, sigma * 30, size=int(first_of_session.sum()))

        cumulative = np.cumsum(returns + gaps)
        close = self.start_price * np.exp(cumulative)
        open_ = np.empty(n)
        open_[0] = self.start_price * np.exp(gaps[0])
        open_[1:] = self.start_price * np.exp(cumulative[:-1] + gaps[1:])

        # Intrabar range: an exponential draw on top of the open-to-close move keeps
        # true range wider than the body, as it is in practice.
        wick = rng.exponential(0.6, size=n) * sigma * seasonality * close
        high = np.maximum(open_, close) + wick * rng.random(n)
        low = np.minimum(open_, close) - wick * rng.random(n)

        volume = np.round(rng.lognormal(mean=3.2, sigma=0.6, size=n) * seasonality)

        frame = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=minutes,
        )
        return normalise_bars(frame)

    @staticmethod
    def _intraday_seasonality(index: pd.DatetimeIndex) -> np.ndarray:
        """Volatility multiplier by UTC hour.

        Asia is quiet, London (07:00) picks up, the London/NY overlap (13:00-16:00)
        is the busiest stretch, and the late US session fades.
        """
        hour = index.hour.to_numpy()
        table = np.full(24, 0.55)
        table[7:10] = 1.15
        table[10:13] = 1.0
        table[13:16] = 1.45
        table[16:19] = 1.0
        table[19:21] = 0.7
        return table[hour]
