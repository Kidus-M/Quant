"""The XAU/USD trading calendar.

Gold trades roughly Sunday 22:00 UTC to Friday 21:00 UTC with a one-hour daily
break at 21:00-22:00 UTC. Two things depend on getting this right:

1. Resampling must never aggregate across a break, or a "1 hour bar" ends up
   spanning nineteen hours of the weekend and prints a fictitious range.
2. Financing is charged on positions held across the daily rollover.

Everything here is vectorised over a DatetimeIndex because it runs on every bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from src.config import Config


def _parse_time(value: str) -> time:
    hh, _, mm = str(value).partition(":")
    return time(int(hh), int(mm or 0))


@dataclass(frozen=True)
class SessionCalendar:
    week_open_weekday: int = 6          # Sunday, using Monday=0
    week_open_time: time = time(22, 0)
    week_close_weekday: int = 4         # Friday
    week_close_time: time = time(21, 0)
    daily_break_start: time = time(21, 0)
    daily_break_end: time = time(22, 0)
    rollover_hour: int = 21

    @classmethod
    def from_config(cls, cfg: Config) -> "SessionCalendar":
        s = cfg.section("session")
        return cls(
            week_open_weekday=int(s.get("week_open_weekday")),
            week_open_time=_parse_time(s.get("week_open_time_utc")),
            week_close_weekday=int(s.get("week_close_weekday")),
            week_close_time=_parse_time(s.get("week_close_time_utc")),
            daily_break_start=_parse_time(s.get("daily_break_start_utc")),
            daily_break_end=_parse_time(s.get("daily_break_end_utc")),
            rollover_hour=int(s.get("rollover_hour_utc")),
        )

    # ------------------------------------------------------------------ #
    # Open/closed
    # ------------------------------------------------------------------ #
    def is_open(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Boolean mask: is the market open at each timestamp?

        A timestamp on a bar label is the *start* of that bar, so this answers
        "would a bar starting here contain trading activity".
        """
        index = _require_utc(index)
        weekday = index.weekday.to_numpy()
        minutes = index.hour.to_numpy() * 60 + index.minute.to_numpy()

        open_min = self.week_open_time.hour * 60 + self.week_open_time.minute
        close_min = self.week_close_time.hour * 60 + self.week_close_time.minute
        brk_start = self.daily_break_start.hour * 60 + self.daily_break_start.minute
        brk_end = self.daily_break_end.hour * 60 + self.daily_break_end.minute

        in_break = (minutes >= brk_start) & (minutes < brk_end)

        sunday_open = (weekday == self.week_open_weekday) & (minutes >= open_min)
        friday_open = (weekday == self.week_close_weekday) & (minutes < close_min)
        # Full trading days: everything strictly between the weekly open and close
        # weekday, minus the daily break.
        midweek = np.isin(weekday, self._midweek_weekdays()) & ~in_break
        return sunday_open | friday_open | midweek

    def _midweek_weekdays(self) -> list[int]:
        # Monday..Thursday for the standard configuration. Derived rather than
        # hardcoded so an exotic calendar in YAML still behaves.
        days: list[int] = []
        d = (self.week_open_weekday + 1) % 7
        while d != self.week_close_weekday:
            days.append(d)
            d = (d + 1) % 7
        return days

    # ------------------------------------------------------------------ #
    # Session identity -- used to stop the resampler crossing a gap
    # ------------------------------------------------------------------ #
    def session_id(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Integer id that increments at every session boundary.

        Two bars share an id only if no market break separates them, so grouping
        by (session_id, time bin) makes it structurally impossible to build a bar
        out of data from either side of the weekend.
        """
        index = _require_utc(index)
        if len(index) == 0:
            return np.zeros(0, dtype=np.int64)
        open_mask = self.is_open(index)
        # A new session starts at the first bar, at any bar that follows a closed
        # period, and after any physical gap longer than the daily break.
        starts = np.zeros(len(index), dtype=bool)
        starts[0] = True
        starts[1:] |= open_mask[1:] & ~open_mask[:-1]
        boundaries = self.boundary_after(index)
        starts[1:] |= boundaries[:-1]
        return np.cumsum(starts) - 1

    def boundary_after(self, index: pd.DatetimeIndex) -> np.ndarray:
        """True where a market break falls between bar i and bar i+1."""
        index = _require_utc(index)
        n = len(index)
        out = np.zeros(n, dtype=bool)
        if n < 2:
            return out
        # Any calendar break shows up as at least one closed minute between the
        # two bars. Compare the number of open minutes elapsed against the raw
        # elapsed time; if they disagree the pair straddles a break.
        crossings = self.rollovers_between(index[:-1], index[1:])
        out[:-1] = crossings > 0
        return out

    # ------------------------------------------------------------------ #
    # Financing
    # ------------------------------------------------------------------ #
    def rollovers_between(self, start: pd.DatetimeIndex, end: pd.DatetimeIndex) -> np.ndarray:
        """Count rollover instants (daily break starts) in the half-open (start, end].

        Positions held across one of these are charged financing.
        """
        start = _require_utc(pd.DatetimeIndex(start))
        end = _require_utc(pd.DatetimeIndex(end))
        roll = pd.Timedelta(hours=self.rollover_hour)
        # Number of whole rollover instants at or before a timestamp, as a count
        # of days since epoch offset so that the difference is exact.
        # Counted as whole days since the epoch rather than via the integer
        # nanosecond view: the underlying datetime unit is not guaranteed to be
        # nanoseconds, and an off-by-1000 here would silently drop every
        # financing charge.
        epoch = pd.Timestamp("1970-01-01", tz="UTC")

        def _count(idx: pd.DatetimeIndex) -> np.ndarray:
            shifted = (idx - roll).normalize()
            return (shifted - epoch).days.to_numpy()

        return (_count(end) - _count(start)).astype(np.int64)


def _require_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(index)
    if index.tz is None:
        raise ValueError(
            "timestamps must be timezone-aware UTC; a naive index is how a "
            "backtest ends up silently shifted by the local offset"
        )
    if str(index.tz) not in ("UTC", "utc"):
        index = index.tz_convert("UTC")
    return index
