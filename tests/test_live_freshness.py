"""The cache must not serve stale bars to a live poller.

The bug this file exists to prevent: ``missing_months`` tolerated a day of
staleness at the cached edge, which is correct for a backtest and silently fatal
for the alert runner. A poller asking for bars up to *now* every five minutes was
told the current month was already covered, and was served the same frame for up
to twenty-four hours. It reported a stale feed once and then evaluated nothing.

The first test below is the regression, written as the observation that found it:
call the loader repeatedly at advancing wall-clock times and watch whether the
newest bar moves.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.cache import CacheKey, ParquetBarCache
from src.data.loader import load_base_bars, load_dataset


def _load_at(cfg, now: pd.Timestamp, *, freshness=None, adapter=None):
    start = (now - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    scoped = cfg.with_overrides({
        "data.start": start,
        "data.end": now.strftime("%Y-%m-%d %H:%M:%S"),
    })
    return load_dataset(scoped, adapter=adapter, freshness=freshness).bars


@pytest.fixture
def live_cfg(cfg):
    # 1-minute run resolution keeps the assertions about bar ages exact.
    return cfg.with_overrides({"data.resolution": "1min", "macro.enabled": False})


def test_live_load_follows_the_clock(live_cfg):
    """With a one-bar freshness bound, the newest bar tracks 'now'."""
    t0 = pd.Timestamp("2023-03-10 12:00", tz="UTC")
    freshness = pd.Timedelta("1min")

    ages = {}
    for delta in ["0min", "5min", "60min", "6h"]:
        now = t0 + pd.Timedelta(delta)
        bars = _load_at(live_cfg, now, freshness=freshness)
        ages[delta] = (now - bars.index[-1]).total_seconds() / 60.0

    # Every poll sees a bar no older than a couple of minutes. Before the fix
    # these were 0, 5, 60 and 360 minutes.
    for delta, age in ages.items():
        assert age <= 2.0, f"at +{delta} the newest bar was {age:.0f} minutes old: {ages}"


def test_backtest_default_still_tolerates_a_partial_day(live_cfg):
    """The research path must not start refetching a month for the final day.

    The one-day tolerance is deliberate there, and removing it globally would
    have made every backtest refetch its final month on every run.
    """
    t0 = pd.Timestamp("2023-03-10 12:00", tz="UTC")
    _load_at(live_cfg, t0)  # populate

    cache = ParquetBarCache.from_config(live_cfg)
    key = CacheKey(
        adapter="synthetic-seed7-p1800.0-v0.15-d0.0",
        symbol="XAUUSD",
        resolution="1min",
    )
    # Whatever the namespace resolves to, use the one actually on disk.
    roots = [p for p in cache.root.iterdir() if p.is_dir()] if cache.root.exists() else []
    assert roots, "expected the load above to have written a cache directory"
    key = CacheKey(adapter=roots[0].name, symbol="XAUUSD", resolution="1min")

    start = t0 - pd.Timedelta(days=30)
    an_hour_later = t0 + pd.Timedelta(hours=1)

    assert cache.missing_months(key, start, an_hour_later) == [], (
        "default (research) freshness should treat an hour-old edge as covered"
    )
    assert cache.missing_months(key, start, an_hour_later, freshness=pd.Timedelta("1min")), (
        "a one-minute freshness bound should mark the current month stale"
    )


def test_live_refetch_asks_only_for_the_gap(live_cfg):
    """A poller must not refetch a whole month every few minutes.

    Against a rate-limited feed that is the difference between working and being
    throttled out within the hour.
    """
    from src.data.synthetic import SyntheticAdapter
    from src.data.sessions import SessionCalendar

    class RecordingAdapter(SyntheticAdapter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        def fetch(self, symbol, start, end):
            self.calls.append((start, end))
            return super().fetch(symbol, start, end)

    adapter = RecordingAdapter(seed=7, calendar=SessionCalendar.from_config(live_cfg))
    t0 = pd.Timestamp("2023-03-10 12:00", tz="UTC")
    freshness = pd.Timedelta("1min")

    _load_at(live_cfg, t0, freshness=freshness, adapter=adapter)
    first_pass = len(adapter.calls)
    assert first_pass > 0

    adapter.calls.clear()
    _load_at(live_cfg, t0 + pd.Timedelta(minutes=30), freshness=freshness, adapter=adapter)

    assert adapter.calls, "the second poll should have fetched the new tail"
    spans = [(end - start) for start, end in adapter.calls]
    assert max(spans) <= pd.Timedelta(hours=2), (
        f"expected a small gap fetch, got spans {spans}. Refetching the whole "
        "month on every poll would exhaust a rate-limited feed."
    )


def test_freshness_is_threaded_through_load_base_bars(live_cfg):
    """Guards the plumbing, so the parameter cannot be quietly dropped."""
    t0 = pd.Timestamp("2023-03-10 12:00", tz="UTC")
    scoped = live_cfg.with_overrides({
        "data.start": (t0 - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        "data.end": t0.strftime("%Y-%m-%d %H:%M:%S"),
    })
    bars, _ = load_base_bars(scoped, freshness=pd.Timedelta("1min"))
    assert not bars.empty

    later = t0 + pd.Timedelta(hours=3)
    scoped_later = live_cfg.with_overrides({
        "data.start": (t0 - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        "data.end": later.strftime("%Y-%m-%d %H:%M:%S"),
    })
    fresh, _ = load_base_bars(scoped_later, freshness=pd.Timedelta("1min"))
    assert fresh.index[-1] > bars.index[-1], "freshness bound did not trigger a refetch"
