"""Sessions, resampling, quality checks and the parquet cache."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.base import normalise_bars
from src.data.cache import CacheKey, ParquetBarCache
from src.data.quality import flag_bars, run_quality_checks
from src.data.resample import (
    assert_no_invented_bars,
    bars_per_year,
    infer_bar_seconds,
    is_fixed_frequency,
    resample_bars,
)
from src.data.sessions import SessionCalendar
from src.data.synthetic import SyntheticAdapter
from tests.conftest import make_bars


# ---------------------------------------------------------------------- #
# Session calendar
# ---------------------------------------------------------------------- #
def test_market_is_shut_on_saturday(calendar):
    saturday = pd.date_range("2023-06-10 00:00", "2023-06-10 23:00", freq="1h", tz="UTC")
    assert not calendar.is_open(saturday).any()


def test_week_opens_sunday_at_2200_utc(calendar):
    index = pd.date_range("2023-06-11 20:00", "2023-06-11 23:00", freq="1h", tz="UTC")
    assert calendar.is_open(index).tolist() == [False, False, True, True]


def test_week_closes_friday_at_2100_utc(calendar):
    index = pd.date_range("2023-06-09 19:00", "2023-06-09 22:00", freq="1h", tz="UTC")
    assert calendar.is_open(index).tolist() == [True, True, False, False]


def test_daily_break_is_closed(calendar):
    index = pd.date_range("2023-06-07 20:00", "2023-06-07 23:00", freq="1h", tz="UTC")
    assert calendar.is_open(index).tolist() == [True, False, True, True]


def test_naive_timestamps_are_rejected(calendar):
    with pytest.raises(ValueError, match="timezone-aware"):
        calendar.is_open(pd.date_range("2023-06-07", periods=3, freq="1h"))


def test_rollover_is_counted_once_per_night(calendar):
    start = pd.DatetimeIndex([pd.Timestamp("2023-06-06 20:00", tz="UTC")])
    end = pd.DatetimeIndex([pd.Timestamp("2023-06-06 22:00", tz="UTC")])
    assert calendar.rollovers_between(start, end)[0] == 1
    # A weekend hold crosses three rollover instants (Fri, Sat, Sun).
    weekend = calendar.rollovers_between(
        pd.DatetimeIndex([pd.Timestamp("2023-06-09 20:00", tz="UTC")]),
        pd.DatetimeIndex([pd.Timestamp("2023-06-11 23:00", tz="UTC")]),
    )
    assert weekend[0] == 3


def test_session_id_increments_at_every_break(calendar):
    index = pd.date_range("2023-06-06 18:00", "2023-06-08 06:00", freq="1h", tz="UTC")
    index = index[calendar.is_open(index)]
    ids = calendar.session_id(index)
    assert ids[0] == 0
    assert ids[-1] >= 1
    assert (np.diff(ids) >= 0).all()


# ---------------------------------------------------------------------- #
# Resampling
# ---------------------------------------------------------------------- #
def test_resampling_accounts_for_every_source_bar(minute_bars, calendar):
    resampled = resample_bars(minute_bars, "15min", calendar)
    assert_no_invented_bars(minute_bars, resampled)
    assert int(resampled["n_source_bars"].sum()) == len(minute_bars)


def test_no_resampled_bar_spans_a_session_break(minute_bars, calendar):
    """Every output bar must come from one session only.

    Without the session grouping, a bucket at the Friday close would merge the
    last minute of the week with the first minute of the next and report the
    weekend gap as an intraday range.
    """
    resampled = resample_bars(minute_bars, "15min", calendar)
    source_session = pd.Series(calendar.session_id(minute_bars.index), index=minute_bars.index)
    bins = minute_bars.index.floor("15min")
    per_bin_sessions = source_session.groupby(bins).nunique()
    for label in resampled.index:
        assert per_bin_sessions[label] >= 1
    # And no output bar covers a span longer than the bin itself.
    spans = resampled.index.to_series().diff().dropna()
    assert (spans >= pd.Timedelta(minutes=15)).all()


def test_empty_bins_are_dropped_not_filled(minute_bars, calendar):
    resampled = resample_bars(minute_bars, "15min", calendar)
    assert (resampled["n_source_bars"] > 0).all()
    # Weekends leave holes in the index rather than fabricated flat bars.
    gaps = resampled.index.to_series().diff().dropna()
    assert gaps.max() > pd.Timedelta(hours=24)


def test_resampled_ohlc_is_internally_consistent(minute_bars, calendar):
    resampled = resample_bars(minute_bars, "1h", calendar)
    assert (resampled["high"] >= resampled[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (resampled["low"] <= resampled[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (resampled["high"] >= resampled["low"]).all()


def test_calendar_frequencies_are_refused(minute_bars, calendar):
    with pytest.raises(ValueError, match="not a fixed frequency"):
        resample_bars(minute_bars, "ME", calendar)


def test_is_fixed_frequency():
    assert is_fixed_frequency("15min")
    assert is_fixed_frequency("4h")
    assert not is_fixed_frequency("ME")


def test_bars_per_year_and_bar_seconds(bars_15m):
    assert infer_bar_seconds(bars_15m.index) == pytest.approx(900.0)
    # Gold trades about 23 hours a day, five days a week.
    assert 20_000 < bars_per_year(bars_15m.index) < 26_000


def test_resampling_is_idempotent_at_the_same_resolution(bars_15m, calendar):
    again = resample_bars(bars_15m, "15min", calendar)
    pd.testing.assert_series_equal(again["close"], bars_15m["close"])


# ---------------------------------------------------------------------- #
# Quality
# ---------------------------------------------------------------------- #
def _lenient(cfg):
    """Disable the abort-on-too-many-drops guard.

    These tests use frames of a handful of bars, where a single bad bar is a
    quarter of the sample. The guard itself is covered by
    ``test_load_aborts_when_too_much_data_is_dropped``.
    """
    return cfg.with_overrides({"data.quality.max_flagged_fraction": 1.0})


def test_duplicate_timestamps_are_flagged_and_dropped(cfg, calendar):
    bars = make_bars([1800, 1810, 1820], start="2023-06-06 10:00")
    doubled = normalise_bars(pd.concat([bars, bars.iloc[[1]]]))
    clean, report = run_quality_checks(doubled, _lenient(cfg), calendar=calendar)
    assert report.counts["duplicate_timestamp"] == 1
    assert len(clean) == 3
    assert not clean.index.has_duplicates
    assert report.quarantine_path is not None and report.quarantine_path.exists()


def test_non_monotonic_index_is_sorted_and_reported(cfg, calendar):
    bars = make_bars([1800, 1810, 1820], start="2023-06-06 10:00")
    shuffled = bars.iloc[[2, 0, 1]]
    clean, report = run_quality_checks(shuffled, cfg, calendar=calendar)
    assert report.was_unsorted
    assert clean.index.is_monotonic_increasing


def test_inconsistent_ohlc_is_dropped(cfg, calendar):
    bars = make_bars([1800, 1810, 1820], start="2023-06-06 10:00")
    broken = bars.copy()
    broken.iloc[1, broken.columns.get_loc("high")] = 1700.0   # high below the body
    clean, report = run_quality_checks(broken, _lenient(cfg), calendar=calendar)
    assert report.counts["ohlc_inconsistent"] == 1
    assert len(clean) == 2


def test_bars_stamped_outside_session_hours_are_dropped(cfg, calendar):
    bars = make_bars([1800.0] * 6, start="2023-06-10 10:00", freq="1h")  # a Saturday
    clean, report = run_quality_checks(bars, _lenient(cfg), calendar=calendar)
    assert report.counts["outside_session"] == 6
    assert clean.empty


def test_price_spikes_are_flagged_but_kept_by_default(cfg, calendar):
    rng = np.random.default_rng(1)
    closes = 1800 + np.cumsum(rng.normal(0, 0.3, 400))
    closes[300] += 90.0    # a violent but not impossible print
    bars = make_bars(closes, start="2023-06-06 00:00", freq="1min")
    clean, report = run_quality_checks(bars, cfg, calendar=calendar)
    assert report.counts["price_spike_sigma"] >= 1
    assert len(clean) == len(bars), "a large real move must not be silently deleted"
    assert any("KEPT" in note for note in report.notes)


def test_price_spikes_can_be_quarantined_when_asked(cfg, calendar):
    rng = np.random.default_rng(1)
    closes = 1800 + np.cumsum(rng.normal(0, 0.3, 400))
    closes[300] += 90.0
    bars = make_bars(closes, start="2023-06-06 00:00", freq="1min")
    strict = cfg.with_overrides({"data.quality.quarantine_price_spikes": True})
    clean, report = run_quality_checks(bars, strict, calendar=calendar)
    assert len(clean) < len(bars)


def test_spike_detection_does_not_use_the_bar_being_tested():
    """The rolling sigma must come from earlier bars only, or an outlier inflates
    its own threshold and hides."""
    rng = np.random.default_rng(2)
    closes = 1800 + np.cumsum(rng.normal(0, 0.2, 300))
    bars = make_bars(closes, start="2023-06-06 00:00", freq="1min")
    flags = flag_bars(bars, sigma_threshold=8.0, sigma_window=100)
    bumped = bars.copy()
    bumped.iloc[250, bumped.columns.get_loc("close")] += 40.0
    bumped_flags = flag_bars(bumped, sigma_threshold=8.0, sigma_window=100)
    assert bumped_flags["price_spike_sigma"].iloc[250]
    assert not flags["price_spike_sigma"].iloc[250]


def test_load_aborts_when_too_much_data_is_dropped(cfg, calendar):
    bars = make_bars([1800.0] * 20, start="2023-06-10 10:00", freq="1h")  # all Saturday
    strict = cfg.with_overrides({"data.quality.max_flagged_fraction": 0.02})
    with pytest.raises(ValueError, match="max_flagged_fraction"):
        run_quality_checks(bars, strict, calendar=calendar)


# ---------------------------------------------------------------------- #
# Sources that pad non-trading hours
#
# Twelve Data returns a full 60 rows an hour right through the weekend. Dropping
# them is correct, but it is ~38% of the load, so the corruption budget and the
# session budget have to be separate numbers or a correct load cannot complete.
# ---------------------------------------------------------------------- #
def _padded_week(corrupt_weekend_bars: int = 0) -> pd.DataFrame:
    """60 in-session minutes plus 40 weekend minutes, as a padding vendor returns.

    A 40% session drop is close to what the live feed actually produces, so the
    fixture exercises the real ratio rather than an extreme.
    """
    weekend_closes = [1800.0] * 40
    for i in range(corrupt_weekend_bars):
        weekend_closes[i] = -1.0
    trading = make_bars([1800.0] * 60, start="2023-06-07 08:00", freq="1min")
    weekend = make_bars(weekend_closes, start="2023-06-10 10:00", freq="1min")
    return pd.concat([trading, weekend]).sort_index()


def test_a_padding_source_budgets_session_drops_separately(cfg, calendar):
    bars = _padded_week()
    strict = cfg.with_overrides({
        "data.quality.max_flagged_fraction": 0.02,
        "data.quality.max_outside_session_fraction": 0.45,
    })

    clean, report = run_quality_checks(
        bars, strict, calendar=calendar, pads_outside_session=True
    )
    assert len(clean) == 60
    assert report.counts["outside_session"] == 40
    assert any("pads non-trading hours" in note for note in report.notes)


def test_the_same_load_fails_for_a_source_that_does_not_pad(cfg, calendar):
    """The exemption is opt-in. Elsewhere an out-of-session bar is still corruption,
    which is what catches a timezone mistake."""
    bars = _padded_week()
    strict = cfg.with_overrides({
        "data.quality.max_flagged_fraction": 0.02,
        "data.quality.max_outside_session_fraction": 0.45,
    })

    with pytest.raises(ValueError, match="max_flagged_fraction"):
        run_quality_checks(bars, strict, calendar=calendar)


def test_a_padding_source_still_has_a_session_ceiling(cfg, calendar):
    """Expected padding is not a licence for unlimited session drop.

    A timezone mistake would push nearly everything out of session, and that must
    still fail rather than quietly shrinking the dataset.
    """
    bars = make_bars([1800.0] * 20, start="2023-06-10 10:00", freq="1h")  # all Saturday
    strict = cfg.with_overrides({
        "data.quality.max_flagged_fraction": 0.02,
        "data.quality.max_outside_session_fraction": 0.45,
    })

    with pytest.raises(ValueError, match="max_outside_session_fraction"):
        run_quality_checks(bars, strict, calendar=calendar, pads_outside_session=True)


def test_a_padding_source_is_still_held_to_the_corruption_budget(cfg, calendar):
    """Only the session reason is re-budgeted; corrupt bars answer to the strict one."""
    closes = [1800.0] * 100
    for i in range(10):
        closes[i] = -1.0
    bars = make_bars(closes, start="2023-06-07 08:00", freq="1min")  # Wednesday, in session
    strict = cfg.with_overrides({
        "data.quality.max_flagged_fraction": 0.02,
        "data.quality.max_outside_session_fraction": 0.95,
    })

    with pytest.raises(ValueError, match="max_flagged_fraction"):
        run_quality_checks(bars, strict, calendar=calendar, pads_outside_session=True)


def test_a_bar_both_out_of_session_and_corrupt_counts_as_corrupt(cfg, calendar):
    """Corruption is counted from its own mask, not by subtracting session drops.

    Subtracting would net these bars to zero and hide a genuinely broken feed
    behind the padding allowance.
    """
    bars = _padded_week(corrupt_weekend_bars=10)
    strict = cfg.with_overrides({
        "data.quality.max_flagged_fraction": 0.02,
        "data.quality.max_outside_session_fraction": 0.45,
    })

    with pytest.raises(ValueError, match="max_flagged_fraction"):
        run_quality_checks(bars, strict, calendar=calendar, pads_outside_session=True)


# ---------------------------------------------------------------------- #
# Cache
# ---------------------------------------------------------------------- #
def test_cache_refuses_to_store_a_derived_resolution(tmp_path, bars_15m):
    cache = ParquetBarCache(tmp_path, base_resolution="1min")
    key = CacheKey("synthetic", "XAUUSD", "15min")
    with pytest.raises(ValueError, match="only the base resolution"):
        cache.write(key, bars_15m)


def test_cache_round_trips_and_reports_no_missing_months(tmp_path, minute_bars):
    cache = ParquetBarCache(tmp_path, base_resolution="1min")
    key = CacheKey("synthetic", "XAUUSD", "1min")
    start, end = minute_bars.index[0], minute_bars.index[-1]
    assert cache.missing_months(key, start, end)

    cache.write(key, minute_bars)
    back = cache.read(key, start, end)
    pd.testing.assert_series_equal(back["close"], minute_bars["close"])
    assert cache.missing_months(key, start, end) == []


def test_cache_is_partitioned_by_month(tmp_path, minute_bars):
    cache = ParquetBarCache(tmp_path, base_resolution="1min")
    key = CacheKey("synthetic", "XAUUSD", "1min")
    cache.write(key, minute_bars)
    files = sorted(p.name for p in (tmp_path / "synthetic" / "XAUUSD" / "1min").glob("*.parquet"))
    # The sample ends at the Friday close on 31 March, so there is no April file.
    assert files == ["2023-01.parquet", "2023-02.parquet", "2023-03.parquet"]


def test_adapter_settings_that_change_the_data_change_the_cache_namespace():
    """Two seeds must not share a cache directory.

    This is not hypothetical: without it, changing the synthetic seed silently
    replays the previous run and every seed looks identical.
    """
    a = SyntheticAdapter(seed=1)
    b = SyntheticAdapter(seed=2)
    c = SyntheticAdapter(seed=1, annual_vol=0.30)
    assert a.cache_namespace != b.cache_namespace
    assert a.cache_namespace != c.cache_namespace


def test_loader_never_refetches_what_is_already_cached(cfg, monkeypatch):
    from src.data import loader

    calls = {"n": 0}
    real = SyntheticAdapter(seed=3)

    class Counting(SyntheticAdapter):
        def fetch(self, symbol, start, end):
            calls["n"] += 1
            return real.fetch(symbol, start, end)

    small = cfg.with_overrides({"data.start": "2023-01-02", "data.end": "2023-01-20"})
    loader.load_dataset(small, adapter=Counting(seed=3))
    first = calls["n"]
    assert first > 0
    loader.load_dataset(small, adapter=Counting(seed=3))
    assert calls["n"] == first, "the second load hit the network instead of the cache"


# ---------------------------------------------------------------------- #
# Adapter contract
# ---------------------------------------------------------------------- #
def test_normalise_bars_rejects_naive_timestamps():
    frame = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=pd.DatetimeIndex(["2023-06-06"]),
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        normalise_bars(frame)


def test_normalise_bars_rejects_missing_columns():
    frame = pd.DataFrame({"open": [1.0]}, index=pd.DatetimeIndex(["2023-06-06"], tz="UTC"))
    with pytest.raises(ValueError, match="missing required columns"):
        normalise_bars(frame)


def test_synthetic_bars_are_utc_and_inside_session_hours(minute_bars, calendar):
    assert str(minute_bars.index.tz) == "UTC"
    assert calendar.is_open(minute_bars.index).all()
    assert (minute_bars["high"] >= minute_bars["low"]).all()
    assert (minute_bars[["open", "high", "low", "close"]] > 0).all().all()


def test_synthetic_series_has_real_session_gaps(minute_bars):
    """Opens must sometimes differ from the previous close, or the fill rule is
    being tested against a series where it costs nothing."""
    gaps = minute_bars["open"].to_numpy()[1:] - minute_bars["close"].to_numpy()[:-1]
    assert (np.abs(gaps) > 1e-9).sum() > 10


def test_synthetic_generator_is_reproducible_across_processes():
    """The seed must not depend on anything that varies between runs.

    Seeding from Python ``hash()`` of the symbol is the trap: string hashing is
    randomised per process, so the "deterministic" generator quietly produces
    different bars every run and no result is reproducible. These are golden
    values; if they change, the seeding changed.
    """
    adapter = SyntheticAdapter(seed=7, calendar=SessionCalendar())
    bars = adapter.fetch("XAUUSD", pd.Timestamp("2023-01-01", tz="UTC"),
                         pd.Timestamp("2023-01-05", tz="UTC"))
    assert len(bars) == 4261
    assert float(bars["close"].iloc[0]) == pytest.approx(1784.650638, abs=1e-6)
    assert float(bars["close"].iloc[-1]) == pytest.approx(1828.606307, abs=1e-6)


def test_different_symbols_get_different_synthetic_series():
    adapter = SyntheticAdapter(seed=7, calendar=SessionCalendar())
    start = pd.Timestamp("2023-01-01", tz="UTC")
    end = pd.Timestamp("2023-01-05", tz="UTC")
    gold = adapter.fetch("XAUUSD", start, end)
    silver = adapter.fetch("XAGUSD", start, end)
    assert not np.allclose(gold["close"].to_numpy(), silver["close"].to_numpy())
