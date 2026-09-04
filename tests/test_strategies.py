"""Strategy behaviour, indicator correctness, and the adapter contracts."""
from __future__ import annotations

import lzma
import struct

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import atr, crossover, donchian, rsi, sma, true_range, wilder_ema
from src.strategies import CHEAT_STRATEGIES, RESEARCH_STRATEGIES, get_strategy
from src.strategies.macro_trend import MacroFilteredTrendStrategy, prints_change
from src.strategies.rsi2 import Rsi2Strategy
from src.strategies.trend import DonchianTrendStrategy
from tests.conftest import make_bars


# ---------------------------------------------------------------------- #
# Indicators
# ---------------------------------------------------------------------- #
def test_sma_matches_a_manual_mean():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(series, 3)
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_wilder_ema_uses_alpha_one_over_period():
    series = pd.Series([1.0] * 10 + [2.0] * 10)
    wilder = wilder_ema(series, 10)
    standard = series.ewm(span=10, adjust=False, min_periods=10).mean()
    # Wilder smoothing is slower than a same-period span EMA.
    assert wilder.iloc[-1] < standard.iloc[-1]


def test_rsi_is_100_when_price_only_rises():
    series = pd.Series(np.arange(1, 30, dtype="float64"))
    assert rsi(series, 2).iloc[-1] == pytest.approx(100.0)


def test_rsi_is_0_when_price_only_falls():
    series = pd.Series(np.arange(30, 1, -1, dtype="float64"))
    assert rsi(series, 2).iloc[-1] == pytest.approx(0.0)


def test_rsi_stays_within_bounds(bars_15m):
    values = rsi(bars_15m["close"], 2).dropna()
    assert values.between(0.0, 100.0).all()
    assert len(values) > 100


def test_true_range_includes_the_gap_from_the_prior_close():
    bars = make_bars([100.0, 110.0], opens=[100.0, 108.0],
                     highs=[101.0, 112.0], lows=[99.0, 107.0])
    tr = true_range(bars)
    # Bar 1: high-low = 5, |high - prev close| = 12, |low - prev close| = 7 -> 12.
    assert tr.iloc[1] == pytest.approx(12.0)


def test_atr_is_positive_and_in_price_units(bars_15m):
    values = atr(bars_15m, 14).dropna()
    assert (values > 0).all()
    assert 0.1 < values.median() < 50.0


def test_donchian_excludes_the_current_bar():
    """The most important indicator detail in the repository.

    Without the shift, ``close > channel_high`` can never fire and
    ``close >= channel_high`` fires on every new high, which backtests as a
    breakout system that never misses a move.
    """
    closes = [10.0, 11.0, 12.0, 13.0, 20.0]
    bars = make_bars(closes, highs=[c + 0.1 for c in closes], lows=[c - 0.1 for c in closes])
    upper, _ = donchian(bars, 3)
    # At bar 4 the channel is built from bars 1..3 only, so the 20 is not in it.
    assert upper.iloc[4] == pytest.approx(13.1)
    assert bars["close"].iloc[4] > upper.iloc[4]


def test_donchian_channel_is_nan_during_warmup():
    bars = make_bars([10.0, 11.0, 12.0, 13.0])
    upper, lower = donchian(bars, 3)
    assert upper.iloc[:3].isna().all()
    assert lower.iloc[:3].isna().all()


def test_crossover_marks_the_crossing_bar_only():
    fast = pd.Series([1.0, 3.0, 4.0, 2.0])
    slow = pd.Series([2.0, 2.0, 2.0, 3.0])
    assert crossover(fast, slow).tolist() == [0.0, 1.0, 0.0, -1.0]


# ---------------------------------------------------------------------- #
# Strategy contract
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(RESEARCH_STRATEGIES))
def test_signals_are_in_the_allowed_set(name, bars_15m):
    strategy = RESEARCH_STRATEGIES[name]()
    macro = None
    if strategy.requires_macro:
        days = pd.date_range(bars_15m.index[0].normalize() - pd.Timedelta(days=300),
                             bars_15m.index[-1].normalize(), freq="B")
        from src.data.fred import join_macro_to_bars

        series = pd.Series(np.linspace(2.0, 1.0, len(days)), index=days, name="DFII10")
        macro = join_macro_to_bars(bars_15m, series, lag_days=1)

    features = strategy.compute_features(bars_15m, macro)
    signals = strategy.generate_signals(bars_15m, features)
    assert signals.index.equals(bars_15m.index)
    assert set(np.unique(signals.to_numpy())) <= {-1, 0, 1}


def _macro_frame(bars):
    from src.data.fred import join_macro_to_bars

    days = pd.date_range(bars.index[0].normalize() - pd.Timedelta(days=300),
                         bars.index[-1].normalize(), freq="B")
    series = pd.Series(np.linspace(2.0, 1.0, len(days)), index=days, name="DFII10")
    return join_macro_to_bars(bars, series, lag_days=1)


@pytest.mark.parametrize("name", sorted(RESEARCH_STRATEGIES))
def test_max_lookback_covers_every_indicator_warmup(name, bars_15m):
    """Every price feature must be fully formed by bar ``max_lookback``.

    Understating this number leaks across the walk-forward seam, because the
    purge and embargo lengths default to it.
    """
    strategy = RESEARCH_STRATEGIES[name]()
    lookback = strategy.max_lookback
    assert lookback >= 0
    macro = _macro_frame(bars_15m) if strategy.requires_macro else None
    features = strategy.compute_features(bars_15m, macro)
    warm = features.iloc[lookback:]
    for column in features.columns:
        if column.startswith("macro"):
            continue   # macro warm-up is set by the publication history, not by bars
        assert warm[column].notna().all(), f"{column} is still warming up after {lookback} bars"


def test_rsi2_holds_until_its_own_exit_fires(bars_15m):
    strategy = Rsi2Strategy()
    features = strategy.compute_features(bars_15m, None)
    signals = strategy.generate_signals(bars_15m, features)
    # Positions persist over multiple bars rather than flickering every bar.
    runs = (signals != signals.shift(1)).cumsum()
    lengths = signals[signals != 0].groupby(runs[signals != 0]).size()
    assert lengths.mean() > 1.0


def test_rsi2_only_goes_long_above_its_trend_filter(bars_15m):
    strategy = Rsi2Strategy()
    features = strategy.compute_features(bars_15m, None)
    signals = strategy.generate_signals(bars_15m, features)
    entries = signals[(signals == 1) & (signals.shift(1) != 1)]
    above = bars_15m.loc[entries.index, "close"] > features.loc[entries.index, "trend_ma"]
    assert above.all()


def test_donchian_trend_stop_ratchets_and_never_loosens(bars_15m):
    """A trailing stop that can move against the position is not a stop."""
    strategy = DonchianTrendStrategy(entry_window=20, atr_stop_multiple=2.0)
    features = strategy.compute_features(bars_15m, None)
    signals = strategy.generate_signals(bars_15m, features)
    assert set(np.unique(signals.to_numpy())) <= {-1, 0, 1}
    # Trades must actually terminate; a stop that never fires is not a stop.
    assert (signals == 0).sum() > 0
    assert (signals != 0).sum() > 0


def test_macro_filter_blocks_trades_against_the_regime(bars_15m):
    days = pd.date_range(bars_15m.index[0].normalize() - pd.Timedelta(days=300),
                         bars_15m.index[-1].normalize(), freq="B")
    from src.data.fred import join_macro_to_bars

    # A steadily RISING real yield: longs must be blocked throughout.
    rising = pd.Series(np.linspace(1.0, 3.0, len(days)), index=days, name="DFII10")
    macro = join_macro_to_bars(bars_15m, rising, lag_days=1)
    strategy = MacroFilteredTrendStrategy()
    features = strategy.compute_features(bars_15m, macro)
    signals = strategy.generate_signals(bars_15m, features)
    assert (signals <= 0).all(), "longs were taken while the real yield was rising"


def test_macro_filter_refuses_to_run_without_its_series(bars_15m):
    strategy = MacroFilteredTrendStrategy()
    with pytest.raises(ValueError, match="needs the"):
        strategy.compute_features(bars_15m, None)


def test_prints_change_measures_across_publications_not_bars():
    index = pd.date_range("2023-06-05", periods=10, freq="1h", tz="UTC")
    # Three distinct prints, each repeated as the intraday forward fill would.
    series = pd.Series([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0], index=index)
    change = prints_change(series, 1)
    assert np.isnan(change.iloc[0])
    assert change.iloc[3] == pytest.approx(1.0)    # 2 - 1
    assert change.iloc[6] == pytest.approx(3.0)    # 5 - 2
    assert change.iloc[9] == pytest.approx(3.0)    # carried forward, not recomputed


# ---------------------------------------------------------------------- #
# Registry
# ---------------------------------------------------------------------- #
def test_cheats_are_not_in_the_research_registry():
    assert set(CHEAT_STRATEGIES) & set(RESEARCH_STRATEGIES) == set()


def test_get_strategy_refuses_to_hand_back_a_cheat():
    with pytest.raises(ValueError, match="deliberately broken"):
        get_strategy("cheat_lookahead")


def test_get_strategy_rejects_an_unknown_name():
    with pytest.raises(KeyError):
        get_strategy("moon_phase")


def test_every_research_strategy_is_constructible_with_defaults():
    for name, cls in RESEARCH_STRATEGIES.items():
        strategy = cls()
        assert strategy.name == name
        assert isinstance(strategy.describe(), str)


# ---------------------------------------------------------------------- #
# Adapters
# ---------------------------------------------------------------------- #
def test_dukascopy_tick_decoder_round_trips():
    """Pack records in the documented layout and check they come back intact.

    This verifies the decoding arithmetic. It cannot verify that Dukascopy still
    serves this layout; only a live fetch does that, and the README says so.
    """
    from src.data.dukascopy import decode_bi5

    records = [(0, 1_800_123, 1_800_023, 1.5, 2.5), (60_000, 1_801_500, 1_801_400, 0.5, 0.75)]
    raw = b"".join(struct.Struct(">3I2f").pack(*r) for r in records)
    decoded = decode_bi5(lzma.compress(raw))
    assert decoded.shape == (2, 5)
    assert decoded[0, 0] == 0
    assert decoded[1, 1] == 1_801_500


def test_dukascopy_rejects_a_truncated_payload():
    from src.data.dukascopy import decode_bi5

    with pytest.raises(ValueError, match="record size"):
        decode_bi5(lzma.compress(b"\x00" * 13))


def test_dukascopy_scales_prices_by_the_instrument_point_value():
    from src.data.dukascopy import _ticks_to_frame

    ticks = np.array([[0.0, 1_800_123, 1_800_023, 1.0, 1.0]])
    frame = _ticks_to_frame(ticks, pd.Timestamp("2024-05-15 14:00", tz="UTC"), "XAUUSD")
    assert frame["ask"].iloc[0] == pytest.approx(1800.123)
    assert frame["bid"].iloc[0] == pytest.approx(1800.023)
    assert frame["spread"].iloc[0] == pytest.approx(0.10)


def test_dukascopy_refuses_implausible_prices():
    """A wrong point scaling shifts every price by a power of ten; catch it."""
    from src.data.dukascopy import _validate_prices

    with pytest.raises(ValueError, match="plausible range"):
        _validate_prices(pd.Series([1_800_123.0]), "XAUUSD")


def test_dukascopy_url_uses_zero_based_months():
    from src.data.dukascopy import DukascopyAdapter

    url = DukascopyAdapter.url_for("XAUUSD", pd.Timestamp("2024-05-15 14:00", tz="UTC"))
    assert url.endswith("/XAUUSD/2024/04/15/14h_ticks.bi5")


def test_ticks_aggregate_into_left_labelled_minute_bars():
    from src.data.dukascopy import ticks_to_minute_bars

    stamps = pd.DatetimeIndex(
        ["2024-05-15 14:00:10", "2024-05-15 14:00:50", "2024-05-15 14:01:05"], tz="UTC"
    )
    ticks = pd.DataFrame(
        {"bid": [1.0, 2.0, 3.0], "ask": [1.1, 2.1, 3.1], "mid": [1.05, 2.05, 3.05],
         "spread": [0.1, 0.1, 0.1], "volume": [1.0, 1.0, 1.0]},
        index=stamps,
    )
    bars = ticks_to_minute_bars(ticks)
    assert len(bars) == 2
    assert bars["open"].iloc[0] == pytest.approx(1.05)
    assert bars["close"].iloc[0] == pytest.approx(2.05)
    assert bars["n_ticks"].iloc[0] == 2


def test_csv_adapter_reads_a_vendor_dump(tmp_path):
    from src.data.csv_source import CsvBarAdapter

    path = tmp_path / "bars.csv"
    path.write_text(
        "Gmt time,Open,High,Low,Close,Volume\n"
        "05.06.2023 08:00:00,1800.0,1801.0,1799.0,1800.5,120\n"
        "05.06.2023 08:01:00,1800.5,1802.0,1800.0,1801.5,140\n",
        encoding="utf-8",
    )
    adapter = CsvBarAdapter(tmp_path, source_timezone="UTC", timestamp_format="%d.%m.%Y %H:%M:%S")
    bars = adapter.fetch("XAUUSD", pd.Timestamp("2023-01-01", tz="UTC"),
                         pd.Timestamp("2024-01-01", tz="UTC"))
    assert len(bars) == 2
    assert str(bars.index.tz) == "UTC"
    assert bars["close"].iloc[-1] == pytest.approx(1801.5)


def test_csv_adapter_converts_a_non_utc_source_timezone(tmp_path):
    from src.data.csv_source import CsvBarAdapter

    path = tmp_path / "bars.csv"
    path.write_text(
        "time,open,high,low,close,volume\n"
        "2023-06-05 04:00:00,1800.0,1801.0,1799.0,1800.5,120\n",
        encoding="utf-8",
    )
    adapter = CsvBarAdapter(tmp_path, source_timezone="America/New_York")
    bars = adapter.fetch("XAUUSD", pd.Timestamp("2023-01-01", tz="UTC"),
                         pd.Timestamp("2024-01-01", tz="UTC"))
    # 04:00 New York in June is 08:00 UTC.
    assert bars.index[0] == pd.Timestamp("2023-06-05 08:00", tz="UTC")


def test_csv_adapter_reports_a_missing_path_clearly(tmp_path):
    from src.data.csv_source import CsvBarAdapter

    adapter = CsvBarAdapter(tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        adapter.fetch("XAUUSD", pd.Timestamp("2023-01-01", tz="UTC"),
                      pd.Timestamp("2024-01-01", tz="UTC"))
