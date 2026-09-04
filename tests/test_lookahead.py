"""Lookahead bias guards.

This is the most important file in the repository. Everything downstream is
worthless if these fail, so each item on the project checklist gets an explicit
test here, and the suite is proved to work by running it against strategies that
cheat on purpose.

Checklist coverage:

1. No indicator uses a bar own close to generate a signal that fills in that bar
   -> ``test_fill_is_next_bar_open`` and ``test_fill_never_uses_high_or_low``
2. No rolling statistic uses ``center=True``           -> ``test_no_centred_windows_in_source``
3. FRED macro lagged at least one day before joining   -> ``test_macro_is_lagged_by_at_least_one_day``
4. Left-closed, left-labelled resampling bins          -> ``test_resample_bins_are_left_closed_left_labelled``
5. No ``.shift(-n)`` outside deliberate label building -> ``test_no_negative_shifts_in_source``
6. Normalisation fitted on the training window only    -> ``test_normaliser_*``
7. Symbol universe is fixed                            -> ``test_symbol_universe_is_a_single_fixed_symbol``

Plus the general guard that subsumes most of them: **truncation invariance**. If a
strategy signal at bar t depends only on bars up to t, then recomputing it on a
truncated series must reproduce exactly the same values. Any peek at the future,
however indirect, breaks that equality.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import _target_from_signals, run_backtest
from src.config import REPO_ROOT
from src.data.fred import available_at, join_macro_to_bars
from src.data.resample import resample_bars
from src.features.normalisation import NotFittedError, RollingZScore, ZScoreNormaliser
from src.strategies import RESEARCH_STRATEGIES
from src.strategies.base import Strategy
from src.strategies.cheating import (
    CentredRollingCheatStrategy,
    FullSampleNormalisationCheatStrategy,
    LookaheadCheatStrategy,
)
from src.strategies.random_entry import RandomEntryStrategy
from tests.conftest import make_bars, make_dataset

SOURCE_FILES = sorted(
    p for p in (REPO_ROOT / "src").rglob("*.py")
    # The cheats are deliberately broken; they are the fixture, not the subject.
    if p.name != "cheating.py"
)


# ---------------------------------------------------------------------- #
# The general guard
# ---------------------------------------------------------------------- #
def signals_for(strategy: Strategy, bars: pd.DataFrame, macro=None) -> pd.Series:
    features = strategy.compute_features(bars, macro)
    return strategy.generate_signals(bars, features)


def truncation_mismatch(strategy: Strategy, bars: pd.DataFrame, cut_points) -> list[int]:
    """Bars at which a truncated recomputation disagrees with the full-series one."""
    full = signals_for(strategy, bars)
    bad = []
    for cut in cut_points:
        truncated = signals_for(strategy, bars.iloc[:cut])
        if not np.array_equal(truncated.to_numpy(), full.iloc[:cut].to_numpy()):
            bad.append(cut)
    return bad


@pytest.mark.parametrize("name", sorted(RESEARCH_STRATEGIES))
def test_research_strategies_are_truncation_invariant(name, bars_15m):
    """A signal computed from bars up to t cannot change when later bars vanish."""
    strategy = RESEARCH_STRATEGIES[name]()
    if strategy.requires_macro:
        pytest.skip("covered by test_macro_strategy_is_truncation_invariant")
    cuts = [800, 1500, 2600, 4000]
    assert truncation_mismatch(strategy, bars_15m, cuts) == []


def test_random_entry_benchmark_is_truncation_invariant(bars_15m):
    strategy = RandomEntryStrategy(entry_probability=0.02, mean_hold_bars=25.0, seed=5)
    assert truncation_mismatch(strategy, bars_15m, [900, 2000, 3300]) == []


def test_macro_strategy_is_truncation_invariant(bars_15m):
    from src.strategies.macro_trend import MacroFilteredTrendStrategy

    macro = _synthetic_macro(bars_15m)
    strategy = MacroFilteredTrendStrategy()
    full = signals_for(strategy, bars_15m, macro)
    for cut in (900, 2000, 3300):
        truncated = signals_for(strategy, bars_15m.iloc[:cut], macro.iloc[:cut])
        assert np.array_equal(truncated.to_numpy(), full.iloc[:cut].to_numpy()), cut


# ---------------------------------------------------------------------- #
# The suite must catch strategies that cheat
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cheat",
    [LookaheadCheatStrategy, CentredRollingCheatStrategy, FullSampleNormalisationCheatStrategy],
    ids=lambda c: c.name,
)
def test_truncation_guard_catches_deliberate_cheats(cheat, bars_15m):
    """If this ever passes, the guard is broken and every result is suspect."""
    mismatches = truncation_mismatch(cheat(), bars_15m, [800, 1500, 2600, 4000])
    assert mismatches, (
        f"{cheat.name} peeks at the future but the truncation guard did not notice. "
        "The guard is broken; do not trust any backtest produced by this engine "
        "until it is fixed."
    )


def test_one_bar_lookahead_cheat_produces_an_absurd_result(dataset, cfg):
    """Sanity anchor: a strategy that knows the next bar makes impossible money.

    This is what a lookahead bug looks like from the outside, and it is worth
    having the number in the test suite so the shape is recognisable.
    """
    honest = run_backtest(dataset, RESEARCH_STRATEGIES["trend_donchian"](), cfg,
                          initial_capital=100_000.0)
    cheat = run_backtest(dataset, LookaheadCheatStrategy(), cfg, initial_capital=100_000.0)
    assert cheat.gross_pnl > 50 * abs(honest.gross_pnl)


# ---------------------------------------------------------------------- #
# 1. Fills happen at the next bar open, never inside the signal bar
# ---------------------------------------------------------------------- #
class _EnterOnceStrategy(Strategy):
    name = "enter_once"

    @classmethod
    def defaults(cls):
        return {"entry_bar": 3, "exit_bar": 6}

    def generate_signals(self, bars, features):
        signal = np.zeros(len(bars), dtype="int8")
        signal[int(self.params["entry_bar"]):int(self.params["exit_bar"])] = 1
        return pd.Series(signal, index=bars.index, dtype="int8")


def test_fill_is_next_bar_open(cfg):
    # Opens deliberately gap away from the prior close, so a fill at the wrong
    # price is unmistakable rather than coincidentally equal.
    closes = [1800, 1805, 1810, 1815, 1830, 1825, 1840, 1850, 1845, 1860]
    opens = [c + 7.0 for c in closes]
    bars = make_bars(closes, opens=opens)
    result = run_backtest(make_dataset(bars), _EnterOnceStrategy(), cfg,
                          initial_capital=100_000.0)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    # The signal is +1 on bars 3, 4 and 5. Each decision executes one bar later,
    # so the position runs from the open of bar 4 to the open of bar 7.
    assert trade["entry_time"] == bars.index[4]
    assert trade["entry_price_raw"] == pytest.approx(bars["open"].iloc[4])
    assert trade["exit_time"] == bars.index[7]
    assert trade["exit_price_raw"] == pytest.approx(bars["open"].iloc[7])


def test_fill_is_never_the_signal_bar_close(cfg):
    closes = [1800, 1805, 1810, 1815, 1830, 1825, 1840, 1850, 1845, 1860]
    opens = [c + 7.0 for c in closes]
    bars = make_bars(closes, opens=opens)
    result = run_backtest(make_dataset(bars), _EnterOnceStrategy(), cfg,
                          initial_capital=100_000.0)
    trade = result.trades.iloc[0]
    assert trade["entry_price_raw"] != pytest.approx(bars["close"].iloc[3])


def test_fill_never_uses_high_or_low(dataset, cfg):
    """Every fill price must be an open, not a high, a low, or anything between.

    Filling at the extreme of a bar assumes knowledge of the path inside the bar,
    which nobody has and which always flatters the backtest.
    """
    result = run_backtest(dataset, RESEARCH_STRATEGIES["trend_donchian"](), cfg,
                          initial_capital=100_000.0)
    assert len(result.trades) > 5
    opens = dataset.bars["open"]
    for _, trade in result.trades.iterrows():
        assert trade["entry_price_raw"] == pytest.approx(opens.loc[trade["entry_time"]])
        if trade["exit_reason"] != "end_of_data":
            assert trade["exit_price_raw"] == pytest.approx(opens.loc[trade["exit_time"]])


def test_target_position_is_shifted_exactly_one_bar():
    signal = pd.Series([0, 1, 1, -1, 0], index=pd.date_range(
        "2023-06-05", periods=5, freq="15min", tz="UTC"), dtype="int8")
    target = _target_from_signals(signal)
    assert target.tolist() == [0, 0, 1, 1, -1]


def test_costs_always_move_the_fill_against_the_trade(flat_cost_model):
    ts = pd.Timestamp("2024-01-03 13:00", tz="UTC")
    assert flat_cost_model.effective_fill_price(1800.0, +1, ts) > 1800.0   # buys pay up
    assert flat_cost_model.effective_fill_price(1800.0, -1, ts) < 1800.0   # sells hit the bid


# ---------------------------------------------------------------------- #
# 2 and 5. Static scan of the source
# ---------------------------------------------------------------------- #
def _walk_source():
    for path in SOURCE_FILES:
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_no_centred_windows_in_source():
    """``center=True`` on a rolling window makes half the window the future."""
    offenders = []
    for path, tree in _walk_source():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg == "center" and not (
                    isinstance(keyword.value, ast.Constant) and keyword.value.value is False
                ):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == [], f"centred rolling windows found: {offenders}"


def test_no_negative_shifts_in_source():
    """``.shift(-n)`` pulls the future backwards.

    Legitimate only for deliberate label construction, which this project does not
    do (no machine learning in version one), so any occurrence is a bug.
    """
    offenders = []
    for path, tree in _walk_source():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in ("shift", "tshift"):
                continue
            for arg in list(node.args) + [k.value for k in node.keywords if k.arg == "periods"]:
                negative = (
                    isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub)
                ) or (isinstance(arg, ast.Constant) and isinstance(arg.value, int) and arg.value < 0)
                if negative:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == [], f"negative shifts found: {offenders}"


def test_no_backfilling_in_source():
    """Backfilling propagates a later value into earlier bars, which is the future."""
    banned = ("bfill(", "backfill(", 'method="bfill"', "method='bfill'",
              'method="backfill"', "limit_direction=\"both\"")
    offenders = []
    for path in SOURCE_FILES:
        text = path.read_text(encoding="utf-8")
        for token in banned:
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains {token!r}")
    assert offenders == [], offenders


# ---------------------------------------------------------------------- #
# 3. Macro series are lagged before joining
# ---------------------------------------------------------------------- #
def _synthetic_macro(bars: pd.DataFrame) -> pd.DataFrame:
    days = pd.date_range(bars.index[0].normalize() - pd.Timedelta(days=400),
                         bars.index[-1].normalize(), freq="B")
    values = pd.Series(np.linspace(1.0, 2.0, len(days)), index=days, name="DFII10")
    return join_macro_to_bars(bars, values, lag_days=1)


def test_available_at_rejects_a_zero_day_lag():
    series = pd.Series([1.0], index=pd.DatetimeIndex(["2023-06-05"]), name="DFII10")
    with pytest.raises(ValueError, match="at least 1"):
        available_at(series, 0)


def test_macro_is_lagged_by_at_least_one_day():
    """A value stamped for day d must not be visible to a bar on day d."""
    days = pd.DatetimeIndex(["2023-06-05", "2023-06-06", "2023-06-07"])
    series = pd.Series([1.0, 2.0, 3.0], index=days, name="DFII10")
    # Two days of bars: 6 and 7 June. The print stamped 7 June only becomes
    # available on 8 June, which is past the end of this window.
    bars = make_bars([1800.0] * 48, start="2023-06-06 00:00", freq="1h")

    joined = join_macro_to_bars(bars, series, lag_days=1)
    on_the_sixth = joined.loc["2023-06-06", "DFII10"]
    # Throughout 6 June the newest usable print is the one stamped 5 June.
    assert (on_the_sixth == 1.0).all()
    on_the_seventh = joined.loc["2023-06-07", "DFII10"]
    assert (on_the_seventh == 2.0).all()
    assert 3.0 not in set(joined["DFII10"].dropna())


def test_macro_join_never_reaches_forward(bars_15m):
    """Property check: at every bar the joined value equals the newest print whose
    availability timestamp is at or before that bar."""
    days = pd.date_range("2022-11-01", "2023-04-01", freq="B")
    rng = np.random.default_rng(4)
    series = pd.Series(rng.normal(1.5, 0.1, len(days)), index=days, name="DFII10")
    joined = join_macro_to_bars(bars_15m, series, lag_days=1)
    stamped = available_at(series, 1)["DFII10"]

    for position in (0, 500, 2000, len(bars_15m) - 1):
        ts = bars_15m.index[position]
        visible = stamped[stamped.index <= ts]
        expected = visible.iloc[-1] if len(visible) else np.nan
        actual = joined["DFII10"].iloc[position]
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected)


def test_macro_values_before_the_first_publication_are_nan():
    days = pd.DatetimeIndex(["2023-06-20"])
    series = pd.Series([1.0], index=days, name="DFII10")
    bars = make_bars([1800.0] * 48, start="2023-06-05 00:00", freq="1h")
    joined = join_macro_to_bars(bars, series, lag_days=1)
    assert joined["DFII10"].isna().all()


# ---------------------------------------------------------------------- #
# 4. Resampling bin convention
# ---------------------------------------------------------------------- #
def test_resample_bins_are_left_closed_left_labelled(minute_bars, calendar):
    """A bar stamped 10:00 covers [10:00, 10:15), so its open is the 10:00 minute
    and its close is the 10:14 minute. Nothing from 10:15 belongs to it."""
    resampled = resample_bars(minute_bars, "15min", calendar)
    label = resampled.index[50]
    window = minute_bars.loc[
        (minute_bars.index >= label) & (minute_bars.index < label + pd.Timedelta(minutes=15))
    ]
    assert resampled.loc[label, "open"] == pytest.approx(window["open"].iloc[0])
    assert resampled.loc[label, "close"] == pytest.approx(window["close"].iloc[-1])
    assert resampled.loc[label, "high"] == pytest.approx(window["high"].max())
    assert resampled.loc[label, "low"] == pytest.approx(window["low"].min())
    # The next minute must not have contributed.
    following = minute_bars.loc[[label + pd.Timedelta(minutes=15)]]
    assert resampled.loc[label, "close"] != pytest.approx(following["close"].iloc[0])


# ---------------------------------------------------------------------- #
# 6. Normalisation is fitted on training data only
# ---------------------------------------------------------------------- #
def test_normaliser_refuses_to_transform_before_fitting():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(NotFittedError):
        ZScoreNormaliser().transform(frame)


def test_normaliser_fitted_on_train_does_not_see_test_statistics():
    train = pd.DataFrame({"x": np.arange(100, dtype="float64")})
    test = pd.DataFrame({"x": np.arange(100, 200, dtype="float64")})
    full = pd.concat([train, test], ignore_index=True)

    train_only = ZScoreNormaliser().fit(train)
    leaky = ZScoreNormaliser().fit(full)

    assert train_only.means["x"] != pytest.approx(leaky.means["x"])
    # Test data normalised on training statistics is far from zero-mean, which is
    # the visible signature of an honest split.
    assert abs(train_only.transform(test)["x"].mean()) > 3.0


def test_rolling_zscore_only_looks_backward():
    values = pd.Series(np.arange(50, dtype="float64"))
    rolling = RollingZScore(window=10).transform(values)
    truncated = RollingZScore(window=10).transform(values.iloc[:30])
    pd.testing.assert_series_equal(rolling.iloc[:30], truncated)


# ---------------------------------------------------------------------- #
# 7. Fixed symbol universe
# ---------------------------------------------------------------------- #
def test_symbol_universe_is_a_single_fixed_symbol(cfg, dataset):
    """No survivorship bias is possible with one instrument, but assert it anyway
    so that adding a universe later cannot slip past unnoticed."""
    symbol = cfg.get("instrument.symbol")
    assert isinstance(symbol, str) and symbol
    assert dataset.provenance.symbol == "XAUUSD"
    assert list(dataset.bars.columns[:5]) == ["open", "high", "low", "close", "volume"]
    assert dataset.bars.index.is_monotonic_increasing
    assert not dataset.bars.index.has_duplicates
