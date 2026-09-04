"""Walk-forward structure: ordering, purging, embargo, trial counting.

Structural properties are what matter here. A walk-forward harness that quietly
overlaps train and test, or that reports in-sample numbers as the result, is worse
than no validation at all because it comes with a certificate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.benchmark import run_random_benchmark
from src.backtest.walkforward import (
    WalkForward,
    build_folds,
    parameter_combinations,
    parameter_sensitivity,
)
from src.strategies import DonchianTrendStrategy, Rsi2Strategy
from src.strategies.random_entry import RandomEntryStrategy, TradeProfile


@pytest.fixture(scope="module")
def long_dataset(base_config, tmp_path_factory):
    """Two and a half years, long enough for several folds."""
    from src.data.loader import BarDataset
    from src.data.base import DataProvenance
    from src.data.quality import QualityReport
    from src.data.resample import resample_bars
    from src.data.sessions import SessionCalendar
    from src.data.synthetic import SyntheticAdapter

    calendar = SessionCalendar()
    minutes = SyntheticAdapter(seed=17, calendar=calendar).fetch(
        "XAUUSD", pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2023-06-30", tz="UTC")
    )
    bars = resample_bars(minutes, "1h", calendar)
    return BarDataset(
        bars=bars, resolution="1h", base_resolution="1min",
        provenance=DataProvenance("synthetic", "XAUUSD", "1min", is_synthetic=True),
        calendar=calendar, quality=QualityReport(len(bars), len(bars)),
        macro=pd.DataFrame(index=bars.index),
    )


@pytest.fixture
def wf_config(cfg, tmp_path):
    return cfg.with_overrides({
        "account.initial_capital_usd": 250_000.0,
        "walk_forward.in_sample_days": 240,
        "walk_forward.out_of_sample_days": 120,
        "walk_forward.min_in_sample_days": 200,
    })


# ---------------------------------------------------------------------- #
# Fold construction
# ---------------------------------------------------------------------- #
def test_folds_are_ordered_and_never_overlap():
    index = pd.date_range("2020-01-01", "2024-01-01", freq="1h", tz="UTC")
    folds = build_folds(index, in_sample_days=365, out_of_sample_days=90, min_in_sample_days=365)
    assert len(folds) > 5
    previous_test_end = None
    for train_start, train_end, test_start, test_end in folds:
        assert train_start < train_end <= test_start < test_end
        # Training never reaches into the test window.
        assert train_end <= test_start
        if previous_test_end is not None:
            # Test windows march forward without overlapping each other.
            assert test_start >= previous_test_end
        previous_test_end = test_end


def test_anchored_folds_keep_the_training_start_pinned():
    index = pd.date_range("2020-01-01", "2023-01-01", freq="1h", tz="UTC")
    folds = build_folds(index, in_sample_days=365, out_of_sample_days=90,
                        min_in_sample_days=365, anchored=True)
    starts = {train_start for train_start, _, _, _ in folds}
    assert len(starts) == 1
    # The training window grows every fold.
    lengths = [(train_end - train_start) for train_start, train_end, _, _ in folds]
    assert lengths == sorted(lengths)


def test_rolling_folds_slide_the_training_start_forward():
    index = pd.date_range("2020-01-01", "2023-01-01", freq="1h", tz="UTC")
    folds = build_folds(index, in_sample_days=365, out_of_sample_days=90,
                        min_in_sample_days=200, anchored=False)
    starts = [train_start for train_start, _, _, _ in folds]
    assert len(set(starts)) > 1
    assert starts == sorted(starts)


def test_no_folds_when_the_history_is_too_short():
    index = pd.date_range("2023-01-01", "2023-03-01", freq="1h", tz="UTC")
    assert build_folds(index, in_sample_days=365, out_of_sample_days=90,
                       min_in_sample_days=365) == []


def test_parameter_combinations_covers_the_whole_grid():
    combos = parameter_combinations(Rsi2Strategy())
    grid = Rsi2Strategy.param_grid
    expected = 1
    for values in grid.values():
        expected *= len(values)
    assert len(combos) == expected
    assert len({tuple(sorted(c.items())) for c in combos}) == expected


def test_a_strategy_with_no_grid_yields_a_single_empty_combination():
    from src.strategies import BuyAndHoldStrategy

    assert parameter_combinations(BuyAndHoldStrategy()) == [{}]


# ---------------------------------------------------------------------- #
# Running it
# ---------------------------------------------------------------------- #
def test_walk_forward_reports_only_out_of_sample_bars(long_dataset, wf_config):
    walk = WalkForward(long_dataset, DonchianTrendStrategy, wf_config).run()
    assert len(walk.folds) >= 2

    equity_index = walk.oos_equity_net.index
    assert equity_index.is_monotonic_increasing
    assert not equity_index.has_duplicates
    # Nothing before the first out-of-sample window may appear in the result.
    assert equity_index.min() >= walk.folds[0].test_start
    for fold in walk.folds:
        inside = equity_index[(equity_index >= fold.test_start) & (equity_index <= fold.test_end)]
        assert len(inside) > 0


def test_every_evaluated_combination_is_counted_as_a_trial(long_dataset, wf_config):
    walk = WalkForward(long_dataset, DonchianTrendStrategy, wf_config).run()
    per_fold = len(parameter_combinations(DonchianTrendStrategy()))
    # Folds that ran contribute their whole grid; the count is never understated.
    assert walk.total_trials >= per_fold * len(walk.folds)
    assert walk.deflated is not None
    assert walk.deflated.n_trials == walk.total_trials


def test_deflated_sharpe_is_never_more_generous_than_the_raw_one(long_dataset, wf_config):
    walk = WalkForward(long_dataset, DonchianTrendStrategy, wf_config).run()
    assert walk.deflated.deflated_sharpe <= walk.deflated.probabilistic_sharpe + 1e-12


def test_embargo_defaults_to_the_maximum_indicator_lookback(long_dataset, wf_config):
    walk = WalkForward(long_dataset, DonchianTrendStrategy, wf_config).run()
    probe = DonchianTrendStrategy()
    assert walk.folds[0].embargo_bars == probe.max_lookback
    assert walk.folds[0].embargo_bars > 0


def test_embargo_can_be_set_explicitly(long_dataset, wf_config):
    configured = wf_config.with_overrides({"walk_forward.embargo_bars": 250})
    walk = WalkForward(long_dataset, DonchianTrendStrategy, configured).run()
    assert all(fold.embargo_bars == 250 for fold in walk.folds)


def test_purging_removes_bars_next_to_the_train_test_boundary(long_dataset, wf_config):
    """The last embargo bars of each training window must not be used for fitting.

    Verified by construction rather than by inspection: with a huge embargo there
    is not enough training data left, and the harness says so instead of quietly
    fitting on the boundary.
    """
    huge = wf_config.with_overrides({"walk_forward.embargo_bars": 100_000})
    with pytest.raises(ValueError, match="no out-of-sample results"):
        WalkForward(long_dataset, DonchianTrendStrategy, huge).run()


def test_stitched_result_is_shaped_like_a_normal_backtest(long_dataset, wf_config):
    from src.metrics.performance import compute_metrics

    walk = WalkForward(long_dataset, DonchianTrendStrategy, wf_config).run()
    stitched = walk.as_backtest_result()
    metrics = compute_metrics(stitched)
    assert metrics.n_trades == len(walk.oos_trades)
    # Gross minus costs must still equal net across the fold seams.
    assert metrics.net_pnl_usd == pytest.approx(
        metrics.gross_pnl_usd - metrics.total_cost_usd, abs=1e-6
    )


def test_parameter_sensitivity_covers_the_whole_grid(long_dataset, wf_config):
    surface = parameter_sensitivity(long_dataset, DonchianTrendStrategy, wf_config)
    assert len(surface) == len(parameter_combinations(DonchianTrendStrategy()))
    assert {"entry_window", "atr_stop_multiple", "objective", "net_pnl_usd"} <= set(surface.columns)
    assert surface["objective"].notna().any()


# ---------------------------------------------------------------------- #
# Random benchmark
# ---------------------------------------------------------------------- #
def test_random_benchmark_matches_the_strategy_trade_profile(dataset, cfg):
    from src.backtest.engine import run_backtest

    result = run_backtest(dataset, DonchianTrendStrategy(), cfg, initial_capital=100_000.0)
    benchmark = run_random_benchmark(dataset, result, cfg, n_runs=120, seed=1)
    assert benchmark.n_runs == 120
    assert benchmark.profile.n_trades == len(result.trades)
    assert benchmark.profile.mean_hold_bars == pytest.approx(result.trades["bars_held"].mean())
    assert 0.0 <= benchmark.strategy_percentile <= 100.0


def test_random_benchmark_percentile_is_consistent_with_the_distribution(dataset, cfg):
    from src.backtest.engine import run_backtest

    result = run_backtest(dataset, Rsi2Strategy(), cfg, initial_capital=100_000.0)
    benchmark = run_random_benchmark(dataset, result, cfg, n_runs=200, seed=2)
    manual = 100.0 * (benchmark.net_pnl < benchmark.strategy_net_pnl).mean()
    assert benchmark.strategy_percentile == pytest.approx(manual)
    assert benchmark.passes == (benchmark.strategy_percentile >= 95.0)


def test_random_benchmark_is_reproducible(dataset, cfg):
    from src.backtest.engine import run_backtest

    result = run_backtest(dataset, Rsi2Strategy(), cfg, initial_capital=100_000.0)
    a = run_random_benchmark(dataset, result, cfg, n_runs=50, seed=42)
    b = run_random_benchmark(dataset, result, cfg, n_runs=50, seed=42)
    np.testing.assert_allclose(a.net_pnl, b.net_pnl)


def test_random_benchmark_handles_a_strategy_that_never_traded(dataset, cfg):
    from src.backtest.engine import run_backtest
    from src.strategies.base import Strategy

    class NeverTrades(Strategy):
        name = "never"

        def generate_signals(self, bars, features):
            return pd.Series(0, index=bars.index, dtype="int8")

    result = run_backtest(dataset, NeverTrades(), cfg, initial_capital=100_000.0)
    benchmark = run_random_benchmark(dataset, result, cfg, n_runs=10)
    assert benchmark.n_runs == 0
    assert "nothing to compare" in " ".join(benchmark.notes)


def test_random_paths_reproduce_the_requested_trade_profile(bars_15m):
    """The null has to trade like the strategy or the comparison is not fair."""
    profile = TradeProfile(n_trades=100, mean_hold_bars=20.0, long_fraction=0.5,
                           n_bars=len(bars_15m))
    counts, holds = [], []
    for seed in range(30):
        signal = RandomEntryStrategy.matching(profile, seed).generate_signals(
            bars_15m, pd.DataFrame(index=bars_15m.index)
        ).to_numpy()
        changes = np.diff(signal, prepend=0)
        counts.append(int(np.count_nonzero((signal != 0) & (changes != 0))))
        holds.append(float(np.count_nonzero(signal)) / max(1, counts[-1]))
    assert 60 < np.mean(counts) < 160
    assert 12 < np.mean(holds) < 30
