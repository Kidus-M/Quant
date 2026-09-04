"""Position sizing, the small-account risk arithmetic, and performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.sizing import (
    PositionSizer,
    format_risk_warning,
    risk_diagnostics,
)
from src.metrics.deflated import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from src.metrics.performance import compute_metrics, drawdown_stats
from src.metrics.stats import norm_cdf, norm_ppf, sample_kurtosis, sample_skew


# ---------------------------------------------------------------------- #
# Sizing
# ---------------------------------------------------------------------- #
def test_fixed_mode_returns_the_configured_lot_size():
    sizer = PositionSizer(mode="fixed", fixed_lots=0.01, contract_size_oz_per_lot=100.0)
    decision = sizer.size(equity=50.0, price=1800.0, atr=3.0)
    assert decision.lots == pytest.approx(0.01)
    assert decision.oz == pytest.approx(1.0)


def test_fixed_fractional_risks_the_configured_fraction():
    """5000 equity, 1% risk = 50 USD. A 2x ATR stop of 2.5 is 5 USD/oz, so 10 oz."""
    sizer = PositionSizer(mode="fixed_fractional", risk_fraction_per_trade=0.01,
                          stop_atr_multiple=2.0, contract_size_oz_per_lot=100.0,
                          min_lot=0.01, lot_step=0.01)
    decision = sizer.size(equity=5000.0, price=1800.0, atr=2.5)
    assert decision.oz == pytest.approx(10.0)
    assert decision.risk_usd == pytest.approx(50.0)


def test_sizing_rounds_down_to_the_lot_step_never_up():
    sizer = PositionSizer(mode="fixed_fractional", risk_fraction_per_trade=0.01,
                          stop_atr_multiple=2.0, contract_size_oz_per_lot=100.0,
                          min_lot=0.01, lot_step=0.01)
    # Budget allows 10.7 oz; the step is 1 oz, so 10 oz, not 11.
    decision = sizer.size(equity=5350.0, price=1800.0, atr=2.5)
    assert decision.oz == pytest.approx(10.0)
    assert decision.risk_usd <= 5350.0 * 0.01 + 1e-9


def test_vol_target_sizing_hits_the_target_daily_volatility():
    """1% of 100000 = 1000 USD/day. Daily vol per oz = 0.01 * sqrt(96) * 1800."""
    sizer = PositionSizer(mode="vol_target", target_daily_vol_fraction=0.01,
                          contract_size_oz_per_lot=100.0, min_lot=0.01, lot_step=0.01)
    decision = sizer.size(equity=100_000.0, price=1800.0, atr=3.0,
                          bar_return_vol=0.001, bars_per_day=96.0)
    expected_oz = 1000.0 / (0.001 * np.sqrt(96.0) * 1800.0)
    assert decision.oz == pytest.approx(np.floor(expected_oz), abs=1.0)


def test_min_lot_override_takes_the_minimum_and_says_so():
    """The situation a $50 account is actually in."""
    sizer = PositionSizer(mode="fixed_fractional", risk_fraction_per_trade=0.01,
                          stop_atr_multiple=2.0, contract_size_oz_per_lot=100.0,
                          min_lot=0.01, lot_step=0.01, allow_min_lot_override=True)
    decision = sizer.size(equity=50.0, price=1800.0, atr=3.0)
    assert decision.oz == pytest.approx(1.0)
    assert decision.min_lot_forced
    assert any("exceeds the configured risk" in note for note in decision.notes)
    # One ounce with a 6 USD stop risks 6 USD, which is 12% of a 50 USD account.
    assert decision.risk_usd == pytest.approx(6.0)


def test_min_lot_override_can_be_switched_off_to_skip_the_trade():
    sizer = PositionSizer(mode="fixed_fractional", risk_fraction_per_trade=0.01,
                          stop_atr_multiple=2.0, contract_size_oz_per_lot=100.0,
                          min_lot=0.01, lot_step=0.01, allow_min_lot_override=False)
    decision = sizer.size(equity=50.0, price=1800.0, atr=3.0)
    assert not decision.is_tradeable
    assert any("trade skipped" in note for note in decision.notes)


def test_no_position_when_equity_is_gone():
    sizer = PositionSizer(mode="fixed")
    assert not sizer.size(equity=0.0, price=1800.0, atr=3.0).is_tradeable
    assert not sizer.size(equity=-5.0, price=1800.0, atr=3.0).is_tradeable


def test_unknown_sizing_mode_raises():
    with pytest.raises(ValueError, match="unknown sizing mode"):
        PositionSizer(mode="kelly").size(equity=100.0, price=1800.0, atr=3.0)


# ---------------------------------------------------------------------- #
# The small-account arithmetic
# ---------------------------------------------------------------------- #
def test_risk_per_atr_is_a_percentage_of_equity():
    sizer = PositionSizer(mode="fixed", contract_size_oz_per_lot=100.0, min_lot=0.01)
    diag = risk_diagnostics(equity=50.0, atr_usd_per_oz=3.0, position_oz=1.0, sizer=sizer,
                            daily_range_usd_per_oz=30.0)
    # 1 oz x 3 USD = 3 USD against 50 USD of equity.
    assert diag.risk_per_atr_pct == pytest.approx(6.0)
    assert diag.risk_per_daily_range_pct == pytest.approx(60.0)
    assert diag.breaches_threshold
    assert diag.min_lot_unaffordable


def test_minimum_viable_account_size_is_computed_both_ways():
    """The number the spec asks for: the equity at which a 1% rule is followable."""
    sizer = PositionSizer(mode="fixed_fractional", risk_fraction_per_trade=0.01,
                          stop_atr_multiple=2.0, contract_size_oz_per_lot=100.0, min_lot=0.01)
    diag = risk_diagnostics(equity=50.0, atr_usd_per_oz=3.0, position_oz=1.0, sizer=sizer,
                            daily_range_usd_per_oz=40.0)
    # Stop-based: 1 oz x (2 x 3) = 6 USD of risk; at 1% that needs 600 USD.
    assert diag.min_viable_equity_usd == pytest.approx(600.0)
    # Daily-range based: 1 oz x 40 USD = 40 USD; at 1% that needs 4000 USD.
    assert diag.min_viable_equity_daily_range_usd == pytest.approx(4000.0)


def test_risk_warning_is_loud_and_names_the_numbers():
    sizer = PositionSizer(mode="fixed", contract_size_oz_per_lot=100.0, min_lot=0.01,
                          risk_fraction_per_trade=0.01, stop_atr_multiple=2.0)
    diag = risk_diagnostics(equity=50.0, atr_usd_per_oz=3.0, position_oz=1.0, sizer=sizer,
                            daily_range_usd_per_oz=40.0)
    text = "\n".join(format_risk_warning(diag))
    assert "POSITION SIZING WARNING" in text
    assert "6.0% of the account" in text
    assert "Minimum viable account size" in text
    assert "4,000 USD" in text


def test_no_warning_when_the_position_is_small_enough():
    sizer = PositionSizer(mode="fixed", contract_size_oz_per_lot=100.0, min_lot=0.01)
    diag = risk_diagnostics(equity=100_000.0, atr_usd_per_oz=3.0, position_oz=1.0, sizer=sizer,
                            daily_range_usd_per_oz=30.0)
    assert not diag.breaches_threshold
    text = "\n".join(format_risk_warning(diag))
    assert "WARNING" not in text
    assert "within the" in text


# ---------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------- #
def _equity(values, freq="15min"):
    index = pd.date_range("2023-01-02", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(np.asarray(values, dtype="float64"), index=index)


def test_drawdown_of_a_monotonic_curve_is_zero():
    stats = drawdown_stats(_equity([100, 101, 102, 103]))
    assert stats.max_drawdown_fraction == pytest.approx(0.0)
    assert stats.time_underwater_fraction == pytest.approx(0.0)


def test_drawdown_is_measured_from_the_running_peak():
    stats = drawdown_stats(_equity([100, 120, 90, 110, 130]))
    # Peak 120 down to 90 is -25%.
    assert stats.max_drawdown_fraction == pytest.approx(-0.25)
    assert stats.max_drawdown_usd == pytest.approx(-30.0)
    assert stats.max_drawdown_recovered is not None


def test_time_underwater_counts_bars_below_the_peak():
    # Peaks are [100, 100, 100, 100, 105], so only the 90 and 95 bars are
    # underwater; recovering exactly to the old peak is not a drawdown.
    stats = drawdown_stats(_equity([100, 90, 95, 100, 105]))
    assert stats.time_underwater_fraction == pytest.approx(2 / 5)


def test_metrics_decompose_gross_costs_and_net(cfg, dataset):
    from src.backtest.engine import run_backtest
    from src.strategies import RESEARCH_STRATEGIES

    result = run_backtest(dataset, RESEARCH_STRATEGIES["rsi2"](), cfg, initial_capital=100_000.0)
    metrics = compute_metrics(result)
    assert metrics.net_pnl_usd == pytest.approx(metrics.gross_pnl_usd - metrics.total_cost_usd)
    assert metrics.total_cost_usd == pytest.approx(sum(metrics.cost_breakdown_usd.values()))
    assert metrics.n_trades == len(result.trades)
    assert 0.0 <= metrics.win_rate <= 1.0
    assert 0.0 <= metrics.exposure <= 1.0


def test_metrics_report_absolute_usd_beside_every_percentage(cfg, dataset):
    from src.backtest.engine import run_backtest
    from src.strategies import RESEARCH_STRATEGIES

    result = run_backtest(dataset, RESEARCH_STRATEGIES["trend_donchian"](), cfg,
                          initial_capital=100_000.0)
    metrics = compute_metrics(result)
    assert metrics.net_pnl_usd == pytest.approx(
        metrics.total_return_net * metrics.initial_capital_usd
    )
    assert metrics.max_drawdown_usd <= 0.0


def test_ratios_are_nan_rather_than_misleading_after_ruin(cfg):
    from src.backtest.engine import run_backtest
    from tests.conftest import make_bars, make_dataset
    from src.strategies.base import Strategy

    class AlwaysLong(Strategy):
        name = "always_long"

        def generate_signals(self, bars, features):
            return pd.Series(1, index=bars.index, dtype="int8")

    bars = make_bars(list(np.linspace(1800, 1700, 80)))
    result = run_backtest(make_dataset(bars), AlwaysLong(), cfg, initial_capital=20.0)
    metrics = compute_metrics(result)
    assert metrics.was_ruined
    assert np.isnan(metrics.cagr_net)
    assert any("reached zero" in note for note in metrics.notes)


def test_profit_factor_and_expectancy_on_a_known_trade_set(cfg):
    from src.backtest.engine import run_backtest
    from src.strategies.base import Strategy
    from tests.conftest import make_bars, make_dataset

    class Alternating(Strategy):
        name = "alternating"

        def generate_signals(self, bars, features):
            values = np.zeros(len(bars), dtype="int8")
            values[1::4] = 1
            return pd.Series(values, index=bars.index, dtype="int8")

    rng = np.random.default_rng(7)
    closes = 1800 + np.cumsum(rng.normal(0, 3.0, 200))
    result = run_backtest(make_dataset(make_bars(closes)), Alternating(), cfg,
                          initial_capital=100_000.0)
    metrics = compute_metrics(result)
    trades = result.trades
    wins = trades.loc[trades.net_pnl > 0, "net_pnl"]
    losses = trades.loc[trades.net_pnl < 0, "net_pnl"]
    assert metrics.win_rate == pytest.approx(len(wins) / len(trades))
    assert metrics.profit_factor == pytest.approx(wins.sum() / -losses.sum())
    assert metrics.expectancy_usd == pytest.approx(trades["net_pnl"].mean())


# ---------------------------------------------------------------------- #
# Deflated Sharpe
# ---------------------------------------------------------------------- #
def test_normal_cdf_and_ppf_are_inverses():
    for p in (0.01, 0.1, 0.5, 0.9, 0.99):
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-6)
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)


def test_sample_moments_of_a_normal_sample():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=200_000)
    assert sample_skew(sample) == pytest.approx(0.0, abs=0.02)
    assert sample_kurtosis(sample) == pytest.approx(3.0, abs=0.05)


def test_expected_max_sharpe_grows_with_the_number_of_trials():
    """Searching harder raises the bar, which is the entire point."""
    variance = 0.01
    values = [expected_max_sharpe(n, variance) for n in (1, 10, 100, 1000)]
    assert values == sorted(values)
    assert values[0] == 0.0
    assert values[-1] > values[1]


def test_probabilistic_sharpe_rises_with_the_sample_size():
    a = probabilistic_sharpe_ratio(0.05, 100)
    b = probabilistic_sharpe_ratio(0.05, 10_000)
    assert b > a


def test_deflated_sharpe_penalises_a_wide_search():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0004, 0.01, 4000)
    single = deflated_sharpe_ratio(returns, n_trials=1)
    searched = deflated_sharpe_ratio(returns, n_trials=500)
    assert searched.deflated_sharpe < single.deflated_sharpe
    assert searched.sharpe_threshold > single.sharpe_threshold
    assert searched.n_trials == 500


def test_pure_noise_does_not_clear_the_deflated_sharpe_bar():
    rng = np.random.default_rng(3)
    best = None
    trial_sharpes = []
    for _ in range(200):
        returns = rng.normal(0.0, 0.01, 2000)
        sharpe = returns.mean() / returns.std(ddof=1)
        trial_sharpes.append(sharpe)
        if best is None or sharpe > best[0]:
            best = (sharpe, returns)
    result = deflated_sharpe_ratio(
        best[1], n_trials=200, trial_sharpes=np.array(trial_sharpes)
    )
    assert not result.is_significant, (
        "the best of 200 pure-noise strategies cleared the deflated Sharpe bar; "
        "the selection-bias adjustment is not working"
    )


def test_deflated_sharpe_needs_either_returns_or_summary_statistics():
    with pytest.raises(ValueError, match="provide either"):
        deflated_sharpe_ratio(n_trials=5)
