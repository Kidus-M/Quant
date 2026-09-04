"""Event loop behaviour: execution order, financing, stop-out, contract enforcement."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.backtest.sizing import PositionSizer
from src.strategies.base import SignalContractError, Strategy, validate_signals
from tests.conftest import make_bars, make_dataset


class _ConstantSignal(Strategy):
    name = "constant"

    @classmethod
    def defaults(cls):
        return {"values": None}

    def generate_signals(self, bars, features):
        return pd.Series(self.params["values"], index=bars.index, dtype="int8")


def _run(bars, values, cfg, **kwargs):
    kwargs.setdefault("initial_capital", 100_000.0)
    return run_backtest(make_dataset(bars), _ConstantSignal(values=values), cfg, **kwargs)


# ---------------------------------------------------------------------- #
def test_first_bar_is_always_flat(cfg):
    bars = make_bars([1800, 1810, 1820, 1830])
    result = _run(bars, [1, 1, 1, 1], cfg)
    assert result.position_oz.iloc[0] == 0.0
    assert result.target_position.iloc[0] == 0


def test_reversal_closes_and_reopens_paying_both_sides(cfg):
    bars = make_bars([1800, 1810, 1820, 1830, 1840, 1850])
    result = _run(bars, [1, 1, -1, -1, -1, -1], cfg)
    assert len(result.trades) == 2
    first, second = result.trades.iloc[0], result.trades.iloc[1]
    assert first.direction == 1 and second.direction == -1
    # Both legs of the reversal execute at the same bar open.
    assert first.exit_time == second.entry_time
    assert first.exit_price_raw == pytest.approx(second.entry_price_raw)
    assert second.entry_reason == "signal_entry"


def test_position_size_is_held_for_the_life_of_a_trade(cfg):
    bars = make_bars(list(np.linspace(1800, 1900, 40)))
    result = _run(bars, [1] * 40, cfg)
    assert len(result.trades) == 1
    assert (result.position_oz[result.position_oz != 0].abs().nunique()) == 1


def test_open_position_is_liquidated_at_the_final_close(cfg):
    bars = make_bars([1800, 1810, 1820, 1830])
    result = _run(bars, [1, 1, 1, 1], cfg)
    trade = result.trades.iloc[0]
    assert trade.exit_reason == "end_of_data"
    assert trade.exit_price_raw == pytest.approx(bars["close"].iloc[-1])
    assert result.position_oz.iloc[-1] == 0.0
    assert any("open at the end of the data" in w for w in result.warnings)


def test_financing_is_charged_across_the_rollover(cfg):
    """A position held across 21:00 UTC pays financing; one closed before does not."""
    held = make_bars([1800.0] * 12, start="2023-06-06 18:00", freq="1h")
    overnight = _run(held, [1] * 12, cfg)
    intraday_bars = make_bars([1800.0] * 12, start="2023-06-06 09:00", freq="1h")
    intraday = _run(intraday_bars, [1] * 12, cfg)

    financing_overnight = overnight.trades["financing_cost"].sum()
    financing_intraday = intraday.trades["financing_cost"].sum()
    assert financing_overnight > 0
    assert financing_intraday == pytest.approx(0.0)


def test_short_positions_receive_a_financing_credit(cfg):
    bars = make_bars([1800.0] * 12, start="2023-06-06 18:00", freq="1h")
    result = _run(bars, [-1] * 12, cfg)
    assert result.trades["financing_cost"].sum() < 0


def test_gross_net_and_costs_decompose_exactly(cfg, dataset):
    from src.strategies import RESEARCH_STRATEGIES

    result = run_backtest(dataset, RESEARCH_STRATEGIES["rsi2"](), cfg, initial_capital=100_000.0)
    reconstructed = result.equity_gross - result.costs_cum
    pd.testing.assert_series_equal(
        reconstructed.rename("equity_net"), result.equity_net, check_exact=False, atol=1e-9
    )
    assert result.net_pnl == pytest.approx(result.gross_pnl - result.total_costs)


def test_costs_are_never_negative_overall(cfg, dataset):
    from src.strategies import RESEARCH_STRATEGIES

    result = run_backtest(dataset, RESEARCH_STRATEGIES["trend_donchian"](), cfg,
                          initial_capital=100_000.0)
    assert result.total_costs > 0
    assert (result.costs_cum.diff().dropna() >= -1e-9).mean() > 0.99


# ---------------------------------------------------------------------- #
# Stop-out
# ---------------------------------------------------------------------- #
def test_account_is_stopped_out_at_zero_equity_and_never_trades_again(cfg):
    # A relentless downtrend on a tiny account.
    closes = list(np.linspace(1800, 1700, 60))
    bars = make_bars(closes)
    result = _run(bars, [1] * 60, cfg, initial_capital=20.0)
    assert result.was_ruined
    assert result.ruin_time is not None
    after = result.position_oz[result.position_oz.index > result.ruin_time]
    assert (after == 0).all(), "a stopped-out account kept trading"
    assert any("ACCOUNT RUINED" in w for w in result.warnings)


def test_a_healthy_account_is_not_flagged_as_ruined(cfg):
    bars = make_bars(list(np.linspace(1800, 1900, 40)))
    result = _run(bars, [1] * 40, cfg, initial_capital=100_000.0)
    assert not result.was_ruined


# ---------------------------------------------------------------------- #
# Contract enforcement
# ---------------------------------------------------------------------- #
def test_signals_outside_minus_one_zero_plus_one_are_rejected(cfg):
    bars = make_bars([1800, 1810, 1820, 1830])
    with pytest.raises(SignalContractError, match=r"\{-1, 0, \+1\}"):
        _run(bars, [0, 2, 1, 0], cfg)


def test_misaligned_signal_index_is_rejected():
    bars = make_bars([1800, 1810, 1820])
    shifted = pd.Series([0, 1, 1], index=bars.index + pd.Timedelta(days=1), dtype="int8")
    with pytest.raises(SignalContractError, match="index does not match"):
        validate_signals(shifted, bars)


def test_wrong_length_signal_is_rejected():
    bars = make_bars([1800, 1810, 1820])
    with pytest.raises(SignalContractError, match="signals for"):
        validate_signals(pd.Series([0, 1], index=bars.index[:2], dtype="int8"), bars)


def test_nan_signals_become_flat_not_phantom_positions():
    bars = make_bars([1800, 1810, 1820])
    signals = validate_signals(pd.Series([1.0, np.nan, -1.0], index=bars.index), bars)
    assert signals.tolist() == [1, 0, -1]


def test_empty_bars_are_refused(cfg):
    from src.data.base import empty_bars

    with pytest.raises(ValueError, match="empty bar set"):
        run_backtest(make_dataset(empty_bars()), _ConstantSignal(values=[]), cfg)


def test_strategy_needing_macro_refuses_to_run_without_it(cfg, dataset):
    from src.strategies.macro_trend import MacroFilteredTrendStrategy

    with pytest.raises(ValueError, match="requires macro"):
        run_backtest(dataset, MacroFilteredTrendStrategy(), cfg)


def test_warmup_bars_are_forced_flat(cfg):
    bars = make_bars(list(np.linspace(1800, 1900, 60)))
    result = _run(bars, [1] * 60, cfg, warmup_bars=20)
    # Warm-up suppresses the signal, and execution lags it by one more bar.
    assert (result.position_oz.iloc[:21] == 0).all()
    assert result.position_oz.iloc[21] != 0


def test_unknown_strategy_parameter_is_rejected():
    from src.strategies import Rsi2Strategy

    with pytest.raises(TypeError, match="unknown parameters"):
        Rsi2Strategy(oversold=10, not_a_real_parameter=3)


# ---------------------------------------------------------------------- #
# Sizing integration
# ---------------------------------------------------------------------- #
def test_fixed_fractional_sizing_scales_with_equity(cfg, dataset):
    from src.strategies import RESEARCH_STRATEGIES

    sizer = PositionSizer.from_config(
        cfg.with_overrides({"sizing.mode": "fixed_fractional",
                            "sizing.allow_min_lot_override": False})
    )
    result = run_backtest(dataset, RESEARCH_STRATEGIES["trend_donchian"](), cfg,
                          initial_capital=5_000_000.0, sizer=sizer)
    assert len(result.trades) > 0
    risked = result.trades["risk_usd"] / 5_000_000.0
    # Every trade risks at most the configured 1%, allowing for lot rounding.
    assert risked.max() <= 0.011
