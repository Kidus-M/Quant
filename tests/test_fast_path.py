"""The vectorised evaluator must agree with the event loop exactly.

The event loop is the reference implementation. The fast path exists only because
the random benchmark runs a thousand backtests, and a second P&L implementation is
a liability unless it is pinned to the first. These tests are that pin.

The awkward cases are all here on purpose: reversals, financing across the
rollover, a position open at the end of the data, and above all a ruined account,
where the two implementations have to agree on the exact bar the broker pulls the
plug and on which bar the position is flattened.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import CostModel
from src.backtest.engine import _target_from_signals, run_backtest
from src.backtest.fast import run_fast
from src.strategies.base import Strategy
from tests.conftest import make_dataset


class _Replay(Strategy):
    name = "replay"

    @classmethod
    def defaults(cls):
        return {"signal": None}

    def generate_signals(self, bars, features):
        return pd.Series(self.params["signal"], index=bars.index, dtype="int8")


def _compare(bars, calendar, signal, capital, cfg, cost_model):
    dataset = make_dataset(bars, calendar)
    engine = run_backtest(dataset, _Replay(signal=signal), cfg,
                          initial_capital=capital, cost_model=cost_model)
    target = _target_from_signals(pd.Series(signal, index=bars.index)).to_numpy()
    fast = run_fast(bars, target, size_oz=1.0, initial_capital=capital,
                    cost_model=cost_model, calendar=calendar)
    return engine, fast


@pytest.mark.parametrize("capital", [25.0, 50.0, 250.0, 5_000.0, 100_000.0])
def test_fast_path_matches_the_event_loop(capital, bars_15m, calendar, cfg, cost_model):
    """Random signal paths, including reversals and stop-outs, must agree to the cent."""
    rng = np.random.default_rng(20240101 + int(capital))
    window = bars_15m.iloc[:3000]
    for _ in range(6):
        signal = rng.choice([-1, 0, 1], size=len(window), p=[0.25, 0.5, 0.25]).astype("int8")
        engine, fast = _compare(window, calendar, signal, capital, cfg, cost_model)

        np.testing.assert_allclose(
            engine.equity_net.to_numpy(), fast.equity_net, atol=1e-8,
            err_msg="net equity diverged between the event loop and the fast path",
        )
        np.testing.assert_allclose(engine.costs_cum.to_numpy(), fast.costs_cum, atol=1e-8)
        np.testing.assert_allclose(engine.equity_gross.to_numpy(), fast.equity_gross, atol=1e-8)
        assert (engine.ruin_time is None) == (fast.ruin_index is None)
        if fast.ruin_index is not None:
            assert engine.ruin_time == window.index[fast.ruin_index]


def test_fast_path_matches_on_a_permanently_long_signal(bars_15m, calendar, cfg, cost_model):
    """Covers financing across many rollovers and a position open at the end."""
    window = bars_15m.iloc[:2000]
    signal = np.ones(len(window), dtype="int8")
    engine, fast = _compare(window, calendar, signal, 100_000.0, cfg, cost_model)
    np.testing.assert_allclose(engine.equity_net.to_numpy(), fast.equity_net, atol=1e-8)
    assert engine.trades["financing_cost"].sum() > 0


def test_fast_path_matches_on_constant_reversals(bars_15m, calendar, cfg, cost_model):
    window = bars_15m.iloc[:600]
    signal = np.where(np.arange(len(window)) % 2 == 0, 1, -1).astype("int8")
    engine, fast = _compare(window, calendar, signal, 100_000.0, cfg, cost_model)
    np.testing.assert_allclose(engine.equity_net.to_numpy(), fast.equity_net, atol=1e-8)
    # A reversal on every bar pays both sides every time.
    assert len(engine.trades) > len(window) / 2 - 2


def test_fast_path_matches_when_never_in_the_market(bars_15m, calendar, cfg, cost_model):
    window = bars_15m.iloc[:500]
    signal = np.zeros(len(window), dtype="int8")
    engine, fast = _compare(window, calendar, signal, 100_000.0, cfg, cost_model)
    np.testing.assert_allclose(engine.equity_net.to_numpy(), fast.equity_net, atol=1e-12)
    assert fast.costs_cum[-1] == pytest.approx(0.0)


def test_fast_path_rejects_a_mismatched_target(bars_15m, calendar, cost_model):
    with pytest.raises(ValueError, match="same length"):
        run_fast(bars_15m.iloc[:100], np.zeros(50), size_oz=1.0, initial_capital=100.0,
                 cost_model=cost_model, calendar=calendar)


def test_fast_path_rejects_a_non_positive_size(bars_15m, calendar, cost_model):
    with pytest.raises(ValueError, match="size_oz must be positive"):
        run_fast(bars_15m.iloc[:100], np.zeros(100), size_oz=0.0, initial_capital=100.0,
                 cost_model=cost_model, calendar=calendar)


def test_hoisted_constants_do_not_change_the_answer(bars_15m, calendar, cost_model):
    """The benchmark precomputes per-bar constants; that must be a pure speed-up."""
    window = bars_15m.iloc[:800]
    rng = np.random.default_rng(5)
    target = rng.choice([-1, 0, 1], size=len(window)).astype("int8")

    plain = run_fast(window, target, size_oz=1.0, initial_capital=10_000.0,
                     cost_model=cost_model, calendar=calendar)

    multipliers = cost_model.spread_multiplier_series(window.index).to_numpy()
    cost_per_oz = (0.5 * cost_model.spread_usd_per_oz_round_trip * multipliers
                   + cost_model.slippage_usd_per_oz_per_side)
    rollovers = np.zeros(len(window), dtype="int64")
    rollovers[1:] = calendar.rollovers_between(window.index[:-1], window.index[1:])
    hoisted = run_fast(window, target, size_oz=1.0, initial_capital=10_000.0,
                       cost_model=cost_model, calendar=calendar,
                       _rollovers=rollovers, _cost_per_oz=cost_per_oz)

    np.testing.assert_allclose(plain.equity_net, hoisted.equity_net, atol=1e-12)


def test_flat_cost_model_makes_gross_and_net_identical_when_flat(bars_15m, calendar):
    free = CostModel(spread_usd_per_oz_round_trip=0.0, slippage_usd_per_oz_per_side=0.0,
                     commission_usd_per_lot_per_side=0.0,
                     swap_long_usd_per_oz_per_night=0.0, swap_short_usd_per_oz_per_night=0.0)
    window = bars_15m.iloc[:400]
    result = run_fast(window, np.ones(len(window)), size_oz=1.0, initial_capital=100_000.0,
                      cost_model=free, calendar=calendar)
    np.testing.assert_allclose(result.equity_net, result.equity_gross, atol=1e-9)
