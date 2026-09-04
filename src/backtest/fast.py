"""Vectorised evaluator, used only where the event loop would be too slow.

The random benchmark runs a thousand backtests. At tens of thousands of bars each,
the Python event loop would take minutes; this takes seconds.

Two code paths computing P&L is a genuine risk: they drift, and the fast one
quietly becomes the optimistic one. The mitigation is that the event loop is the
reference implementation and ``tests/test_fast_path.py`` asserts the two agree to
the cent on random signal sequences, including reversals, financing and ruin. If
they ever disagree, the fast path is wrong by definition.

Restriction: fixed position sizing only. Volatility targeting and fixed-fractional
sizing depend on the equity path, so they are not vectorisable without changing
what they mean, and this refuses to run rather than approximating them.

The P&L identity, for position ``p[i]`` held from the open of bar ``i``::

    gross[i] = sum_{k<i} p[k] * (open[k+1] - open[k])  +  p[i] * (close[i] - open[i])

which is the telescoped form of marking each held segment from entry open to exit
open, with the current bar marked to its close.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.costs import CostModel
from src.data.sessions import SessionCalendar


@dataclass
class FastResult:
    equity_net: np.ndarray
    equity_gross: np.ndarray
    costs_cum: np.ndarray
    position_oz: np.ndarray
    ruin_index: int | None
    n_trades: int
    ruin_before_execution: bool = False

    @property
    def net_pnl(self) -> float:
        return float(self.equity_net[-1] - self.equity_net[0] + 0.0)


def run_fast(
    bars: pd.DataFrame,
    target: np.ndarray,
    *,
    size_oz: float,
    initial_capital: float,
    cost_model: CostModel,
    calendar: SessionCalendar,
    _rollovers: np.ndarray | None = None,
    _cost_per_oz: np.ndarray | None = None,
) -> FastResult:
    """Evaluate a fixed-size strategy from an already-shifted target array.

    ``target`` must already carry the next-bar-open execution delay; this function
    does not shift it. That keeps the delay in exactly one place, the engine.

    The two private arguments let a caller hoist per-bar constants out of a loop
    over many random runs; they are derived from ``bars`` when omitted.
    """
    n = len(bars)
    if len(target) != n:
        raise ValueError("target and bars must be the same length")
    if size_oz <= 0:
        raise ValueError("size_oz must be positive")

    open_px = bars["open"].to_numpy(dtype="float64")
    close_px = bars["close"].to_numpy(dtype="float64")
    index = bars.index

    if _cost_per_oz is None:
        multipliers = cost_model.spread_multiplier_series(index).to_numpy(dtype="float64")
        _cost_per_oz = (
            0.5 * cost_model.spread_usd_per_oz_round_trip * multipliers
            + cost_model.slippage_usd_per_oz_per_side
        )
    if _rollovers is None:
        _rollovers = np.zeros(n, dtype="int64")
        if n > 1:
            _rollovers[1:] = calendar.rollovers_between(index[:-1], index[1:])

    position = np.asarray(target, dtype="float64") * float(size_oz)
    result = _evaluate(
        position, open_px, close_px, _cost_per_oz, _rollovers,
        initial_capital, cost_model, size_oz,
    )
    if result.ruin_index is None:
        return result

    # Stopped out. The engine checks equity twice per bar, so which bar the
    # position is flattened on depends on where the account went under:
    #   * before execution (a financing charge took it under): flatten at THIS
    #     bar open, because the engine refuses to trade on this bar at all
    #   * at the mark (the close): flatten at the NEXT bar open
    # Equity up to the ruin bar is unaffected by suppressing later positions, so
    # the ruin index is stable and one re-evaluation is exact, not iterative.
    r = result.ruin_index
    position = position.copy()
    position[r + (0 if result.ruin_before_execution else 1):] = 0.0
    return _evaluate(
        position, open_px, close_px, _cost_per_oz, _rollovers,
        initial_capital, cost_model, size_oz,
        forced_ruin=r, forced_ruin_pre=result.ruin_before_execution,
    )


def _evaluate(
    position: np.ndarray,
    open_px: np.ndarray,
    close_px: np.ndarray,
    cost_per_oz: np.ndarray,
    rollovers: np.ndarray,
    initial_capital: float,
    cost_model: CostModel,
    size_oz: float,
    forced_ruin: int | None = None,
    forced_ruin_pre: bool = False,
) -> FastResult:
    n = position.size

    # Gross: each bar contributes the move from its open to the next open, and the
    # currently-held position is marked from this bar open to this bar close.
    open_to_open = np.zeros(n, dtype="float64")
    open_to_open[:-1] = np.diff(open_px)
    carried = np.concatenate(([0.0], np.cumsum(position[:-1] * open_to_open[:-1])))
    gross_cum = carried + position * (close_px - open_px)

    # Transaction costs at each position change, including the final liquidation.
    delta = np.diff(position, prepend=0.0)
    traded = np.abs(delta)
    trade_cost = traded * cost_per_oz + (traded / cost_model.contract_size_oz_per_lot) * (
        cost_model.commission_usd_per_lot_per_side
    )
    if position[-1] != 0.0:
        # The engine closes an open position at the final close.
        trade_cost[-1] += abs(position[-1]) * cost_per_oz[-1] + (
            abs(position[-1]) / cost_model.contract_size_oz_per_lot
        ) * cost_model.commission_usd_per_lot_per_side

    # Financing on the position carried into each bar, charged before execution.
    carried_position = np.concatenate(([0.0], position[:-1]))
    swap_rate = np.where(
        carried_position > 0,
        cost_model.swap_long_usd_per_oz_per_night,
        cost_model.swap_short_usd_per_oz_per_night,
    )
    financing = -swap_rate * np.abs(carried_position) * rollovers

    costs_cum = np.cumsum(trade_cost + financing)
    equity_gross = initial_capital + gross_cum
    equity_net = equity_gross - costs_cum

    # Equity as the engine sees it at execution time: marked to the previous
    # close, with this bar financing charge already taken.
    equity_pre_exec = np.empty(n, dtype="float64")
    equity_pre_exec[0] = initial_capital - financing[0]
    equity_pre_exec[1:] = equity_net[:-1] - financing[1:]

    if forced_ruin is not None:
        ruin_index, ruin_pre = forced_ruin, forced_ruin_pre
    else:
        under = np.flatnonzero((equity_net <= 0) | (equity_pre_exec <= 0))
        if under.size:
            ruin_index = int(under[0])
            ruin_pre = bool(equity_pre_exec[ruin_index] <= 0)
        else:
            ruin_index, ruin_pre = None, False

    n_trades = int(np.count_nonzero((position != 0) & (np.diff(position, prepend=0.0) != 0)))

    return FastResult(
        equity_net=equity_net,
        equity_gross=equity_gross,
        costs_cum=costs_cum,
        position_oz=position,
        ruin_index=ruin_index,
        n_trades=n_trades,
        ruin_before_execution=ruin_pre,
    )
