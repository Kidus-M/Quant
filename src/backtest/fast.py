"""Vectorised evaluator, used only where the event loop would be too slow.

The random benchmark runs a thousand backtests. At tens of thousands of bars each
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

Reproducing the stop-out exactly is the fiddly part, because the engine looks at
equity three times per bar: after financing, again at the moment it sizes an entry
(which happens *after* any closing trade on the same bar has been realised), and
finally at the mark. On a $50 account those three moments genuinely differ, so all
three are reconstructed here.
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
    # True when the account went under before or during execution, in which case
    # the ruin bar itself is already flat; False when it went under at the mark,
    # which flattens from the following bar.
    flat_from_ruin_bar: bool = False


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
    state = _evaluate(position, open_px, close_px, _cost_per_oz, _rollovers,
                      initial_capital, cost_model)

    ruin = _first_ruin(state, initial_capital)
    if ruin is None:
        return _as_result(state, position, None, False)

    ruin_index, flat_from_ruin_bar = ruin
    # Everything before the ruin bar is untouched by flattening what comes after,
    # so the ruin index is stable and one re-evaluation is exact, not iterative.
    position = position.copy()
    position[ruin_index + (0 if flat_from_ruin_bar else 1):] = 0.0
    state = _evaluate(position, open_px, close_px, _cost_per_oz, _rollovers,
                      initial_capital, cost_model)
    return _as_result(state, position, ruin_index, flat_from_ruin_bar)


@dataclass
class _State:
    equity_net: np.ndarray
    equity_gross: np.ndarray
    costs_cum: np.ndarray
    equity_pre_exec: np.ndarray
    equity_at_entry: np.ndarray
    entry_attempted: np.ndarray


def _evaluate(
    position: np.ndarray,
    open_px: np.ndarray,
    close_px: np.ndarray,
    cost_per_oz: np.ndarray,
    rollovers: np.ndarray,
    initial_capital: float,
    cost_model: CostModel,
) -> _State:
    n = position.size
    contract = cost_model.contract_size_oz_per_lot
    commission = cost_model.commission_usd_per_lot_per_side
    prev_position = np.concatenate(([0.0], position[:-1]))

    # Gross P&L marked to each bar open (everything flat) and to each bar close.
    open_to_open = np.zeros(n, dtype="float64")
    open_to_open[:-1] = np.diff(open_px)
    gross_at_open = np.concatenate(([0.0], np.cumsum(position[:-1] * open_to_open[:-1])))
    gross_cum = gross_at_open + position * (close_px - open_px)

    changed = position != prev_position
    closed_oz = np.where(changed, np.abs(prev_position), 0.0)
    opened_oz = np.where(changed, np.abs(position), 0.0)

    def transact(quantity, per_oz) -> np.ndarray:
        return quantity * per_oz + (quantity / contract) * commission

    close_cost = transact(closed_oz, cost_per_oz)
    open_cost = transact(opened_oz, cost_per_oz)
    trade_cost = close_cost + open_cost
    if position[-1] != 0.0:
        # The engine liquidates any position still open at the final close. The
        # per-ounce cost must be the LAST bar one; letting the full-length
        # cost_per_oz array broadcast here silently charges the first bar spread.
        trade_cost[-1] += transact(abs(float(position[-1])), float(cost_per_oz[-1]))

    swap_rate = np.where(
        prev_position > 0,
        cost_model.swap_long_usd_per_oz_per_night,
        cost_model.swap_short_usd_per_oz_per_night,
    )
    financing = -swap_rate * np.abs(prev_position) * rollovers

    costs_cum = np.cumsum(trade_cost + financing)
    equity_gross = initial_capital + gross_cum
    equity_net = equity_gross - costs_cum

    # Equity at the two intra-bar moments the engine inspects.
    costs_before_bar = np.concatenate(([0.0], costs_cum[:-1]))
    # Before execution the mark still sits at the previous close, so this is
    # simply the previous bar net equity less this bar financing charge.
    equity_pre_exec = np.concatenate(
        ([initial_capital - financing[0]], equity_net[:-1] - financing[1:])
    )
    # At the moment an entry is sized, any closing trade on this bar has already
    # been realised at this bar open and its cost already paid.
    equity_at_entry = (
        initial_capital + gross_at_open - costs_before_bar - financing - close_cost
    )
    entry_attempted = changed & (position != 0.0)

    return _State(
        equity_net=equity_net,
        equity_gross=equity_gross,
        costs_cum=costs_cum,
        equity_pre_exec=equity_pre_exec,
        equity_at_entry=equity_at_entry,
        entry_attempted=entry_attempted,
    )


def _first_ruin(state: _State, initial_capital: float) -> tuple[int, bool] | None:
    """Earliest bar at which the account is stopped out, and whether that bar is
    already flat.

    Three triggers, matching the three moments the engine checks:

    * equity non-positive after financing, before execution
    * equity non-positive at the instant an entry would be sized, which is after
      any closing trade on the same bar has been realised
    * equity non-positive at the mark

    The first two flatten the ruin bar itself; the third flattens from the next
    bar. An entry refused for want of equity is always immediately followed by a
    non-positive mark on the same bar, so it is a stop-out rather than a skip.
    """
    flatten_here = (state.equity_pre_exec <= 0) | (
        state.entry_attempted & (state.equity_at_entry <= 0)
    )
    flatten_next = state.equity_net <= 0
    candidates = np.flatnonzero(flatten_here | flatten_next)
    if not candidates.size:
        return None
    index = int(candidates[0])
    return index, bool(flatten_here[index])


def _as_result(
    state: _State, position: np.ndarray, ruin_index: int | None, flat_from_ruin_bar: bool
) -> FastResult:
    changed = position != np.concatenate(([0.0], position[:-1]))
    n_trades = int(np.count_nonzero(changed & (position != 0.0)))
    return FastResult(
        equity_net=state.equity_net,
        equity_gross=state.equity_gross,
        costs_cum=state.costs_cum,
        position_oz=position,
        ruin_index=ruin_index,
        n_trades=n_trades,
        flat_from_ruin_bar=flat_from_ruin_bar,
    )
