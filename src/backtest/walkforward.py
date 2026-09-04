"""Anchored walk-forward validation.

This is the part that decides whether a result means anything, so the rules are
strict and there is no configuration switch that relaxes them:

* **No random splits, ever.** Time-series data is always ordered. A k-fold split
  trains on next year to predict last year, which is not a thing anyone can do.
* **Anchored windows.** The training window always starts at the beginning of the
  data and grows; the test window is the period immediately after it and rolls
  forward. Parameters are chosen on training data alone.
* **Only concatenated out-of-sample results are reported as "the result".** The
  in-sample numbers are recorded for diagnosis and are never the headline.
* **Purge and embargo.** The last ``embargo_bars`` of each training window are
  dropped so that parameter selection is not influenced by bars sitting right
  against the test boundary, where indicator windows straddle the seam. The test
  window is preceded by a warm-up region in which trading is suppressed, so the
  first out-of-sample signal is produced by fully-formed indicators without any
  trade being taken on partially-warm data. Default embargo is the strategy
  maximum indicator lookback.
* **Every parameter combination evaluated is counted** and fed to the deflated
  Sharpe adjustment. Searching harder makes the bar higher, which is the point.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.engine import BacktestResult, _target_from_signals, run_backtest
from src.backtest.fast import run_fast
from src.backtest.sizing import PositionSizer
from src.config import Config
from src.data.loader import BarDataset
from src.metrics.deflated import DeflatedSharpeResult, deflated_sharpe_ratio
from src.metrics.performance import per_observation_sharpe
from src.strategies.base import Strategy, validate_signals

log = logging.getLogger(__name__)


@dataclass
class Fold:
    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_bars: int
    best_params: dict
    in_sample_objective: float
    n_combinations: int
    result: BacktestResult | None = None

    def as_row(self) -> dict:
        return {
            "fold": self.number,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "embargo_bars": self.embargo_bars,
            "combinations": self.n_combinations,
            **{f"param_{k}": v for k, v in self.best_params.items()},
            "in_sample_objective": self.in_sample_objective,
            "oos_net_pnl": float(self.result.net_pnl) if self.result else float("nan"),
            "oos_trades": int(len(self.result.trades)) if self.result else 0,
        }


@dataclass
class WalkForwardResult:
    strategy: str
    folds: list[Fold]
    oos_equity_net: pd.Series
    oos_equity_gross: pd.Series
    oos_costs_cum: pd.Series
    oos_trades: pd.DataFrame
    initial_capital: float
    bars_per_year: float
    total_trials: int
    trial_sharpes: np.ndarray
    deflated: DeflatedSharpeResult | None = None
    sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)

    def folds_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.as_row() for f in self.folds])

    def as_backtest_result(self) -> BacktestResult:
        """The concatenated out-of-sample record, shaped like a single backtest so
        the metrics and reporting code needs no special case."""
        position = pd.Series(0.0, index=self.oos_equity_net.index, name="position_oz")
        empty = pd.Series(0, index=self.oos_equity_net.index, dtype="int8")
        ruin = self.oos_equity_net[self.oos_equity_net <= 0]
        return BacktestResult(
            strategy=f"{self.strategy} (walk-forward OOS)",
            params={"folds": len(self.folds), "trials": self.total_trials},
            equity_net=self.oos_equity_net,
            equity_gross=self.oos_equity_gross,
            costs_cum=self.oos_costs_cum,
            position_oz=position,
            signal=empty,
            target_position=empty,
            trades=self.oos_trades,
            initial_capital=self.initial_capital,
            bars_per_year=self.bars_per_year,
            bars_per_day=float("nan"),
            resolution="",
            warnings=list(self.warnings),
            ruin_time=ruin.index[0] if len(ruin) else None,
        )


# ---------------------------------------------------------------------- #
def parameter_combinations(strategy: Strategy) -> list[dict]:
    grid = dict(getattr(strategy, "param_grid", {}) or {})
    if not grid:
        return [{}]
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def build_folds(
    index: pd.DatetimeIndex,
    *,
    in_sample_days: int,
    out_of_sample_days: int,
    min_in_sample_days: int,
    anchored: bool = True,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Train/test boundaries as timestamps. Anchored keeps the training start
    pinned to the beginning of the data; rolling slides it forward."""
    start, end = index[0], index[-1]
    train_span = pd.Timedelta(days=in_sample_days)
    test_span = pd.Timedelta(days=out_of_sample_days)
    minimum = pd.Timedelta(days=min_in_sample_days)

    folds = []
    train_end = start + train_span
    while train_end + test_span <= end + pd.Timedelta(seconds=1):
        train_start = start if anchored else max(start, train_end - train_span)
        if train_end - train_start < minimum:
            train_end += test_span
            continue
        test_end = min(end, train_end + test_span)
        folds.append((train_start, train_end, train_end, test_end))
        train_end += test_span
    return folds


def _objective_from_equity(equity: np.ndarray, objective: str) -> float:
    if equity.size < 3:
        return float("-inf")
    values = equity.astype("float64")
    nonpositive = np.flatnonzero(values <= 0)
    if nonpositive.size:
        values = values[: nonpositive[0] + 1]
    if values.size < 3:
        return float("-inf")
    rets = np.diff(values) / values[:-1]
    rets = rets[np.isfinite(rets)]
    if objective == "net_pnl":
        return float(equity[-1] - equity[0])
    std = rets.std(ddof=1) if rets.size > 1 else 0.0
    if std <= 0:
        return float("-inf")
    return float(rets.mean() / std)


class WalkForward:
    def __init__(
        self,
        dataset: BarDataset,
        strategy_class: type[Strategy],
        cfg: Config,
        *,
        base_params: dict | None = None,
        objective: str = "sharpe",
    ):
        self.dataset = dataset
        self.strategy_class = strategy_class
        self.cfg = cfg
        self.base_params = dict(base_params or {})
        self.objective = objective
        self.cost_model = CostModel.from_config(cfg)
        self.sizer = PositionSizer.from_config(cfg)
        self.capital = float(cfg.get("account.initial_capital_usd"))

    # ------------------------------------------------------------------ #
    def _embargo_bars(self, strategy: Strategy) -> int:
        configured = self.cfg.get("walk_forward.embargo_bars", None)
        if configured is not None:
            return int(configured)
        return int(strategy.max_lookback)

    def _fast_objective(self, strategy: Strategy, bars: pd.DataFrame, macro) -> float:
        """In-sample scoring.

        Uses the vectorised evaluator when sizing is fixed. That path is asserted
        equal to the event loop to the cent in the test suite, so this is a speed
        optimisation and not a second set of rules.
        """
        features = strategy.compute_features(bars, macro)
        signals = validate_signals(strategy.generate_signals(bars, features), bars, strategy.name)
        warmup = strategy.max_lookback
        if warmup > 0:
            if warmup >= len(bars):
                return float("-inf")
            signals = signals.copy()
            signals.iloc[:warmup] = 0
        target = _target_from_signals(signals).to_numpy()
        if not np.any(target):
            return float("-inf")

        size_oz = self.sizer.fixed_lots * self.sizer.contract_size_oz_per_lot
        result = run_fast(
            bars, target, size_oz=size_oz, initial_capital=self.capital,
            cost_model=self.cost_model, calendar=self.dataset.calendar,
        )
        return _objective_from_equity(result.equity_net, self.objective)

    def _slow_objective(self, strategy: Strategy, slice_dataset: BarDataset) -> float:
        try:
            result = run_backtest(slice_dataset, strategy, self.cfg,
                                  initial_capital=self.capital,
                                  cost_model=self.cost_model, sizer=self.sizer)
        except ValueError:
            return float("-inf")
        return _objective_from_equity(result.equity_net.to_numpy(), self.objective)

    # ------------------------------------------------------------------ #
    def run(self) -> WalkForwardResult:
        cfg = self.cfg
        index = self.dataset.bars.index
        anchored = str(cfg.get("walk_forward.mode", "anchored")) == "anchored"
        boundaries = build_folds(
            index,
            in_sample_days=int(cfg.get("walk_forward.in_sample_days")),
            out_of_sample_days=int(cfg.get("walk_forward.out_of_sample_days")),
            min_in_sample_days=int(cfg.get("walk_forward.min_in_sample_days")),
            anchored=anchored,
        )
        if not boundaries:
            raise ValueError(
                "no walk-forward folds fit in the data: shorten in_sample_days or "
                "out_of_sample_days, or load a longer history"
            )

        probe = self.strategy_class(**self.base_params)
        embargo = self._embargo_bars(probe)
        combinations = parameter_combinations(probe)
        use_fast = self.sizer.mode == "fixed"

        folds: list[Fold] = []
        trial_sharpes: list[float] = []
        equity_pieces: list[pd.Series] = []
        cost_pieces: list[pd.Series] = []
        trade_frames: list[pd.DataFrame] = []
        warnings: list[str] = []
        running_capital = self.capital
        total_trials = 0

        for number, (train_start, train_end, test_start, test_end) in enumerate(boundaries, start=1):
            train_mask = (index >= train_start) & (index < train_end)
            train_index = index[train_mask]
            if embargo > 0:
                # Purge: drop the bars adjacent to the boundary from training.
                train_index = train_index[:-embargo] if len(train_index) > embargo else train_index[:0]
            if len(train_index) < max(50, embargo * 2):
                warnings.append(f"fold {number}: too few training bars after purging, skipped")
                continue

            train_bars = self.dataset.bars.loc[train_index]
            train_macro = (
                self.dataset.macro.loc[train_index]
                if len(self.dataset.macro.columns) else None
            )
            train_slice = self.dataset.slice(train_index[0], train_index[-1])

            best_params, best_score = None, float("-inf")
            for params in combinations:
                candidate = self.strategy_class(**{**self.base_params, **params})
                score = (
                    self._fast_objective(candidate, train_bars, train_macro)
                    if use_fast
                    else self._slow_objective(candidate, train_slice)
                )
                total_trials += 1
                if np.isfinite(score):
                    trial_sharpes.append(score)
                if score > best_score:
                    best_params, best_score = params, score

            if best_params is None:
                warnings.append(f"fold {number}: no parameter set produced a usable result")
                continue

            # Out-of-sample. The warm-up region before the test window lets the
            # indicators form on data the test itself does not trade.
            strategy = self.strategy_class(**{**self.base_params, **best_params})
            warmup_needed = max(strategy.max_lookback, embargo)
            first_test = int(np.searchsorted(index, test_start, side="left"))
            warm_start = max(0, first_test - warmup_needed)
            oos_index = index[warm_start:][index[warm_start:] <= test_end]
            if len(oos_index) <= warmup_needed + 1:
                warnings.append(f"fold {number}: out-of-sample window too short, skipped")
                continue

            oos_dataset = self.dataset.slice(oos_index[0], oos_index[-1])
            suppressed = int(first_test - warm_start)
            result = run_backtest(
                oos_dataset, strategy, cfg,
                initial_capital=running_capital,
                cost_model=self.cost_model, sizer=self.sizer,
                warmup_bars=max(suppressed, strategy.max_lookback),
            )

            keep = result.equity_net.index >= test_start
            equity_pieces.append(result.equity_net[keep])
            cost_pieces.append(result.costs_cum[keep] - float(result.costs_cum[keep].iloc[0]))
            if len(result.trades):
                trade_frames.append(result.trades.assign(fold=number))

            running_capital = float(result.equity_net[keep].iloc[-1]) if keep.any() else running_capital
            folds.append(Fold(
                number=number, train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end, embargo_bars=embargo,
                best_params=best_params, in_sample_objective=best_score,
                n_combinations=len(combinations), result=result,
            ))
            log.info(
                "fold %d: train %s..%s -> test %s..%s, best %s, OOS net %.2f USD",
                number, train_start.date(), train_end.date(), test_start.date(),
                test_end.date(), best_params, result.net_pnl,
            )

            if running_capital <= 0:
                warnings.append(
                    f"the account reached zero during fold {number}; later folds are "
                    "not run because there is nothing left to trade"
                )
                break

        if not equity_pieces:
            raise ValueError("walk-forward produced no out-of-sample results")

        oos_equity = pd.concat(equity_pieces)
        oos_costs = _stitch_costs(cost_pieces)
        # Gross is rebuilt as net plus cumulative costs so the gross/net/cost
        # decomposition stays consistent across the fold seams, where each fold
        # restarts its own accounting from the running capital.
        oos_gross = oos_equity + oos_costs

        trades = (
            pd.concat(trade_frames).sort_values("entry_time").reset_index(drop=True)
            if trade_frames else pd.DataFrame()
        )

        from src.data.resample import bars_per_year

        stitched = WalkForwardResult(
            strategy=probe.name,
            folds=folds,
            oos_equity_net=oos_equity.rename("equity_net"),
            oos_equity_gross=oos_gross.rename("equity_gross"),
            oos_costs_cum=oos_costs.rename("costs_cum"),
            oos_trades=trades,
            initial_capital=self.capital,
            bars_per_year=bars_per_year(oos_equity.index),
            total_trials=total_trials,
            trial_sharpes=np.asarray(trial_sharpes, dtype="float64"),
            warnings=warnings,
        )

        sharpe, n_obs, skew, kurt = per_observation_sharpe(stitched.as_backtest_result())
        stitched.deflated = deflated_sharpe_ratio(
            sharpe=sharpe, n_observations=n_obs, n_trials=max(1, total_trials),
            trial_sharpes=stitched.trial_sharpes, skew=skew, kurtosis=kurt,
        )
        return stitched


def _stitch_costs(pieces: list[pd.Series]) -> pd.Series:
    """Make the per-fold cost curves continuous across seams."""
    running = 0.0
    out = []
    for piece in pieces:
        out.append(piece + running)
        running = float(out[-1].iloc[-1]) if len(out[-1]) else running
    return pd.concat(out)


def parameter_sensitivity(
    dataset: BarDataset,
    strategy_class: type[Strategy],
    cfg: Config,
    *,
    base_params: dict | None = None,
    objective: str = "sharpe",
) -> pd.DataFrame:
    """Score every parameter combination over the whole sample.

    **This is an in-sample diagnostic, not a performance claim.** It exists to
    answer one question: is the chosen parameter set sitting on a broad plateau or
    on a lonely spike? A sharp peak surrounded by losses is an artifact of the
    search. A broad plateau might be real.
    """
    cost_model = CostModel.from_config(cfg)
    sizer = PositionSizer.from_config(cfg)
    capital = float(cfg.get("account.initial_capital_usd"))
    probe = strategy_class(**(base_params or {}))
    combinations = parameter_combinations(probe)

    bars = dataset.bars
    macro = dataset.macro if len(dataset.macro.columns) else None
    rows = []
    for params in combinations:
        strategy = strategy_class(**{**(base_params or {}), **params})
        features = strategy.compute_features(bars, macro)
        signals = validate_signals(strategy.generate_signals(bars, features), bars, strategy.name)
        warmup = strategy.max_lookback
        if warmup > 0 and warmup < len(bars):
            signals = signals.copy()
            signals.iloc[:warmup] = 0
        target = _target_from_signals(signals).to_numpy()

        if sizer.mode == "fixed":
            size_oz = sizer.fixed_lots * sizer.contract_size_oz_per_lot
            evaluated = run_fast(bars, target, size_oz=size_oz, initial_capital=capital,
                                 cost_model=cost_model, calendar=dataset.calendar)
            equity = evaluated.equity_net
        else:
            evaluated = run_backtest(dataset, strategy, cfg, initial_capital=capital,
                                     cost_model=cost_model, sizer=sizer)
            equity = evaluated.equity_net.to_numpy()

        rows.append({
            **params,
            "objective": _objective_from_equity(equity, objective),
            "net_pnl_usd": float(equity[-1] - capital),
            "ruined": bool(np.any(equity <= 0)),
        })
    return pd.DataFrame(rows)
