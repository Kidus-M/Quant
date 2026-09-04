"""The random-entry benchmark, run as a distribution rather than a single number.

A strategy result on its own is not evidence. The question is whether it is better
than trading gold at random with the same frequency, the same long/short mix, the
same average holding period and the same costs. This runs that null a thousand
times and reports where the strategy falls in the resulting distribution.

The reported verdict is deliberately blunt. If the strategy does not clear the
95th percentile of the random distribution, the summary says it has not
demonstrated an edge, regardless of how good the equity curve looks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.engine import BacktestResult
from src.backtest.fast import run_fast
from src.backtest.sizing import PositionSizer
from src.config import Config
from src.data.loader import BarDataset
from src.strategies.random_entry import TradeProfile, profile_from_result, random_signal_array

log = logging.getLogger(__name__)


@dataclass
class RandomBenchmarkResult:
    n_runs: int
    seed: int
    profile: TradeProfile
    size_oz: float
    strategy_net_pnl: float
    net_pnl: np.ndarray
    percentile_threshold: float
    initial_capital: float
    notes: list[str] = field(default_factory=list)

    # -------------------------------------------------------------- #
    @property
    def threshold_value(self) -> float:
        return float(np.percentile(self.net_pnl, self.percentile_threshold))

    @property
    def strategy_percentile(self) -> float:
        """Where the strategy sits in the random distribution, 0-100."""
        if not self.net_pnl.size:
            return float("nan")
        return float(100.0 * (self.net_pnl < self.strategy_net_pnl).mean())

    @property
    def passes(self) -> bool:
        return bool(self.strategy_percentile >= self.percentile_threshold)

    @property
    def ruin_fraction(self) -> float:
        return float((self.net_pnl <= -self.initial_capital + 1e-9).mean()) if self.net_pnl.size else 0.0

    def summary_lines(self) -> list[str]:
        if not self.net_pnl.size:
            return ["random benchmark: not run (the strategy took no trades)"]
        verdict = (
            "BEATS the random benchmark"
            if self.passes
            else "does NOT beat the random benchmark"
        )
        lines = [
            f"Random benchmark ({self.n_runs} runs matched to {self.profile.describe()}):",
            f"  strategy net P&L      {self.strategy_net_pnl:>10,.2f} USD",
            f"  random median         {np.median(self.net_pnl):>10,.2f} USD",
            f"  random {self.percentile_threshold:.0f}th percentile "
            f"{self.threshold_value:>10,.2f} USD",
            f"  strategy sits at the {self.strategy_percentile:.1f}th percentile "
            f"of random -- it {verdict}.",
        ]
        if self.ruin_fraction > 0:
            lines.append(
                f"  {self.ruin_fraction:.0%} of random runs wiped out the account entirely."
            )
        lines.extend(f"  note: {n}" for n in self.notes)
        return lines


def run_random_benchmark(
    dataset: BarDataset,
    reference: BacktestResult,
    cfg: Config,
    *,
    n_runs: int | None = None,
    seed: int | None = None,
    cost_model: CostModel | None = None,
    sizer: PositionSizer | None = None,
) -> RandomBenchmarkResult:
    cost_model = cost_model or CostModel.from_config(cfg)
    sizer = sizer or PositionSizer.from_config(cfg)
    n_runs = int(n_runs if n_runs is not None else cfg.get("random_benchmark.n_runs", 1000))
    seed = int(seed if seed is not None else cfg.get("random_benchmark.seed", 0))
    threshold = float(cfg.get("random_benchmark.percentile_threshold", 95))
    capital = reference.initial_capital

    profile = profile_from_result(reference)
    notes: list[str] = []

    if profile.n_trades == 0:
        return RandomBenchmarkResult(
            n_runs=0, seed=seed, profile=profile, size_oz=0.0,
            strategy_net_pnl=reference.net_pnl, net_pnl=np.zeros(0),
            percentile_threshold=threshold, initial_capital=capital,
            notes=["the strategy took no trades, so there is nothing to compare"],
        )

    if sizer.mode == "fixed":
        size_oz = sizer.fixed_lots * sizer.contract_size_oz_per_lot
    else:
        # The vectorised evaluator only supports constant size. Matching the median
        # size the strategy actually traded keeps the comparison in the same units;
        # the benchmark is then a like-for-like null, not an identical mechanism.
        size_oz = float(reference.trades["size_oz"].median())
        notes.append(
            f"strategy uses {sizer.mode} sizing; the benchmark holds size constant "
            f"at the strategy median of {size_oz:.2f} oz"
        )

    bars = dataset.bars
    index = bars.index
    n = len(bars)

    # Hoist the per-bar constants out of the run loop.
    multipliers = cost_model.spread_multiplier_series(index).to_numpy(dtype="float64")
    cost_per_oz = (
        0.5 * cost_model.spread_usd_per_oz_round_trip * multipliers
        + cost_model.slippage_usd_per_oz_per_side
    )
    rollovers = np.zeros(n, dtype="int64")
    if n > 1:
        rollovers[1:] = dataset.calendar.rollovers_between(index[:-1], index[1:])

    rng = np.random.default_rng(seed)
    net_pnl = np.empty(n_runs, dtype="float64")

    log.info("running %d random-entry benchmark paths matched to %s", n_runs, profile.describe())
    for run in range(n_runs):
        signal = random_signal_array(
            n,
            entry_probability=profile.entry_probability,
            mean_hold_bars=profile.mean_hold_bars,
            long_fraction=profile.long_fraction,
            rng=rng,
        )
        # Same one-bar execution delay the engine applies. Without it the benchmark
        # would be the easier game, and beating it would mean nothing.
        target = np.concatenate(([0], signal[:-1]))
        result = run_fast(
            bars, target, size_oz=size_oz, initial_capital=capital,
            cost_model=cost_model, calendar=dataset.calendar,
            _rollovers=rollovers, _cost_per_oz=cost_per_oz,
        )
        net_pnl[run] = result.equity_net[-1] - capital

    return RandomBenchmarkResult(
        n_runs=n_runs, seed=seed, profile=profile, size_oz=size_oz,
        strategy_net_pnl=reference.net_pnl, net_pnl=net_pnl,
        percentile_threshold=threshold, initial_capital=capital, notes=notes,
    )
