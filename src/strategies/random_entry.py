"""Random entry benchmark.

This is the honest null hypothesis and it is a first-class output, not a footnote.
It enters at random with the same trade frequency, same long/short mix and same
average holding period as the strategy under test, and pays the same costs. Run a
thousand times, it gives a distribution of net returns that a coin flip with this
trade profile would have produced on this data.

If the strategy under test does not land outside the 95th percentile of that
distribution, it has demonstrated nothing. It has traded gold as often as a random
number generator and done no better.

The generator never looks at prices at all, so it cannot leak. Its only inputs are
the bar count, a seed, and the trade profile it is asked to match.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategies.base import Strategy

if TYPE_CHECKING:  # avoids a cycle: the engine imports the strategy registry
    from src.backtest.engine import BacktestResult


@dataclass(frozen=True)
class TradeProfile:
    """The shape of a strategy trading behaviour, for the benchmark to imitate."""

    n_trades: int
    mean_hold_bars: float
    long_fraction: float
    n_bars: int

    @property
    def entry_probability(self) -> float:
        """Probability of opening on any given flat bar.

        Chosen so the expected number of trades over the sample matches. Bars spent
        holding are unavailable for entry, hence the subtraction.
        """
        if self.n_trades <= 0:
            return 0.0
        flat_bars = max(1.0, self.n_bars - self.n_trades * self.mean_hold_bars)
        return float(min(1.0, self.n_trades / flat_bars))

    def describe(self) -> str:
        return (
            f"{self.n_trades} trades, mean hold {self.mean_hold_bars:.1f} bars, "
            f"{self.long_fraction:.0%} long"
        )


def profile_from_result(result: "BacktestResult") -> TradeProfile:
    trades = result.trades
    n_bars = len(result.equity_net)
    if trades.empty:
        return TradeProfile(0, 0.0, 0.5, n_bars)
    return TradeProfile(
        n_trades=int(len(trades)),
        mean_hold_bars=float(max(1.0, trades["bars_held"].mean())),
        long_fraction=float((trades["direction"] > 0).mean()),
        n_bars=n_bars,
    )


def random_signal_array(
    n_bars: int,
    *,
    entry_probability: float,
    mean_hold_bars: float,
    long_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One random position path in {-1, 0, +1}.

    Costs O(number of trades) rather than O(number of bars): flat gaps and holding
    periods are both drawn as geometric waiting times and written as slices. That
    matters because the benchmark runs this a thousand times over tens of
    thousands of bars.

    This is the single definition of a random path. The engine route and the fast
    route both call it, so the null hypothesis cannot quietly differ between the
    two places it is used.
    """
    signal = np.zeros(n_bars, dtype="int8")
    if entry_probability <= 0 or n_bars == 0:
        return signal
    mean_hold = max(1.0, float(mean_hold_bars))
    p_entry = min(1.0, float(entry_probability))

    position = 0
    while True:
        position += int(rng.geometric(p_entry))   # bars spent flat before entering
        if position >= n_bars:
            break
        hold = int(rng.geometric(1.0 / mean_hold))
        end = min(n_bars, position + hold)
        signal[position:end] = 1 if rng.random() < long_fraction else -1
        position = end
    return signal


class RandomEntryStrategy(Strategy):
    """Random entries with a configurable trade profile.

    Holding periods are drawn from a geometric distribution with the requested
    mean, which reproduces the memoryless holding behaviour of a random exit rule
    without pinning every trade to the same length.
    """

    name = "random_entry"
    param_grid: dict = {}

    @classmethod
    def defaults(cls) -> dict:
        return {
            "entry_probability": 0.01,
            "mean_hold_bars": 20.0,
            "long_fraction": 0.5,
            "seed": 0,
        }

    def generate_signals(self, bars: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        signal = random_signal_array(
            len(bars),
            entry_probability=float(self.params["entry_probability"]),
            mean_hold_bars=float(self.params["mean_hold_bars"]),
            long_fraction=float(self.params["long_fraction"]),
            rng=np.random.default_rng(int(self.params["seed"])),
        )
        return pd.Series(signal, index=bars.index, dtype="int8")

    @classmethod
    def matching(cls, profile: TradeProfile, seed: int) -> "RandomEntryStrategy":
        return cls(
            entry_probability=profile.entry_probability,
            mean_hold_bars=profile.mean_hold_bars,
            long_fraction=profile.long_fraction,
            seed=seed,
        )
