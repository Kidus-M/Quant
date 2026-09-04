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
        n = len(bars)
        rng = np.random.default_rng(int(self.params["seed"]))
        p_entry = float(self.params["entry_probability"])
        mean_hold = max(1.0, float(self.params["mean_hold_bars"]))
        long_fraction = float(self.params["long_fraction"])

        entries = rng.random(n) < p_entry
        directions = np.where(rng.random(n) < long_fraction, 1, -1).astype("int8")
        # Geometric holding times with the requested mean. p = 1/mean.
        holds = rng.geometric(1.0 / mean_hold, size=n)

        signal = np.zeros(n, dtype="int8")
        i = 0
        while i < n:
            if entries[i]:
                hold = int(holds[i])
                end = min(n, i + hold)
                signal[i:end] = directions[i]
                i = end
            else:
                i += 1
        return pd.Series(signal, index=bars.index, dtype="int8")

    @classmethod
    def matching(cls, profile: TradeProfile, seed: int) -> "RandomEntryStrategy":
        return cls(
            entry_probability=profile.entry_probability,
            mean_hold_bars=profile.mean_hold_bars,
            long_fraction=profile.long_fraction,
            seed=seed,
        )
