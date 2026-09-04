"""Strategy base class and the contract the engine enforces.

A strategy sees bars and features. It does not see the cost model, the portfolio,
the equity curve, or any bar later than the one it is deciding on. That is not a
convention here, it is enforced: ``generate_signals`` is handed only ``bars`` and
``features``, and ``tests/test_lookahead.py`` re-runs every strategy on truncated
data and asserts the signals are unchanged.

The output is a target position in {-1, 0, +1} at each bar. The value at bar ``t``
is a decision made from information available at the close of bar ``t``; the
engine fills it at the open of bar ``t+1``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np
import pandas as pd

VALID_SIGNALS = frozenset({-1, 0, 1})


class SignalContractError(ValueError):
    pass


class Strategy(ABC):
    """Subclass, set ``name``, declare ``param_grid``, implement the two methods."""

    name: str = "unnamed"
    # Searched by the walk-forward harness. Every combination counts as a trial
    # towards the deflated Sharpe adjustment, so keep grids honest and small.
    param_grid: dict[str, Iterable[Any]] = {}
    # Set True by strategies that need macro columns present.
    requires_macro: bool = False

    def __init__(self, **params: Any):
        self.params: dict[str, Any] = {**self.defaults(), **params}
        unknown = set(params) - set(self.defaults())
        if unknown:
            raise TypeError(f"{type(self).__name__} got unknown parameters: {sorted(unknown)}")

    # ------------------------------------------------------------------ #
    @classmethod
    def defaults(cls) -> dict[str, Any]:
        return {}

    @property
    def max_lookback(self) -> int:
        """Longest indicator window, in bars.

        Used for two things: dropping the warm-up region, and choosing the default
        purge/embargo length at walk-forward boundaries. Understating it produces
        leakage across the train/test seam.
        """
        return 0

    def compute_features(self, bars: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
        """Indicators this strategy needs, indexed exactly like ``bars``."""
        return pd.DataFrame(index=bars.index)

    @abstractmethod
    def generate_signals(self, bars: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """Target position in {-1, 0, +1} for every bar."""

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        if not self.params:
            return self.name
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({rendered})"

    def with_params(self, **params: Any) -> "Strategy":
        return type(self)(**{**self.params, **params})

    def __repr__(self) -> str:
        return f"<{self.describe()}>"


def validate_signals(signals: pd.Series, bars: pd.DataFrame, strategy_name: str = "strategy") -> pd.Series:
    """Enforce the signal contract, loudly.

    Every check here has caught a real bug at some point in a backtester of this
    shape: a reindexed series silently misaligning, a boolean mask leaking through
    as True/False, a NaN turning into a phantom position.
    """
    if not isinstance(signals, pd.Series):
        raise SignalContractError(f"{strategy_name}: generate_signals must return a Series")
    if len(signals) != len(bars):
        raise SignalContractError(
            f"{strategy_name}: returned {len(signals)} signals for {len(bars)} bars"
        )
    if not signals.index.equals(bars.index):
        raise SignalContractError(
            f"{strategy_name}: signal index does not match the bar index. A reindexed "
            "or reordered signal series shifts every decision relative to its bar."
        )
    filled = signals.fillna(0)
    values = filled.to_numpy()
    if not np.all(np.isin(values, [-1, 0, 1])):
        offenders = np.unique(values[~np.isin(values, [-1, 0, 1])])[:5]
        raise SignalContractError(
            f"{strategy_name}: signals must be in {{-1, 0, +1}}, found {offenders}"
        )
    return pd.Series(values.astype("int8"), index=bars.index, name="signal")
