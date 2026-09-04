"""Deliberately broken strategies. Test fixtures, never for research use.

These exist so the lookahead guards can be shown to work. A test suite that only
ever runs on correct code proves nothing: it passes whether the guards fire or
not. ``tests/test_lookahead.py`` asserts that every strategy here is caught.

If a change to the guards lets one of these through, the guards are broken and
every result the engine has ever produced is suspect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import Strategy


class LookaheadCheatStrategy(Strategy):
    """Peeks exactly one bar ahead and trades the answer.

    The subtle part, and the reason this specific cheat is the fixture: shifting
    by -1 and filling at the next bar open looks superficially like a legitimate
    next-bar execution. The signal at bar t is built from the close of bar t+1, so
    the fill at the open of t+1 happens before the information exists. It produces
    a beautiful equity curve and the trade log looks entirely normal.
    """

    name = "cheat_lookahead"
    is_test_fixture = True

    @classmethod
    def defaults(cls) -> dict:
        return {"horizon": 1}

    def generate_signals(self, bars: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        horizon = int(self.params["horizon"])
        future = bars["close"].shift(-horizon)
        direction = np.sign(future - bars["close"]).fillna(0.0)
        return pd.Series(direction.to_numpy().astype("int8"), index=bars.index, dtype="int8")


class CentredRollingCheatStrategy(Strategy):
    """Uses a centred moving average, half of which is the future.

    Quieter than the shift cheat and correspondingly easier to ship by accident:
    there is no negative shift anywhere, just ``center=True`` on a rolling window.
    """

    name = "cheat_centred_ma"
    is_test_fixture = True

    @classmethod
    def defaults(cls) -> dict:
        return {"window": 20}

    @property
    def max_lookback(self) -> int:
        return int(self.params["window"])

    def generate_signals(self, bars: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        window = int(self.params["window"])
        centred = bars["close"].rolling(window, center=True, min_periods=1).mean()
        signal = np.where(bars["close"].to_numpy() > centred.to_numpy(), 1, -1).astype("int8")
        return pd.Series(signal, index=bars.index, dtype="int8")


class FullSampleNormalisationCheatStrategy(Strategy):
    """Z-scores the close against statistics fitted on the entire sample.

    Nothing here reads a future bar directly, which is what makes it dangerous.
    The mean and standard deviation encode the whole period, so the strategy knows
    in 2019 what counted as expensive in 2024.
    """

    name = "cheat_full_sample_zscore"
    is_test_fixture = True

    @classmethod
    def defaults(cls) -> dict:
        return {"threshold": 1.0}

    def generate_signals(self, bars: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        z = (close - close.mean()) / close.std()
        threshold = float(self.params["threshold"])
        signal = np.where(z < -threshold, 1, np.where(z > threshold, -1, 0)).astype("int8")
        return pd.Series(signal, index=bars.index, dtype="int8")
