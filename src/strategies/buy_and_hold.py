"""Buy and hold. The benchmark every other strategy is measured against.

Long from the first bar to the last. It pays one round trip of costs and carries
financing every night, which for long gold is a real drag, so its net result is
not the same as the price change over the period. That difference is itself
informative: it is the floor a trading strategy has to clear before its activity
has bought anything.
"""
from __future__ import annotations

import pandas as pd

from src.strategies.base import Strategy


class BuyAndHoldStrategy(Strategy):
    name = "buy_and_hold"
    param_grid: dict = {}

    def generate_signals(self, bars: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=bars.index, dtype="int8")
