"""Trend following baseline: Donchian breakout with an ATR trailing stop.

This family has the longest live track record of anything in the comparison set,
which is why it belongs here even if it does not win on this data. If a novel
strategy cannot beat a Donchian breakout net of costs, that is worth knowing
before any more time goes into it.

Rules:

* Long when the close exceeds the highest high of the previous ``entry_window``
  bars. Short on the mirror image.
* Exit on an ATR trailing stop: the stop trails the best close achieved since
  entry by ``atr_stop_multiple`` ATRs and never moves against the position.
* A breakout in the opposite direction reverses.

**Stops are evaluated on closes, and the exit fills at the next open.** No
intrabar stop fills. Filling a stop intrabar requires assuming the order in which
the high and low were reached inside the bar, and every such assumption flatters
the backtest. The cost is that a fast reversal is exited one bar late, which is
pessimistic, which is the right direction to be wrong in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.indicators import atr, donchian
from src.strategies.base import Strategy


class DonchianTrendStrategy(Strategy):
    name = "trend_donchian"
    param_grid = {
        "entry_window": [20, 40, 55, 100],
        "atr_stop_multiple": [2.0, 3.0, 4.0],
    }

    @classmethod
    def defaults(cls) -> dict:
        return {
            "entry_window": 55,
            "atr_period": 14,
            "atr_stop_multiple": 3.0,
            "allow_shorts": True,
        }

    @property
    def max_lookback(self) -> int:
        # +1 because the Donchian channel is shifted to exclude the current bar.
        return int(max(self.params["entry_window"] + 1, self.params["atr_period"]))

    def compute_features(self, bars: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
        upper, lower = donchian(bars, int(self.params["entry_window"]))
        return pd.DataFrame(
            {
                "donchian_upper": upper,
                "donchian_lower": lower,
                "atr": atr(bars, int(self.params["atr_period"])),
            },
            index=bars.index,
        )

    def generate_signals(self, bars: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        close = bars["close"].to_numpy(dtype="float64")
        upper = features["donchian_upper"].to_numpy(dtype="float64")
        lower = features["donchian_lower"].to_numpy(dtype="float64")
        atr_v = features["atr"].to_numpy(dtype="float64")

        allow_shorts = bool(self.params["allow_shorts"])
        stop_mult = float(self.params["atr_stop_multiple"])

        long_entry = close > upper
        short_entry = (close < lower) if allow_shorts else np.zeros(len(close), dtype=bool)

        signal = np.zeros(len(close), dtype="int8")
        state = 0
        best = np.nan     # best close achieved since entry, in the trade direction
        stop = np.nan

        for i in range(len(close)):
            price = close[i]
            a = atr_v[i]

            if state == 1:
                if price > best:
                    best = price
                if np.isfinite(a):
                    # The stop ratchets: it can tighten but never loosen.
                    candidate = best - stop_mult * a
                    stop = candidate if not np.isfinite(stop) else max(stop, candidate)
                if np.isfinite(stop) and price < stop:
                    state = 0
            elif state == -1:
                if price < best:
                    best = price
                if np.isfinite(a):
                    candidate = best + stop_mult * a
                    stop = candidate if not np.isfinite(stop) else min(stop, candidate)
                if np.isfinite(stop) and price > stop:
                    state = 0

            if state == 0:
                if long_entry[i] and np.isfinite(a):
                    state, best, stop = 1, price, price - stop_mult * a
                elif short_entry[i] and np.isfinite(a):
                    state, best, stop = -1, price, price + stop_mult * a
            elif state == 1 and short_entry[i] and np.isfinite(a):
                state, best, stop = -1, price, price + stop_mult * a
            elif state == -1 and long_entry[i] and np.isfinite(a):
                state, best, stop = 1, price, price - stop_mult * a

            signal[i] = state
        return pd.Series(signal, index=bars.index, dtype="int8")
