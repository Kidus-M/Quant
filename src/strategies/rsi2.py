"""RSI(2) mean reversion, after Connors and Alvarez.

Original rules, on daily bars:

* Long when RSI(2) is below 10 and price is above its 200-period moving average
* Exit when price closes above its 5-period moving average
* Symmetric short rules below the 200-period moving average

**Read the results on this one with suspicion.** The rules were published in 2008,
designed for daily bars on US equity indices, in a market with a strong mean
reverting tendency at the index level and a structural long bias. None of that is
true of intraday spot gold. Applying them to 5-minute bars changes the holding
period by two orders of magnitude while leaving the cost per trade untouched,
which is exactly the regime where a strategy can look excellent gross and lose
money net.

A strong intraday result here is more likely to be one of: the 200-period MA
acting as a slow trend filter that happens to fit this sample, or the exit rule
capturing bid-ask bounce that the cost model has not fully priced. Check it
against the random benchmark before believing any of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.indicators import atr, rsi, sma
from src.strategies.base import Strategy


class Rsi2Strategy(Strategy):
    name = "rsi2"
    param_grid = {
        "oversold": [5, 10, 15, 20],
        "trend_ma": [100, 200, 400],
        "exit_ma": [5, 10],
    }

    @classmethod
    def defaults(cls) -> dict:
        return {
            "rsi_period": 2,
            "oversold": 10,
            "overbought": 90,
            "trend_ma": 200,
            "exit_ma": 5,
            "allow_shorts": True,
            "atr_period": 14,
        }

    @property
    def max_lookback(self) -> int:
        return int(max(self.params["trend_ma"], self.params["exit_ma"], self.params["rsi_period"],
                       self.params["atr_period"]))

    def compute_features(self, bars: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
        close = bars["close"]
        return pd.DataFrame(
            {
                "rsi": rsi(close, int(self.params["rsi_period"])),
                "trend_ma": sma(close, int(self.params["trend_ma"])),
                "exit_ma": sma(close, int(self.params["exit_ma"])),
                "atr": atr(bars, int(self.params["atr_period"])),
            },
            index=bars.index,
        )

    def generate_signals(self, bars: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        close = bars["close"].to_numpy(dtype="float64")
        rsi_v = features["rsi"].to_numpy(dtype="float64")
        trend = features["trend_ma"].to_numpy(dtype="float64")
        exit_ma = features["exit_ma"].to_numpy(dtype="float64")

        oversold = float(self.params["oversold"])
        overbought = float(self.params["overbought"])
        allow_shorts = bool(self.params["allow_shorts"])

        long_entry = (rsi_v < oversold) & (close > trend)
        short_entry = (rsi_v > overbought) & (close < trend) if allow_shorts else np.zeros(len(close), dtype=bool)
        long_exit = close > exit_ma
        short_exit = close < exit_ma

        # A state machine rather than a vectorised mask: entry and exit conditions
        # can both be true on the same bar, and the position must be carried until
        # its own exit fires. Everything read here is at index i or earlier.
        signal = np.zeros(len(close), dtype="int8")
        state = 0
        for i in range(len(close)):
            if state == 0:
                if long_entry[i]:
                    state = 1
                elif short_entry[i]:
                    state = -1
            elif state == 1:
                if long_exit[i]:
                    state = 0
                    if short_entry[i]:
                        state = -1
            elif state == -1:
                if short_exit[i]:
                    state = 0
                    if long_entry[i]:
                        state = 1
            signal[i] = state
        return pd.Series(signal, index=bars.index, dtype="int8")
