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

from src.features.indicators import atr, rsi, rsi_components, sma
from src.strategies.base import EntryLevel, Strategy


class Rsi2Strategy(Strategy):
    name = "rsi2"
    # 54 combinations, against 24 before the stop was added. ``oversold`` gives up
    # 15 to pay for it, because every combination is a trial against the deflated
    # Sharpe bar and a wider search has to clear a higher one.
    #
    # ``max_hold_bars`` is deliberately NOT searched. Measured on 15-minute bars it
    # never binds: with a 5- or 10-bar exit MA the close crosses back through it
    # long before any plausible time limit, so at 24 bars the time stop fired on
    # 0.0% of exits. Searching a parameter that changes nothing doubles the trial
    # count and raises the bar for free. It stays available as a knob because a
    # wider exit MA or a coarser bar size would make it bind again.
    param_grid = {
        "oversold": [5, 10, 20],
        "trend_ma": [100, 200, 400],
        "exit_ma": [5, 10],
        "atr_stop_multiple": [1.5, 2.0, 3.0],
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
            # Loss cap, in ATRs from the entry close. Fixed, not trailing: a
            # trailing stop on a mean-reversion entry would cut the trade exactly
            # when the move it is betting on begins. 0 disables it. At 2.0 the
            # stop accounts for roughly a fifth of exits on 15-minute bars; at 3.0
            # it is loose enough to fire on under a tenth.
            "atr_stop_multiple": 2.0,
            # Bars a position may be held before it is closed regardless. 0
            # disables it, which is the default because on 15-minute bars it never
            # binds -- see the note on param_grid. It exists for the case where a
            # wider exit MA leaves a position with no exit rule that fires on an
            # adverse move.
            "max_hold_bars": 0,
            # UTC hours (start, end) in which a position may be OPENED, half-open
            # and wrapping. None trades every hour, which is what the published
            # results were produced with. A round trip costs 0.95 USD/oz in the
            # Asian session against 0.50 in London/NY under the configured spread
            # multipliers, so restricting entries to (7, 16) or (12, 21) is a cost
            # decision before it is a signal one. Exits are never restricted.
            "trade_hours_utc": None,
        }

    @property
    def max_lookback(self) -> int:
        # max_hold_bars is deliberately absent: it bounds how long a position is
        # carried, not how far back a feature reads, so it cannot leak across a
        # walk-forward seam and must not inflate the embargo.
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

    def _walk(self, bars: pd.DataFrame, features: pd.DataFrame):
        """Run the entry/exit state machine once, returning signals and stops.

        Both ``generate_signals`` and ``current_stop`` call this, so the level an
        alert quotes as "your stop" is produced by the same code that decides the
        backtested exit, exactly as in the Donchian strategy. A stop written
        separately for the alerting would be a second definition of the strategy.

        Three things can close a position, in this order of precedence:

        1. the ATR loss cap, because a stop a slower rule can override is not a stop
        2. the time stop
        3. the original close-through-the-exit-MA rule

        The first two are the additions. Without them the only exit was the MA
        cross, which fires on a favourable move and never on an unfavourable one,
        so losers were held until they reverted or the sample ended. That is
        visible in the published numbers as a 58% win rate paired with an average
        loss almost twice the average win.
        """
        close = bars["close"].to_numpy(dtype="float64")
        rsi_v = features["rsi"].to_numpy(dtype="float64")
        trend = features["trend_ma"].to_numpy(dtype="float64")
        exit_ma = features["exit_ma"].to_numpy(dtype="float64")
        atr_v = features["atr"].to_numpy(dtype="float64")

        oversold = float(self.params["oversold"])
        overbought = float(self.params["overbought"])
        allow_shorts = bool(self.params["allow_shorts"])
        stop_mult = float(self.params["atr_stop_multiple"])
        max_hold = int(self.params["max_hold_bars"])

        long_entry = (rsi_v < oversold) & (close > trend)
        short_entry = (rsi_v > overbought) & (close < trend) if allow_shorts else np.zeros(len(close), dtype=bool)
        long_exit = close > exit_ma
        short_exit = close < exit_ma

        def entry_stop(price: float, a: float, direction: int) -> float:
            if stop_mult <= 0 or not np.isfinite(a):
                return np.nan
            return price - stop_mult * a if direction > 0 else price + stop_mult * a

        # A state machine rather than a vectorised mask: entry and exit conditions
        # can both be true on the same bar, and the position must be carried until
        # its own exit fires. Everything read here is at index i or earlier.
        signal = np.zeros(len(close), dtype="int8")
        stops = np.full(len(close), np.nan, dtype="float64")
        state = 0
        stop = np.nan
        held = 0

        for i in range(len(close)):
            price = close[i]
            a = atr_v[i]

            if state == 0:
                if long_entry[i]:
                    state, stop, held = 1, entry_stop(price, a, 1), 0
                elif short_entry[i]:
                    state, stop, held = -1, entry_stop(price, a, -1), 0
            elif state == 1:
                held += 1
                stopped = np.isfinite(stop) and price < stop
                timed_out = max_hold > 0 and held >= max_hold
                if stopped or timed_out or long_exit[i]:
                    state, stop, held = 0, np.nan, 0
                    if short_entry[i]:
                        state, stop = -1, entry_stop(price, a, -1)
            elif state == -1:
                held += 1
                stopped = np.isfinite(stop) and price > stop
                timed_out = max_hold > 0 and held >= max_hold
                if stopped or timed_out or short_exit[i]:
                    state, stop, held = 0, np.nan, 0
                    if long_entry[i]:
                        state, stop = 1, entry_stop(price, a, 1)

            signal[i] = state
            stops[i] = stop if state != 0 else np.nan
        return signal, stops

    def generate_signals(self, bars: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        signal, _ = self._walk(bars, features)
        signal = self._apply_trade_hours(signal, bars.index)
        return pd.Series(signal, index=bars.index, dtype="int8")

    def current_stop(self, bars: pd.DataFrame, features: pd.DataFrame) -> float | None:
        _, stops = self._walk(bars, features)
        value = float(stops[-1]) if len(stops) else float("nan")
        return value if np.isfinite(value) else None

    # ------------------------------------------------------------------ #
    # Phase 2: what price would actually trigger this setup on the next bar
    # ------------------------------------------------------------------ #
    def entry_levels(self, bars: pd.DataFrame, features: pd.DataFrame) -> list[EntryLevel]:
        """Invert Wilder RSI to get the trigger price.

        RSI is usually treated as unreadable in price terms, but for a fixed
        period it inverts in closed form, which is what makes an "approaching"
        alert possible at all. With alpha = 1/n and the running averages
        ``ag`` and ``al`` at the current bar, a next-bar close change of ``d``
        gives::

            d <= 0:  ag' = ag(1-a),            al' = al(1-a) - d*a
            d >= 0:  ag' = ag(1-a) + d*a,      al' = al(1-a)

        Setting RSI' to the threshold T and solving:

            long  (RSI falls to T):   d = (n-1) * ( al - ag*(100-T)/T )
            short (RSI rises to T):   d = (n-1) * ( T*al/(100-T) - ag )

        The sign of ``d`` is the sanity check: a long trigger needs a down move,
        so a positive ``d`` means RSI is already below the threshold and the
        setup is live rather than approaching.
        """
        if len(bars) < 2:
            return []

        period = int(self.params["rsi_period"])
        avg_gain, avg_loss = rsi_components(bars["close"], period)
        ag = float(avg_gain.iloc[-1])
        al = float(avg_loss.iloc[-1])
        close = float(bars["close"].iloc[-1])
        trend_ma = float(features["trend_ma"].iloc[-1])
        smoothing = float(period - 1)   # (1 - alpha) / alpha for alpha = 1/period

        if not (np.isfinite(ag) and np.isfinite(al)):
            return []

        levels: list[EntryLevel] = []

        oversold = float(self.params["oversold"])
        if oversold > 0:
            move = smoothing * (al - ag * (100.0 - oversold) / oversold)
            price = close + move
            blocked: list[str] = []
            if move > 0:
                blocked.append(f"RSI({period}) is already below {oversold:g}")
            if np.isfinite(trend_ma) and price <= trend_ma:
                blocked.append(
                    f"the trigger price sits below the {int(self.params['trend_ma'])}-period "
                    "MA, so the long filter would reject it"
                )
            if not np.isfinite(trend_ma):
                blocked.append("trend filter still warming up")
            levels.append(EntryLevel(
                direction=1, price=price, kind="mean_reversion",
                blocked_by=tuple(blocked),
                note=f"close that drives RSI({period}) below {oversold:g} while above the trend MA",
            ))

        overbought = float(self.params["overbought"])
        if bool(self.params["allow_shorts"]) and overbought < 100:
            move = smoothing * (overbought * al / (100.0 - overbought) - ag)
            price = close + move
            blocked = []
            if move < 0:
                blocked.append(f"RSI({period}) is already above {overbought:g}")
            if np.isfinite(trend_ma) and price >= trend_ma:
                blocked.append(
                    f"the trigger price sits above the {int(self.params['trend_ma'])}-period "
                    "MA, so the short filter would reject it"
                )
            if not np.isfinite(trend_ma):
                blocked.append("trend filter still warming up")
            levels.append(EntryLevel(
                direction=-1, price=price, kind="mean_reversion",
                blocked_by=tuple(blocked),
                note=f"close that drives RSI({period}) above {overbought:g} while below the trend MA",
            ))

        return levels
