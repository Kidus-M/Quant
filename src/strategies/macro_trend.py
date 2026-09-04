"""Trend following with a real-yield regime filter.

Identical to ``trend_donchian`` except that direction is gated on the 10-year TIPS
real yield (``DFII10``):

* longs only while the real yield is falling on a 20-print basis
* shorts only while it is rising

The economic story is the standard one: gold pays no coupon, so a falling real
yield lowers the opportunity cost of holding it. The point of running this side by
side with the unfiltered version is to find out whether the macro filter actually
adds anything over price alone, or whether it is an expensive way to trade less.

Two details keep this honest:

* The macro series arrives already lagged by at least one publication day and
  forward-filled backward-only (see ``src/data/fred.py``).
* The 20-day change is computed across *prints*, at the bar where each new value
  first appears, not across calendar bars. Diffing the forward-filled intraday
  series would compare a value against itself for most of the day, and the
  first bar of a new print would be the only one carrying information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import EntryLevel, Strategy
from src.strategies.trend import DonchianTrendStrategy


def prints_change(series: pd.Series, periods: int) -> pd.Series:
    """Change over the last ``periods`` distinct published values.

    ``series`` is the forward-filled intraday view of a daily series. Consecutive
    identical prints are indistinguishable from a value that did not change, so a
    genuinely flat series is treated as no new information. For a yield quoted to
    two decimals that is rare, and treating it as no-change is the conservative
    reading.
    """
    changed = series.ne(series.shift(1)) & series.notna()
    prints = series[changed]
    delta = prints.diff(periods)
    return delta.reindex(series.index).ffill()


class MacroFilteredTrendStrategy(DonchianTrendStrategy):
    name = "trend_macro_filtered"
    requires_macro = True
    param_grid = {
        "entry_window": [20, 40, 55, 100],
        "atr_stop_multiple": [2.0, 3.0, 4.0],
        "macro_lookback_prints": [10, 20, 40],
    }

    @classmethod
    def defaults(cls) -> dict:
        return {
            **DonchianTrendStrategy.defaults(),
            "macro_series": "DFII10",
            "macro_lookback_prints": 20,
        }

    def compute_features(self, bars: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
        features = super().compute_features(bars, macro)
        column = str(self.params["macro_series"])
        if macro is None or column not in macro.columns:
            raise ValueError(
                f"{self.name} needs the {column!r} macro column. Without it the "
                "filter would be a no-op and this would silently become the "
                "unfiltered trend strategy."
            )
        series = macro[column].astype("float64")
        features["macro_level"] = series
        features["macro_change"] = prints_change(series, int(self.params["macro_lookback_prints"]))
        return features

    def generate_signals(self, bars: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        base = super().generate_signals(bars, features).to_numpy()
        change = features["macro_change"].to_numpy(dtype="float64")

        # Unknown regime (before the first full lookback) means no trade, not a
        # free pass. Filling it either way would invent a regime.
        allow_long = change < 0
        allow_short = change > 0

        filtered = np.where(
            (base > 0) & allow_long, 1,
            np.where((base < 0) & allow_short, -1, 0),
        ).astype("int8")
        return pd.Series(filtered, index=bars.index, dtype="int8")

    def entry_levels(self, bars: pd.DataFrame, features: pd.DataFrame) -> list[EntryLevel]:
        """Donchian levels, gated by the current real-yield regime.

        A blocked level is still reported rather than hidden. "Price is at the
        breakout but the macro filter is holding it back" is the single most
        useful thing this strategy can say, and dropping the level entirely would
        make it look like there was simply no setup.
        """
        levels = super().entry_levels(bars, features)
        if not levels or "macro_change" not in features.columns:
            return levels

        change = float(features["macro_change"].iloc[-1])
        series = str(self.params["macro_series"])
        prints = int(self.params["macro_lookback_prints"])

        if not np.isfinite(change):
            reason = f"{series} regime unknown (fewer than {prints} prints available)"
            return [
                EntryLevel(l.direction, l.price, l.kind, l.blocked_by + (reason,), l.note)
                for l in levels
            ]

        gated = []
        for level in levels:
            blocked = list(level.blocked_by)
            if level.direction > 0 and change >= 0:
                blocked.append(
                    f"{series} is rising over {prints} prints ({change:+.3f}), so longs are filtered out"
                )
            if level.direction < 0 and change <= 0:
                blocked.append(
                    f"{series} is falling over {prints} prints ({change:+.3f}), so shorts are filtered out"
                )
            gated.append(EntryLevel(level.direction, level.price, level.kind,
                                    tuple(blocked), level.note))
        return gated
