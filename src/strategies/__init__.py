"""Strategy registry.

``RESEARCH_STRATEGIES`` is what the CLI and the reports iterate over by default.
``RETIRED_STRATEGIES`` can still be named explicitly but are not searched, so
their parameter trials no longer raise the deflated Sharpe bar for the rest. The
cheats live in a separate mapping so there is no path by which a deliberately
broken strategy ends up in a comparison table.
"""
from __future__ import annotations

from src.strategies.base import Strategy, validate_signals
from src.strategies.buy_and_hold import BuyAndHoldStrategy
from src.strategies.cheating import (
    CentredRollingCheatStrategy,
    FullSampleNormalisationCheatStrategy,
    LookaheadCheatStrategy,
)
from src.strategies.macro_trend import MacroFilteredTrendStrategy
from src.strategies.random_entry import RandomEntryStrategy, TradeProfile, profile_from_result
from src.strategies.rsi2 import Rsi2Strategy
from src.strategies.trend import DonchianTrendStrategy

RESEARCH_STRATEGIES: dict[str, type[Strategy]] = {
    BuyAndHoldStrategy.name: BuyAndHoldStrategy,
    DonchianTrendStrategy.name: DonchianTrendStrategy,
}

# Retired 2026-09-17 on the first walk-forward over real 2019-2026 bars, at
# five bar sizes. Both are kept, tested, and runnable by name; neither is
# searched by default, because every trial they consume raises the bar the
# remaining strategy has to clear. See README, "First results on real gold".
#
# rsi2: gross per trade 0.16 -> -0.17 -> -2.83 from 15min to 4h. The direction
#   calls lose before costs are counted. Not a tuning problem.
# trend_macro_filtered: 87th, 24th, 52nd, 20th, 18th percentile as bars
#   lengthen. It fails to inherit Donchian's monotonic improvement, so the
#   real-yield filter removes good trades as readily as bad ones.
RETIRED_STRATEGIES: dict[str, type[Strategy]] = {
    Rsi2Strategy.name: Rsi2Strategy,
    MacroFilteredTrendStrategy.name: MacroFilteredTrendStrategy,
}

BENCHMARK_STRATEGIES: dict[str, type[Strategy]] = {
    RandomEntryStrategy.name: RandomEntryStrategy,
}

# Test fixtures only. Deliberately not merged into the registry above.
CHEAT_STRATEGIES: dict[str, type[Strategy]] = {
    LookaheadCheatStrategy.name: LookaheadCheatStrategy,
    CentredRollingCheatStrategy.name: CentredRollingCheatStrategy,
    FullSampleNormalisationCheatStrategy.name: FullSampleNormalisationCheatStrategy,
}


def get_strategy(name: str, **params) -> Strategy:
    if name in RESEARCH_STRATEGIES:
        return RESEARCH_STRATEGIES[name](**params)
    if name in RETIRED_STRATEGIES:
        return RETIRED_STRATEGIES[name](**params)
    if name in BENCHMARK_STRATEGIES:
        return BENCHMARK_STRATEGIES[name](**params)
    if name in CHEAT_STRATEGIES:
        raise ValueError(
            f"{name!r} is a deliberately broken test fixture and cannot be run as "
            "research. Import it from src.strategies.cheating if you are testing "
            "the lookahead guards."
        )
    raise KeyError(
        f"unknown strategy {name!r}. Available: {sorted(RESEARCH_STRATEGIES)}; "
        f"retired but runnable by name: {sorted(RETIRED_STRATEGIES)}"
    )


__all__ = [
    "Strategy",
    "validate_signals",
    "BuyAndHoldStrategy",
    "Rsi2Strategy",
    "DonchianTrendStrategy",
    "MacroFilteredTrendStrategy",
    "RandomEntryStrategy",
    "TradeProfile",
    "profile_from_result",
    "RESEARCH_STRATEGIES",
    "RETIRED_STRATEGIES",
    "BENCHMARK_STRATEGIES",
    "CHEAT_STRATEGIES",
    "get_strategy",
]
