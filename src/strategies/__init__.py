"""Strategy registry.

``RESEARCH_STRATEGIES`` is what the CLI and the reports iterate over. The cheats
live in a separate mapping so there is no path by which a deliberately broken
strategy ends up in a comparison table.
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
    Rsi2Strategy.name: Rsi2Strategy,
    DonchianTrendStrategy.name: DonchianTrendStrategy,
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
    if name in BENCHMARK_STRATEGIES:
        return BENCHMARK_STRATEGIES[name](**params)
    if name in CHEAT_STRATEGIES:
        raise ValueError(
            f"{name!r} is a deliberately broken test fixture and cannot be run as "
            "research. Import it from src.strategies.cheating if you are testing "
            "the lookahead guards."
        )
    raise KeyError(f"unknown strategy {name!r}. Available: {sorted(RESEARCH_STRATEGIES)}")


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
    "BENCHMARK_STRATEGIES",
    "CHEAT_STRATEGIES",
    "get_strategy",
]
