"""Transaction cost model.

Pessimistic by default, configurable, and applied against the trade every time.

The model separates the two things a naive backtester conflates:

* **Gross P&L** is computed from the raw next-bar open. It is what the price did.
* **Costs** are accumulated separately as spread, slippage, commission and
  financing. Net P&L is gross minus costs.

An equivalent way to write the same trade is to move the fill price against you by
the half-spread plus slippage. Both are computed here, and
``tests/test_cost_model.py`` asserts they agree to the cent, so the decomposition
in the report is not a different number from the one the trade log implies.

On a $50 account this file decides the outcome. A 0.30 round-trip spread on one
ounce is $0.30, which is 0.6% of the account per trade before the market moves at
all. Ten trades a day is 6% a day of pure cost drag.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Config

BUY = 1
SELL = -1


@dataclass(frozen=True)
class TradeCosts:
    """Costs for one side of one trade, in USD. All values are non-negative
    charges except ``financing``, which can be a credit for short positions."""

    spread: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0
    financing: float = 0.0

    @property
    def total(self) -> float:
        return self.spread + self.slippage + self.commission + self.financing

    def __add__(self, other: "TradeCosts") -> "TradeCosts":
        return TradeCosts(
            spread=self.spread + other.spread,
            slippage=self.slippage + other.slippage,
            commission=self.commission + other.commission,
            financing=self.financing + other.financing,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "spread": self.spread,
            "slippage": self.slippage,
            "commission": self.commission,
            "financing": self.financing,
            "total": self.total,
        }


@dataclass(frozen=True)
class CostModel:
    # Spread is quoted round trip, the way a broker advertises it, and halved here
    # so that each side pays its share.
    spread_usd_per_oz_round_trip: float = 0.30
    slippage_usd_per_oz_per_side: float = 0.10
    commission_usd_per_lot_per_side: float = 0.0
    swap_long_usd_per_oz_per_night: float = -0.15
    swap_short_usd_per_oz_per_night: float = 0.05
    contract_size_oz_per_lot: float = 100.0
    # Multiplier on the spread by UTC hour. Missing hours are 1.0.
    spread_session_multipliers: dict[int, float] = field(default_factory=dict)
    news_spread_multiplier: float = 1.0
    news_window_minutes: int = 15
    news_timestamps: tuple[pd.Timestamp, ...] = ()

    @classmethod
    def from_config(cls, cfg: Config, news_timestamps=()) -> "CostModel":
        c = cfg.section("costs")
        raw_mult = c.get("spread_session_multipliers", {}) or {}
        multipliers = {int(k): float(v) for k, v in raw_mult.items()}
        return cls(
            spread_usd_per_oz_round_trip=float(c.get("spread_usd_per_oz_round_trip")),
            slippage_usd_per_oz_per_side=float(c.get("slippage_usd_per_oz_per_side")),
            commission_usd_per_lot_per_side=float(c.get("commission_usd_per_lot_per_side")),
            swap_long_usd_per_oz_per_night=float(c.get("swap_long_usd_per_oz_per_night")),
            swap_short_usd_per_oz_per_night=float(c.get("swap_short_usd_per_oz_per_night")),
            contract_size_oz_per_lot=float(cfg.get("instrument.contract_size_oz_per_lot", 100.0)),
            spread_session_multipliers=multipliers,
            news_spread_multiplier=float(c.get("news_spread_multiplier", 1.0)),
            news_window_minutes=int(c.get("news_window_minutes", 15)),
            news_timestamps=tuple(pd.Timestamp(t) for t in news_timestamps),
        )

    # ------------------------------------------------------------------ #
    # Spread
    # ------------------------------------------------------------------ #
    def hourly_multiplier_table(self) -> np.ndarray:
        table = np.ones(24, dtype="float64")
        for hour, mult in self.spread_session_multipliers.items():
            table[int(hour) % 24] = float(mult)
        return table

    def spread_multiplier(self, ts: pd.Timestamp) -> float:
        mult = self.hourly_multiplier_table()[pd.Timestamp(ts).hour]
        if self._near_news(ts):
            mult *= self.news_spread_multiplier
        return float(mult)

    def spread_multiplier_series(self, index: pd.DatetimeIndex) -> pd.Series:
        table = self.hourly_multiplier_table()
        values = table[pd.DatetimeIndex(index).hour.to_numpy()]
        if self.news_timestamps:
            window = pd.Timedelta(minutes=self.news_window_minutes)
            near = np.zeros(len(index), dtype=bool)
            idx = pd.DatetimeIndex(index)
            for event in self.news_timestamps:
                near |= (idx >= event - window) & (idx <= event + window)
            values = np.where(near, values * self.news_spread_multiplier, values)
        return pd.Series(values, index=index, name="spread_multiplier")

    def _near_news(self, ts: pd.Timestamp) -> bool:
        if not self.news_timestamps:
            return False
        ts = pd.Timestamp(ts)
        window = pd.Timedelta(minutes=self.news_window_minutes)
        return any(abs(ts - event) <= window for event in self.news_timestamps)

    def half_spread_per_oz(self, ts: pd.Timestamp) -> float:
        """Cost per ounce for one side, in USD."""
        return 0.5 * self.spread_usd_per_oz_round_trip * self.spread_multiplier(ts)

    def cost_per_oz(self, ts: pd.Timestamp) -> float:
        """Total per-ounce price concession for one side: half spread plus slippage."""
        return self.half_spread_per_oz(ts) + self.slippage_usd_per_oz_per_side

    # ------------------------------------------------------------------ #
    # Fills
    # ------------------------------------------------------------------ #
    def effective_fill_price(self, raw_price: float, side: int, ts: pd.Timestamp) -> float:
        """Move the fill against the trade. Buys pay up, sells hit the bid.

        There is no code path that moves a fill in the favourable direction.
        """
        if side not in (BUY, SELL):
            raise ValueError(f"side must be +1 (buy) or -1 (sell), got {side!r}")
        return float(raw_price + side * self.cost_per_oz(ts))

    def trade_costs(self, oz: float, side: int, ts: pd.Timestamp) -> TradeCosts:
        """Costs in USD for transacting ``oz`` ounces on one side."""
        quantity = abs(float(oz))
        if quantity == 0.0:
            return TradeCosts()
        lots = quantity / self.contract_size_oz_per_lot
        return TradeCosts(
            spread=quantity * self.half_spread_per_oz(ts),
            slippage=quantity * self.slippage_usd_per_oz_per_side,
            commission=lots * self.commission_usd_per_lot_per_side,
        )

    # ------------------------------------------------------------------ #
    # Financing
    # ------------------------------------------------------------------ #
    def financing_cost(self, position_oz: float, nights: int) -> float:
        """USD cost of holding ``position_oz`` across ``nights`` rollovers.

        Positive is a charge. Long gold normally pays; a short normally receives a
        small credit. Intraday strategies barely touch this, which is exactly why
        it must be present: without it, a strategy that quietly drifts into
        overnight holds is never penalised for it.
        """
        if position_oz == 0.0 or nights <= 0:
            return 0.0
        rate = (
            self.swap_long_usd_per_oz_per_night
            if position_oz > 0
            else self.swap_short_usd_per_oz_per_night
        )
        # Config states the swap as a P&L adjustment (negative = you pay), and this
        # returns a cost (positive = you pay), hence the sign flip.
        return float(-rate * abs(position_oz) * nights)

    # ------------------------------------------------------------------ #
    def round_trip_cost_usd(self, oz: float, ts: pd.Timestamp) -> float:
        """What one complete in-and-out costs. The single most useful number for
        judging whether a strategy can clear its own friction."""
        return self.trade_costs(oz, BUY, ts).total + self.trade_costs(oz, SELL, ts).total

    def breakeven_move_usd_per_oz(self, ts: pd.Timestamp) -> float:
        """How far price must move, in USD per ounce, for a round trip to break even."""
        per_oz = 2.0 * self.cost_per_oz(ts)
        commission_per_oz = (
            2.0 * self.commission_usd_per_lot_per_side / self.contract_size_oz_per_lot
        )
        return per_oz + commission_per_oz

    def describe(self) -> list[str]:
        base_ts = pd.Timestamp("2024-01-03 13:00", tz="UTC")  # a normal London/NY hour
        return [
            f"spread: {self.spread_usd_per_oz_round_trip:.3f} USD/oz round trip "
            f"(x{min(self.hourly_multiplier_table()):.2f} to "
            f"x{max(self.hourly_multiplier_table()):.2f} by session)",
            f"slippage: {self.slippage_usd_per_oz_per_side:.3f} USD/oz per side, always adverse",
            f"commission: {self.commission_usd_per_lot_per_side:.3f} USD/lot per side",
            f"financing: long {self.swap_long_usd_per_oz_per_night:+.3f}, "
            f"short {self.swap_short_usd_per_oz_per_night:+.3f} USD/oz/night",
            f"breakeven move at a normal hour: "
            f"{self.breakeven_move_usd_per_oz(base_ts):.3f} USD/oz round trip",
        ]
