"""Portfolio accounting.

Positions are held in ounces, signed. The accounting keeps gross and net apart at
every instant:

    gross_equity = initial_capital + realised_gross + unrealised_gross
    net_equity   = gross_equity - cumulative_costs

Costs are charged the moment they are incurred, so opening a position drops net
equity immediately by the spread and slippage paid to get in. That is what
actually happens to a real account, and it is the reason a strategy that trades
constantly bleeds even when it is right about direction.

**Sizing is fixed for the life of a trade.** Strategies emit a target position in
{-1, 0, +1}, so there is nothing here to express a scale-in. A signal that keeps
its sign leaves the position untouched; a sign flip closes and reopens, paying
both sides. Pyramiding would need a richer strategy contract and is deliberately
out of scope rather than half-implemented.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.backtest.costs import BUY, SELL, CostModel, TradeCosts


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int                  # +1 long, -1 short
    size_oz: float
    size_lots: float
    entry_price_raw: float
    entry_price_eff: float
    exit_price_raw: float
    exit_price_eff: float
    gross_pnl: float
    spread_cost: float
    slippage_cost: float
    commission_cost: float
    financing_cost: float
    total_cost: float
    net_pnl: float
    bars_held: int
    stop_distance_usd_per_oz: float
    risk_usd: float
    r_multiple: float
    entry_reason: str = ""
    exit_reason: str = ""
    min_lot_forced: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _OpenPosition:
    entry_time: pd.Timestamp
    entry_bar: int
    direction: int
    size_oz: float
    entry_price_raw: float
    entry_price_eff: float
    entry_costs: TradeCosts
    stop_distance_usd_per_oz: float
    risk_usd: float
    reason: str
    min_lot_forced: bool
    financing_accrued: float = 0.0


class Portfolio:
    def __init__(self, initial_capital: float, cost_model: CostModel):
        if initial_capital <= 0:
            raise ValueError("initial capital must be positive")
        self.initial_capital = float(initial_capital)
        self.costs = cost_model

        self.position_oz: float = 0.0
        self._open: _OpenPosition | None = None

        self.realised_gross: float = 0.0
        self.cost_totals = TradeCosts()
        self.trades: list[Trade] = []

        self._mark_price: float = float("nan")
        self._bar_index: int = -1

    # ------------------------------------------------------------------ #
    # Valuation
    # ------------------------------------------------------------------ #
    @property
    def unrealised_gross(self) -> float:
        if self._open is None or not np.isfinite(self._mark_price):
            return 0.0
        return self._open.size_oz * (self._mark_price - self._open.entry_price_raw)

    @property
    def gross_equity(self) -> float:
        return self.initial_capital + self.realised_gross + self.unrealised_gross

    @property
    def cumulative_costs(self) -> float:
        return self.cost_totals.total

    @property
    def net_equity(self) -> float:
        return self.gross_equity - self.cumulative_costs

    def mark(self, price: float, bar_index: int) -> None:
        self._mark_price = float(price)
        self._bar_index = int(bar_index)

    # ------------------------------------------------------------------ #
    # Financing
    # ------------------------------------------------------------------ #
    def accrue_financing(self, nights: int) -> float:
        """Charge financing for ``nights`` rollovers crossed while holding."""
        if self._open is None or nights <= 0:
            return 0.0
        charge = self.costs.financing_cost(self._open.size_oz, nights)
        self._open.financing_accrued += charge
        self.cost_totals = self.cost_totals + TradeCosts(financing=charge)
        return charge

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def open_position(
        self,
        *,
        ts: pd.Timestamp,
        bar_index: int,
        direction: int,
        size_oz: float,
        raw_price: float,
        stop_distance_usd_per_oz: float = float("nan"),
        risk_usd: float = float("nan"),
        reason: str = "",
        min_lot_forced: bool = False,
    ) -> None:
        if self._open is not None:
            raise RuntimeError("open_position called while a position is already open")
        if direction not in (1, -1) or size_oz <= 0:
            raise ValueError(f"invalid position: direction={direction}, size_oz={size_oz}")

        side = BUY if direction > 0 else SELL
        entry_costs = self.costs.trade_costs(size_oz, side, ts)
        self.cost_totals = self.cost_totals + entry_costs

        self._open = _OpenPosition(
            entry_time=ts,
            entry_bar=bar_index,
            direction=direction,
            size_oz=direction * abs(size_oz),
            entry_price_raw=float(raw_price),
            entry_price_eff=self.costs.effective_fill_price(raw_price, side, ts),
            entry_costs=entry_costs,
            stop_distance_usd_per_oz=float(stop_distance_usd_per_oz),
            risk_usd=float(risk_usd),
            reason=reason,
            min_lot_forced=min_lot_forced,
        )
        self.position_oz = self._open.size_oz
        self.mark(raw_price, bar_index)

    def close_position(
        self, *, ts: pd.Timestamp, bar_index: int, raw_price: float, reason: str = ""
    ) -> Trade | None:
        if self._open is None:
            return None
        pos = self._open
        side = SELL if pos.direction > 0 else BUY
        exit_costs = self.costs.trade_costs(pos.size_oz, side, ts)
        self.cost_totals = self.cost_totals + exit_costs

        gross = pos.size_oz * (float(raw_price) - pos.entry_price_raw)
        self.realised_gross += gross

        all_costs = pos.entry_costs + exit_costs + TradeCosts(financing=pos.financing_accrued)
        net = gross - all_costs.total
        risk = pos.risk_usd
        r_multiple = net / risk if np.isfinite(risk) and risk > 0 else float("nan")

        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=ts,
            direction=pos.direction,
            size_oz=abs(pos.size_oz),
            size_lots=abs(pos.size_oz) / self.costs.contract_size_oz_per_lot,
            entry_price_raw=pos.entry_price_raw,
            entry_price_eff=pos.entry_price_eff,
            exit_price_raw=float(raw_price),
            exit_price_eff=self.costs.effective_fill_price(raw_price, side, ts),
            gross_pnl=gross,
            spread_cost=all_costs.spread,
            slippage_cost=all_costs.slippage,
            commission_cost=all_costs.commission,
            financing_cost=all_costs.financing,
            total_cost=all_costs.total,
            net_pnl=net,
            bars_held=int(bar_index - pos.entry_bar),
            stop_distance_usd_per_oz=pos.stop_distance_usd_per_oz,
            risk_usd=risk,
            r_multiple=r_multiple,
            entry_reason=pos.reason,
            exit_reason=reason,
            min_lot_forced=pos.min_lot_forced,
        )
        self.trades.append(trade)
        self._open = None
        self.position_oz = 0.0
        self.mark(raw_price, bar_index)
        return trade

    # ------------------------------------------------------------------ #
    @property
    def is_open(self) -> bool:
        return self._open is not None

    @property
    def direction(self) -> int:
        return self._open.direction if self._open else 0

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=[f.name for f in Trade.__dataclass_fields__.values()])
        frame = pd.DataFrame([t.as_dict() for t in self.trades])
        return frame.sort_values("entry_time").reset_index(drop=True)

    def reconciliation(self) -> dict[str, float]:
        """Cross-check: net P&L implied by the effective fill prices must equal
        gross minus costs. Two independent routes to the same number, so a sign
        error in the cost model cannot hide.
        """
        frame = self.trades_frame()
        if frame.empty:
            return {"from_effective_prices": 0.0, "from_gross_minus_costs": 0.0, "difference": 0.0}
        eff = (
            frame["direction"] * frame["size_oz"]
            * (frame["exit_price_eff"] - frame["entry_price_eff"])
            - frame["commission_cost"] - frame["financing_cost"]
        ).sum()
        decomposed = (frame["gross_pnl"] - frame["total_cost"]).sum()
        return {
            "from_effective_prices": float(eff),
            "from_gross_minus_costs": float(decomposed),
            "difference": float(eff - decomposed),
        }
