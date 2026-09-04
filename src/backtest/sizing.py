"""Position sizing, and the risk arithmetic that a $50 account has to face.

Three modes, all configurable:

1. ``fixed``            a constant lot size
2. ``vol_target``       size so expected daily P&L volatility is a target fraction of equity
3. ``fixed_fractional`` size so a stop-loss hit costs a fixed fraction of equity

The second half of this module is the part that matters most for the account this
is being built for. With a 0.01 lot minimum, one ounce of gold moves roughly
$30-50 on an ordinary day. On $50 of equity, the smallest position it is possible
to open risks most of the account on ordinary noise. ``risk_diagnostics`` computes
that number and ``format_risk_warning`` makes it impossible to miss in the report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.config import Config


@dataclass(frozen=True)
class SizingDecision:
    oz: float
    lots: float
    mode: str
    stop_distance_usd_per_oz: float
    risk_usd: float
    min_lot_forced: bool = False
    notes: tuple[str, ...] = ()

    @property
    def is_tradeable(self) -> bool:
        return self.oz > 0.0


@dataclass
class PositionSizer:
    mode: str = "fixed"
    fixed_lots: float = 0.01
    contract_size_oz_per_lot: float = 100.0
    min_lot: float = 0.01
    lot_step: float = 0.01
    target_daily_vol_fraction: float = 0.01
    risk_fraction_per_trade: float = 0.01
    stop_atr_multiple: float = 2.0
    # When the risk budget cannot buy even one minimum lot, do we take the minimum
    # lot anyway (what a small account holder actually does) or skip the trade?
    allow_min_lot_override: bool = True

    @classmethod
    def from_config(cls, cfg: Config) -> "PositionSizer":
        s = cfg.section("sizing")
        return cls(
            mode=str(s.get("mode", "fixed")),
            fixed_lots=float(s.get("fixed_lots", 0.01)),
            contract_size_oz_per_lot=float(cfg.get("instrument.contract_size_oz_per_lot", 100.0)),
            min_lot=float(cfg.get("instrument.min_lot", 0.01)),
            lot_step=float(cfg.get("instrument.lot_step", 0.01)),
            target_daily_vol_fraction=float(s.get("target_daily_vol_fraction", 0.01)),
            risk_fraction_per_trade=float(s.get("risk_fraction_per_trade", 0.01)),
            stop_atr_multiple=float(s.get("stop_atr_multiple", 2.0)),
            allow_min_lot_override=bool(s.get("allow_min_lot_override", True)),
        )

    # ------------------------------------------------------------------ #
    @property
    def min_position_oz(self) -> float:
        return self.min_lot * self.contract_size_oz_per_lot

    def _round_lots(self, lots: float) -> float:
        """Round down to the broker lot step. Rounding up would quietly take more
        risk than the configured budget allows."""
        if self.lot_step <= 0:
            return lots
        steps = math.floor(lots / self.lot_step + 1e-9)
        return max(0.0, steps * self.lot_step)

    def size(
        self,
        *,
        equity: float,
        price: float,
        atr: float,
        bar_return_vol: float | None = None,
        bars_per_day: float | None = None,
    ) -> SizingDecision:
        notes: list[str] = []
        stop_distance = float(self.stop_atr_multiple * atr) if np.isfinite(atr) else float("nan")

        if equity <= 0:
            return SizingDecision(0.0, 0.0, self.mode, stop_distance, 0.0, notes=("equity is zero or negative",))

        if self.mode == "fixed":
            lots = self.fixed_lots
        elif self.mode == "fixed_fractional":
            if not np.isfinite(stop_distance) or stop_distance <= 0:
                return SizingDecision(0.0, 0.0, self.mode, stop_distance, 0.0,
                                      notes=("ATR unavailable, cannot size by risk",))
            budget = equity * self.risk_fraction_per_trade
            lots = (budget / stop_distance) / self.contract_size_oz_per_lot
        elif self.mode == "vol_target":
            if not bar_return_vol or not np.isfinite(bar_return_vol) or bar_return_vol <= 0:
                return SizingDecision(0.0, 0.0, self.mode, stop_distance, 0.0,
                                      notes=("return volatility unavailable, cannot size",))
            if not bars_per_day or bars_per_day <= 0:
                raise ValueError("vol_target sizing needs bars_per_day")
            daily_vol_usd_per_oz = bar_return_vol * math.sqrt(bars_per_day) * price
            if daily_vol_usd_per_oz <= 0:
                return SizingDecision(0.0, 0.0, self.mode, stop_distance, 0.0,
                                      notes=("estimated daily volatility is zero",))
            target_usd = equity * self.target_daily_vol_fraction
            lots = (target_usd / daily_vol_usd_per_oz) / self.contract_size_oz_per_lot
        else:
            raise ValueError(f"unknown sizing mode {self.mode!r}")

        rounded = self._round_lots(lots)
        min_lot_forced = False
        if rounded < self.min_lot:
            if self.allow_min_lot_override and lots > 0:
                rounded = self.min_lot
                min_lot_forced = True
                notes.append(
                    f"risk budget allowed {lots:.4f} lots but the broker minimum is "
                    f"{self.min_lot:.2f}; took the minimum, which exceeds the configured risk"
                )
            else:
                notes.append(
                    f"risk budget allowed {lots:.4f} lots, below the {self.min_lot:.2f} "
                    "minimum; trade skipped"
                )
                return SizingDecision(0.0, 0.0, self.mode, stop_distance, 0.0,
                                      min_lot_forced=False, notes=tuple(notes))

        oz = rounded * self.contract_size_oz_per_lot
        risk_usd = oz * stop_distance if np.isfinite(stop_distance) else float("nan")
        return SizingDecision(
            oz=oz, lots=rounded, mode=self.mode,
            stop_distance_usd_per_oz=stop_distance, risk_usd=risk_usd,
            min_lot_forced=min_lot_forced, notes=tuple(notes),
        )


# ---------------------------------------------------------------------- #
# The numbers a small account needs shoved in its face
# ---------------------------------------------------------------------- #
@dataclass
class RiskDiagnostics:
    equity: float
    atr_usd_per_oz: float
    daily_range_usd_per_oz: float
    position_oz: float
    min_position_oz: float
    risk_per_atr_pct: float
    min_risk_per_atr_pct: float
    risk_per_daily_range_pct: float
    min_viable_equity_usd: float
    min_viable_equity_daily_range_usd: float
    risk_fraction_per_trade: float
    stop_atr_multiple: float
    warning_threshold_pct: float
    notes: list[str] = field(default_factory=list)

    @property
    def breaches_threshold(self) -> bool:
        return bool(np.isfinite(self.risk_per_atr_pct) and self.risk_per_atr_pct > self.warning_threshold_pct)

    @property
    def min_lot_unaffordable(self) -> bool:
        return bool(
            np.isfinite(self.min_risk_per_atr_pct)
            and self.min_risk_per_atr_pct > self.warning_threshold_pct
        )


def risk_diagnostics(
    *,
    equity: float,
    atr_usd_per_oz: float,
    position_oz: float,
    sizer: PositionSizer,
    daily_range_usd_per_oz: float | None = None,
    warning_threshold_pct: float = 5.0,
) -> RiskDiagnostics:
    """Percentage of equity at risk per ATR of adverse movement, plus the equity
    level at which the configured risk rule becomes possible at all.

    ``min_viable_equity_usd`` answers the question the spec asks: given a 0.01 lot
    minimum and this instrument, how much equity does a 1%-risk-per-trade rule
    need before it can be followed rather than approximated?
    """
    min_oz = sizer.min_position_oz
    daily_range = daily_range_usd_per_oz if daily_range_usd_per_oz is not None else float("nan")

    def pct(oz: float, move: float) -> float:
        if equity <= 0 or not np.isfinite(move):
            return float("nan")
        return 100.0 * oz * move / equity

    stop_distance = sizer.stop_atr_multiple * atr_usd_per_oz
    def viable_equity(adverse_move: float) -> float:
        if not np.isfinite(adverse_move) or sizer.risk_fraction_per_trade <= 0:
            return float("nan")
        return (adverse_move * min_oz) / sizer.risk_fraction_per_trade

    # Two readings of the same question. The stop-based number is what the
    # configured rule needs; the daily-range number is what surviving an ordinary
    # day needs, and it is the larger and more honest of the two.
    min_viable = viable_equity(stop_distance)
    min_viable_daily = viable_equity(daily_range)

    return RiskDiagnostics(
        equity=equity,
        atr_usd_per_oz=atr_usd_per_oz,
        daily_range_usd_per_oz=daily_range,
        position_oz=position_oz,
        min_position_oz=min_oz,
        risk_per_atr_pct=pct(position_oz, atr_usd_per_oz),
        min_risk_per_atr_pct=pct(min_oz, atr_usd_per_oz),
        risk_per_daily_range_pct=pct(position_oz, daily_range),
        min_viable_equity_usd=min_viable,
        min_viable_equity_daily_range_usd=min_viable_daily,
        risk_fraction_per_trade=sizer.risk_fraction_per_trade,
        stop_atr_multiple=sizer.stop_atr_multiple,
        warning_threshold_pct=warning_threshold_pct,
    )


def format_risk_warning(diag: RiskDiagnostics) -> list[str]:
    """Loud, plain-language block for the top of the report.

    This is the single most important output for a small account, so it is written
    to be readable by someone who skips the tables.
    """
    lines: list[str] = []
    if not diag.breaches_threshold and not diag.min_lot_unaffordable:
        lines.append(
            f"Position risk: {diag.risk_per_atr_pct:.2f}% of equity per ATR "
            f"(ATR {diag.atr_usd_per_oz:.2f} USD/oz), within the "
            f"{diag.warning_threshold_pct:.0f}% threshold."
        )
        return lines

    lines.append("=" * 78)
    lines.append("POSITION SIZING WARNING")
    lines.append("=" * 78)
    lines.append(
        f"Account equity {diag.equity:,.2f} USD. One ATR of adverse movement "
        f"({diag.atr_usd_per_oz:.2f} USD/oz) against the configured position of "
        f"{diag.position_oz:.2f} oz costs {diag.risk_per_atr_pct:.1f}% of the account."
    )
    if np.isfinite(diag.risk_per_daily_range_pct):
        lines.append(
            f"A typical full daily range ({diag.daily_range_usd_per_oz:.2f} USD/oz) "
            f"is {diag.risk_per_daily_range_pct:.1f}% of the account."
        )
    if diag.min_lot_unaffordable:
        lines.append(
            f"The broker minimum position of {diag.min_position_oz:.2f} oz "
            f"({diag.min_position_oz / 100:.2f} lots) alone risks "
            f"{diag.min_risk_per_atr_pct:.1f}% of equity per ATR. This account "
            "cannot open the smallest tradeable position without taking risk far "
            "above any sane per-trade limit. Ordinary daily noise, not a bad "
            "strategy, is enough to end it."
        )
    if np.isfinite(diag.min_viable_equity_usd):
        lines.append(
            f"Minimum viable account size for a {diag.risk_fraction_per_trade:.1%} "
            f"risk-per-trade rule with a {diag.stop_atr_multiple:g}x ATR stop: "
            f"{diag.min_viable_equity_usd:,.0f} USD. Below that, the "
            f"{diag.min_position_oz / 100:.2f} lot minimum forces more risk per "
            "trade than the rule permits."
        )
    if np.isfinite(diag.min_viable_equity_daily_range_usd):
        lines.append(
            f"Measured against a typical DAILY range rather than an intraday stop, "
            f"the same rule needs {diag.min_viable_equity_daily_range_usd:,.0f} USD. "
            "That is the figure to plan around: it is the equity at which one "
            "ordinary day of gold movement costs "
            f"{diag.risk_fraction_per_trade:.1%} of the account rather than half of it."
        )
    lines.append("=" * 78)
    return lines
