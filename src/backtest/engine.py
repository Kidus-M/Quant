"""The event loop.

One rule dominates everything else in this file:

    A signal generated on bar t is filled at the OPEN of bar t+1, plus slippage.

Never the close of bar t (that price was needed to generate the signal, so filling
there is time travel). Never the high or the low (the backtester does not know the
path within a bar, and any assumption about it flatters the result). The shift is
applied once, here, in ``_target_from_signals``, so no strategy can opt out.

Order of operations within bar ``i``:

1. Charge financing for any rollover crossed since bar ``i-1``, on the position
   held over that interval.
2. Execute the position change decided at bar ``i-1``, at the open of bar ``i``.
3. Mark to the close of bar ``i`` and record equity.

Sizing inputs (equity, ATR, volatility) are read at bar ``i-1``, the decision bar.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.portfolio import Portfolio
from src.backtest.sizing import PositionSizer, RiskDiagnostics, risk_diagnostics
from src.config import Config
from src.data.base import DataProvenance
from src.data.loader import BarDataset
from src.data.resample import bars_per_year, infer_bar_seconds
from src.features.indicators import atr as atr_indicator
from src.features.indicators import rolling_return_vol
from src.strategies.base import Strategy, validate_signals

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    strategy: str
    params: dict
    equity_net: pd.Series
    equity_gross: pd.Series
    costs_cum: pd.Series
    position_oz: pd.Series
    signal: pd.Series
    target_position: pd.Series
    trades: pd.DataFrame
    initial_capital: float
    bars_per_year: float
    bars_per_day: float
    resolution: str
    provenance: DataProvenance | None = None
    risk: RiskDiagnostics | None = None
    warnings: list[str] = field(default_factory=list)
    cost_lines: list[str] = field(default_factory=list)
    reconciliation: dict = field(default_factory=dict)
    # Set when net equity first reached zero. A $50 account that goes to zero has
    # not "underperformed"; it has stopped existing, and no ratio conveys that.
    ruin_time: pd.Timestamp | None = None

    # -------------------------------------------------------------- #
    @property
    def gross_pnl(self) -> float:
        return float(self.equity_gross.iloc[-1] - self.initial_capital) if len(self.equity_gross) else 0.0

    @property
    def total_costs(self) -> float:
        return float(self.costs_cum.iloc[-1]) if len(self.costs_cum) else 0.0

    @property
    def net_pnl(self) -> float:
        return float(self.equity_net.iloc[-1] - self.initial_capital) if len(self.equity_net) else 0.0

    @property
    def costs_exceed_gross_profit(self) -> bool:
        return self.gross_pnl > 0 and self.total_costs > self.gross_pnl

    @property
    def was_ruined(self) -> bool:
        return self.ruin_time is not None

    @property
    def exposure(self) -> float:
        if not len(self.position_oz):
            return 0.0
        return float((self.position_oz != 0).mean())


def _target_from_signals(signals: pd.Series) -> pd.Series:
    """The single place the fill delay is applied.

    ``target[t]`` is the position to hold from the open of bar ``t``, decided at
    the close of bar ``t-1``. The first bar is necessarily flat.
    """
    shifted = signals.shift(1)
    shifted = shifted.astype("float64").fillna(0.0).astype("int8")
    return shifted.rename("target_position")


def run_backtest(
    dataset: BarDataset,
    strategy: Strategy,
    cfg: Config,
    *,
    initial_capital: float | None = None,
    cost_model: CostModel | None = None,
    sizer: PositionSizer | None = None,
    warmup_bars: int | None = None,
) -> BacktestResult:
    bars = dataset.bars
    if bars.empty:
        raise ValueError("cannot backtest an empty bar set")

    cost_model = cost_model or CostModel.from_config(cfg)
    sizer = sizer or PositionSizer.from_config(cfg)
    capital = float(initial_capital if initial_capital is not None else cfg.get("account.initial_capital_usd"))
    warnings: list[str] = []

    macro = dataset.macro if len(dataset.macro.columns) else None
    if strategy.requires_macro and macro is None:
        raise ValueError(
            f"{strategy.name} requires macro columns but none were loaded. Enable "
            "macro in config and make sure the FRED fetch succeeded; running it "
            "without the filter would silently test a different strategy."
        )

    features = strategy.compute_features(bars, macro)
    if not features.index.equals(bars.index):
        raise ValueError(f"{strategy.name}: feature index does not match the bar index")

    signals = validate_signals(strategy.generate_signals(bars, features), bars, strategy.name)

    # Warm-up: the first max_lookback bars cannot have valid indicators, so any
    # signal there is an artefact of partially-filled windows.
    warmup = strategy.max_lookback if warmup_bars is None else warmup_bars
    if warmup > 0:
        if warmup >= len(bars):
            raise ValueError(
                f"{strategy.name} needs {warmup} warm-up bars but only {len(bars)} are available"
            )
        signals = signals.copy()
        signals.iloc[:warmup] = 0

    target = _target_from_signals(signals)

    bar_seconds = infer_bar_seconds(bars.index)
    bars_per_day = 86400.0 / bar_seconds if np.isfinite(bar_seconds) and bar_seconds > 0 else float("nan")

    atr_period = int(cfg.get("sizing.atr_period", 14))
    atr_series = (
        features["atr"] if "atr" in features.columns else atr_indicator(bars, atr_period)
    ).to_numpy(dtype="float64")
    vol_series = (
        features["return_vol"]
        if "return_vol" in features.columns
        else rolling_return_vol(bars["close"], int(cfg.get("sizing.vol_lookback_bars", 96)))
    ).to_numpy(dtype="float64")

    calendar = dataset.calendar
    index = bars.index
    rollovers = np.zeros(len(index), dtype="int64")
    if len(index) > 1:
        rollovers[1:] = calendar.rollovers_between(index[:-1], index[1:])

    open_px = bars["open"].to_numpy(dtype="float64")
    close_px = bars["close"].to_numpy(dtype="float64")
    target_arr = target.to_numpy()

    portfolio = Portfolio(capital, cost_model)
    n = len(index)
    equity_net = np.empty(n, dtype="float64")
    equity_gross = np.empty(n, dtype="float64")
    costs_cum = np.empty(n, dtype="float64")
    position = np.zeros(n, dtype="float64")

    forced_min_lot = 0
    skipped_unaffordable = 0
    skipped_no_equity = 0
    ruin_index: int | None = None

    for i in range(n):
        ts = index[i]

        # 1. financing on the position carried into this bar
        if rollovers[i] > 0:
            portfolio.accrue_financing(int(rollovers[i]))

        # 2. execute the decision made at bar i-1, at this bar open
        desired = int(target_arr[i])
        if desired != portfolio.direction:
            fill_price = open_px[i]
            if portfolio.is_open:
                portfolio.close_position(
                    ts=ts, bar_index=i, raw_price=fill_price,
                    reason="signal_exit" if desired == 0 else "signal_reverse",
                )
            if desired != 0:
                decision_price = close_px[i - 1] if i > 0 else open_px[i]
                decision_atr = atr_series[i - 1] if i > 0 else np.nan
                decision_vol = vol_series[i - 1] if i > 0 else np.nan
                decision = sizer.size(
                    equity=portfolio.net_equity,
                    price=decision_price,
                    atr=decision_atr,
                    bar_return_vol=decision_vol,
                    bars_per_day=bars_per_day,
                )
                if decision.is_tradeable:
                    if decision.min_lot_forced:
                        forced_min_lot += 1
                    portfolio.open_position(
                        ts=ts, bar_index=i, direction=desired, size_oz=decision.oz,
                        raw_price=fill_price,
                        stop_distance_usd_per_oz=decision.stop_distance_usd_per_oz,
                        risk_usd=decision.risk_usd,
                        reason="signal_entry",
                        min_lot_forced=decision.min_lot_forced,
                    )
                elif portfolio.net_equity <= 0:
                    skipped_no_equity += 1
                else:
                    skipped_unaffordable += 1

        # 3. mark to this bar close
        portfolio.mark(close_px[i], i)
        if ruin_index is None and portfolio.net_equity <= 0:
            ruin_index = i
        equity_net[i] = portfolio.net_equity
        equity_gross[i] = portfolio.gross_equity
        costs_cum[i] = portfolio.cumulative_costs
        position[i] = portfolio.position_oz

    if portfolio.is_open:
        # Close on the final close so the trade log is complete. This is a
        # bookkeeping liquidation, not a signal, and it is labelled as such.
        portfolio.close_position(
            ts=index[-1], bar_index=n - 1, raw_price=close_px[-1], reason="end_of_data"
        )
        equity_net[-1] = portfolio.net_equity
        equity_gross[-1] = portfolio.gross_equity
        costs_cum[-1] = portfolio.cumulative_costs
        position[-1] = 0.0
        warnings.append("a position was open at the end of the data and was closed at the final close")

    if forced_min_lot:
        warnings.append(
            f"{forced_min_lot} entries were forced up to the {sizer.min_lot} lot broker "
            "minimum, taking more risk than the configured budget allows"
        )
    if skipped_unaffordable:
        warnings.append(
            f"{skipped_unaffordable} entries were skipped because the risk budget "
            "could not cover the minimum lot"
        )
    trades = portfolio.trades_frame()
    ruin_time = None
    if ruin_index is not None:
        # The single most important thing that can happen to a $50 account, so it
        # is stated in dollars and elapsed time rather than buried in a ratio.
        ruin_time = index[ruin_index]
        n_before = int((trades["entry_time"] <= ruin_time).sum()) if len(trades) else 0
        warnings.append(
            f"ACCOUNT RUINED: net equity reached zero on {ruin_time}, "
            f"{(ruin_time - index[0]).days} days in, after {n_before} trades. "
            "Everything past that point is arithmetic, not a tradeable result."
        )
    if skipped_no_equity:
        warnings.append(
            f"{skipped_no_equity} entries after that point were skipped for having no equity to trade"
        )

    equity_series = pd.Series(equity_net, index=index, name="equity_net")

    risk = _risk_report(bars, trades, cfg, sizer, capital)
    if risk is not None and (risk.breaches_threshold or risk.min_lot_unaffordable):
        warnings.append(
            f"position risk per ATR is {risk.risk_per_atr_pct:.1f}% of equity, above "
            f"the {risk.warning_threshold_pct:.0f}% threshold"
        )

    result = BacktestResult(
        strategy=strategy.name,
        params=dict(strategy.params),
        equity_net=equity_series,
        equity_gross=pd.Series(equity_gross, index=index, name="equity_gross"),
        costs_cum=pd.Series(costs_cum, index=index, name="costs_cum"),
        position_oz=pd.Series(position, index=index, name="position_oz"),
        signal=signals,
        target_position=target,
        trades=trades,
        initial_capital=capital,
        bars_per_year=bars_per_year(index),
        bars_per_day=bars_per_day,
        resolution=dataset.resolution,
        provenance=dataset.provenance,
        risk=risk,
        warnings=warnings,
        cost_lines=cost_model.describe(),
        reconciliation=portfolio.reconciliation(),
        ruin_time=ruin_time,
    )

    difference = abs(result.reconciliation.get("difference", 0.0))
    if difference > 1e-6:
        raise AssertionError(
            "cost accounting does not reconcile: net P&L from effective fill prices "
            f"differs from gross minus costs by {difference:.6f} USD"
        )
    return result


def _risk_report(
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    cfg: Config,
    sizer: PositionSizer,
    capital: float,
) -> RiskDiagnostics | None:
    """Risk arithmetic at typical conditions, not at a cherry-picked bar."""
    atr_period = int(cfg.get("sizing.atr_period", 14))
    atr_values = atr_indicator(bars, atr_period).dropna()
    if atr_values.empty:
        return None
    median_atr = float(atr_values.median())

    daily = bars.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    daily_range = float((daily["high"] - daily["low"]).median()) if len(daily) else float("nan")

    if len(trades):
        position_oz = float(trades["size_oz"].median())
    else:
        position_oz = sizer.min_position_oz

    return risk_diagnostics(
        equity=capital,
        atr_usd_per_oz=median_atr,
        position_oz=position_oz,
        sizer=sizer,
        daily_range_usd_per_oz=daily_range,
        warning_threshold_pct=float(cfg.get("sizing.risk_per_atr_warning_pct", 5.0)),
    )
