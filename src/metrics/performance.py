"""Performance statistics.

Everything is computed on the **net** equity curve unless the name says gross, and
absolute USD sits next to every percentage. On a $50 account a 40% return is $20,
and quoting only the percentage makes an unreachable result look like a plan.

Annualisation uses the empirical bars-per-year of the actual index rather than a
hardcoded constant, because the gold calendar does not divide neatly into anything.

Ratios on a ruined account are meaningless and are reported as NaN rather than as
a number that implies the account still existed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult
from src.metrics.stats import sample_kurtosis, sample_skew


@dataclass
class DrawdownStats:
    max_drawdown_fraction: float
    max_drawdown_usd: float
    max_drawdown_start: pd.Timestamp | None
    max_drawdown_trough: pd.Timestamp | None
    max_drawdown_recovered: pd.Timestamp | None
    longest_drawdown_days: float
    time_underwater_fraction: float


@dataclass
class PerformanceMetrics:
    strategy: str
    params: dict

    initial_capital_usd: float
    final_equity_net_usd: float
    final_equity_gross_usd: float

    gross_pnl_usd: float
    total_cost_usd: float
    net_pnl_usd: float
    cost_drag_pct_of_gross_profit: float
    cost_per_trade_usd: float

    total_return_net: float
    total_return_gross: float
    cagr_net: float
    cagr_gross: float

    sharpe: float
    sortino: float
    calmar: float

    max_drawdown_fraction: float
    max_drawdown_usd: float
    longest_drawdown_days: float
    time_underwater_fraction: float

    n_trades: int
    trades_per_year: float
    win_rate: float
    profit_factor: float
    avg_win_usd: float
    avg_loss_usd: float
    expectancy_usd: float
    expectancy_r: float

    exposure: float
    years: float
    bars: int
    was_ruined: bool
    ruin_time: pd.Timestamp | None

    cost_breakdown_usd: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _returns(equity: pd.Series) -> np.ndarray:
    """Simple period returns, truncated at ruin.

    Once equity is non-positive a percentage return is undefined, so the series is
    cut there instead of producing a large meaningless number.
    """
    values = equity.to_numpy(dtype="float64")
    nonpositive = np.flatnonzero(values <= 0)
    if nonpositive.size:
        values = values[: nonpositive[0] + 1]
    if values.size < 2:
        return np.zeros(0, dtype="float64")
    prev = values[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, (values[1:] - prev) / prev, np.nan)
    return rets[np.isfinite(rets)]


def drawdown_stats(equity: pd.Series) -> DrawdownStats:
    values = equity.to_numpy(dtype="float64")
    if values.size == 0:
        return DrawdownStats(0.0, 0.0, None, None, None, 0.0, 0.0)
    peaks = np.maximum.accumulate(values)
    drawdown_usd = values - peaks
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown_frac = np.where(peaks > 0, drawdown_usd / peaks, np.nan)

    trough_i = int(np.nanargmin(drawdown_frac)) if np.any(np.isfinite(drawdown_frac)) else 0
    peak_i = int(np.argmax(values[: trough_i + 1])) if trough_i > 0 else 0

    recovered_at = None
    after = np.flatnonzero(values[trough_i:] >= peaks[trough_i])
    if after.size:
        recovered_at = equity.index[trough_i + int(after[0])]

    underwater = drawdown_usd < -1e-12
    time_underwater = float(underwater.mean()) if underwater.size else 0.0

    # Longest unbroken underwater stretch, in days.
    longest_days = 0.0
    if underwater.any():
        start = None
        for i, flag in enumerate(underwater):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                span = (equity.index[i] - equity.index[start]).total_seconds() / 86400.0
                longest_days = max(longest_days, span)
                start = None
        if start is not None:
            span = (equity.index[-1] - equity.index[start]).total_seconds() / 86400.0
            longest_days = max(longest_days, span)

    return DrawdownStats(
        max_drawdown_fraction=float(np.nanmin(drawdown_frac)) if np.any(np.isfinite(drawdown_frac)) else 0.0,
        max_drawdown_usd=float(drawdown_usd[trough_i]),
        max_drawdown_start=equity.index[peak_i],
        max_drawdown_trough=equity.index[trough_i],
        max_drawdown_recovered=recovered_at,
        longest_drawdown_days=longest_days,
        time_underwater_fraction=time_underwater,
    )


def compute_metrics(result: BacktestResult) -> PerformanceMetrics:
    equity = result.equity_net
    index = equity.index
    notes: list[str] = []

    years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 3600) if len(index) > 1 else 0.0
    periods_per_year = result.bars_per_year

    rets = _returns(equity)
    if rets.size and rets.size < len(index) - 1:
        notes.append("return series truncated at the point equity reached zero")

    mean, std = (float(rets.mean()), float(rets.std(ddof=1))) if rets.size > 1 else (0.0, 0.0)
    sharpe = float(mean / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")

    downside = rets[rets < 0]
    downside_dev = float(np.sqrt((downside**2).mean())) if downside.size else 0.0
    sortino = float(mean / downside_dev * np.sqrt(periods_per_year)) if downside_dev > 0 else float("nan")

    dd = drawdown_stats(equity)

    capital = result.initial_capital
    final_net = float(equity.iloc[-1])
    final_gross = float(result.equity_gross.iloc[-1])

    total_return_net = (final_net - capital) / capital
    total_return_gross = (final_gross - capital) / capital

    def cagr(final: float) -> float:
        if years <= 0 or capital <= 0 or final <= 0:
            return float("nan")
        return float((final / capital) ** (1.0 / years) - 1.0)

    cagr_net, cagr_gross = cagr(final_net), cagr(final_gross)
    if np.isnan(cagr_net) and final_net <= 0:
        notes.append("net CAGR is undefined: the account reached zero or below")

    calmar = (
        float(cagr_net / abs(dd.max_drawdown_fraction))
        if np.isfinite(cagr_net) and dd.max_drawdown_fraction < 0
        else float("nan")
    )

    trades = result.trades
    n_trades = int(len(trades))
    if n_trades:
        net_pnls = trades["net_pnl"].to_numpy(dtype="float64")
        wins, losses = net_pnls[net_pnls > 0], net_pnls[net_pnls < 0]
        win_rate = float(len(wins) / n_trades)
        gross_wins, gross_losses = float(wins.sum()), float(-losses.sum())
        profit_factor = float(gross_wins / gross_losses) if gross_losses > 0 else float("inf")
        avg_win = float(wins.mean()) if wins.size else 0.0
        avg_loss = float(losses.mean()) if losses.size else 0.0
        expectancy_usd = float(net_pnls.mean())
        r_values = trades["r_multiple"].to_numpy(dtype="float64")
        r_values = r_values[np.isfinite(r_values)]
        expectancy_r = float(r_values.mean()) if r_values.size else float("nan")
        cost_per_trade = float(trades["total_cost"].mean())
        cost_breakdown = {
            k: float(trades[f"{k}_cost"].sum())
            for k in ("spread", "slippage", "commission", "financing")
        }
    else:
        win_rate = profit_factor = avg_win = avg_loss = 0.0
        expectancy_usd = cost_per_trade = 0.0
        expectancy_r = float("nan")
        cost_breakdown = {}
        notes.append("no trades were taken")

    gross_pnl = result.gross_pnl
    total_costs = result.total_costs
    cost_drag_pct = float(100.0 * total_costs / gross_pnl) if gross_pnl > 0 else float("nan")

    if result.costs_exceed_gross_profit:
        notes.append(
            f"costs of {total_costs:,.2f} USD exceeded gross profit of {gross_pnl:,.2f} USD"
        )

    return PerformanceMetrics(
        strategy=result.strategy,
        params=dict(result.params),
        initial_capital_usd=capital,
        final_equity_net_usd=final_net,
        final_equity_gross_usd=final_gross,
        gross_pnl_usd=gross_pnl,
        total_cost_usd=total_costs,
        net_pnl_usd=result.net_pnl,
        cost_drag_pct_of_gross_profit=cost_drag_pct,
        cost_per_trade_usd=cost_per_trade,
        total_return_net=total_return_net,
        total_return_gross=total_return_gross,
        cagr_net=cagr_net,
        cagr_gross=cagr_gross,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown_fraction=dd.max_drawdown_fraction,
        max_drawdown_usd=dd.max_drawdown_usd,
        longest_drawdown_days=dd.longest_drawdown_days,
        time_underwater_fraction=dd.time_underwater_fraction,
        n_trades=n_trades,
        trades_per_year=float(n_trades / years) if years > 0 else float("nan"),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win_usd=avg_win,
        avg_loss_usd=avg_loss,
        expectancy_usd=expectancy_usd,
        expectancy_r=expectancy_r,
        exposure=result.exposure,
        years=years,
        bars=len(index),
        was_ruined=result.was_ruined,
        ruin_time=result.ruin_time,
        cost_breakdown_usd=cost_breakdown,
        notes=notes,
    )


def per_observation_sharpe(result: BacktestResult) -> tuple[float, int, float, float]:
    """Sharpe per bar plus the moments the deflated Sharpe needs.

    Deliberately not annualised: the deflated Sharpe formula is defined on the
    per-observation statistic, and annualising first inflates it.
    """
    rets = _returns(result.equity_net)
    if rets.size < 2:
        return 0.0, int(rets.size), 0.0, 3.0
    std = float(rets.std(ddof=1))
    sharpe = float(rets.mean() / std) if std > 0 else 0.0
    return sharpe, int(rets.size), sample_skew(rets), sample_kurtosis(rets)
