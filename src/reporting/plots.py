"""Equity, drawdown and parameter-sensitivity plots.

Uses the Agg backend so this runs headless. Every equity plot shows the gross and
the net curve together, because the gap between them is the whole story on a small
account and a net-only chart hides how much of the strategy work went to the broker.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger(__name__)

_NET = "#1b5e20"
_GROSS = "#9e9e9e"
_COST = "#c62828"


def plot_equity(
    equity_net: pd.Series,
    equity_gross: pd.Series,
    costs_cum: pd.Series,
    *,
    title: str,
    path: Path,
    initial_capital: float,
    ruin_time: pd.Timestamp | None = None,
) -> Path:
    fig, (ax, ax_cost) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax.plot(equity_gross.index, equity_gross.to_numpy(), color=_GROSS, lw=1.1,
            label="gross equity (before costs)")
    ax.plot(equity_net.index, equity_net.to_numpy(), color=_NET, lw=1.4,
            label="net equity (after costs)")
    ax.axhline(initial_capital, color="#444", lw=0.8, ls="--", label="starting capital")
    ax.axhline(0.0, color=_COST, lw=0.8, ls=":")
    if ruin_time is not None:
        ax.axvline(ruin_time, color=_COST, lw=1.4)
        ax.annotate(
            "account reached zero",
            xy=(ruin_time, 0.0),
            xytext=(6, 14),
            textcoords="offset points",
            color=_COST,
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylabel("equity (USD)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)

    ax_cost.fill_between(costs_cum.index, 0, costs_cum.to_numpy(), color=_COST, alpha=0.35)
    ax_cost.plot(costs_cum.index, costs_cum.to_numpy(), color=_COST, lw=1.0)
    ax_cost.set_ylabel("cumulative\ncost (USD)")
    ax_cost.set_xlabel("UTC")
    ax_cost.grid(alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_drawdown(equity_net: pd.Series, *, title: str, path: Path) -> Path:
    values = equity_net.to_numpy(dtype="float64")
    peaks = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peaks > 0, (values - peaks) / peaks, np.nan) * 100.0

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.fill_between(equity_net.index, drawdown, 0, color=_COST, alpha=0.4)
    ax.plot(equity_net.index, drawdown, color=_COST, lw=1.0)
    ax.set_ylabel("drawdown (%)")
    ax.set_xlabel("UTC")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_random_benchmark(
    net_pnl: np.ndarray,
    strategy_net_pnl: float,
    *,
    percentile: float,
    title: str,
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(net_pnl, bins=60, color="#90a4ae", edgecolor="white", linewidth=0.4)
    threshold = float(np.percentile(net_pnl, percentile))
    ax.axvline(threshold, color="#455a64", lw=1.4, ls="--",
               label=f"random {percentile:.0f}th pct = {threshold:,.2f} USD")
    ax.axvline(strategy_net_pnl, color=_NET, lw=2.0,
               label=f"strategy = {strategy_net_pnl:,.2f} USD")
    ax.set_xlabel("net P&L over the test period (USD)")
    ax.set_ylabel("random runs")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_parameter_sensitivity(
    sensitivity: pd.DataFrame,
    *,
    metric: str = "objective",
    title: str,
    path: Path,
) -> Path | None:
    """Plot the metric surface over the parameter grid.

    A broad plateau is weak evidence the result is not pure curve fit. A lonely
    spike surrounded by losses is evidence it is. Two parameters render as a
    heatmap; one renders as a line; more than two fall back to a sorted bar chart
    because a readable 3-D parameter surface does not exist.
    """
    param_cols = [c for c in sensitivity.columns if c not in (metric, "net_pnl_usd", "ruined")]
    if sensitivity.empty or not param_cols:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)

    if len(param_cols) == 2:
        rows, cols = param_cols
        pivot = sensitivity.pivot_table(index=rows, columns=cols, values=metric)
        fig, ax = plt.subplots(figsize=(1.6 * len(pivot.columns) + 4, 1.0 * len(pivot.index) + 3))
        image = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
        ax.set_xlabel(cols)
        ax.set_ylabel(rows)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.to_numpy()[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label=metric)
    elif len(param_cols) == 1:
        col = param_cols[0]
        ordered = sensitivity.sort_values(col)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(ordered[col], ordered[metric], marker="o", color=_NET)
        ax.set_xlabel(col)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
    else:
        ordered = sensitivity.sort_values(metric, ascending=False)
        labels = [
            ", ".join(f"{c}={row[c]}" for c in param_cols) for _, row in ordered.iterrows()
        ]
        fig, ax = plt.subplots(figsize=(10, 0.32 * len(labels) + 2.5))
        ax.barh(range(len(labels)), ordered[metric].to_numpy(), color="#607d8b")
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(metric)
        ax.grid(alpha=0.25, axis="x")

    ax.set_title(f"{title}\n(in-sample diagnostic, NOT a performance claim)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
