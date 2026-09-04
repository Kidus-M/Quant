"""Message composition.

Every alert carries the same block of context, because a notification that only
says "RSI2 long setup" is unactionable and, worse, invites the reader to fill in
the missing numbers optimistically:

* symbol and current price
* the entry level and the distance to it, in USD and in ATRs
* which strategy fired, and on what rule
* an ATR-based suggested stop, and what one ATR costs at the configured size
* the position sizing warning, whenever it applies
* the evidence caveat, always
* PAPER SIGNAL, always

The last two are not decoration. This engine is wired to a strategy that has not
cleared its own benchmark, and a message that omits that reads like a
recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.alerts.telegram import html_escape
from src.backtest.sizing import RiskDiagnostics
from src.strategies.base import EntryLevel

APPROACHING = "approaching"
TRIGGERED = "triggered"


@dataclass
class AlertContext:
    """Everything one message needs. Assembled by the runner."""

    event: str                      # APPROACHING | TRIGGERED
    strategy: str
    strategy_params: dict
    symbol: str
    resolution: str
    bar_time: pd.Timestamp
    price: float
    level: EntryLevel
    distance_usd: float
    distance_atr: float
    atr: float
    suggested_stop: float
    stop_source: str                # "strategy trailing stop" | "ATR suggestion"
    position: int                   # current strategy position at the last bar
    position_oz: float
    risk: RiskDiagnostics | None
    evidence_note: str
    is_synthetic: bool
    now: pd.Timestamp


def _fmt_price(value: float) -> str:
    return f"{value:,.2f}" if np.isfinite(value) else "n/a"


def format_alert(context: AlertContext) -> str:
    direction = context.level.side
    arrow = "▲" if context.level.direction > 0 else "▼"
    headline = (
        f"{arrow} <b>{html_escape(direction)} {html_escape(context.symbol)}</b> "
        f"- {html_escape(context.event)}"
    )

    lines = [
        headline,
        f"<b>{html_escape(context.strategy)}</b> on {html_escape(context.resolution)} bars",
        "",
        f"Price        <b>{_fmt_price(context.price)}</b>",
        f"Entry level  <b>{_fmt_price(context.level.price)}</b>",
    ]

    if context.event == APPROACHING:
        lines.append(
            f"Distance     {_fmt_price(abs(context.distance_usd))} USD/oz "
            f"({context.distance_atr:.2f} ATR)"
        )
    else:
        lines.append("Distance     level reached")

    lines += [
        f"ATR          {_fmt_price(context.atr)} USD/oz",
        f"Suggested stop  <b>{_fmt_price(context.suggested_stop)}</b> "
        f"({html_escape(context.stop_source)})",
    ]

    stop_distance = abs(context.level.price - context.suggested_stop)
    if np.isfinite(stop_distance) and context.position_oz > 0:
        risk_usd = stop_distance * context.position_oz
        lines.append(
            f"Risk at {context.position_oz:g} oz  {_fmt_price(risk_usd)} USD if the stop is hit"
        )

    if context.level.note:
        lines += ["", f"Rule: {html_escape(context.level.note)}"]

    if context.level.blocked_by:
        lines.append("")
        lines.append("<b>Filters currently blocking this setup:</b>")
        lines += [f"- {html_escape(reason)}" for reason in context.level.blocked_by]

    if context.position != 0:
        held = "long" if context.position > 0 else "short"
        lines += ["", f"Note: the strategy is already {held} as of the last closed bar."]

    warning = format_risk_line(context.risk)
    if warning:
        lines += ["", warning]

    lines += [
        "",
        f"Bar {html_escape(context.bar_time.strftime('%Y-%m-%d %H:%M UTC'))}"
        f" | sent {html_escape(context.now.strftime('%H:%M UTC'))}",
    ]
    if context.is_synthetic:
        lines.append(
            "<b>SYNTHETIC DATA</b> - this runner is pointed at the offline generator, "
            "not a market feed."
        )
    lines += [
        f"<i>{html_escape(context.evidence_note)}</i>",
        "<b>PAPER SIGNAL - no order was placed and this engine cannot place one.</b>",
    ]
    return "\n".join(lines)


def format_risk_line(risk: RiskDiagnostics | None) -> str:
    """One-line version of the position sizing warning, for a phone screen."""
    if risk is None or not risk.needs_warning:
        return ""
    parts = [
        "<b>POSITION SIZING WARNING</b>",
        f"At {risk.equity:,.2f} USD equity, the minimum {risk.min_position_oz:g} oz "
        f"position risks {risk.min_risk_per_atr_pct:.1f}% of the account per ATR "
        f"and {risk.risk_per_daily_range_pct:.0f}% per typical daily range.",
    ]
    if np.isfinite(risk.min_viable_equity_daily_range_usd):
        parts.append(
            f"A {risk.risk_fraction_per_trade:.0%} risk rule needs about "
            f"{risk.min_viable_equity_daily_range_usd:,.0f} USD."
        )
    return "\n".join(parts)


def format_heartbeat(
    *,
    symbol: str,
    now: pd.Timestamp,
    last_bar: pd.Timestamp | None,
    bars_seen: int,
    checks: int,
    strategy_states: list[str],
    armed_setups: int,
    market_open: bool,
    evidence_note: str,
    is_synthetic: bool,
    stale: bool = False,
) -> str:
    """Periodic proof of life, so silence means something is broken."""
    lines = [
        f"✓ <b>Heartbeat</b> - {html_escape(symbol)} watcher alive",
        "",
        f"Now          {html_escape(now.strftime('%Y-%m-%d %H:%M UTC'))}",
        f"Last bar     {html_escape(last_bar.strftime('%Y-%m-%d %H:%M UTC')) if last_bar is not None else 'none'}",
        f"Bars loaded  {bars_seen:,}",
        f"Checks run   {checks:,}",
        f"Market       {'open' if market_open else 'closed'}",
        f"Setups alerted and awaiting reset: {armed_setups}",
    ]
    if strategy_states:
        lines += ["", "<b>Current state</b>"]
        lines += [f"- {html_escape(state)}" for state in strategy_states]
    if stale:
        lines += ["", "<b>Warning: bar data is stale.</b> The feed may be down."]
    if is_synthetic:
        lines.append("")
        lines.append("<b>SYNTHETIC DATA</b> - pointed at the offline generator.")
    lines += ["", f"<i>{html_escape(evidence_note)}</i>"]
    return "\n".join(lines)


def format_stale_warning(
    *, symbol: str, last_bar: pd.Timestamp | None, age_minutes: float, limit_minutes: float
) -> str:
    return "\n".join([
        f"⚠ <b>Stale data</b> - {html_escape(symbol)}",
        "",
        f"Newest bar is {age_minutes:,.0f} minutes old "
        f"(limit {limit_minutes:,.0f}).",
        f"Last bar: {html_escape(last_bar.strftime('%Y-%m-%d %H:%M UTC')) if last_bar is not None else 'none'}",
        "",
        "No signals are being evaluated on fresh data. Silence from here is a "
        "feed problem, not an absence of setups.",
    ])
