"""Message composition.

An alert is read on a phone, usually in a hurry, so it leads with the three
numbers a reader actually acts on and pushes everything else below them:

1. **direction** -- LONG or SHORT, in the first line
2. **entry**
3. **stop**

then a take-profit zone, then the context that explains where those came from.

Anything that cannot be acted on has been cut. What has NOT been cut, and must
not be:

* the position sizing warning, whenever it applies. On a 50 USD account it is the
  single most important line in the message.
* the evidence caveat, always. This engine is wired to strategies that have not
  cleared their own random-entry benchmark, and a message that omits that reads
  like a recommendation.
* PAPER SIGNAL, always.

**On the take-profit probabilities.** They are the driftless first-passage
result: for a target ``a`` away and a stop ``b`` away, the chance of touching the
target first is ``b / (a + b)``, so a 2R target is hit about a third of the time.
That is the null hypothesis, not a forecast -- it is what the numbers look like
when the strategy has no edge at all, which is exactly the comparison a reader
needs and the one they are least likely to make unaided. It assumes no drift and
continuous prices, and it prices in no costs; a gap through either level makes it
optimistic. The message says so in one line rather than leaving it implied.
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


# R multiples shown in the take-profit zone.
TP_MULTIPLES = (1.0, 2.0, 3.0)


def take_profit_zone(
    entry: float, stop: float, direction: int, multiples=TP_MULTIPLES
) -> list[tuple[float, float, float]]:
    """``(multiple, price, probability)`` for each take-profit level.

    The probability is the driftless first-passage result -- with the stop ``b``
    away and the target ``a`` away, the chance of touching the target first is
    ``b / (a + b)``, which for a target at ``m`` times the risk is ``1 / (1 + m)``.
    No drift, continuous prices, no costs. It is the null, and it is here so that
    a 3R target is read as "about one in four" rather than as a plan.
    """
    risk = abs(entry - stop)
    if not (np.isfinite(risk) and np.isfinite(entry) and risk > 0):
        return []
    sign = 1.0 if direction > 0 else -1.0
    return [
        (float(m), entry + sign * float(m) * risk, 1.0 / (1.0 + float(m)))
        for m in multiples
    ]


def format_alert(context: AlertContext) -> str:
    direction = context.level.side
    arrow = "\u25b2" if context.level.direction > 0 else "\u25bc"
    entry = context.level.price
    stop = context.suggested_stop
    risk_per_oz = abs(entry - stop)

    lines = [
        f"{arrow} <b>{html_escape(direction)} {html_escape(context.symbol)}</b>"
        f" - {html_escape(context.event)}",
        "",
        f"Entry  <b>{_fmt_price(entry)}</b>",
    ]

    # The stop line carries its own distance, so the reader never has to subtract
    # two five-figure numbers on a phone to find out what the trade risks.
    if np.isfinite(risk_per_oz) and risk_per_oz > 0:
        stop_bits = f"{_fmt_price(risk_per_oz)} USD/oz"
        if np.isfinite(context.atr) and context.atr > 0:
            stop_bits += f", {risk_per_oz / context.atr:.1f} ATR"
        if context.position_oz > 0:
            stop_bits += f", {_fmt_price(risk_per_oz * context.position_oz)} USD at {context.position_oz:g} oz"
        lines.append(f"Stop   <b>{_fmt_price(stop)}</b>  ({stop_bits})")
    else:
        lines.append(f"Stop   <b>{_fmt_price(stop)}</b>")

    targets = take_profit_zone(entry, stop, context.level.direction)
    if targets:
        lines += ["", "<b>Take profit</b>  (chance of reaching before the stop)"]
        lines += [
            f"TP{i}    {_fmt_price(price)}   {multiple:g}R   {probability:.0%}"
            for i, (multiple, price, probability) in enumerate(targets, start=1)
        ]

    # Context, below the actionable numbers.
    context_bits = [html_escape(context.strategy), html_escape(context.resolution) + " bars"]
    if context.event == APPROACHING and np.isfinite(context.distance_usd):
        context_bits.append(
            f"{_fmt_price(abs(context.distance_usd))} USD/oz away ({context.distance_atr:.1f} ATR)"
        )
    lines += ["", " \u00b7 ".join(context_bits)]
    if context.level.note:
        lines.append(f"Rule: {html_escape(context.level.note)}")

    if context.level.blocked_by:
        lines.append("")
        lines.append("<b>Blocked:</b> " + html_escape("; ".join(context.level.blocked_by)))

    if context.position != 0:
        held = "long" if context.position > 0 else "short"
        lines.append(f"Already {held} as of the last closed bar.")

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
        "<i>TP odds assume no drift and ignore costs.</i>",
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
