"""The scheduled runner.

One pass does exactly this:

1. Pull the latest bars through the same data layer the backtester uses.
2. Refuse to evaluate anything if the newest bar is stale, and say so once.
3. For each watched strategy, compute features, the current position, and the
   price levels that would trigger an entry.
4. Decide whether each level is newly worth a message (approach threshold plus
   hysteresis, handled by the alert store).
5. Send, rate limited, deduplicated.
6. Emit a heartbeat if one is due.

Deliberate design choices:

* **It reuses the backtest data layer and strategy code unchanged.** A separate
  "live" implementation of a strategy is a second definition that drifts from the
  tested one, and the drift is invisible until money is involved.
* **Only closed bars are used.** The most recent bar from a live feed may be
  partial, and evaluating a rule on a half-formed bar is the live-trading
  equivalent of the lookahead bug the backtester works so hard to avoid.
* **It notifies. It cannot trade.** There is no broker client in this repository.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.alerts.formatter import (
    APPROACHING,
    TRIGGERED,
    AlertContext,
    format_alert,
    format_heartbeat,
    format_stale_warning,
)
from src.alerts.state import AlertStore
from src.alerts.telegram import RateLimiter, TelegramClient, TelegramCredentials
from src.backtest.sizing import PositionSizer, risk_diagnostics
from src.config import Config
from src.data.loader import load_dataset
from src.features.indicators import atr as atr_indicator
from src.strategies import RESEARCH_STRATEGIES
from src.strategies.base import EntryLevel, Strategy, validate_signals

log = logging.getLogger(__name__)


@dataclass
class AlertSettings:
    enabled: bool = True
    strategies: tuple[str, ...] = ("trend_donchian",)
    symbol_label: str = "XAU/USD"
    poll_interval_seconds: int = 300
    approach_atr_multiple: float = 0.5
    reset_atr_multiple: float = 1.5
    level_move_atr_multiple: float = 1.0
    notify_on_trigger: bool = True
    stop_atr_multiple: float = 2.0
    max_bar_age_minutes: float = 90.0
    lookback_days: int = 30
    heartbeat_hours: float = 6.0
    send_startup_heartbeat: bool = True
    min_seconds_between_messages: float = 3.0
    max_messages_per_hour: int = 20
    state_path: str = "data/alerts_state.json"
    evidence_note: str = ""
    dry_run: bool = False

    @classmethod
    def from_config(cls, cfg: Config) -> "AlertSettings":
        a = cfg.section("alerts")
        return cls(
            enabled=bool(a.get("enabled", True)),
            strategies=tuple(a.get("strategies", ["trend_donchian"])),
            symbol_label=str(a.get("symbol_label", cfg.get("instrument.symbol", "XAUUSD"))),
            poll_interval_seconds=int(a.get("poll_interval_seconds", 300)),
            approach_atr_multiple=float(a.get("approach_atr_multiple", 0.5)),
            reset_atr_multiple=float(a.get("reset_atr_multiple", 1.5)),
            level_move_atr_multiple=float(a.get("level_move_atr_multiple", 1.0)),
            notify_on_trigger=bool(a.get("notify_on_trigger", True)),
            stop_atr_multiple=float(a.get("stop_atr_multiple", cfg.get("sizing.stop_atr_multiple", 2.0))),
            max_bar_age_minutes=float(a.get("max_bar_age_minutes", 90)),
            lookback_days=int(a.get("lookback_days", 30)),
            heartbeat_hours=float(a.get("heartbeat_hours", 6)),
            send_startup_heartbeat=bool(a.get("send_startup_heartbeat", True)),
            min_seconds_between_messages=float(a.get("rate_limit.min_seconds_between_messages", 3)),
            max_messages_per_hour=int(a.get("rate_limit.max_messages_per_hour", 20)),
            state_path=str(a.get("state_path", "data/alerts_state.json")),
            evidence_note=str(a.get("evidence_note", "")).strip(),
            dry_run=bool(a.get("dry_run", False)),
        )

    def validate(self) -> None:
        if self.reset_atr_multiple <= self.approach_atr_multiple:
            raise ValueError(
                "alerts.reset_atr_multiple must be greater than approach_atr_multiple. "
                "Without a gap between them there is no hysteresis, and price sitting "
                "at the threshold produces one alert per bar."
            )
        if self.approach_atr_multiple <= 0:
            raise ValueError("alerts.approach_atr_multiple must be positive")


@dataclass
class SentMessage:
    kind: str          # "alert" | "heartbeat" | "stale"
    key: str
    text: str
    delivered: bool


@dataclass
class CheckOutcome:
    sent: list[SentMessage] = field(default_factory=list)
    market_open: bool = False
    stale: bool = False
    last_bar: pd.Timestamp | None = None
    strategy_states: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class AlertRunner:
    def __init__(
        self,
        cfg: Config,
        *,
        client: TelegramClient | None = None,
        store: AlertStore | None = None,
        settings: AlertSettings | None = None,
        adapter=None,
        clock=None,
    ):
        self.cfg = cfg
        self.settings = settings or AlertSettings.from_config(cfg)
        self.settings.validate()
        self.adapter = adapter
        self._clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))

        self.store = store or AlertStore.load(self.settings.state_path)
        if client is not None:
            self.client = client
        else:
            credentials = TelegramCredentials.from_env()
            self.client = TelegramClient(
                credentials,
                limiter=RateLimiter(
                    min_seconds_between_messages=self.settings.min_seconds_between_messages,
                    max_messages_per_hour=self.settings.max_messages_per_hour,
                    recent=list(self.store.send_times),
                ),
                dry_run=self.settings.dry_run,
            )

        self.sizer = PositionSizer.from_config(cfg)
        unknown = [s for s in self.settings.strategies if s not in RESEARCH_STRATEGIES]
        if unknown:
            raise ValueError(
                f"alerts.strategies names unknown strategies {unknown}; "
                f"available: {sorted(RESEARCH_STRATEGIES)}"
            )

    # ------------------------------------------------------------------ #
    def _now(self) -> pd.Timestamp:
        return self._clock()

    def _persist(self) -> None:
        self.store.send_times = list(self.client.limiter.recent)
        self.store.save()

    def _send(self, kind: str, key: str, text: str, outcome: CheckOutcome) -> bool:
        try:
            delivered = self.client.send(text)
        except Exception as exc:
            # A delivery failure must not kill the loop, and must not be silent.
            message = f"{type(exc).__name__}: {exc}"
            log.error("failed to send %s alert (%s): %s", kind, key, message)
            outcome.errors.append(message)
            return False
        outcome.sent.append(SentMessage(kind=kind, key=key, text=text, delivered=delivered))
        return delivered

    # ------------------------------------------------------------------ #
    def load_bars(self):
        """Latest bars through the normal data layer.

        The window is bounded by ``lookback_days``, which must comfortably exceed
        the longest indicator lookback or the strategy state will be wrong.

        ``freshness`` is what makes this a *live* load rather than a research one.
        The cache tolerates a day of staleness by default, which is correct for a
        backtest and silently fatal here: without this the poller is served the
        same frame for up to twenty-four hours, reports a stale feed once, and
        then evaluates nothing. One base bar is the tightest useful bound.
        """
        now = self._now()
        start = (now - pd.Timedelta(days=self.settings.lookback_days)).strftime("%Y-%m-%d")
        cfg = self.cfg.with_overrides({
            "data.start": start,
            "data.end": now.strftime("%Y-%m-%d %H:%M:%S"),
        })
        freshness = pd.Timedelta(self.cfg.get("data.base_resolution", "1min"))
        return load_dataset(cfg, adapter=self.adapter, freshness=freshness)

    # ------------------------------------------------------------------ #
    def check_once(self) -> CheckOutcome:
        outcome = CheckOutcome()
        now = self._now()
        self.store.record_check(now.timestamp())

        if not self.settings.enabled:
            log.info("alerts are disabled in config; nothing to do")
            return outcome

        try:
            dataset = self.load_bars()
        except Exception as exc:
            message = f"data load failed: {type(exc).__name__}: {exc}"
            log.error(message)
            outcome.errors.append(message)
            self._persist()
            return outcome

        bars = dataset.bars
        if bars.empty:
            outcome.errors.append("no bars available")
            self._persist()
            return outcome

        # Only closed bars. The newest bar from a live feed can be partial, and a
        # rule evaluated on a half-formed bar is a live lookahead bug.
        bars = bars.iloc[:-1] if len(bars) > 1 else bars
        last_bar = bars.index[-1]
        outcome.last_bar = last_bar
        outcome.market_open = bool(dataset.calendar.is_open(pd.DatetimeIndex([now]))[0])

        age_minutes = (now - last_bar).total_seconds() / 60.0
        outcome.stale = age_minutes > self.settings.max_bar_age_minutes and outcome.market_open
        if outcome.stale:
            if not self.store.stale_data_notified:
                text = format_stale_warning(
                    symbol=self.settings.symbol_label, last_bar=last_bar,
                    age_minutes=age_minutes, limit_minutes=self.settings.max_bar_age_minutes,
                )
                self._send("stale", "stale_data", text, outcome)
                self.store.stale_data_notified = True
            self._maybe_heartbeat(outcome, dataset, bars, now)
            self._persist()
            return outcome
        self.store.stale_data_notified = False

        macro = dataset.macro if len(dataset.macro.columns) else None
        for name in self.settings.strategies:
            try:
                self._check_strategy(name, dataset, bars, macro, now, outcome)
            except Exception as exc:
                message = f"{name}: {type(exc).__name__}: {exc}"
                log.error("strategy check failed for %s", message)
                outcome.errors.append(message)

        self._maybe_heartbeat(outcome, dataset, bars, now)
        self._persist()
        return outcome

    # ------------------------------------------------------------------ #
    def _check_strategy(self, name, dataset, bars, macro, now, outcome: CheckOutcome) -> None:
        strategy: Strategy = RESEARCH_STRATEGIES[name]()
        if len(bars) <= strategy.max_lookback + 2:
            outcome.errors.append(
                f"{name}: only {len(bars)} bars loaded but {strategy.max_lookback} are "
                "needed for warm-up; raise alerts.lookback_days"
            )
            return

        features = strategy.compute_features(bars, macro)
        signals = validate_signals(strategy.generate_signals(bars, features), bars, name)
        position = int(signals.iloc[-1])
        previous = int(signals.iloc[-2]) if len(signals) > 1 else 0

        atr_series = (
            features["atr"] if "atr" in features.columns
            else atr_indicator(bars, int(self.cfg.get("sizing.atr_period", 14)))
        )
        atr_value = float(atr_series.iloc[-1])
        price = float(bars["close"].iloc[-1])

        if not np.isfinite(atr_value) or atr_value <= 0:
            outcome.errors.append(f"{name}: ATR unavailable, skipping")
            return

        levels = strategy.entry_levels(bars, features)
        outcome.strategy_states.append(
            f"{name}: {_position_word(position)}, price {price:,.2f}, ATR {atr_value:,.2f}"
            + (f", {len(levels)} level(s) watched" if levels else ", no price levels exposed")
        )

        risk = self._risk(bars, atr_value)
        position_oz = self.sizer.min_position_oz

        # A signal that just fired is the actionable moment; report it regardless
        # of how far price travelled to get there.
        if self.settings.notify_on_trigger and position != 0 and position != previous:
            level = next(
                (lv for lv in levels if lv.direction == position),
                EntryLevel(direction=position, price=price, kind="signal", note="signal fired"),
            )
            key = f"{name}:{position}:trigger"
            state = self.store.get(key)
            if state.status != "alerted" or state.last_level != price:
                context = self._context(
                    TRIGGERED, name, strategy, dataset, bars, level, price, price,
                    0.0, 0.0, atr_value, position, position_oz, risk, now, features,
                )
                if self._send("alert", key, format_alert(context), outcome):
                    self.store.record_alert(key, level=price, distance=0.0, now=now.timestamp())
        elif position == 0:
            self.store.rearm(f"{name}:1:trigger")
            self.store.rearm(f"{name}:-1:trigger")

        if position != 0:
            # Already in the trade: an approach alert would be noise.
            return

        approach = self.settings.approach_atr_multiple * atr_value
        reset = self.settings.reset_atr_multiple * atr_value
        tolerance = self.settings.level_move_atr_multiple * atr_value

        for level in levels:
            if not np.isfinite(level.price):
                continue
            key = f"{name}:{level.direction}:approach"
            if level.blocked_by:
                # A blocked setup cannot fire, so it re-arms rather than alerting.
                self.store.rearm(key)
                continue

            distance = level.price - price
            # Only approaches from the side the trade would come from count. Price
            # already through a breakout level is a fired signal, not an approach.
            if level.direction > 0 and distance < 0:
                self.store.rearm(key)
                continue
            if level.direction < 0 and distance > 0:
                self.store.rearm(key)
                continue

            magnitude = abs(distance)
            if not self.store.should_alert(
                key, level=level.price, distance=magnitude,
                approach_threshold=approach, reset_threshold=reset,
                level_move_tolerance=tolerance,
            ):
                continue

            context = self._context(
                APPROACHING, name, strategy, dataset, bars, level, price, level.price,
                distance, magnitude / atr_value, atr_value, position, position_oz,
                risk, now, features,
            )
            if self._send("alert", key, format_alert(context), outcome):
                self.store.record_alert(key, level=level.price, distance=magnitude,
                                        now=now.timestamp())

    # ------------------------------------------------------------------ #
    def _context(self, event, name, strategy, dataset, bars, level, price, _target,
                 distance_usd, distance_atr, atr_value, position, position_oz,
                 risk, now, features) -> AlertContext:
        strategy_stop = strategy.current_stop(bars, features)
        if strategy_stop is not None and np.isfinite(strategy_stop):
            suggested_stop, source = float(strategy_stop), "strategy trailing stop"
        else:
            offset = self.settings.stop_atr_multiple * atr_value
            reference = level.price if np.isfinite(level.price) else price
            suggested_stop = reference - offset if level.direction > 0 else reference + offset
            source = f"{self.settings.stop_atr_multiple:g}x ATR from the entry level"

        return AlertContext(
            event=event,
            strategy=strategy.describe(),
            strategy_params=dict(strategy.params),
            symbol=self.settings.symbol_label,
            resolution=dataset.resolution,
            bar_time=bars.index[-1],
            price=price,
            level=level,
            distance_usd=distance_usd,
            distance_atr=distance_atr,
            atr=atr_value,
            suggested_stop=suggested_stop,
            stop_source=source,
            position=position,
            position_oz=position_oz,
            risk=risk,
            evidence_note=self.settings.evidence_note,
            is_synthetic=dataset.provenance.is_synthetic,
            now=now,
        )

    def _risk(self, bars, atr_value):
        daily = bars.resample("1D").agg({"high": "max", "low": "min"}).dropna()
        daily_range = float((daily["high"] - daily["low"]).median()) if len(daily) else float("nan")
        return risk_diagnostics(
            equity=float(self.cfg.get("account.initial_capital_usd")),
            atr_usd_per_oz=atr_value,
            position_oz=self.sizer.min_position_oz,
            sizer=self.sizer,
            daily_range_usd_per_oz=daily_range,
            warning_threshold_pct=float(self.cfg.get("sizing.risk_per_atr_warning_pct", 5.0)),
        )

    def _maybe_heartbeat(self, outcome: CheckOutcome, dataset, bars, now) -> None:
        first_ever = self.store.last_heartbeat_epoch is None
        if first_ever and not self.settings.send_startup_heartbeat:
            # Record the time so the interval starts now rather than firing on the
            # next pass anyway.
            self.store.record_heartbeat(now.timestamp())
            return
        if not self.store.heartbeat_due(interval_hours=self.settings.heartbeat_hours,
                                        now=now.timestamp()):
            return
        text = format_heartbeat(
            symbol=self.settings.symbol_label,
            now=now,
            last_bar=outcome.last_bar,
            bars_seen=len(bars),
            checks=self.store.checks,
            strategy_states=outcome.strategy_states,
            armed_setups=len(self.store.active_setups()),
            market_open=outcome.market_open,
            evidence_note=self.settings.evidence_note,
            is_synthetic=dataset.provenance.is_synthetic,
            stale=outcome.stale,
        )
        if self._send("heartbeat", "heartbeat", text, outcome):
            self.store.record_heartbeat(now.timestamp())

    # ------------------------------------------------------------------ #
    def run_forever(self, *, interval_seconds: int | None = None, max_iterations: int | None = None,
                    sleeper=time.sleep) -> int:
        interval = interval_seconds or self.settings.poll_interval_seconds
        if self.settings.send_startup_heartbeat:
            self.store.last_heartbeat_epoch = None   # force one on the first pass

        iterations = 0
        log.info(
            "alert runner started: watching %s every %ss (dry_run=%s)",
            ", ".join(self.settings.strategies), interval, self.settings.dry_run,
        )
        while max_iterations is None or iterations < max_iterations:
            outcome = self.check_once()
            for message in outcome.sent:
                log.info("sent %s (%s), delivered=%s", message.kind, message.key, message.delivered)
            for error in outcome.errors:
                log.error("check error: %s", error)
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
            sleeper(interval)
        return iterations


def _position_word(position: int) -> str:
    return {1: "long", -1: "short", 0: "flat"}[int(position)]
