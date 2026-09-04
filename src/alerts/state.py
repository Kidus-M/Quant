"""Persistent alert state: deduplication, hysteresis, heartbeat and rate limiting.

The requirement is "one alert per setup, not one per bar". A naive threshold test
fires on every bar while price sits near a level, which trains the recipient to
ignore the bot within a day.

Each setup is a small state machine:

    armed  --price comes within the approach threshold-->  alerted
    alerted  --price moves back beyond the reset threshold-->  armed
    alerted  --the level itself moves by more than the tolerance-->  re-alert

The gap between the approach and reset thresholds is deliberate hysteresis. The
state lives on disk so a restart does not replay every alert that was already
sent, and the rate limiter shares the file for the same reason.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.config import resolve_path

log = logging.getLogger(__name__)

ARMED = "armed"
ALERTED = "alerted"


@dataclass
class SetupState:
    key: str
    status: str = ARMED
    last_alert_epoch: float | None = None
    last_level: float | None = None
    last_distance: float | None = None
    alert_count: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class AlertStore:
    path: Path
    setups: dict[str, SetupState] = field(default_factory=dict)
    last_heartbeat_epoch: float | None = None
    last_check_epoch: float | None = None
    stale_data_notified: bool = False
    send_times: list[float] = field(default_factory=list)
    checks: int = 0

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path) -> "AlertStore":
        path = resolve_path(path)
        store = cls(path=path)
        if not path.exists():
            return store
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # A corrupt state file must not stop alerting, but starting from a
            # clean slate can replay alerts, so say so loudly.
            log.warning("alert state at %s is unreadable (%s); starting from empty", path, exc)
            return store

        store.setups = {
            key: SetupState(**value) for key, value in (raw.get("setups") or {}).items()
        }
        store.last_heartbeat_epoch = raw.get("last_heartbeat_epoch")
        store.last_check_epoch = raw.get("last_check_epoch")
        store.stale_data_notified = bool(raw.get("stale_data_notified", False))
        store.send_times = list(raw.get("send_times") or [])
        store.checks = int(raw.get("checks", 0))
        return store

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "setups": {key: value.as_dict() for key, value in self.setups.items()},
            "last_heartbeat_epoch": self.last_heartbeat_epoch,
            "last_check_epoch": self.last_check_epoch,
            "stale_data_notified": self.stale_data_notified,
            "send_times": self.send_times[-200:],
            "checks": self.checks,
        }
        # Write via a temporary file so a crash mid-write cannot leave the state
        # truncated, which would replay alerts on the next start.
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    # ------------------------------------------------------------------ #
    def get(self, key: str) -> SetupState:
        return self.setups.setdefault(key, SetupState(key=key))

    def record_alert(self, key: str, *, level: float, distance: float, now: float | None = None) -> None:
        now = time.time() if now is None else now
        state = self.get(key)
        state.status = ALERTED
        state.last_alert_epoch = now
        state.last_level = float(level)
        state.last_distance = float(distance)
        state.alert_count += 1

    def rearm(self, key: str) -> None:
        state = self.get(key)
        if state.status != ARMED:
            log.debug("setup %s re-armed", key)
        state.status = ARMED
        state.last_distance = None

    def should_alert(
        self,
        key: str,
        *,
        level: float,
        distance: float,
        approach_threshold: float,
        reset_threshold: float,
        level_move_tolerance: float,
    ) -> bool:
        """The whole deduplication decision, in one place.

        Returns True only when this is a genuinely new setup, not the same one
        still sitting near its trigger.
        """
        state = self.get(key)

        if distance > reset_threshold:
            self.rearm(key)
            return False
        if distance > approach_threshold:
            # In the hysteresis band: not close enough to alert, not far enough
            # to re-arm. Leave the state alone.
            return False

        if state.status == ARMED:
            return True

        # Already alerted. Only speak again if the level itself has moved enough
        # that the previous message is misleading.
        if state.last_level is not None and level_move_tolerance > 0:
            if abs(level - state.last_level) > level_move_tolerance:
                return True
        return False

    # ------------------------------------------------------------------ #
    def heartbeat_due(self, *, interval_hours: float, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        if self.last_heartbeat_epoch is None:
            return True
        return (now - self.last_heartbeat_epoch) >= interval_hours * 3600.0

    def record_heartbeat(self, now: float | None = None) -> None:
        self.last_heartbeat_epoch = time.time() if now is None else now

    def record_check(self, now: float | None = None) -> None:
        self.last_check_epoch = time.time() if now is None else now
        self.checks += 1

    def active_setups(self) -> dict[str, SetupState]:
        return {k: v for k, v in self.setups.items() if v.status == ALERTED}
