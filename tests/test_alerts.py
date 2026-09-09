"""Phase 2 alerting: transport, deduplication, levels, and the runner.

The behaviours worth guarding here are the ones that make an alert bot either
useful or ignorable:

* one alert per setup, not one per bar
* the bot token never leaks into a log, an exception or the state file
* silence is always explained, by a heartbeat or a stale-data warning
* every message carries the position sizing warning and the paper-signal caveat
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.alerts.formatter import (
    APPROACHING,
    TRIGGERED,
    format_heartbeat,
    format_stale_warning,
)
from src.alerts.runner import AlertRunner, AlertSettings
from src.alerts.state import ALERTED, ARMED, AlertStore
from src.alerts.telegram import (
    MAX_MESSAGE_CHARS,
    RateLimiter,
    TelegramClient,
    TelegramCredentials,
    TelegramError,
)
from src.config import load_configs
from src.data.base import DataProvenance
from src.data.loader import BarDataset
from src.data.quality import QualityReport
from src.data.sessions import SessionCalendar
from tests.conftest import make_bars

FAKE_TOKEN = "1234567890:AAFakeTokenValueForTestingOnly_not_real"


# ---------------------------------------------------------------------- #
# Fakes
# ---------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"ok": True, "result": {}}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []
        self.headers = {}

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        if self.responses:
            return self.responses.pop(0)
        return FakeResponse()


class RecordingClient:
    """Stands in for TelegramClient in runner tests."""

    def __init__(self, fail_with: Exception | None = None):
        self.messages: list[str] = []
        self.limiter = RateLimiter()
        self.dry_run = False
        self.fail_with = fail_with

    def send(self, text: str) -> bool:
        if self.fail_with is not None:
            raise self.fail_with
        self.messages.append(text)
        return True


@pytest.fixture
def credentials():
    return TelegramCredentials(bot_token=FAKE_TOKEN, chat_id="42")


@pytest.fixture
def alert_cfg(tmp_path):
    cfg = load_configs("config/backtest.yaml", "config/alerts.yaml")
    return cfg.with_overrides({
        # Pinned rather than inherited: these tests drive the full runner, so a
        # live production default would have them fetching real bars over a
        # rate-limited API on every run.
        "data.adapter": "synthetic",
        "data.cache_dir": str(tmp_path / "cache"),
        "data.quarantine_dir": str(tmp_path / "quarantine"),
        "macro.enabled": False,
        "alerts.state_path": str(tmp_path / "alerts_state.json"),
        "alerts.strategies": ["trend_donchian"],
        "alerts.send_startup_heartbeat": False,
    })


# ---------------------------------------------------------------------- #
# Credentials and redaction
# ---------------------------------------------------------------------- #
def test_missing_credentials_raise_with_actionable_advice(monkeypatch):
    monkeypatch.setattr("src.config.load_dotenv", lambda *a, **k: {})
    monkeypatch.setattr("src.alerts.telegram.load_dotenv", lambda *a, **k: {})
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(TelegramError, match="BotFather"):
        TelegramCredentials.from_env()


def test_missing_credentials_can_be_optional(monkeypatch):
    monkeypatch.setattr("src.alerts.telegram.load_dotenv", lambda *a, **k: {})
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert TelegramCredentials.from_env(required=False) is None


def test_redaction_removes_the_token_and_the_bot_id(credentials):
    leaky = f"HTTPSConnectionPool: POST https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage failed"
    cleaned = credentials.redact(leaky)
    assert FAKE_TOKEN not in cleaned
    assert "1234567890" not in cleaned
    assert "<TELEGRAM_BOT_TOKEN>" in cleaned


def test_transport_errors_never_contain_the_token(credentials):
    session = FakeSession([FakeResponse(status_code=401, payload={"ok": False,
                                                                 "description": f"bad token {FAKE_TOKEN}"})])
    client = TelegramClient(credentials, session=session, sleeper=lambda s: None)
    with pytest.raises(TelegramError) as excinfo:
        client.send("hello")
    assert FAKE_TOKEN not in str(excinfo.value)
    assert "bot token looks wrong" in str(excinfo.value)


def test_network_failures_are_redacted_too(credentials):
    class Boom(FakeSession):
        def post(self, url, json=None, timeout=None):
            raise RuntimeError(f"connection to {url} refused")

    client = TelegramClient(credentials, session=Boom(), retries=2, sleeper=lambda s: None)
    with pytest.raises(TelegramError) as excinfo:
        client.send("hello")
    assert FAKE_TOKEN not in str(excinfo.value)


def test_permanent_rejections_are_not_retried(credentials):
    session = FakeSession([FakeResponse(status_code=403, payload={"ok": False, "description": "forbidden"})])
    client = TelegramClient(credentials, session=session, retries=5, sleeper=lambda s: None)
    with pytest.raises(TelegramError, match="TELEGRAM_CHAT_ID"):
        client.send("hello")
    assert len(session.calls) == 1, "a 403 was retried, burning the rate limit for nothing"


def test_successful_send_posts_the_expected_payload(credentials):
    session = FakeSession()
    client = TelegramClient(credentials, session=session, sleeper=lambda s: None)
    assert client.send("hello <b>world</b>") is True
    url, payload = session.calls[0]
    assert url.endswith("/sendMessage")
    assert payload["chat_id"] == "42"
    assert payload["parse_mode"] == "HTML"
    assert payload["text"] == "hello <b>world</b>"


def test_over_long_messages_are_truncated(credentials):
    session = FakeSession()
    client = TelegramClient(credentials, session=session, sleeper=lambda s: None)
    client.send("x" * (MAX_MESSAGE_CHARS * 2))
    _, payload = session.calls[0]
    assert len(payload["text"]) <= MAX_MESSAGE_CHARS
    assert payload["text"].endswith("(message truncated)")


def test_dry_run_never_touches_the_network(credentials):
    session = FakeSession()
    client = TelegramClient(credentials, session=session, dry_run=True)
    assert client.send("hello") is True
    assert session.calls == []


def test_rate_limited_responses_are_retried_after_the_requested_delay(credentials):
    slept = []
    session = FakeSession([
        FakeResponse(status_code=429, payload={"ok": False, "parameters": {"retry_after": 7}}),
        FakeResponse(),
    ])
    client = TelegramClient(credentials, session=session, sleeper=slept.append)
    assert client.send("hello") is True
    assert 7 in slept


# ---------------------------------------------------------------------- #
# Rate limiting
# ---------------------------------------------------------------------- #
def test_rate_limiter_enforces_minimum_spacing():
    limiter = RateLimiter(min_seconds_between_messages=5, max_messages_per_hour=100)
    limiter.record(1000.0)
    assert limiter.seconds_until_allowed(1002.0) == pytest.approx(3.0)
    assert limiter.seconds_until_allowed(1006.0) == 0.0


def test_rate_limiter_enforces_the_hourly_cap():
    limiter = RateLimiter(min_seconds_between_messages=0, max_messages_per_hour=3)
    for i in range(3):
        limiter.record(1000.0 + i)
    wait = limiter.seconds_until_allowed(1010.0)
    assert wait > 3000     # must wait until the oldest message ages out
    assert limiter.seconds_until_allowed(1000.0 + 3601) == 0.0


def test_hourly_cap_drops_the_message_rather_than_blocking(credentials):
    session = FakeSession()
    limiter = RateLimiter(min_seconds_between_messages=0, max_messages_per_hour=1)
    limiter.record(1000.0)
    client = TelegramClient(credentials, session=session, limiter=limiter,
                            sleeper=lambda s: None, clock=lambda: 1001.0)
    assert client.send("hello") is False
    assert session.calls == []


# ---------------------------------------------------------------------- #
# Deduplication state machine
# ---------------------------------------------------------------------- #
def _store(tmp_path) -> AlertStore:
    return AlertStore.load(tmp_path / "state.json")


def test_first_approach_alerts_and_the_second_does_not(tmp_path):
    """The core requirement: one alert per setup, not one per bar."""
    store = _store(tmp_path)
    args = dict(approach_threshold=1.0, reset_threshold=3.0, level_move_tolerance=2.0)

    assert store.should_alert("k", level=1800.0, distance=0.5, **args) is True
    store.record_alert("k", level=1800.0, distance=0.5)
    for _ in range(20):
        assert store.should_alert("k", level=1800.0, distance=0.4, **args) is False


def test_setup_rearms_once_price_moves_away_and_can_alert_again(tmp_path):
    store = _store(tmp_path)
    args = dict(approach_threshold=1.0, reset_threshold=3.0, level_move_tolerance=2.0)

    assert store.should_alert("k", level=1800.0, distance=0.5, **args) is True
    store.record_alert("k", level=1800.0, distance=0.5)
    # Inside the hysteresis band: still quiet, still alerted.
    assert store.should_alert("k", level=1800.0, distance=2.0, **args) is False
    assert store.get("k").status == ALERTED
    # Beyond the reset threshold: re-arm.
    assert store.should_alert("k", level=1800.0, distance=5.0, **args) is False
    assert store.get("k").status == ARMED
    # Coming back is a new setup.
    assert store.should_alert("k", level=1800.0, distance=0.5, **args) is True


def test_a_materially_moved_level_alerts_again(tmp_path):
    store = _store(tmp_path)
    args = dict(approach_threshold=1.0, reset_threshold=3.0, level_move_tolerance=2.0)
    assert store.should_alert("k", level=1800.0, distance=0.5, **args) is True
    store.record_alert("k", level=1800.0, distance=0.5)
    assert store.should_alert("k", level=1801.0, distance=0.5, **args) is False
    assert store.should_alert("k", level=1805.0, distance=0.5, **args) is True


def test_state_survives_a_restart(tmp_path):
    store = _store(tmp_path)
    store.record_alert("k", level=1800.0, distance=0.5)
    store.record_heartbeat(1234.0)
    store.record_check(1235.0)
    store.save()

    reloaded = AlertStore.load(tmp_path / "state.json")
    assert reloaded.get("k").status == ALERTED
    assert reloaded.get("k").alert_count == 1
    assert reloaded.last_heartbeat_epoch == 1234.0
    assert reloaded.checks == 1


def test_a_corrupt_state_file_does_not_stop_alerting(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{ this is not json", encoding="utf-8")
    store = AlertStore.load(path)
    assert store.setups == {}
    assert store.get("k").status == ARMED


def test_state_is_written_atomically(tmp_path):
    store = _store(tmp_path)
    store.record_alert("k", level=1.0, distance=0.0)
    store.save()
    assert (tmp_path / "state.json").exists()
    assert not (tmp_path / "state.json.tmp").exists()
    json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))


def test_heartbeat_is_due_on_a_fresh_store_then_respects_the_interval(tmp_path):
    store = _store(tmp_path)
    assert store.heartbeat_due(interval_hours=6, now=1000.0) is True
    store.record_heartbeat(1000.0)
    assert store.heartbeat_due(interval_hours=6, now=1000.0 + 3600) is False
    assert store.heartbeat_due(interval_hours=6, now=1000.0 + 6 * 3600) is True


def test_the_state_file_never_contains_credentials(tmp_path, alert_cfg):
    store = AlertStore.load(tmp_path / "state.json")
    store.record_alert("k", level=1800.0, distance=1.0)
    store.save()
    text = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert "TELEGRAM" not in text.upper()
    assert ":AA" not in text


# ---------------------------------------------------------------------- #
# Entry levels
# ---------------------------------------------------------------------- #
def test_donchian_entry_levels_are_the_channel_edges(bars_15m):
    from src.strategies import DonchianTrendStrategy

    strategy = DonchianTrendStrategy()
    features = strategy.compute_features(bars_15m, None)
    levels = strategy.entry_levels(bars_15m, features)
    assert {lv.direction for lv in levels} == {1, -1}
    long_level = next(lv for lv in levels if lv.direction > 0)
    short_level = next(lv for lv in levels if lv.direction < 0)
    assert long_level.price == pytest.approx(features["donchian_upper"].iloc[-1])
    assert short_level.price == pytest.approx(features["donchian_lower"].iloc[-1])
    assert long_level.side == "LONG"


def test_donchian_current_stop_matches_the_backtested_stop(bars_15m):
    """The stop quoted in an alert must come from the same code the backtest uses."""
    from src.strategies import DonchianTrendStrategy

    strategy = DonchianTrendStrategy()
    features = strategy.compute_features(bars_15m, None)
    signals, stops = strategy._walk(bars_15m, features)
    reported = strategy.current_stop(bars_15m, features)
    if signals[-1] == 0:
        assert reported is None
    else:
        assert reported == pytest.approx(stops[-1])


def test_rsi2_trigger_price_actually_produces_the_threshold_rsi(bars_15m):
    """Property check on the closed-form RSI inversion.

    Append a bar closing exactly at the reported level and confirm RSI lands on
    the threshold. A level that does not do this is a lie to the recipient.
    """
    from src.features.indicators import rsi
    from src.strategies import Rsi2Strategy

    strategy = Rsi2Strategy()
    checked = 0
    for cut in range(600, 4000, 211):
        window = bars_15m.iloc[:cut]
        features = strategy.compute_features(window, None)
        for level in strategy.entry_levels(window, features):
            if any("already" in reason for reason in level.blocked_by):
                continue      # setup is live, not approaching; the formula branch differs
            nxt = window.iloc[[-1]].copy()
            nxt.index = nxt.index + pd.Timedelta(minutes=15)
            nxt.loc[:, ["open", "high", "low", "close"]] = level.price
            achieved = float(rsi(pd.concat([window, nxt])["close"], 2).iloc[-1])
            target = (strategy.params["oversold"] if level.direction > 0
                      else strategy.params["overbought"])
            assert achieved == pytest.approx(float(target), abs=1e-6)
            checked += 1
    assert checked > 10, "the inversion was never exercised"


def test_macro_filter_blocks_the_level_and_says_why(bars_15m):
    from src.data.fred import join_macro_to_bars
    from src.strategies.macro_trend import MacroFilteredTrendStrategy

    days = pd.date_range(bars_15m.index[0].normalize() - pd.Timedelta(days=300),
                         bars_15m.index[-1].normalize(), freq="B")
    rising = pd.Series(np.linspace(1.0, 3.0, len(days)), index=days, name="DFII10")
    macro = join_macro_to_bars(bars_15m, rising, lag_days=1)

    strategy = MacroFilteredTrendStrategy()
    features = strategy.compute_features(bars_15m, macro)
    levels = strategy.entry_levels(bars_15m, features)
    long_level = next(lv for lv in levels if lv.direction > 0)
    assert not long_level.is_reachable
    assert any("rising" in reason for reason in long_level.blocked_by)


def test_a_strategy_with_no_price_level_reports_none(bars_15m):
    from src.strategies import BuyAndHoldStrategy

    strategy = BuyAndHoldStrategy()
    assert strategy.entry_levels(bars_15m, strategy.compute_features(bars_15m)) == []


# ---------------------------------------------------------------------- #
# Runner
# ---------------------------------------------------------------------- #
class StubRunner(AlertRunner):
    """Runner with the data layer replaced, so tests control price exactly."""

    def __init__(self, *args, bars=None, **kwargs):
        self._bars = bars
        super().__init__(*args, **kwargs)

    def load_bars(self):
        calendar = SessionCalendar()
        return BarDataset(
            bars=self._bars, resolution="15min", base_resolution="1min",
            provenance=DataProvenance("test", "XAUUSD", "1min", is_synthetic=False),
            calendar=calendar, quality=QualityReport(len(self._bars), len(self._bars)),
            macro=pd.DataFrame(index=self._bars.index),
        )


def _rising_bars(n=400, start_price=1800.0, step=0.4, tail=None):
    closes = list(start_price + np.arange(n) * step)
    if tail is not None:
        closes = closes[: n - len(tail)] + list(tail)
    return make_bars(closes, start="2023-06-05 08:00", freq="15min")


def _breakout_bars(n=400, start_price=1800.0):
    """Range-bound, then a decisive break above the channel on the last closed bar.

    The runner discards the newest bar as potentially partial, so the breakout is
    placed at index -2 of the frame to land on the bar the runner actually sees.
    """
    closes = list(start_price + 5.0 * np.sin(np.arange(n) / 7.0))
    closes += [start_price + 40.0, start_price + 41.0]
    return make_bars(closes, start="2023-06-05 08:00", freq="15min")


def _runner(alert_cfg, bars, client=None, now=None, **overrides):
    settings = AlertSettings.from_config(alert_cfg)
    for key, value in overrides.items():
        setattr(settings, key, value)
    settings.validate()
    clock = (lambda: pd.Timestamp(now)) if now else (lambda: bars.index[-1] + pd.Timedelta(minutes=15))
    return StubRunner(
        alert_cfg, bars=bars, client=client or RecordingClient(),
        settings=settings, clock=clock,
    )


def test_runner_sends_one_alert_per_setup_not_one_per_bar(alert_cfg):
    bars = _rising_bars()
    client = RecordingClient()
    runner = _runner(alert_cfg, bars, client, approach_atr_multiple=20.0,
                     reset_atr_multiple=60.0, notify_on_trigger=False)
    first = runner.check_once()
    assert any(m.kind == "alert" for m in first.sent)
    before = len(client.messages)

    for _ in range(5):
        runner.check_once()
    assert len(client.messages) == before, "the runner re-alerted on an unchanged setup"


def test_runner_stays_quiet_when_price_is_far_from_every_level(alert_cfg):
    bars = _rising_bars()
    client = RecordingClient()
    runner = _runner(alert_cfg, bars, client, approach_atr_multiple=0.01,
                     reset_atr_multiple=0.02)
    outcome = runner.check_once()
    assert [m for m in outcome.sent if m.kind == "alert"] == []


def test_runner_does_not_send_approach_alerts_while_in_a_position(alert_cfg):
    # A steady uptrend puts the Donchian strategy long and keeps it there.
    bars = _rising_bars(n=500, step=1.0)
    client = RecordingClient()
    runner = _runner(alert_cfg, bars, client, approach_atr_multiple=50.0,
                     reset_atr_multiple=200.0, notify_on_trigger=False)
    outcome = runner.check_once()
    assert "long" in " ".join(outcome.strategy_states)
    assert [m for m in outcome.sent if m.kind == "alert"] == []


def test_runner_alerts_when_a_signal_actually_fires(alert_cfg):
    bars = _breakout_bars()
    client = RecordingClient()
    runner = _runner(alert_cfg, bars, client, notify_on_trigger=True,
                     approach_atr_multiple=0.01, reset_atr_multiple=0.02)
    outcome = runner.check_once()
    triggered = [m for m in outcome.sent if m.kind == "alert" and "trigger" in m.key]
    assert triggered, "a breakout on the last closed bar produced no trigger alert"
    assert "triggered" in triggered[0].text
    assert "LONG" in triggered[0].text

    # And it does not repeat on the next pass while the position is still open.
    before = len(client.messages)
    runner.check_once()
    assert len(client.messages) == before


def test_alert_message_contains_everything_the_spec_requires(alert_cfg):
    bars = _rising_bars()
    client = RecordingClient()
    runner = _runner(alert_cfg, bars, client, approach_atr_multiple=20.0,
                     reset_atr_multiple=60.0, notify_on_trigger=False)
    outcome = runner.check_once()
    text = next(m.text for m in outcome.sent if m.kind == "alert")

    assert "XAU/USD" in text                      # symbol
    assert "Price" in text                        # current price
    assert "Entry level" in text                  # entry level
    assert "Distance" in text                     # distance
    assert "trend_donchian" in text               # which strategy fired
    assert "Suggested stop" in text               # ATR-based suggested stop
    assert "ATR" in text
    assert "POSITION SIZING WARNING" in text      # sizing warning, $50 account
    assert "PAPER SIGNAL" in text                 # it notifies, it does not trade
    assert "random-entry benchmark" in text       # evidence caveat


def test_blocked_setups_do_not_alert(alert_cfg):
    """A level the filters would reject is not an actionable alert."""
    from src.strategies.base import EntryLevel
    from src.strategies.trend import DonchianTrendStrategy

    bars = _rising_bars()
    client = RecordingClient()

    original = DonchianTrendStrategy.entry_levels

    def blocked(self, bars_, features):
        return [
            EntryLevel(level.direction, level.price, level.kind,
                       blocked_by=("a filter says no",), note=level.note)
            for level in original(self, bars_, features)
        ]

    DonchianTrendStrategy.entry_levels = blocked
    try:
        runner = _runner(alert_cfg, bars, client, approach_atr_multiple=20.0,
                         reset_atr_multiple=60.0, notify_on_trigger=False)
        outcome = runner.check_once()
    finally:
        DonchianTrendStrategy.entry_levels = original
    assert [m for m in outcome.sent if m.kind == "alert"] == []


def test_stale_data_warns_once_rather_than_going_quiet(alert_cfg):
    bars = _rising_bars()
    client = RecordingClient()
    # Pretend it is a trading day, long after the last bar.
    late = bars.index[-1] + pd.Timedelta(hours=6)
    runner = _runner(alert_cfg, bars, client, now=late, max_bar_age_minutes=30,
                     heartbeat_hours=999)
    first = runner.check_once()
    assert any(m.kind == "stale" for m in first.sent)
    assert "Stale data" in first.sent[0].text

    second = runner.check_once()
    assert [m for m in second.sent if m.kind == "stale"] == [], "stale warning repeated"


def test_heartbeat_is_sent_when_due_and_reports_state(alert_cfg):
    bars = _rising_bars()
    client = RecordingClient()
    runner = _runner(alert_cfg, bars, client, send_startup_heartbeat=True,
                     heartbeat_hours=6, approach_atr_multiple=0.01,
                     reset_atr_multiple=0.02)
    outcome = runner.check_once()
    heartbeat = next(m.text for m in outcome.sent if m.kind == "heartbeat")
    assert "Heartbeat" in heartbeat
    assert "trend_donchian" in heartbeat
    assert "Last bar" in heartbeat

    again = runner.check_once()
    assert [m for m in again.sent if m.kind == "heartbeat"] == []


def test_startup_heartbeat_can_be_suppressed(alert_cfg):
    bars = _rising_bars()
    client = RecordingClient()
    runner = _runner(alert_cfg, bars, client, send_startup_heartbeat=False,
                     approach_atr_multiple=0.01, reset_atr_multiple=0.02)
    outcome = runner.check_once()
    assert [m for m in outcome.sent if m.kind == "heartbeat"] == []


def test_a_send_failure_is_recorded_and_does_not_crash_the_loop(alert_cfg):
    bars = _rising_bars()
    client = RecordingClient(fail_with=TelegramError("network down"))
    runner = _runner(alert_cfg, bars, client, approach_atr_multiple=20.0,
                     reset_atr_multiple=60.0, send_startup_heartbeat=True)
    outcome = runner.check_once()
    assert outcome.errors
    assert "network down" in outcome.errors[0]


def test_a_failed_alert_is_not_marked_as_sent(alert_cfg):
    """A dropped message must be retried, not silently deduplicated away."""
    bars = _rising_bars()
    failing = RecordingClient(fail_with=TelegramError("network down"))
    runner = _runner(alert_cfg, bars, failing, approach_atr_multiple=20.0,
                     reset_atr_multiple=60.0, notify_on_trigger=False,
                     send_startup_heartbeat=False)
    runner.check_once()
    assert all(setup.status == ARMED for setup in runner.store.setups.values())

    runner.client = RecordingClient()
    outcome = runner.check_once()
    assert any(m.kind == "alert" for m in outcome.sent)


def test_run_forever_stops_after_max_iterations(alert_cfg):
    bars = _rising_bars()
    client = RecordingClient()
    runner = _runner(alert_cfg, bars, client, approach_atr_multiple=0.01,
                     reset_atr_multiple=0.02)
    iterations = runner.run_forever(interval_seconds=1, max_iterations=3, sleeper=lambda s: None)
    assert iterations == 3
    assert runner.store.checks == 3


def test_runner_uses_only_closed_bars(alert_cfg):
    """The newest bar from a live feed can be partial; using it is a live
    lookahead bug."""
    bars = _rising_bars()
    client = RecordingClient()
    runner = _runner(alert_cfg, bars, client, approach_atr_multiple=0.01,
                     reset_atr_multiple=0.02, send_startup_heartbeat=True)
    outcome = runner.check_once()
    assert outcome.last_bar == bars.index[-2]


def test_unknown_strategy_in_config_is_rejected(alert_cfg):
    bars = _rising_bars()
    with pytest.raises(ValueError, match="unknown strategies"):
        _runner(alert_cfg.with_overrides({"alerts.strategies": ["moon_phase"]}), bars,
                strategies=("moon_phase",))


def test_hysteresis_settings_must_leave_a_gap(alert_cfg):
    settings = AlertSettings.from_config(alert_cfg)
    settings.approach_atr_multiple = 2.0
    settings.reset_atr_multiple = 1.0
    with pytest.raises(ValueError, match="hysteresis"):
        settings.validate()


def test_disabled_alerts_do_nothing(alert_cfg):
    bars = _rising_bars()
    client = RecordingClient()
    runner = _runner(alert_cfg, bars, client, enabled=False)
    outcome = runner.check_once()
    assert outcome.sent == []
    assert client.messages == []


# ---------------------------------------------------------------------- #
# Formatting helpers
# ---------------------------------------------------------------------- #
def test_heartbeat_makes_silence_diagnosable():
    text = format_heartbeat(
        symbol="XAU/USD", now=pd.Timestamp("2024-01-03 12:00", tz="UTC"),
        last_bar=pd.Timestamp("2024-01-03 11:45", tz="UTC"), bars_seen=1000,
        checks=12, strategy_states=["trend_donchian: flat"], armed_setups=1,
        market_open=True, evidence_note="not validated", is_synthetic=False,
    )
    assert "Heartbeat" in text and "Last bar" in text and "Checks run" in text
    assert "not validated" in text


def test_stale_warning_explains_that_silence_is_a_feed_problem():
    text = format_stale_warning(
        symbol="XAU/USD", last_bar=pd.Timestamp("2024-01-03 08:00", tz="UTC"),
        age_minutes=240, limit_minutes=90,
    )
    assert "Stale data" in text
    assert "feed problem" in text


def test_html_special_characters_are_escaped():
    from src.alerts.telegram import html_escape

    assert html_escape("a < b & c > d") == "a &lt; b &amp; c &gt; d"
