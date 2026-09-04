"""Telegram transport.

Two things this module is careful about:

* **The bot token never reaches a log, an exception message, or a report.** The
  token is in the URL, so a bare ``requests`` exception string contains it and
  would end up in a traceback on a shared terminal or in a CI log. Every string
  that leaves this module goes through ``redact``.
* **Rate limits are respected and persisted.** Telegram throttles roughly one
  message per second to a chat and around twenty per minute to a group. The
  limiter state lives in the alert store, so restarting the runner in a loop
  cannot be used to bypass its own limit.

Credentials come from a gitignored ``.env``. Nothing here writes them anywhere.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from html import escape

from src.config import load_dotenv

log = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
MAX_MESSAGE_CHARS = 4096


class TelegramError(RuntimeError):
    """Raised for credential and transport failures, always with the token redacted."""


@dataclass(frozen=True)
class TelegramCredentials:
    bot_token: str
    chat_id: str

    @classmethod
    def from_env(cls, *, required: bool = True) -> "TelegramCredentials | None":
        load_dotenv()
        token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
        chat_id = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
        if not token or not chat_id:
            if not required:
                return None
            missing = [
                name for name, value in
                (("TELEGRAM_BOT_TOKEN", token), ("TELEGRAM_CHAT_ID", chat_id))
                if not value
            ]
            raise TelegramError(
                f"missing credentials: {', '.join(missing)}. Create a bot with "
                "@BotFather, put the token and your chat id in a .env file at the "
                "repository root, and keep that file gitignored (it already is)."
            )
        return cls(bot_token=token, chat_id=chat_id)

    def redact(self, text: str) -> str:
        """Remove the token from any string before it is logged or raised."""
        if not text:
            return text
        cleaned = text.replace(self.bot_token, "<TELEGRAM_BOT_TOKEN>")
        head, _, _ = self.bot_token.partition(":")
        if head and len(head) > 4:
            cleaned = cleaned.replace(head, "<TELEGRAM_BOT_ID>")
        return cleaned

    @property
    def _url_base(self) -> str:
        return f"{API_BASE}/bot{self.bot_token}"


@dataclass
class RateLimiter:
    """Minimum spacing plus an hourly cap.

    ``recent`` holds epoch seconds of recent sends and is persisted by the caller,
    so the limit survives a restart.
    """

    min_seconds_between_messages: float = 3.0
    max_messages_per_hour: int = 20
    recent: list[float] = field(default_factory=list)

    def prune(self, now: float) -> None:
        cutoff = now - 3600.0
        self.recent = [t for t in self.recent if t >= cutoff]

    def seconds_until_allowed(self, now: float) -> float:
        self.prune(now)
        wait = 0.0
        if self.recent:
            wait = max(wait, self.min_seconds_between_messages - (now - max(self.recent)))
        if len(self.recent) >= self.max_messages_per_hour:
            oldest = min(self.recent)
            wait = max(wait, (oldest + 3600.0) - now)
        return max(0.0, wait)

    def record(self, now: float) -> None:
        self.recent.append(now)
        self.prune(now)


class TelegramClient:
    def __init__(
        self,
        credentials: TelegramCredentials,
        *,
        session=None,
        limiter: RateLimiter | None = None,
        dry_run: bool = False,
        timeout: float = 20.0,
        retries: int = 3,
        sleeper=time.sleep,
        clock=time.time,
    ):
        self.credentials = credentials
        self.limiter = limiter or RateLimiter()
        self.dry_run = dry_run
        self.timeout = timeout
        self.retries = retries
        self._session = session
        self._sleep = sleeper
        self._clock = clock

    def _require_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "xauusd-backtester-alerts/0.1"})
        return self._session

    # ------------------------------------------------------------------ #
    def check_credentials(self) -> str:
        """Confirm the token works and return the bot username.

        Called by ``run.py alerts check`` so a misconfigured bot is discovered
        deliberately rather than by noticing that no alerts ever arrive.
        """
        payload = self._call("getMe", {})
        result = payload.get("result", {})
        return str(result.get("username") or result.get("first_name") or "unknown")

    def send(self, text: str) -> bool:
        """Send one message. Returns False if it was dropped by the rate limit."""
        text = self._truncate(text)
        if self.dry_run:
            log.info("[alerts][dry-run] would send:\n%s", text)
            return True

        now = self._clock()
        wait = self.limiter.seconds_until_allowed(now)
        if wait > 0:
            if wait > 60:
                log.warning(
                    "alert suppressed by the hourly rate limit (next slot in %.0f s). "
                    "This is a guard against an alert storm, not a delivery failure.",
                    wait,
                )
                return False
            self._sleep(wait)
            now = self._clock()

        self._call("sendMessage", {
            "chat_id": self.credentials.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        })
        self.limiter.record(self._clock())
        return True

    # ------------------------------------------------------------------ #
    def _call(self, method: str, payload: dict) -> dict:
        session = self._require_session()
        url = f"{self.credentials._url_base}/{method}"
        last_error = "no attempt was made"
        for attempt in range(self.retries):
            try:
                response = session.post(url, json=payload, timeout=self.timeout)
            except Exception as exc:
                last_error = self.credentials.redact(f"{type(exc).__name__}: {exc}")
                self._sleep(min(2**attempt, 8))
                continue

            if response.status_code == 429:
                # Telegram tells us exactly how long to wait.
                retry_after = 5
                try:
                    retry_after = int(response.json().get("parameters", {}).get("retry_after", 5))
                except Exception:
                    pass
                log.warning("telegram asked us to back off for %ss", retry_after)
                self._sleep(retry_after)
                last_error = "HTTP 429 (rate limited)"
                continue

            try:
                body = response.json()
            except Exception:
                body = {}

            if response.status_code == 200 and body.get("ok"):
                return body

            description = str(body.get("description") or response.text or "")[:300]
            last_error = self.credentials.redact(
                f"HTTP {response.status_code}: {description}"
            )
            if response.status_code in (400, 401, 403, 404):
                # Bad token, wrong chat id, or the user never messaged the bot.
                # Retrying will not help and would just burn the rate limit.
                raise TelegramError(f"{method} rejected: {last_error}{_hint(response.status_code)}")
            self._sleep(min(2**attempt, 8))

        raise TelegramError(f"{method} failed after {self.retries} attempts: {last_error}")

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= MAX_MESSAGE_CHARS:
            return text
        keep = MAX_MESSAGE_CHARS - 40
        return text[:keep] + "\n... (message truncated)"


def _hint(status: int) -> str:
    if status == 401:
        return " -- the bot token looks wrong; check TELEGRAM_BOT_TOKEN in .env."
    if status in (400, 403):
        return (
            " -- check TELEGRAM_CHAT_ID, and note that a bot cannot message a "
            "person until that person has sent it a message first."
        )
    return ""


def html_escape(text: str) -> str:
    """Escape user-facing text for Telegram HTML parse mode."""
    return escape(str(text), quote=False)
