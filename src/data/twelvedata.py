"""Twelve Data time-series adapter.

Why this source exists alongside ``oanda.py``: OANDA is a broker, and a broker
account cannot be opened from every country. Twelve Data is a data vendor, so a
free key is issued on signup with no residency check. That makes it the source
of last resort that actually works, rather than the best one on paper.

What it gives up relative to OANDA, stated plainly because the cost model cares:

* **No bid/ask, so no measured spread.** OANDA's ``price=BA`` turns the assumed
  0.30 USD/oz round-trip spread into a measurement. Here it stays an assumption,
  and ``costs.spread_usd_per_oz_round_trip`` is doing real work again.
* **No volume for metals.** The API returns OHLC only for ``XAU/USD``. Volume is
  reported as 0.0 rather than invented, which means every bar trips the quality
  layer's ``zero_volume`` flag. That flag is informational and kept by default
  (``quality.quarantine_zero_volume: false``); it is not a data fault, and the
  provenance notes say so, so the quality report cannot be misread later.
* **History starts around 2020**, not 2019. A request for a window the plan does
  not cover returns HTTP 400 with "No data is available on the specified dates",
  which is treated as an empty window rather than an error, so a backfill that
  runs off the edge of coverage shortens instead of crashing.

Two behaviours of this API are traps, and both are handled here rather than left
for a caller to discover:

* **The timezone is the account's, not UTC, unless you ask.** Without an explicit
  ``timezone=UTC`` the same bar comes back stamped ten hours off. Every session
  boundary in this project would shift and nothing would fail loudly. The
  parameter is therefore hardcoded, not configurable -- there is no legitimate
  reason to request bars in a local timezone here.
* **An over-long window truncates silently.** Asking for a month of 1-minute bars
  returns the newest 5000 with HTTP 200 and no indication that the other 26 days
  were dropped. Requests are chunked to stay under the cap, and every response is
  checked for truncation anyway; see ``_fetch_chunk``.

The key travels in the query string, so it can surface in a URL inside an
exception, a log line or a traceback. Every string leaving this module goes
through ``redact``.

Credentials come from a gitignored ``.env``: ``TWELVEDATA_API_KEY``.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, replace

import pandas as pd

from src.config import load_dotenv
from src.data.base import (
    BarAdapter,
    empty_bars,
    normalise_bars,
    validate_price_range,
)

log = logging.getLogger(__name__)

API_BASE = "https://api.twelvedata.com"

# The only path this adapter is permitted to build. Twelve Data is read-only by
# nature, but pinning the endpoint keeps the module's surface honest and small.
TIME_SERIES_PATH = "/time_series"

# Hard cap on rows in a single response. Asking for more is an error; asking for a
# window that contains more is silently truncated to the newest MAX_OUTPUTSIZE.
MAX_OUTPUTSIZE = 5000

# Bars requested per chunk. Deliberately below MAX_OUTPUTSIZE so that a market
# session denser than expected cannot push a chunk over the cap and trigger the
# silent truncation described in the module docstring.
DEFAULT_CHUNK_BARS = 4000

# Free ("basic") plan allows 8 requests per minute. Exceeding it earns a 429.
DEFAULT_REQUESTS_PER_MINUTE = 8

INTERVALS = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "45min": "45min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1day",
}

# Twelve Data separates base and quote with a slash. Getting this wrong yields a
# 400 that reads like an auth failure, so it is explicit rather than inferred.
SYMBOLS = {
    "XAUUSD": "XAU/USD",
    "XAGUSD": "XAG/USD",
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
}

# Substrings that mark a 400 as "this window is outside coverage" rather than a
# real failure. Matched case-insensitively against the API's message.
_NO_DATA_MARKERS = ("no data is available", "not found within the specified dates")

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


class TwelveDataError(RuntimeError):
    """Transport or credential failure, always with the API key redacted."""


def to_symbol(symbol: str) -> str:
    """``XAUUSD`` -> ``XAU/USD``."""
    cleaned = symbol.upper().replace("/", "").replace("_", "")
    if cleaned in SYMBOLS:
        return SYMBOLS[cleaned]
    if len(cleaned) == 6:
        return f"{cleaned[:3]}/{cleaned[3:]}"
    raise ValueError(
        f"cannot map {symbol!r} to a Twelve Data symbol. Add it to "
        "src/data/twelvedata.py:SYMBOLS rather than guessing at the call site."
    )


def to_interval(resolution: str) -> str:
    try:
        return INTERVALS[str(resolution).lower()]
    except KeyError:
        raise ValueError(
            f"no Twelve Data interval for resolution {resolution!r}; "
            f"known: {sorted(INTERVALS)}"
        ) from None


@dataclass(frozen=True)
class TwelveDataCredentials:
    api_key: str

    @classmethod
    def from_env(cls, *, required: bool = True) -> "TwelveDataCredentials | None":
        load_dotenv()
        key = (os.environ.get("TWELVEDATA_API_KEY") or "").strip()
        if not key:
            if not required:
                return None
            raise TwelveDataError(
                "missing credentials: TWELVEDATA_API_KEY. Sign up free at "
                "twelvedata.com, copy the key from the dashboard, and put it in a "
                ".env file at the repository root (already gitignored). Remember "
                "to set the same variable wherever the alert runner is deployed: "
                "a .env file is not shipped to the server."
            )
        return cls(api_key=key)

    def redact(self, text: str) -> str:
        """Strip the key from any string before it is logged or raised.

        The key is a query parameter, so it appears in full inside request URLs
        that land in exception text.
        """
        if not text or not self.api_key:
            return text
        return text.replace(self.api_key, "<TWELVEDATA_API_KEY>")


class TwelveDataAdapter(BarAdapter):
    name = "twelvedata"
    native_resolution = "1min"
    is_synthetic = False
    # Measured against the live API: it returns a full 60 rows an hour right
    # through Saturday and most of Sunday, when gold does not trade. Those rows
    # are forward-filled padding, not quotes. The session filter drops them,
    # which is correct, and this tells the quality layer to expect it.
    pads_outside_session = True

    def __init__(
        self,
        *,
        credentials: TwelveDataCredentials | None = None,
        native_resolution: str = "1min",
        chunk_bars: int = DEFAULT_CHUNK_BARS,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        timeout: float = 60.0,
        retries: int = 3,
        retry_backoff: float = 1.5,
        session=None,
        sleeper=time.sleep,
        clock=time.monotonic,
    ):
        self.credentials = credentials or TwelveDataCredentials.from_env()
        self.native_resolution = native_resolution
        if not 0 < chunk_bars <= MAX_OUTPUTSIZE:
            raise ValueError(f"chunk_bars must be in 1..{MAX_OUTPUTSIZE}, got {chunk_bars}")
        self.chunk_bars = chunk_bars
        self.requests_per_minute = max(1, int(requests_per_minute))
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff
        self._session = session
        self._sleep = sleeper
        self._clock = clock
        self._last_request_at: float | None = None

    def provenance(self, symbol: str):
        # Carried into every report. Without these, a quality summary showing
        # every bar flagged zero_volume and 38% dropped looks like a broken feed
        # rather than a source that does not publish volume for metals and pads
        # the weekend.
        return replace(
            super().provenance(symbol),
            notes=(
                "volume is not published for this instrument and is reported as 0.0",
                "no bid/ask available: the cost model's spread remains an assumption",
                "non-trading hours are padded by the source and dropped on load",
            ),
        )

    # ------------------------------------------------------------------ #
    def _require_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update(
                {"User-Agent": "xauusd-backtester/0.1 (research, read-only)"}
            )
        return self._session

    def _throttle(self) -> None:
        """Space requests to stay inside the plan's per-minute allowance."""
        min_interval = 60.0 / self.requests_per_minute
        if self._last_request_at is not None:
            elapsed = self._clock() - self._last_request_at
            if elapsed < min_interval:
                self._sleep(min_interval - elapsed)
        self._last_request_at = self._clock()

    def _get(self, path: str, params: dict) -> dict:
        if path != TIME_SERIES_PATH:
            raise TwelveDataError(
                f"refusing to call {path!r}. This adapter reads exactly one "
                f"endpoint, {TIME_SERIES_PATH}."
            )

        url = f"{API_BASE}{path}"
        query = dict(params)
        query["apikey"] = self.credentials.api_key
        session = self._require_session()
        last_error = "no attempt was made"

        for attempt in range(self.retries):
            self._throttle()
            try:
                response = session.get(url, params=query, timeout=self.timeout)
            except Exception as exc:
                # The URL carries the key, and requests puts the URL in the
                # exception text.
                last_error = self.credentials.redact(f"{type(exc).__name__}: {exc}")
                self._sleep(self.retry_backoff**attempt)
                continue

            if response.status_code == 429:
                wait = float(response.headers.get("Retry-After", 60.0 / self.requests_per_minute))
                log.warning("twelve data rate limited; backing off %.1fs", wait)
                self._sleep(wait)
                last_error = "HTTP 429 (rate limited)"
                continue

            try:
                body = response.json()
            except Exception as exc:
                last_error = self.credentials.redact(f"unparseable response body: {exc}")
                self._sleep(self.retry_backoff**attempt)
                continue

            # An error may arrive as a non-200, or as HTTP 200 with status=error
            # in the body. Both are failures and both are handled here.
            if response.status_code == 200 and body.get("status") != "error":
                return body

            message = str(body.get("message") or "")[:300]
            code = body.get("code", response.status_code)
            last_error = self.credentials.redact(f"HTTP {code}: {message}")

            if _is_no_data(message):
                # Outside the plan's coverage. A legitimately empty window, not a
                # failure: let the caller shorten the backfill rather than abort.
                log.info("twelve data reports no coverage for the requested window")
                return {"values": []}

            if int(code) in (401, 403):
                raise TwelveDataError(f"{last_error}{_hint(int(code))}")
            if int(code) in (400, 404):
                raise TwelveDataError(f"{last_error}{_hint(int(code))}")
            self._sleep(self.retry_backoff**attempt)

        raise TwelveDataError(
            f"time_series request failed after {self.retries} attempts: {last_error}"
        )

    # ------------------------------------------------------------------ #
    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        td_symbol = to_symbol(symbol)
        interval = to_interval(self.native_resolution)
        start = pd.Timestamp(start).tz_convert("UTC")
        end = pd.Timestamp(end).tz_convert("UTC")
        if end <= start:
            return empty_bars()

        step = pd.Timedelta(self.native_resolution)
        span = step * self.chunk_bars

        frames: list[pd.DataFrame] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + span, end)
            frame = self._fetch_chunk(td_symbol, interval, cursor, chunk_end)
            if len(frame):
                frames.append(frame)
            # Advance past the window just requested regardless of what came back.
            # An empty chunk is a closed market or a gap in coverage, not a reason
            # to stall or to re-request the same window forever.
            cursor = chunk_end + step

        if not frames:
            return empty_bars()

        bars = pd.concat(frames)
        bars = bars[~bars.index.duplicated(keep="last")].sort_index()
        bars = bars[(bars.index >= start) & (bars.index <= end)]
        if bars.empty:
            return empty_bars()

        validate_price_range(bars["close"], symbol, source="twelvedata")
        return normalise_bars(bars)

    def _fetch_chunk(
        self, td_symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        payload = self._get(TIME_SERIES_PATH, {
            "symbol": td_symbol,
            "interval": interval,
            # Never omit. See the module docstring: without it the API answers in
            # the account's local timezone and every session boundary shifts.
            "timezone": "UTC",
            "start_date": start.strftime(_TIMESTAMP_FORMAT),
            "end_date": end.strftime(_TIMESTAMP_FORMAT),
            "outputsize": MAX_OUTPUTSIZE,
            "order": "ASC",
        })
        values = payload.get("values") or []
        if not values:
            return pd.DataFrame()

        frame = _values_to_frame(values)

        # Truncation guard. A response pinned to the cap that does not reach back
        # to the start of the window dropped bars silently. Chunking should make
        # this unreachable, so it means the assumption is wrong somewhere, and a
        # quiet hole in the middle of a backtest is worse than a loud failure.
        if len(values) >= MAX_OUTPUTSIZE and len(frame) and frame.index[0] > start:
            raise TwelveDataError(
                f"response for {start}..{end} hit the {MAX_OUTPUTSIZE}-row cap and "
                f"starts at {frame.index[0]}, so earlier bars in the window were "
                "dropped silently. Lower data.twelvedata.chunk_bars."
            )
        return frame


def _values_to_frame(values: list[dict]) -> pd.DataFrame:
    """Rows into the canonical bar shape.

    ``order=ASC`` is requested, but the frame is sorted at the end of ``fetch``
    anyway rather than trusting the server to honour it.
    """
    index = pd.DatetimeIndex(
        pd.to_datetime([v["datetime"] for v in values], utc=True),
        name="timestamp",
    )
    return pd.DataFrame(
        {
            "open": [float(v["open"]) for v in values],
            "high": [float(v["high"]) for v in values],
            "low": [float(v["low"]) for v in values],
            "close": [float(v["close"]) for v in values],
            # Absent for metals. Reported as 0.0 rather than fabricated; see the
            # module docstring and the provenance notes.
            "volume": [float(v.get("volume") or 0.0) for v in values],
        },
        index=index,
    )


def _is_no_data(message: str) -> bool:
    lowered = (message or "").lower()
    return any(marker in lowered for marker in _NO_DATA_MARKERS)


def _hint(status: int) -> str:
    if status == 401:
        return " -- TWELVEDATA_API_KEY looks wrong or has been revoked."
    if status == 403:
        return (
            " -- the key is valid but this data is outside the plan. Metals and "
            "forex are on the free tier; some exchanges are not."
        )
    if status == 400:
        return " -- usually a bad symbol; see src/data/twelvedata.py:SYMBOLS."
    return ""
