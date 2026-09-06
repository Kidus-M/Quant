"""OANDA v20 candle adapter.

Why this source, after Dukascopy stopped serving the free datafeed (it now answers
``429 Too Many Requests`` on a first request, and ``503`` behind a browser
User-Agent, for every symbol and every date):

* A practice account is free and gives years of M1 history, which is what a
  walk-forward study needs.
* It can return **separate bid and ask candles** (``price=BA``). That turns the
  single most load-bearing number in the whole cost model -- the assumed
  0.30 USD/oz round-trip spread -- from an assertion into a measurement. The mid
  OHLC drives the strategies; the measured spread rides along in a ``spread``
  column for the cost model to check itself against.

**This module cannot place an order, and the design is what stops it, not the
intention.**

* It calls exactly one endpoint, ``/v3/instruments/{instrument}/candles``. That
  is enforced at runtime in ``_get``, not merely by convention.
* It never receives, reads or stores an OANDA *account id*. Every OANDA endpoint
  that can create, modify or close a position is addressed as
  ``/v3/accounts/{accountID}/...``. Without an account id, no such URL can be
  constructed here even by a caller trying to.

``tests/test_oanda.py`` asserts both properties against the whole repository, so
the boundary fails the build rather than eroding quietly.

Credentials come from a gitignored ``.env``: ``OANDA_API_TOKEN``, and optionally
``OANDA_ENVIRONMENT`` (``practice`` or ``live``, default ``practice``). Nothing
here writes them anywhere, and every string that leaves this module is redacted.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import pandas as pd

from src.config import load_dotenv
from src.data.base import (
    BarAdapter,
    empty_bars,
    normalise_bars,
    validate_price_range,
)

log = logging.getLogger(__name__)

HOSTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}

# The only path this module is permitted to build. See the module docstring.
CANDLES_PATH = "/v3/instruments/{instrument}/candles"

# OANDA caps a single candles response at 5000.
MAX_COUNT = 5000

GRANULARITY = {
    "1min": "M1",
    "5min": "M5",
    "15min": "M15",
    "30min": "M30",
    "1h": "H1",
    "4h": "H4",
    "1d": "D",
}

# OANDA writes instruments with an underscore. Getting this wrong yields a 400
# that reads like an auth problem, so it is explicit rather than inferred.
INSTRUMENTS = {
    "XAUUSD": "XAU_USD",
    "XAGUSD": "XAG_USD",
    "EURUSD": "EUR_USD",
    "GBPUSD": "GBP_USD",
    "USDJPY": "USD_JPY",
}


class OandaError(RuntimeError):
    """Transport or credential failure, always with the token redacted."""


def to_instrument(symbol: str) -> str:
    """``XAUUSD`` -> ``XAU_USD``."""
    symbol = symbol.upper().replace("/", "").replace("_", "")
    if symbol in INSTRUMENTS:
        return INSTRUMENTS[symbol]
    if len(symbol) == 6:
        return f"{symbol[:3]}_{symbol[3:]}"
    raise ValueError(
        f"cannot map {symbol!r} to an OANDA instrument name. Add it to "
        "src/data/oanda.py:INSTRUMENTS rather than guessing at the call site."
    )


def to_granularity(resolution: str) -> str:
    try:
        return GRANULARITY[str(resolution).lower()]
    except KeyError:
        raise ValueError(
            f"no OANDA granularity for resolution {resolution!r}; "
            f"known: {sorted(GRANULARITY)}"
        ) from None


@dataclass(frozen=True)
class OandaCredentials:
    api_token: str
    environment: str = "practice"

    @classmethod
    def from_env(cls, *, environment: str | None = None, required: bool = True):
        load_dotenv()
        token = (os.environ.get("OANDA_API_TOKEN") or "").strip()
        env = (environment or os.environ.get("OANDA_ENVIRONMENT") or "practice").strip().lower()
        if env not in HOSTS:
            raise OandaError(f"OANDA_ENVIRONMENT must be one of {sorted(HOSTS)}, got {env!r}")
        if not token:
            if not required:
                return None
            raise OandaError(
                "missing credentials: OANDA_API_TOKEN. Create a free practice "
                "account at oanda.com, generate a personal access token, and put "
                "it in a .env file at the repository root (already gitignored). "
                "No account id is needed: this adapter only reads candles."
            )
        return cls(api_token=token, environment=env)

    @property
    def host(self) -> str:
        return HOSTS[self.environment]

    def redact(self, text: str) -> str:
        """Strip the token from any string before it is logged or raised."""
        if not text or not self.api_token:
            return text
        cleaned = text.replace(self.api_token, "<OANDA_API_TOKEN>")
        head, _, _ = self.api_token.partition("-")
        if head and len(head) > 8:
            cleaned = cleaned.replace(head, "<OANDA_API_TOKEN>")
        return cleaned


class OandaAdapter(BarAdapter):
    name = "oanda"
    native_resolution = "1min"
    is_synthetic = False

    def __init__(
        self,
        *,
        credentials: OandaCredentials | None = None,
        native_resolution: str = "1min",
        price: str = "BA",
        timeout: float = 30.0,
        retries: int = 3,
        retry_backoff: float = 1.5,
        session=None,
        sleeper=time.sleep,
    ):
        self.credentials = credentials or OandaCredentials.from_env()
        self.native_resolution = native_resolution
        if price not in ("BA", "M"):
            raise ValueError("price must be 'BA' (bid/ask, gives measured spread) or 'M' (mid)")
        self.price = price
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff
        self._session = session
        self._sleep = sleeper

    @property
    def cache_namespace(self) -> str:
        # Environment and price mode both change the bars that come back, so both
        # belong in the cache key. Omitting them would serve mid-only bars, with no
        # spread column, to a run that asked for bid/ask.
        return f"{self.name}-{self.credentials.environment}-{self.price}"

    # ------------------------------------------------------------------ #
    def _require_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.credentials.api_token}",
            "Accept-Datetime-Format": "RFC3339",
            "User-Agent": "xauusd-backtester/0.1 (research, read-only)",
        }

    def _get(self, path: str, params: dict) -> dict:
        # The execution boundary, enforced at runtime. Any path that is not an
        # instrument candles read is a programming error, and a loud one.
        if not (path.startswith("/v3/instruments/") and path.endswith("/candles")):
            raise OandaError(
                f"refusing to call {path!r}. This adapter is permitted exactly one "
                "endpoint, /v3/instruments/{instrument}/candles. Order placement is "
                "out of scope for this project by design."
            )

        url = f"{self.credentials.host}{path}"
        session = self._require_session()
        last_error = "no attempt was made"

        for attempt in range(self.retries):
            try:
                response = session.get(
                    url, params=params, headers=self._headers(), timeout=self.timeout
                )
            except Exception as exc:
                last_error = self.credentials.redact(f"{type(exc).__name__}: {exc}")
                self._sleep(self.retry_backoff**attempt)
                continue

            if response.status_code == 200:
                try:
                    return response.json()
                except Exception as exc:
                    raise OandaError(
                        self.credentials.redact(f"unparseable response body: {exc}")
                    ) from None

            body = ""
            try:
                body = str(response.json().get("errorMessage") or "")
            except Exception:
                body = (response.text or "")[:300]
            last_error = self.credentials.redact(f"HTTP {response.status_code}: {body}")

            if response.status_code in (400, 401, 403, 404):
                raise OandaError(f"{last_error}{_hint(response.status_code)}")
            if response.status_code == 429:
                wait = float(response.headers.get("Retry-After", 2))
                log.warning("oanda rate limited; backing off %.0fs", wait)
                self._sleep(wait)
                continue
            self._sleep(self.retry_backoff**attempt)

        raise OandaError(f"candles request failed after {self.retries} attempts: {last_error}")

    # ------------------------------------------------------------------ #
    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        instrument = to_instrument(symbol)
        granularity = to_granularity(self.native_resolution)
        start = pd.Timestamp(start).tz_convert("UTC")
        end = pd.Timestamp(end).tz_convert("UTC")
        if end <= start:
            return empty_bars()

        step = pd.Timedelta(self.native_resolution)
        path = CANDLES_PATH.format(instrument=instrument)
        cursor = start
        frames: list[pd.DataFrame] = []
        # One page is MAX_COUNT bars; the bound stops a malformed response from
        # spinning forever, and is generous enough for a multi-year backfill.
        max_pages = int((end - start) / (step * MAX_COUNT)) + 10

        for page in range(max_pages):
            payload = self._get(path, {
                "granularity": granularity,
                "price": self.price,
                "from": cursor.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                "count": MAX_COUNT,
                # Only the very first page may include the candle stamped exactly
                # at the cursor; afterwards it is the previous page's last bar.
                "includeFirst": "true" if page == 0 else "false",
            })
            candles = payload.get("candles") or []
            if not candles:
                break

            frame = _candles_to_frame(candles, price=self.price)
            last_time = pd.to_datetime(candles[-1]["time"], utc=True, format="ISO8601")
            if len(frame):
                frames.append(frame)

            if last_time <= cursor and page > 0:
                # No forward progress. Refuse to loop rather than hammer the API.
                log.warning("oanda pagination stalled at %s; stopping", cursor)
                break
            cursor = last_time + step
            if cursor >= end or len(candles) < MAX_COUNT:
                break
        else:
            log.warning(
                "oanda pagination hit its %d page bound before reaching %s", max_pages, end
            )

        if not frames:
            return empty_bars()

        bars = pd.concat(frames)
        bars = bars[~bars.index.duplicated(keep="last")].sort_index()
        bars = bars[(bars.index >= start) & (bars.index <= end)]
        if bars.empty:
            return empty_bars()

        validate_price_range(bars["close"], symbol, source="oanda")
        return normalise_bars(bars)


def _candles_to_frame(candles: list[dict], *, price: str) -> pd.DataFrame:
    """Mid OHLC plus the measured open spread, from bid/ask candles.

    Incomplete candles are dropped. The final candle of a live response is the one
    currently forming; treating it as a bar would mean computing a signal from a
    half-built bar, which is the live equivalent of reading tomorrow's close.
    """
    rows = [c for c in candles if c.get("complete", False)]
    if not rows:
        return pd.DataFrame()

    index = pd.DatetimeIndex(
        pd.to_datetime([c["time"] for c in rows], utc=True, format="ISO8601"),
        name="timestamp",
    )

    if price == "M":
        mid = [c["mid"] for c in rows]
        frame = pd.DataFrame(
            {
                "open": [float(m["o"]) for m in mid],
                "high": [float(m["h"]) for m in mid],
                "low": [float(m["l"]) for m in mid],
                "close": [float(m["c"]) for m in mid],
                "volume": [float(c.get("volume", 0)) for c in rows],
            },
            index=index,
        )
        return frame

    bid = [c["bid"] for c in rows]
    ask = [c["ask"] for c in rows]

    def mid_of(field: str) -> list[float]:
        return [(float(b[field]) + float(a[field])) / 2.0 for b, a in zip(bid, ask)]

    frame = pd.DataFrame(
        {
            "open": mid_of("o"),
            "high": mid_of("h"),
            "low": mid_of("l"),
            "close": mid_of("c"),
            "volume": [float(c.get("volume", 0)) for c in rows],
            # Quoted at the bar *open*, because the engine fills at the open of the
            # bar after the signal. A bar-average spread would be a different
            # number answering a different question.
            "spread": [float(a["o"]) - float(b["o"]) for b, a in zip(bid, ask)],
        },
        index=index,
    )
    return frame


def _hint(status: int) -> str:
    if status == 401:
        return " -- OANDA_API_TOKEN looks wrong or has been revoked."
    if status == 403:
        return (
            " -- the token is valid but not permitted here. Check that "
            "OANDA_ENVIRONMENT matches where the token was issued: a practice "
            "token does not work against the live host, or the reverse."
        )
    if status == 400:
        return " -- usually a bad instrument name; see src/data/oanda.py:INSTRUMENTS."
    return ""
