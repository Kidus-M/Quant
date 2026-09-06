"""OANDA candle adapter, and the execution boundary that keeps it read-only.

The project's first stated non-goal is: no live order placement, no broker
execution API, not in phase 1, not as a stub. Introducing a broker's API as a
data source puts that constraint under real pressure for the first time, so the
boundary is asserted here rather than left to good intentions:

* the adapter refuses at runtime to build any path but the candles read
* no module under ``src/`` contains an OANDA execution endpoint in executable code
* nothing anywhere reads an OANDA account id, without which no order URL can be
  addressed at all

The parsing tests matter for a different reason. ``price=BA`` is what makes the
cost model checkable against the tape, and a silent error in deriving mid from
bid/ask would corrupt every price in the study.
"""
from __future__ import annotations

import ast

import pandas as pd
import pytest

from src.data.oanda import (
    CANDLES_PATH,
    MAX_COUNT,
    OandaAdapter,
    OandaCredentials,
    OandaError,
    to_granularity,
    to_instrument,
)
from tests.test_lookahead import REPO_ROOT, SOURCE_FILES


# ---------------------------------------------------------------------- #
# Fakes
# ---------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Returns queued responses and records every request made."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.requests.append({"url": url, "params": params or {}, "headers": headers or {}})
        if not self.responses:
            return FakeResponse(200, {"candles": []})
        nxt = self.responses.pop(0)
        return nxt() if callable(nxt) else nxt


EPOCH = pd.Timestamp("2024-06-11 09:00", tz="UTC")


def candle(minute: int, *, bid_open=2000.0, spread=0.30, complete=True, base=EPOCH):
    """One BA candle, ``minute`` minutes after ``base``.

    Ask sits ``spread`` above bid at every OHLC point, so the expected mid and
    the expected spread are both known exactly.
    """
    b = bid_open
    bid = {"o": f"{b:.5f}", "h": f"{b + 1:.5f}", "l": f"{b - 1:.5f}", "c": f"{b + 0.5:.5f}"}
    ask = {k: f"{float(v) + spread:.5f}" for k, v in bid.items()}
    stamp = base + pd.Timedelta(minutes=minute)
    return {
        "complete": complete,
        "volume": 42,
        "time": stamp.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "bid": bid,
        "ask": ask,
    }


def make_adapter(responses, *, price="BA", environment="practice"):
    return OandaAdapter(
        credentials=OandaCredentials(api_token="tok-secret-123456789", environment=environment),
        session=FakeSession(responses),
        price=price,
        sleeper=lambda _s: None,
    )


WINDOW = (pd.Timestamp("2024-06-11 09:00", tz="UTC"), pd.Timestamp("2024-06-11 10:00", tz="UTC"))


# ---------------------------------------------------------------------- #
# Naming
# ---------------------------------------------------------------------- #
def test_instrument_names_use_oanda_underscores():
    assert to_instrument("XAUUSD") == "XAU_USD"
    assert to_instrument("xauusd") == "XAU_USD"
    assert to_instrument("XAU_USD") == "XAU_USD"
    assert to_instrument("AUDNZD") == "AUD_NZD"  # not in the table, inferred


def test_unmappable_symbol_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="INSTRUMENTS"):
        to_instrument("GOLD")


def test_granularity_mapping():
    assert to_granularity("1min") == "M1"
    assert to_granularity("15min") == "M15"
    with pytest.raises(ValueError, match="no OANDA granularity"):
        to_granularity("7min")


# ---------------------------------------------------------------------- #
# Parsing
# ---------------------------------------------------------------------- #
def test_mid_is_the_average_of_bid_and_ask():
    adapter = make_adapter([FakeResponse(200, {"candles": [candle(0, bid_open=2000.0, spread=0.40)]})])
    bars = adapter.fetch("XAUUSD", *WINDOW)

    assert len(bars) == 1
    row = bars.iloc[0]
    assert row["open"] == pytest.approx(2000.20)   # (2000.00 + 2000.40) / 2
    assert row["high"] == pytest.approx(2001.20)
    assert row["low"] == pytest.approx(1999.20)
    assert row["close"] == pytest.approx(2000.70)
    assert row["volume"] == pytest.approx(42.0)


def test_measured_spread_is_carried_and_quoted_at_the_open():
    adapter = make_adapter([FakeResponse(200, {"candles": [candle(0, spread=0.37)]})])
    bars = adapter.fetch("XAUUSD", *WINDOW)
    assert "spread" in bars.columns
    assert bars["spread"].iloc[0] == pytest.approx(0.37)


def test_index_is_utc_and_named():
    adapter = make_adapter([FakeResponse(200, {"candles": [candle(0), candle(1)]})])
    bars = adapter.fetch("XAUUSD", *WINDOW)
    assert str(bars.index.tz) == "UTC"
    assert bars.index.name == "timestamp"
    assert bars.index.is_monotonic_increasing


def test_incomplete_candles_are_dropped():
    """The forming candle is a half-built bar; using it is a live lookahead bug."""
    adapter = make_adapter([FakeResponse(200, {"candles": [
        candle(0, complete=True),
        candle(1, complete=True),
        candle(2, complete=False),
    ]})])
    bars = adapter.fetch("XAUUSD", *WINDOW)
    assert len(bars) == 2
    assert bars.index[-1] == pd.Timestamp("2024-06-11 09:01", tz="UTC")


def test_mid_only_mode_has_no_spread_column():
    payload = {"candles": [{
        "complete": True, "volume": 5, "time": "2024-06-11T09:00:00.000000000Z",
        "mid": {"o": "2000.0", "h": "2001.0", "l": "1999.0", "c": "2000.5"},
    }]}
    adapter = make_adapter([FakeResponse(200, payload)], price="M")
    bars = adapter.fetch("XAUUSD", *WINDOW)
    assert "spread" not in bars.columns
    assert bars["close"].iloc[0] == pytest.approx(2000.5)


def test_implausible_prices_are_refused():
    """A factor-of-ten scaling error must not reach the cache."""
    adapter = make_adapter([FakeResponse(200, {"candles": [candle(0, bid_open=2_000_000.0)]})])
    with pytest.raises(ValueError, match="plausible range"):
        adapter.fetch("XAUUSD", *WINDOW)


# ---------------------------------------------------------------------- #
# Pagination
# ---------------------------------------------------------------------- #
def test_pagination_advances_and_does_not_duplicate_the_seam():
    first = {"candles": [candle(m) for m in range(MAX_COUNT)]}
    # The next page picks up where the first left off.
    second = {"candles": [candle(MAX_COUNT + m) for m in range(3)]}
    adapter = make_adapter([FakeResponse(200, first), FakeResponse(200, second)])

    bars = adapter.fetch(
        "XAUUSD",
        pd.Timestamp("2024-06-11 09:00", tz="UTC"),
        pd.Timestamp("2024-06-15 23:00", tz="UTC"),
    )

    assert not bars.index.has_duplicates
    session = adapter._session
    assert len(session.requests) == 2
    assert session.requests[0]["params"]["includeFirst"] == "true"
    # The second page must not re-include the previous page's last bar.
    assert session.requests[1]["params"]["includeFirst"] == "false"
    assert session.requests[1]["params"]["from"] > session.requests[0]["params"]["from"]


def test_a_short_page_ends_pagination():
    adapter = make_adapter([FakeResponse(200, {"candles": [candle(0), candle(1)]})])
    adapter.fetch("XAUUSD", *WINDOW)
    assert len(adapter._session.requests) == 1


def test_empty_response_yields_empty_bars():
    adapter = make_adapter([FakeResponse(200, {"candles": []})])
    bars = adapter.fetch("XAUUSD", *WINDOW)
    assert bars.empty
    assert list(bars.columns)[:5] == ["open", "high", "low", "close", "volume"]


def test_requests_ask_for_the_configured_price_and_granularity():
    adapter = make_adapter([FakeResponse(200, {"candles": [candle(0)]})])
    adapter.fetch("XAUUSD", *WINDOW)
    params = adapter._session.requests[0]["params"]
    assert params["granularity"] == "M1"
    assert params["price"] == "BA"
    assert params["count"] == MAX_COUNT
    assert adapter._session.requests[0]["url"].endswith("/v3/instruments/XAU_USD/candles")


# ---------------------------------------------------------------------- #
# Credentials
# ---------------------------------------------------------------------- #
def test_missing_token_names_the_variable(monkeypatch):
    monkeypatch.setenv("OANDA_API_TOKEN", "")
    with pytest.raises(OandaError, match="OANDA_API_TOKEN"):
        OandaCredentials.from_env()


def test_unknown_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("OANDA_API_TOKEN", "tok")
    with pytest.raises(OandaError, match="practice"):
        OandaCredentials.from_env(environment="demo")


def test_redact_removes_the_token():
    creds = OandaCredentials(api_token="abcdef123456-xyz", environment="practice")
    message = "failed with token abcdef123456-xyz in the header"
    assert "abcdef123456" not in creds.redact(message)


def test_auth_failure_does_not_leak_the_token():
    adapter = make_adapter([FakeResponse(401, {"errorMessage": "Insufficient authorization"})])
    with pytest.raises(OandaError) as excinfo:
        adapter.fetch("XAUUSD", *WINDOW)
    assert "tok-secret-123456789" not in str(excinfo.value)
    assert "OANDA_API_TOKEN" in str(excinfo.value)


def test_client_errors_are_not_retried():
    """Retrying a 400 burns the rate limit and cannot succeed."""
    adapter = make_adapter([FakeResponse(400, {"errorMessage": "Invalid instrument"})])
    with pytest.raises(OandaError):
        adapter.fetch("XAUUSD", *WINDOW)
    assert len(adapter._session.requests) == 1


def test_server_errors_are_retried_then_reported():
    adapter = make_adapter([FakeResponse(503, None, text="upstream down")] * 3)
    with pytest.raises(OandaError, match="after 3 attempts"):
        adapter.fetch("XAUUSD", *WINDOW)
    assert len(adapter._session.requests) == 3


def test_rate_limit_backs_off_and_retries():
    adapter = make_adapter([
        FakeResponse(429, {"errorMessage": "slow down"}, headers={"Retry-After": "0"}),
        FakeResponse(200, {"candles": [candle(0)]}),
    ])
    bars = adapter.fetch("XAUUSD", *WINDOW)
    assert len(bars) == 1


# ---------------------------------------------------------------------- #
# Cache identity
# ---------------------------------------------------------------------- #
def test_cache_namespace_separates_environment_and_price_mode():
    """Bars differ between these settings, so the cache key must too."""
    practice_ba = make_adapter([], price="BA", environment="practice").cache_namespace
    practice_m = make_adapter([], price="M", environment="practice").cache_namespace
    live_ba = make_adapter([], price="BA", environment="live").cache_namespace
    assert len({practice_ba, practice_m, live_ba}) == 3


def test_cache_namespace_never_contains_the_token():
    assert "secret" not in make_adapter([]).cache_namespace


# ---------------------------------------------------------------------- #
# The execution boundary
# ---------------------------------------------------------------------- #
def test_adapter_refuses_any_path_but_candles():
    adapter = make_adapter([FakeResponse(200, {})])
    with pytest.raises(OandaError, match="permitted exactly one endpoint"):
        adapter._get("/v3/accounts/001-001-1234567-001/orders", {})
    assert adapter._session.requests == [], "the request must not be attempted at all"


def test_the_only_endpoint_constant_is_a_candles_read():
    assert CANDLES_PATH == "/v3/instruments/{instrument}/candles"


FORBIDDEN_FRAGMENTS = (
    "/orders",
    "/trades",
    "/positions",
    "/pendingOrders",
    "/openTrades",
    "/v3/accounts",
    "pricing/stream",
)


def _executable_strings(tree: ast.AST):
    """Every string constant that is not a docstring.

    Docstrings are excluded on purpose: this file and ``src/data/oanda.py`` both
    have to *name* the endpoints they refuse to call.
    """
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                yield node.value


def test_no_execution_endpoint_appears_in_executable_source():
    """The stated non-goal, enforced across the whole package.

    'No live order placement. No broker execution API. Not in phase 1, not as a
    stub.' A broker data feed is the moment that constraint becomes easy to
    erode, so the build fails rather than the boundary drifting.
    """
    offenders = []
    for path in SOURCE_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for text in _executable_strings(tree):
            for fragment in FORBIDDEN_FRAGMENTS:
                if fragment in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {fragment!r} in {text!r}")
    assert not offenders, "broker execution endpoints in source:\n" + "\n".join(offenders)


def test_nothing_reads_an_oanda_account_id():
    """Without an account id, no order URL is addressable even by mistake."""
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in SOURCE_FILES
        if "OANDA_ACCOUNT_ID" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"account id read in: {offenders}"
