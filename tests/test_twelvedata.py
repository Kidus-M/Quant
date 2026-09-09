"""Twelve Data adapter.

Most of these tests exist because of two specific ways this API misleads a
caller, both found by probing the live endpoint rather than by reading the docs:

* **It answers in the account's local timezone unless told otherwise.** The same
  bar came back stamped 23:28 and 13:28 depending only on whether ``timezone``
  was sent. Nothing about that failure is loud: the bars parse, the backtest
  runs, and every session boundary is silently wrong. So the parameter is
  asserted on every outbound request.
* **It truncates an over-long window silently.** A one-month request returned the
  newest 5000 rows with HTTP 200 and no flag, quietly discarding 26 days from the
  middle of the range. The adapter chunks to avoid it and checks for it anyway.

The credential tests matter because the key is a *query parameter* here, not a
header, so it lands inside any URL that appears in an exception or a log line.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.base import DataProvenance
from src.data.twelvedata import (
    MAX_OUTPUTSIZE,
    TIME_SERIES_PATH,
    TwelveDataAdapter,
    TwelveDataCredentials,
    TwelveDataError,
    to_interval,
    to_symbol,
)


# ---------------------------------------------------------------------- #
# Fakes
# ---------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
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

    def get(self, url, params=None, timeout=None):
        self.requests.append({"url": url, "params": params or {}})
        if not self.responses:
            return FakeResponse(200, {"values": []})
        nxt = self.responses.pop(0)
        return nxt() if callable(nxt) else nxt


EPOCH = pd.Timestamp("2024-06-11 09:00", tz="UTC")


def bar(minute: int, *, open_=2000.0, volume=None, base=EPOCH) -> dict:
    """One row in the shape the API actually returns.

    Volume is omitted by default because the live API omits it for XAU/USD.
    """
    stamp = base + pd.Timedelta(minutes=minute)
    row = {
        "datetime": stamp.strftime("%Y-%m-%d %H:%M:%S"),
        "open": f"{open_:.5f}",
        "high": f"{open_ + 1:.5f}",
        "low": f"{open_ - 1:.5f}",
        "close": f"{open_ + 0.5:.5f}",
    }
    if volume is not None:
        row["volume"] = str(volume)
    return row


def make_adapter(responses, **kwargs):
    kwargs.setdefault("chunk_bars", 4000)
    return TwelveDataAdapter(
        credentials=TwelveDataCredentials(api_key="SECRET-KEY-123"),
        session=FakeSession(responses),
        sleeper=lambda _s: None,
        **kwargs,
    )


def ok(values) -> FakeResponse:
    return FakeResponse(200, {"values": values})


# ---------------------------------------------------------------------- #
# Symbol and interval mapping
# ---------------------------------------------------------------------- #
def test_symbol_names_use_a_slash():
    assert to_symbol("XAUUSD") == "XAU/USD"
    assert to_symbol("xauusd") == "XAU/USD"
    assert to_symbol("XAU/USD") == "XAU/USD"
    assert to_symbol("EURUSD") == "EUR/USD"


def test_unmappable_symbol_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="cannot map"):
        to_symbol("GOLD")


def test_interval_mapping():
    assert to_interval("1min") == "1min"
    assert to_interval("15min") == "15min"
    assert to_interval("1d") == "1day"
    with pytest.raises(ValueError, match="no Twelve Data interval"):
        to_interval("3min")


# ---------------------------------------------------------------------- #
# The timezone trap
# ---------------------------------------------------------------------- #
def test_every_request_pins_the_timezone_to_utc():
    """Without this the API answers in the account's local timezone.

    That shifts every session boundary in the project and fails silently, so it
    is asserted on every request rather than trusted to a default.
    """
    adapter = make_adapter([ok([bar(0), bar(1)])])
    adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=2))

    assert adapter._session.requests, "no request was made"
    for request in adapter._session.requests:
        assert request["params"]["timezone"] == "UTC"


def test_timezone_is_not_configurable():
    """There is no legitimate reason to request bars in a local timezone here."""
    adapter = make_adapter([ok([bar(0)])])
    with pytest.raises(TypeError):
        TwelveDataAdapter(
            credentials=TwelveDataCredentials(api_key="k"), timezone="Europe/London"
        )
    del adapter


def test_naive_timestamps_are_interpreted_as_utc():
    adapter = make_adapter([ok([bar(0)])])
    bars = adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=1))

    assert str(bars.index.tz) == "UTC"
    assert bars.index.name == "timestamp"
    assert bars.index[0] == EPOCH


# ---------------------------------------------------------------------- #
# Parsing
# ---------------------------------------------------------------------- #
def test_ohlc_is_parsed_as_float():
    adapter = make_adapter([ok([bar(0, open_=2000.0)])])
    bars = adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=1))

    row = bars.iloc[0]
    assert row["open"] == pytest.approx(2000.0)
    assert row["high"] == pytest.approx(2001.0)
    assert row["low"] == pytest.approx(1999.0)
    assert row["close"] == pytest.approx(2000.5)
    assert bars["close"].dtype == "float64"


def test_absent_volume_becomes_zero_rather_than_being_invented():
    """The API publishes no volume for metals. 0.0 is honest; a guess is not."""
    adapter = make_adapter([ok([bar(0), bar(1)])])
    bars = adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=2))

    assert (bars["volume"] == 0.0).all()


def test_volume_is_used_when_the_source_supplies_it():
    adapter = make_adapter([ok([bar(0, volume=17)])])
    bars = adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=1))

    assert bars["volume"].iloc[0] == pytest.approx(17.0)


def test_descending_rows_are_sorted_ascending():
    """The live API returns newest-first by default."""
    adapter = make_adapter([ok([bar(2), bar(1), bar(0)])])
    bars = adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=3))

    assert bars.index.is_monotonic_increasing
    assert bars.index[0] == EPOCH


def test_no_spread_column_is_produced():
    """This source has no bid/ask, so the cost model must not think it measured one."""
    adapter = make_adapter([ok([bar(0)])])
    bars = adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=1))

    assert "spread" not in bars.columns


def test_implausible_prices_are_refused():
    adapter = make_adapter([ok([bar(0, open_=3.5)])])
    with pytest.raises(ValueError, match="plausible range"):
        adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=1))


def test_empty_response_yields_empty_bars():
    adapter = make_adapter([ok([])])
    bars = adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=5))

    assert bars.empty
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]


def test_end_before_start_fetches_nothing():
    adapter = make_adapter([])
    bars = adapter.fetch("XAUUSD", EPOCH, EPOCH - pd.Timedelta(minutes=5))

    assert bars.empty
    assert adapter._session.requests == []


# ---------------------------------------------------------------------- #
# Silent truncation
# ---------------------------------------------------------------------- #
def test_truncated_response_raises_rather_than_leaving_a_hole():
    """A capped response that does not reach the window start dropped bars.

    The live API does this with HTTP 200 and no error field. A quiet hole in the
    middle of a backtest is worse than a loud failure.
    """
    start = EPOCH
    # Rows arrive starting an hour late: the earlier part of the window is gone.
    values = [bar(60 + i) for i in range(MAX_OUTPUTSIZE)]
    adapter = make_adapter([ok(values)], chunk_bars=MAX_OUTPUTSIZE)

    with pytest.raises(TwelveDataError, match="dropped silently"):
        adapter.fetch("XAUUSD", start, start + pd.Timedelta(minutes=MAX_OUTPUTSIZE + 120))


def test_a_full_page_that_covers_the_window_is_not_treated_as_truncated():
    start = EPOCH
    values = [bar(i) for i in range(MAX_OUTPUTSIZE)]
    adapter = make_adapter([ok(values)], chunk_bars=MAX_OUTPUTSIZE)

    bars = adapter.fetch("XAUUSD", start, start + pd.Timedelta(minutes=MAX_OUTPUTSIZE))
    assert len(bars) == MAX_OUTPUTSIZE


def test_chunk_bars_must_stay_within_the_response_cap():
    with pytest.raises(ValueError, match="chunk_bars"):
        TwelveDataAdapter(
            credentials=TwelveDataCredentials(api_key="k"), chunk_bars=MAX_OUTPUTSIZE + 1
        )


# ---------------------------------------------------------------------- #
# Chunking
# ---------------------------------------------------------------------- #
def test_a_long_window_is_split_into_several_requests():
    adapter = make_adapter(
        [ok([bar(0)]), ok([bar(10)]), ok([bar(20)])], chunk_bars=10
    )
    adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=25))

    assert len(adapter._session.requests) >= 3


def test_chunks_do_not_re_request_the_same_window():
    adapter = make_adapter(
        [ok([bar(0)]), ok([bar(10)]), ok([bar(20)])], chunk_bars=10
    )
    adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=25))

    starts = [r["params"]["start_date"] for r in adapter._session.requests]
    assert len(starts) == len(set(starts)), "a chunk window was requested twice"


def test_an_empty_chunk_does_not_stall_the_walk():
    """A closed market mid-range must not cause an infinite re-request loop."""
    adapter = make_adapter(
        [ok([bar(0)]), ok([]), ok([bar(20)])], chunk_bars=10
    )
    bars = adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=25))

    assert len(adapter._session.requests) == 3
    assert len(bars) == 2


def test_bars_outside_the_requested_window_are_dropped():
    adapter = make_adapter([ok([bar(-5), bar(0), bar(1), bar(99)])], chunk_bars=4000)
    bars = adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=2))

    assert bars.index.min() >= EPOCH
    assert bars.index.max() <= EPOCH + pd.Timedelta(minutes=2)


def test_duplicate_timestamps_across_chunks_collapse():
    adapter = make_adapter([ok([bar(0), bar(1)]), ok([bar(1), bar(2)])], chunk_bars=1)
    bars = adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=3))

    assert bars.index.is_unique


# ---------------------------------------------------------------------- #
# Coverage gaps are not failures
# ---------------------------------------------------------------------- #
def test_a_window_outside_coverage_yields_empty_bars_rather_than_raising():
    """Free-tier history starts around 2020. Running off the edge must not crash.

    A backfill that reaches past the plan's coverage should shorten, so that a
    long backtest range degrades into less history rather than no run at all.
    """
    payload = {
        "code": 400,
        "status": "error",
        "message": "No data is available on the specified dates. Try setting different start/end dates.",
    }
    adapter = make_adapter([FakeResponse(400, payload)])
    bars = adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=5))

    assert bars.empty


def test_a_real_client_error_still_raises():
    payload = {"code": 400, "status": "error", "message": "**symbol** not found"}
    adapter = make_adapter([FakeResponse(400, payload)])

    with pytest.raises(TwelveDataError, match="symbol"):
        adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=5))


def test_error_in_a_200_body_is_treated_as_a_failure():
    """This API reports some failures with HTTP 200 and status=error."""
    payload = {"code": 401, "status": "error", "message": "bad key"}
    adapter = make_adapter([FakeResponse(200, payload)])

    with pytest.raises(TwelveDataError):
        adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=5))


# ---------------------------------------------------------------------- #
# Credentials and redaction
# ---------------------------------------------------------------------- #
def test_missing_key_names_the_variable(monkeypatch):
    monkeypatch.setattr("src.data.twelvedata.load_dotenv", lambda *a, **k: {})
    monkeypatch.delenv("TWELVEDATA_API_KEY", raising=False)

    with pytest.raises(TwelveDataError, match="TWELVEDATA_API_KEY"):
        TwelveDataCredentials.from_env()


def test_missing_key_mentions_deploying_the_variable_separately(monkeypatch):
    """A .env file is not shipped to the server; that has already bitten once."""
    monkeypatch.setattr("src.data.twelvedata.load_dotenv", lambda *a, **k: {})
    monkeypatch.delenv("TWELVEDATA_API_KEY", raising=False)

    with pytest.raises(TwelveDataError, match="not shipped to the server"):
        TwelveDataCredentials.from_env()


def test_redact_removes_the_key():
    credentials = TwelveDataCredentials(api_key="SECRET-KEY-123")
    text = "https://api.twelvedata.com/time_series?apikey=SECRET-KEY-123&symbol=XAU/USD"

    cleaned = credentials.redact(text)
    assert "SECRET-KEY-123" not in cleaned
    assert "<TWELVEDATA_API_KEY>" in cleaned


def test_transport_failure_does_not_leak_the_key():
    """The key is a query parameter, so it rides inside the URL in exceptions."""

    class Boom:
        def get(self, url, params=None, timeout=None):
            raise OSError(
                "failed connecting to "
                "https://api.twelvedata.com/time_series?apikey=SECRET-KEY-123"
            )

    adapter = TwelveDataAdapter(
        credentials=TwelveDataCredentials(api_key="SECRET-KEY-123"),
        session=Boom(),
        sleeper=lambda _s: None,
        retries=2,
    )

    with pytest.raises(TwelveDataError) as excinfo:
        adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=5))
    assert "SECRET-KEY-123" not in str(excinfo.value)


def test_auth_failure_does_not_leak_the_key():
    payload = {"code": 401, "status": "error", "message": "key SECRET-KEY-123 is invalid"}
    adapter = make_adapter([FakeResponse(401, payload)])

    with pytest.raises(TwelveDataError) as excinfo:
        adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=5))
    assert "SECRET-KEY-123" not in str(excinfo.value)


def test_cache_namespace_never_contains_the_key():
    adapter = make_adapter([])
    assert "SECRET-KEY-123" not in adapter.cache_namespace
    assert adapter.cache_namespace == "twelvedata"


# ---------------------------------------------------------------------- #
# Transport behaviour
# ---------------------------------------------------------------------- #
def test_rate_limit_backs_off_and_retries():
    slept: list[float] = []
    adapter = TwelveDataAdapter(
        credentials=TwelveDataCredentials(api_key="k"),
        session=FakeSession([
            FakeResponse(429, {"code": 429, "status": "error", "message": "limit"},
                         headers={"Retry-After": "4"}),
            ok([bar(0)]),
        ]),
        sleeper=slept.append,
        retries=3,
    )

    bars = adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=1))
    assert len(bars) == 1
    assert 4.0 in slept


def test_client_errors_are_not_retried():
    session = FakeSession([
        FakeResponse(401, {"code": 401, "status": "error", "message": "nope"}),
        ok([bar(0)]),
    ])
    adapter = TwelveDataAdapter(
        credentials=TwelveDataCredentials(api_key="k"),
        session=session,
        sleeper=lambda _s: None,
        retries=3,
    )

    with pytest.raises(TwelveDataError):
        adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=1))
    assert len(session.requests) == 1, "a client error should not be retried"


def test_requests_are_spaced_to_respect_the_plan_limit():
    slept: list[float] = []
    clock = iter([0.0] * 40)
    adapter = TwelveDataAdapter(
        credentials=TwelveDataCredentials(api_key="k"),
        session=FakeSession([ok([bar(0)]), ok([bar(10)]), ok([bar(20)])]),
        sleeper=slept.append,
        clock=lambda: next(clock),
        requests_per_minute=8,
        chunk_bars=10,
    )
    adapter.fetch("XAUUSD", EPOCH, EPOCH + pd.Timedelta(minutes=25))

    # 8 per minute is one every 7.5 seconds.
    assert any(s == pytest.approx(7.5) for s in slept)


def test_adapter_refuses_any_path_but_the_time_series_read():
    adapter = make_adapter([])
    with pytest.raises(TwelveDataError, match="refusing to call"):
        adapter._get("/v1/orders", {})


def test_the_only_endpoint_constant_is_a_time_series_read():
    assert TIME_SERIES_PATH == "/time_series"


# ---------------------------------------------------------------------- #
# Provenance
# ---------------------------------------------------------------------- #
def test_provenance_is_not_marked_synthetic():
    adapter = make_adapter([])
    provenance = adapter.provenance("XAUUSD")

    assert isinstance(provenance, DataProvenance)
    assert provenance.is_synthetic is False
    assert "twelvedata" in provenance.describe()


def test_provenance_records_that_volume_is_absent():
    """Otherwise a quality report flagging every bar looks like a broken feed."""
    adapter = make_adapter([])
    notes = " ".join(adapter.provenance("XAUUSD").notes).lower()

    assert "volume" in notes
    assert "spread" in notes
