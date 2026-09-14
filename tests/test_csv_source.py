"""CSV adapter, with the emphasis on the timezone conversion.

A vendor dump stamped in a local timezone is the one data problem that does not
announce itself. Read HistData's US Eastern timestamps as UTC and every bar lands
four or five hours from where it belongs: the session filter then drops the London
open and keeps the middle of the night, the spread multipliers are applied to the
wrong hours, and the backtest still runs and still prints a number. So these tests
assert the conversion itself, not merely that a file parses.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.csv_source import FORMATS, CsvBarAdapter

# One January row (EST, UTC-5) and one July row (EDT, UTC-4). Any implementation
# that applies a fixed offset instead of a real timezone gets exactly one wrong.
HISTDATA_ROWS = [
    "20240102 000000;2062.61;2063.19;2062.25;2062.75;0",
    "20240102 000100;2062.75;2062.90;2062.40;2062.50;0",
    "20240701 120000;2330.10;2331.00;2329.80;2330.55;0",
]

FULL_RANGE = (pd.Timestamp("2000-01-01", tz="UTC"), pd.Timestamp("2030-01-01", tz="UTC"))


def write_histdata(directory, rows=HISTDATA_ROWS, name="DAT_ASCII_XAUUSD_M1_202401.csv"):
    path = directory / name
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def fetch(directory, **kwargs) -> pd.DataFrame:
    adapter = CsvBarAdapter(directory, **kwargs)
    return adapter.fetch("XAUUSD", *FULL_RANGE)


class TestHistDataPreset:
    def test_headerless_semicolon_file_parses(self, tmp_path):
        write_histdata(tmp_path)
        bars = fetch(tmp_path, format="histdata")

        assert len(bars) == 3
        assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
        assert bars["open"].iloc[0] == pytest.approx(2062.61)
        assert bars["close"].iloc[0] == pytest.approx(2062.75)

    def test_eastern_timestamps_are_converted_to_utc(self, tmp_path):
        """The whole reason the preset exists."""
        write_histdata(tmp_path)
        bars = fetch(tmp_path, format="histdata")

        # 00:00 EST (UTC-5) -> 05:00 UTC
        assert bars.index[0] == pd.Timestamp("2024-01-02 05:00:00", tz="UTC")
        # 12:00 EDT (UTC-4) -> 16:00 UTC, not 17:00. A fixed -5 offset fails here.
        assert bars.index[-1] == pd.Timestamp("2024-07-01 16:00:00", tz="UTC")

    def test_index_is_utc_and_named(self, tmp_path):
        write_histdata(tmp_path)
        bars = fetch(tmp_path, format="histdata")

        assert str(bars.index.tz) == "UTC"
        assert bars.index.name == "timestamp"
        assert bars.index.is_monotonic_increasing

    def test_reading_as_utc_would_have_given_a_different_answer(self, tmp_path):
        """Guards the test above from passing vacuously."""
        write_histdata(tmp_path)
        correct = fetch(tmp_path, format="histdata")
        naive = fetch(tmp_path, format="histdata", source_timezone="UTC")

        assert not correct.index.equals(naive.index)
        assert (correct.index[0] - naive.index[0]) == pd.Timedelta(hours=5)

    def test_explicit_argument_overrides_the_preset(self, tmp_path):
        write_histdata(tmp_path)
        bars = fetch(tmp_path, format="histdata", source_timezone="UTC")

        assert bars.index[0] == pd.Timestamp("2024-01-02 00:00:00", tz="UTC")

    def test_preset_does_not_mutate_when_overridden(self, tmp_path):
        """An override must not leak into the module-level preset dict."""
        write_histdata(tmp_path)
        fetch(tmp_path, format="histdata", source_timezone="UTC")

        assert FORMATS["histdata"]["source_timezone"] == "America/New_York"


class TestDaylightSavingSeams:
    """Gold is shut at 02:00 Eastern on a Sunday, so these rows should not exist.

    If a file ever does contain them, the load must fail loudly rather than place
    the bars an hour from where they belong.
    """

    def test_ambiguous_hour_is_refused_not_guessed(self, tmp_path):
        # 2024-11-03 01:30 Eastern occurs twice. One occurrence alone is unresolvable.
        write_histdata(tmp_path, rows=["20241103 013000;2740.10;2741.00;2739.80;2740.55;0"])

        with pytest.raises(ValueError, match="daylight-saving"):
            fetch(tmp_path, format="histdata")

    def test_nonexistent_hour_is_refused_not_shifted(self, tmp_path):
        # 2024-03-10 02:30 Eastern never happened.
        write_histdata(tmp_path, rows=["20240310 023000;2180.10;2181.00;2179.80;2180.55;0"])

        with pytest.raises(ValueError, match="daylight-saving"):
            fetch(tmp_path, format="histdata")

    def test_error_names_the_file_and_says_nothing_was_loaded(self, tmp_path):
        write_histdata(
            tmp_path,
            rows=["20241103 013000;2740.10;2741.00;2739.80;2740.55;0"],
            name="DAT_ASCII_XAUUSD_M1_202411.csv",
        )

        with pytest.raises(ValueError) as excinfo:
            fetch(tmp_path, format="histdata")

        message = str(excinfo.value)
        assert "DAT_ASCII_XAUUSD_M1_202411.csv" in message
        assert "NOT been loaded" in message

    def test_a_year_of_ordinary_eastern_timestamps_localises(self, tmp_path):
        """The seams must not make the common case fail."""
        stamps = pd.date_range("2024-01-01", "2024-12-31", freq="7h")
        # Drop the two transition windows, as a real gold feed does.
        keep = [t for t in stamps if not (
            (t.month == 3 and t.day == 10 and t.hour == 2)
            or (t.month == 11 and t.day == 3 and t.hour == 1)
        )]
        rows = [f"{t.strftime('%Y%m%d %H%M%S')};2000.0;2001.0;1999.0;2000.5;0" for t in keep]
        write_histdata(tmp_path, rows=rows)

        bars = fetch(tmp_path, format="histdata")

        assert len(bars) == len(keep)
        assert bars.index.is_monotonic_increasing


class TestGenericOptions:
    def test_headerless_file_without_column_names_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="column_names"):
            CsvBarAdapter(tmp_path, has_header=False)

    def test_unknown_preset_names_the_known_ones(self, tmp_path):
        with pytest.raises(ValueError, match="histdata"):
            CsvBarAdapter(tmp_path, format="nosuchvendor")

    def test_ordinary_comma_file_with_a_header_still_works(self, tmp_path):
        (tmp_path / "bars.csv").write_text(
            "timestamp,open,high,low,close,volume\n"
            "2024-01-02 05:00:00,2062.61,2063.19,2062.25,2062.75,10\n",
            encoding="utf-8",
        )

        bars = fetch(tmp_path)

        assert len(bars) == 1
        assert bars.index[0] == pd.Timestamp("2024-01-02 05:00:00", tz="UTC")
        assert bars["volume"].iloc[0] == pytest.approx(10.0)

    def test_custom_delimiter_and_column_names(self, tmp_path):
        (tmp_path / "bars.csv").write_text(
            "2024-01-02 05:00:00|1.0|2.0|0.5|1.5|7\n", encoding="utf-8"
        )

        bars = fetch(
            tmp_path,
            delimiter="|",
            has_header=False,
            column_names=["timestamp", "open", "high", "low", "close", "volume"],
        )

        assert len(bars) == 1
        assert bars["high"].iloc[0] == pytest.approx(2.0)
