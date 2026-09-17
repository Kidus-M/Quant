"""CSV / Parquet loader for bars already on disk.

Handles the formats a vendor dump normally arrives in: a column of timestamps in
some timezone, OHLCV under any of a dozen spellings. The timezone of the source
must be stated explicitly in config. There is no autodetection, because a silent
timezone guess moves every session boundary and shows up only as a strategy that
mysteriously works.

Named presets exist for sources whose layout is fixed and publicly documented, so
that using one is a single config key rather than five that must agree. See
``FORMATS``; ``data.csv.format: histdata`` is the one that matters here, because
HistData.com is the only free source of XAUUSD 1-minute history going back before
2020 that needs no account and applies no residency rule. Its timezone is a fixed
UTC-5 with no daylight saving -- the vendor says so explicitly -- and the preset
encodes exactly that.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import resolve_path
from src.data.base import BarAdapter, empty_bars, normalise_bars

log = logging.getLogger(__name__)

_ALIASES = {
    "open": ["open", "o", "bidopen", "open_", "openbid"],
    "high": ["high", "h", "bidhigh", "highbid"],
    "low": ["low", "l", "bidlow", "lowbid"],
    "close": ["close", "c", "bidclose", "closebid", "last"],
    "volume": ["volume", "v", "vol", "tickvolume", "tick_volume", "volumebid"],
}
_TIME_ALIASES = ["timestamp", "time", "datetime", "date", "gmt time", "gmt_time", "local time"]


# Presets for vendor dumps with a fixed, documented layout. Each is exactly the
# set of constructor arguments that layout implies; nothing is inferred at read
# time. A preset never overrides a value the caller stated explicitly.
FORMATS: dict[str, dict] = {
    # HistData.com "ASCII M1" archives, e.g. DAT_ASCII_XAUUSD_M1_202401.csv.
    # Headerless, semicolon-delimited, one row per minute:
    #     20240102 000000;2062.61;2063.19;2062.25;2062.75;0
    #
    # Timestamps are, in the vendor's own words, "Eastern Standard Time (EST)
    # time-zone WITHOUT Day Light Savings adjustments": a fixed UTC-5 all year.
    # That is NOT America/New_York, which observes DST and would place every bar
    # from March to November one hour early. Etc/GMT+5 is the POSIX spelling of a
    # fixed UTC-5 (the sign is inverted by convention). This preset originally
    # used America/New_York; the vendor spec was checked and it was wrong.
    #
    # Prices are BID quotes, not mid. A long fills at the ask, so a bid tape
    # understates long entries and overstates long exits by half the spread --
    # the cost model's spread assumption is doing that work here, not the data.
    # Volume is always 0 for metals.
    "histdata": {
        "delimiter": ";",
        "has_header": False,
        "column_names": ["timestamp", "open", "high", "low", "close", "volume"],
        "timestamp_column": "timestamp",
        "timestamp_format": "%Y%m%d %H%M%S",
        "source_timezone": "Etc/GMT+5",
        "glob": "*.csv",
    },
}


class CsvBarAdapter(BarAdapter):
    """Reads one file or a directory of files matching a glob."""

    name = "csv"

    def __init__(
        self,
        path: str | Path,
        *,
        source_timezone: str | None = None,
        native_resolution: str = "1min",
        glob: str | None = None,
        timestamp_column: str | None = None,
        timestamp_format: str | None = None,
        delimiter: str | None = None,
        has_header: bool | None = None,
        column_names: list[str] | None = None,
        format: str | None = None,
    ):
        preset = self._preset(format)
        # An explicitly passed argument always wins over the preset, so a vendor
        # who changes one field does not force the preset to be abandoned whole.
        def pick(name, value, fallback):
            return value if value is not None else preset.get(name, fallback)

        self.path = resolve_path(path)
        self.format = format
        self.source_timezone = pick("source_timezone", source_timezone, "UTC")
        self.native_resolution = native_resolution
        self.glob = pick("glob", glob, "*.csv")
        self.timestamp_column = pick("timestamp_column", timestamp_column, None)
        self.timestamp_format = pick("timestamp_format", timestamp_format, None)
        self.delimiter = pick("delimiter", delimiter, ",")
        self.has_header = bool(pick("has_header", has_header, True))
        self.column_names = pick("column_names", column_names, None)
        if not self.has_header and not self.column_names:
            raise ValueError(
                "a headerless file needs data.csv.column_names, or a data.csv.format "
                f"preset that supplies them. Known presets: {sorted(FORMATS)}"
            )
        # Parsed files, keyed by path and mtime. The loader fills its cache one
        # month at a time, so a seven-year history means ~90 calls to ``fetch``;
        # re-parsing 170 MB of CSV on each one turned a one-minute load into
        # forty-five. A network adapter has to pay per call. A local file does not.
        self._parsed: dict[tuple[Path, float], pd.DataFrame] = {}

    @staticmethod
    def _preset(name: str | None) -> dict:
        if not name:
            return {}
        key = str(name).strip().lower()
        if key not in FORMATS:
            raise ValueError(
                f"unknown data.csv.format {name!r}. Known presets: {sorted(FORMATS)}"
            )
        return FORMATS[key]

    def _files(self) -> list[Path]:
        if self.path.is_file():
            return [self.path]
        if not self.path.exists():
            raise FileNotFoundError(
                f"data path {self.path} does not exist. Point data.csv.path at your "
                "downloaded bars, or set data.adapter to synthetic to run offline."
            )
        return sorted(self.path.glob(self.glob))

    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        frames = []
        for file in self._files():
            frames.append(self._read_cached(file))
        if not frames:
            return empty_bars()
        bars = pd.concat(frames).sort_index()
        bars = bars[(bars.index >= start) & (bars.index <= end)]
        return normalise_bars(bars)

    def _read_cached(self, file: Path) -> pd.DataFrame:
        key = (file, file.stat().st_mtime)
        if key not in self._parsed:
            # Drop any stale entry for the same path so an edited file is re-read
            # and the old parse does not linger in memory beside the new one.
            for stale in [k for k in self._parsed if k[0] == file]:
                del self._parsed[stale]
            self._parsed[key] = self._read_one(file)
        return self._parsed[key]

    def _read_one(self, file: Path) -> pd.DataFrame:
        if file.suffix.lower() in (".parquet", ".pq"):
            raw = pd.read_parquet(file)
        elif self.has_header:
            raw = pd.read_csv(file, sep=self.delimiter)
        else:
            raw = pd.read_csv(
                file, sep=self.delimiter, header=None, names=list(self.column_names)
            )
        raw.columns = [str(c).strip() for c in raw.columns]
        lowered = {c.lower(): c for c in raw.columns}

        if isinstance(raw.index, pd.DatetimeIndex):
            ts = raw.index.to_series()
        else:
            ts_col = self.timestamp_column or next(
                (lowered[a] for a in _TIME_ALIASES if a in lowered), None
            )
            if ts_col is None:
                raise ValueError(
                    f"{file.name}: no timestamp column found. Set data.csv.timestamp_column."
                )
            ts = raw[ts_col]

        parsed = pd.to_datetime(ts, format=self.timestamp_format, utc=False, errors="raise")
        parsed = pd.DatetimeIndex(parsed)
        if parsed.tz is None:
            parsed = self._localise(parsed, file).tz_convert("UTC")
        else:
            parsed = parsed.tz_convert("UTC")

        out = {}
        for canonical, aliases in _ALIASES.items():
            col = next((lowered[a] for a in aliases if a in lowered), None)
            if col is None:
                if canonical == "volume":
                    out[canonical] = 0.0
                    continue
                raise ValueError(f"{file.name}: could not find a {canonical!r} column")
            out[canonical] = pd.to_numeric(raw[col], errors="coerce").to_numpy()

        frame = pd.DataFrame(out, index=parsed)
        frame.index.name = "timestamp"
        log.info("loaded %d rows from %s", len(frame), file.name)
        return frame


    def _localise(self, parsed: pd.DatetimeIndex, file: Path) -> pd.DatetimeIndex:
        """Attach ``source_timezone``, refusing to guess at a DST seam.

        A source stamped in a zone that observes daylight saving has two hours a
        year that need a decision: one that occurs twice, and one that does not
        occur at all. Both are silent corruption if resolved by default -- an
        hour of bars stamped an hour away from where they belong, at a seam, once
        a year, is close to undiscoverable later.

        For XAUUSD specifically the question should never arise: US transitions
        happen at 02:00 Eastern on a Sunday, and gold does not reopen until 18:00
        Eastern, so there are no bars in either window. ``ambiguous="infer"``
        therefore resolves the ordinary case and anything it cannot resolve is
        raised with the file named, rather than silently placed.
        """
        try:
            return parsed.tz_localize(self.source_timezone, ambiguous="infer")
        except ValueError as exc:
            raise ValueError(
                f"{file.name}: timestamps could not be placed in "
                f"{self.source_timezone!r} across a daylight-saving transition "
                f"({exc}). The file may already be UTC (set "
                "data.csv.source_timezone: UTC), or it straddles a transition in a "
                "way that needs an explicit decision. It has NOT been loaded with "
                "a guess."
            ) from exc


class ParquetBarAdapter(CsvBarAdapter):
    name = "parquet"

    def __init__(self, path: str | Path, **kwargs):
        kwargs.setdefault("glob", "*.parquet")
        super().__init__(path, **kwargs)
