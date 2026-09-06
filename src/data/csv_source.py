"""CSV / Parquet loader for bars already on disk.

Handles the formats a vendor dump normally arrives in: a column of timestamps in
some timezone, OHLCV under any of a dozen spellings. The timezone of the source
must be stated explicitly in config. There is no autodetection, because a silent
timezone guess moves every session boundary and shows up only as a strategy that
mysteriously works.
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


class CsvBarAdapter(BarAdapter):
    """Reads one file or a directory of files matching a glob."""

    name = "csv"

    def __init__(
        self,
        path: str | Path,
        *,
        source_timezone: str = "UTC",
        native_resolution: str = "1min",
        glob: str = "*.csv",
        timestamp_column: str | None = None,
        timestamp_format: str | None = None,
    ):
        self.path = resolve_path(path)
        self.source_timezone = source_timezone
        self.native_resolution = native_resolution
        self.glob = glob
        self.timestamp_column = timestamp_column
        self.timestamp_format = timestamp_format

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
            frames.append(self._read_one(file))
        if not frames:
            return empty_bars()
        bars = pd.concat(frames).sort_index()
        bars = bars[(bars.index >= start) & (bars.index <= end)]
        return normalise_bars(bars)

    def _read_one(self, file: Path) -> pd.DataFrame:
        if file.suffix.lower() in (".parquet", ".pq"):
            raw = pd.read_parquet(file)
        else:
            raw = pd.read_csv(file)
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
            parsed = parsed.tz_localize(self.source_timezone).tz_convert("UTC")
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


class ParquetBarAdapter(CsvBarAdapter):
    name = "parquet"

    def __init__(self, path: str | Path, **kwargs):
        kwargs.setdefault("glob", "*.parquet")
        super().__init__(path, **kwargs)
