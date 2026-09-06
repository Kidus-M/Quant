"""Parquet bar cache.

Design decisions worth stating, because both prevent a class of quiet corruption:

* **Only the base resolution is ever written.** ``ParquetBarCache`` refuses to
  store anything other than the configured source resolution. A cached 15-minute
  file is indistinguishable from source data six months later, and once a derived
  file is in the cache, every downstream resample is building on an aggregate of
  an aggregate.

* **Month-partitioned with a manifest.** Coverage is tracked per calendar month,
  so a request for a range already on disk never hits the network. A month is
  only marked complete once its data extends to the month end, which stops a
  half-fetched current month from being treated as final.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import Config, resolve_path
from src.data.base import MEASURED, OHLCV, normalise_bars

log = logging.getLogger(__name__)

MANIFEST_NAME = "_manifest.json"


@dataclass(frozen=True)
class CacheKey:
    adapter: str
    symbol: str
    resolution: str

    def as_path(self, root: Path) -> Path:
        return root / self.adapter / self.symbol.upper() / self.resolution


class ParquetBarCache:
    def __init__(self, root: str | Path, base_resolution: str = "1min"):
        self.root = resolve_path(root)
        self.base_resolution = base_resolution

    @classmethod
    def from_config(cls, cfg: Config) -> "ParquetBarCache":
        return cls(
            root=cfg.get("data.cache_dir", "data/cache"),
            base_resolution=cfg.get("data.base_resolution", "1min"),
        )

    # ------------------------------------------------------------------ #
    def _dir(self, key: CacheKey) -> Path:
        return key.as_path(self.root)

    def _manifest_path(self, key: CacheKey) -> Path:
        return self._dir(key) / MANIFEST_NAME

    def _load_manifest(self, key: CacheKey) -> dict:
        path = self._manifest_path(key)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("cache manifest at %s is unreadable; treating cache as empty", path)
            return {}

    def _save_manifest(self, key: CacheKey, manifest: dict) -> None:
        path = self._manifest_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # ------------------------------------------------------------------ #
    def coverage(self, key: CacheKey, period: pd.Period) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """``(covered_start, covered_end)`` for one cached month, or ``(None, None)``.

        Callers use this to fetch only the gap at the end of a partially cached
        month instead of the whole month.
        """
        entry = self._load_manifest(key).get(str(period))
        if entry is None or not (self._dir(key) / f"{period}.parquet").exists():
            return None, None
        return pd.Timestamp(entry["covered_start"]), pd.Timestamp(entry["covered_end"])

    def missing_months(
        self,
        key: CacheKey,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        freshness: pd.Timedelta | None = None,
    ) -> list[pd.Period]:
        """Months in [start, end] whose cached coverage is incomplete.

        ``freshness`` is how far behind the requested end a partially cached month
        is allowed to fall before it counts as missing. The default of one day is
        right for a backtest, where the final partial day changes nothing and
        refetching a month to obtain it is pure waste.

        It is badly wrong for a live caller. A poller asking for bars up to *now*
        every five minutes will, under the one-day default, be told the current
        month is already covered and be served the same stale frame until the
        cached edge falls a full day behind. Live callers must pass a freshness on
        the order of one bar. See ``load_base_bars``.
        """
        tolerance = pd.Timedelta(days=1) if freshness is None else pd.Timedelta(freshness)
        manifest = self._load_manifest(key)
        wanted = pd.period_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="M")
        missing = []
        for period in wanted:
            entry = manifest.get(str(period))
            if entry is None or not (self._dir(key) / f"{period}.parquet").exists():
                missing.append(period)
                continue
            if entry.get("complete"):
                continue
            # Incomplete month: refetch only if the request reaches past what is stored.
            covered_end = pd.Timestamp(entry["covered_end"])
            month_end = min(end, period.end_time.tz_localize("UTC"))
            if covered_end < month_end - tolerance:
                missing.append(period)
        return missing

    def write(self, key: CacheKey, bars: pd.DataFrame) -> None:
        if key.resolution != self.base_resolution:
            raise ValueError(
                f"refusing to cache resolution {key.resolution!r}: only the base "
                f"resolution {self.base_resolution!r} may be stored as source data. "
                "Derived resolutions are recomputed on demand so that an aggregate "
                "can never be mistaken for a source bar."
            )
        if bars.empty:
            return
        bars = normalise_bars(bars)
        bars = bars[OHLCV + [c for c in MEASURED if c in bars.columns]]
        out_dir = self._dir(key)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest(key)
        now = pd.Timestamp.now(tz="UTC")

        for month, chunk in bars.groupby(bars.index.strftime("%Y-%m")):
            period = pd.Period(month, freq="M")
            path = out_dir / f"{period}.parquet"
            if path.exists():
                existing = pd.read_parquet(path)
                chunk = pd.concat([existing, chunk])
                chunk = chunk[~chunk.index.duplicated(keep="last")].sort_index()
            chunk.to_parquet(path)
            month_end = period.end_time.tz_localize("UTC")
            manifest[str(period)] = {
                "rows": int(len(chunk)),
                "covered_start": str(chunk.index.min()),
                "covered_end": str(chunk.index.max()),
                # A month is only final once its data runs to the month end and the
                # month itself is in the past.
                "complete": bool(chunk.index.max() >= month_end - pd.Timedelta(days=3) and month_end < now),
                "written_at": str(now),
            }
        self._save_manifest(key, manifest)

    def read(self, key: CacheKey, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        out_dir = self._dir(key)
        if not out_dir.exists():
            from src.data.base import empty_bars

            return empty_bars()
        periods = pd.period_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="M")
        frames = []
        for period in periods:
            path = out_dir / f"{period}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
        if not frames:
            from src.data.base import empty_bars

            return empty_bars()
        bars = pd.concat(frames).sort_index()
        bars = bars[~bars.index.duplicated(keep="last")]
        return normalise_bars(bars.loc[(bars.index >= start) & (bars.index <= end)])

    def clear(self, key: CacheKey) -> None:
        out_dir = self._dir(key)
        if out_dir.exists():
            for path in out_dir.glob("*"):
                path.unlink()


class SeriesCache:
    """Tiny parquet cache for daily macro series. Same directory conventions."""

    def __init__(self, root: str | Path):
        self.root = resolve_path(root)

    def path(self, series_id: str) -> Path:
        return self.root / f"{series_id}.parquet"

    def read(self, series_id: str) -> pd.Series | None:
        path = self.path(series_id)
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        return frame[series_id]

    def write(self, series_id: str, values: pd.Series) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        values.rename(series_id).to_frame().to_parquet(self.path(series_id))
