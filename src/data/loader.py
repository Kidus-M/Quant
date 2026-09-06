"""Assembles the data layer: adapter -> cache -> quality -> resample.

This is the only module strategy code (indirectly) depends on for bars, and it is
the only place that knows which adapter is configured.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.config import Config
from src.data.base import BarAdapter, DataProvenance, empty_bars
from src.data.cache import CacheKey, ParquetBarCache
from src.data.quality import QualityReport, run_quality_checks
from src.data.resample import assert_no_invented_bars, bars_per_year, resample_bars
from src.data.sessions import SessionCalendar

log = logging.getLogger(__name__)


@dataclass
class BarDataset:
    """Bars at the run resolution, plus everything needed to interpret them."""

    bars: pd.DataFrame
    resolution: str
    base_resolution: str
    provenance: DataProvenance
    calendar: SessionCalendar
    quality: QualityReport
    macro: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.bars.index

    @property
    def bars_per_year(self) -> float:
        return bars_per_year(self.bars.index)

    def slice(self, start, end) -> "BarDataset":
        mask = (self.bars.index >= start) & (self.bars.index <= end)
        macro = self.macro.loc[mask] if len(self.macro) else self.macro
        return BarDataset(
            bars=self.bars.loc[mask],
            resolution=self.resolution,
            base_resolution=self.base_resolution,
            provenance=self.provenance,
            calendar=self.calendar,
            quality=self.quality,
            macro=macro,
        )

    def describe(self) -> list[str]:
        lines = [
            f"source: {self.provenance.describe()}",
            f"resolution: {self.resolution} (resampled from {self.base_resolution})",
        ]
        if len(self.bars):
            lines.append(
                f"span: {self.bars.index[0]} to {self.bars.index[-1]} "
                f"({len(self.bars):,} bars, {self.bars_per_year:,.0f}/year)"
            )
        lines.extend(self.quality.summary_lines())
        return lines


def build_adapter(cfg: Config) -> BarAdapter:
    name = str(cfg.get("data.adapter", "synthetic")).lower()
    if name == "synthetic":
        from src.data.synthetic import SyntheticAdapter

        return SyntheticAdapter(
            seed=int(cfg.get("data.synthetic.seed", 7)),
            start_price=float(cfg.get("data.synthetic.start_price", 1800.0)),
            annual_vol=float(cfg.get("data.synthetic.annual_vol", 0.15)),
            annual_drift=float(cfg.get("data.synthetic.annual_drift", 0.0)),
            calendar=SessionCalendar.from_config(cfg),
        )
    if name in ("csv", "parquet"):
        from src.data.csv_source import CsvBarAdapter, ParquetBarAdapter

        cls = CsvBarAdapter if name == "csv" else ParquetBarAdapter
        return cls(
            path=cfg.get("data.csv.path"),
            source_timezone=cfg.get("data.csv.source_timezone", "UTC"),
            native_resolution=cfg.get("data.base_resolution", "1min"),
            timestamp_column=cfg.get("data.csv.timestamp_column", None),
            timestamp_format=cfg.get("data.csv.timestamp_format", None),
        )
    if name == "dukascopy":
        from src.data.dukascopy import DukascopyAdapter

        return DukascopyAdapter(max_workers=int(cfg.get("data.dukascopy.max_workers", 8)))
    if name == "oanda":
        from src.data.oanda import OandaAdapter, OandaCredentials

        return OandaAdapter(
            credentials=OandaCredentials.from_env(
                environment=cfg.get("data.oanda.environment", "practice")
            ),
            native_resolution=cfg.get("data.base_resolution", "1min"),
            price=cfg.get("data.oanda.price", "BA"),
        )
    raise ValueError(f"unknown data adapter {name!r}")


def load_base_bars(
    cfg: Config,
    *,
    adapter: BarAdapter | None = None,
    refresh: bool = False,
    freshness: pd.Timedelta | None = None,
) -> tuple[pd.DataFrame, DataProvenance]:
    """Base-resolution bars, served from the parquet cache wherever possible.

    ``freshness`` bounds how stale the cached edge of the current month may be
    before it is refetched. Leave it unset for research runs. A live caller
    polling for new bars must set it to roughly one bar, or the cache will serve
    the same frame for up to a day; see ``ParquetBarCache.missing_months``.
    """
    adapter = adapter or build_adapter(cfg)
    symbol = cfg.get("instrument.symbol")
    base_resolution = cfg.get("data.base_resolution", "1min")
    start = pd.Timestamp(cfg.get("data.start"), tz="UTC")
    end = pd.Timestamp(cfg.get("data.end"), tz="UTC")
    if end <= start:
        raise ValueError(f"data.end ({end}) must be after data.start ({start})")

    cache = ParquetBarCache.from_config(cfg)
    key = CacheKey(adapter=adapter.cache_namespace, symbol=symbol, resolution=base_resolution)

    if refresh:
        cache.clear(key)

    missing = cache.missing_months(key, start, end, freshness=freshness)
    if missing:
        log.info(
            "cache miss for %s %s: fetching %d month(s) [%s .. %s]",
            symbol, base_resolution, len(missing), missing[0], missing[-1],
        )
        for period in missing:
            chunk_start = max(start, period.start_time.tz_localize("UTC"))
            chunk_end = min(end, period.end_time.tz_localize("UTC"))
            _, covered_end = cache.coverage(key, period)
            if covered_end is not None and chunk_start <= covered_end < chunk_end:
                # Partially cached month: ask only for the tail we are missing.
                # A live poller refetching a whole month every few minutes would
                # exhaust a rate-limited feed within the hour. The overlap of one
                # bar is deliberate -- it rewrites the edge bar, which may have
                # been stored while it was still forming.
                chunk_start = covered_end
            fetched = adapter.fetch(symbol, chunk_start, chunk_end)
            if len(fetched):
                cache.write(key, fetched)
    else:
        log.info("cache hit for %s %s over the full requested range", symbol, base_resolution)

    bars = cache.read(key, start, end)
    return bars, adapter.provenance(symbol)


def load_dataset(
    cfg: Config,
    *,
    adapter: BarAdapter | None = None,
    resolution: str | None = None,
    refresh: bool = False,
    with_macro: bool | None = None,
    freshness: pd.Timedelta | None = None,
) -> BarDataset:
    """Full load: fetch or read cache, quality check, then resample to the run
    resolution. The base bars are never persisted at a derived resolution.

    ``freshness`` is forwarded to ``load_base_bars`` and only matters to live
    callers polling for new bars."""
    calendar = SessionCalendar.from_config(cfg)
    base_resolution = cfg.get("data.base_resolution", "1min")
    resolution = resolution or cfg.get("data.resolution", base_resolution)

    base_bars, provenance = load_base_bars(
        cfg, adapter=adapter, refresh=refresh, freshness=freshness
    )
    if base_bars.empty:
        return BarDataset(
            bars=empty_bars(), resolution=resolution, base_resolution=base_resolution,
            provenance=provenance, calendar=calendar, quality=QualityReport(),
        )

    clean, report = run_quality_checks(base_bars, cfg, calendar=calendar, label=f"{provenance.symbol}_{base_resolution}")

    if resolution == base_resolution:
        bars = clean.copy()
        bars["n_source_bars"] = 1
    else:
        bars = resample_bars(clean, resolution, calendar)
        assert_no_invented_bars(clean, bars)

    macro = pd.DataFrame(index=bars.index)
    want_macro = bool(cfg.get("macro.enabled", False)) if with_macro is None else with_macro
    if want_macro:
        from src.data.fred import load_macro

        try:
            macro = load_macro(cfg, bars)
        except Exception as exc:
            # A macro fetch failure must not silently turn a macro-filtered strategy
            # into an unfiltered one; the strategy checks for the column and refuses.
            log.error("macro load failed (%s: %s); macro columns will be absent", type(exc).__name__, exc)
            macro = pd.DataFrame(index=bars.index)

    dataset = BarDataset(
        bars=bars,
        resolution=resolution,
        base_resolution=base_resolution,
        provenance=provenance,
        calendar=calendar,
        quality=report,
        macro=macro,
    )
    for line in dataset.describe():
        log.info("[data] %s", line)
    return dataset
