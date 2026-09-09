"""Shared fixtures.

Tests build bars directly from the synthetic adapter rather than through the
loader, so nothing here touches the parquet cache or the network. Anything that
needs the cache builds its own temporary directory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import CostModel
from src.backtest.sizing import PositionSizer
from src.config import Config, load_config
from src.data.base import normalise_bars
from src.data.loader import BarDataset
from src.data.quality import QualityReport
from src.data.resample import resample_bars
from src.data.sessions import SessionCalendar
from src.data.synthetic import SyntheticAdapter


@pytest.fixture(scope="session")
def base_config() -> Config:
    return load_config("config/backtest.yaml")


@pytest.fixture
def cfg(base_config, tmp_path) -> Config:
    """Config pointed at a throwaway cache, with macro off so tests never call out."""
    return base_config.with_overrides({
        # Pinned, not inherited. Tests must not change behaviour because the
        # production default in config/backtest.yaml moved, and a suite that
        # silently starts calling a live API is slow, flaky, and spends a rate
        # limit that the alert runner needs.
        "data.adapter": "synthetic",
        "data.start": "2023-01-01",
        "data.end": "2023-04-01",
        "data.cache_dir": str(tmp_path / "cache"),
        "data.quarantine_dir": str(tmp_path / "quarantine"),
        "macro.enabled": False,
        "reporting.output_dir": str(tmp_path / "reports"),
    })


@pytest.fixture(scope="session")
def calendar() -> SessionCalendar:
    return SessionCalendar()


@pytest.fixture(scope="session")
def minute_bars(calendar) -> pd.DataFrame:
    adapter = SyntheticAdapter(seed=99, calendar=calendar)
    return adapter.fetch(
        "XAUUSD",
        pd.Timestamp("2023-01-01", tz="UTC"),
        pd.Timestamp("2023-04-01", tz="UTC"),
    )


@pytest.fixture(scope="session")
def bars_15m(minute_bars, calendar) -> pd.DataFrame:
    return resample_bars(minute_bars, "15min", calendar)


@pytest.fixture
def dataset(bars_15m, calendar) -> BarDataset:
    from src.data.base import DataProvenance

    return BarDataset(
        bars=bars_15m,
        resolution="15min",
        base_resolution="1min",
        provenance=DataProvenance("synthetic", "XAUUSD", "1min", is_synthetic=True),
        calendar=calendar,
        quality=QualityReport(n_input=len(bars_15m), n_output=len(bars_15m)),
        macro=pd.DataFrame(index=bars_15m.index),
    )


@pytest.fixture
def cost_model(cfg) -> CostModel:
    return CostModel.from_config(cfg)


@pytest.fixture
def sizer(cfg) -> PositionSizer:
    return PositionSizer.from_config(cfg)


@pytest.fixture
def flat_cost_model() -> CostModel:
    """Costs with no session variation, for hand-computed arithmetic."""
    return CostModel(
        spread_usd_per_oz_round_trip=0.30,
        slippage_usd_per_oz_per_side=0.10,
        commission_usd_per_lot_per_side=0.0,
        swap_long_usd_per_oz_per_night=-0.15,
        swap_short_usd_per_oz_per_night=0.05,
        contract_size_oz_per_lot=100.0,
        spread_session_multipliers={},
    )


def make_bars(closes, *, start="2023-06-05 08:00", freq="15min", opens=None,
              highs=None, lows=None, volume=100.0) -> pd.DataFrame:
    """Hand-built bars for arithmetic tests.

    Defaults put every open at the prior close so a test can reason about fills
    without also reasoning about gaps; pass ``opens`` when the gap is the point.
    """
    closes = np.asarray(closes, dtype="float64")
    n = len(closes)
    index = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="timestamp")
    if opens is None:
        opens = np.concatenate(([closes[0]], closes[:-1]))
    opens = np.asarray(opens, dtype="float64")
    if highs is None:
        highs = np.maximum(opens, closes) + 0.5
    if lows is None:
        lows = np.minimum(opens, closes) - 0.5
    frame = pd.DataFrame(
        {"open": opens, "high": np.asarray(highs, dtype="float64"),
         "low": np.asarray(lows, dtype="float64"), "close": closes,
         "volume": np.full(n, volume, dtype="float64")},
        index=index,
    )
    return normalise_bars(frame)


def make_dataset(bars: pd.DataFrame, calendar: SessionCalendar | None = None) -> BarDataset:
    from src.data.base import DataProvenance

    calendar = calendar or SessionCalendar()
    return BarDataset(
        bars=bars,
        resolution="15min",
        base_resolution="1min",
        provenance=DataProvenance("test", "XAUUSD", "1min", is_synthetic=True),
        calendar=calendar,
        quality=QualityReport(n_input=len(bars), n_output=len(bars)),
        macro=pd.DataFrame(index=bars.index),
    )
