"""Feature normalisation with an explicit fit/transform split.

A z-score fitted on the whole series leaks the future into every training bar:
the mean and standard deviation encode data the strategy could not have seen. The
split here is not stylistic. ``transform`` refuses to run before ``fit``, so the
fit-on-train-only rule cannot be forgotten in a walk-forward loop.

``RollingZScore`` is the alternative that needs no fitting at all, because its
window only ever looks backward. Prefer it when a rolling normalisation is
acceptable, since there is then nothing to leak.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class NotFittedError(RuntimeError):
    pass


@dataclass
class ZScoreNormaliser:
    """Fit on the in-sample window, apply everywhere."""

    columns: list[str] | None = None
    means: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    stds: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    fitted: bool = False
    fit_span: tuple[pd.Timestamp, pd.Timestamp] | None = None

    def fit(self, frame: pd.DataFrame) -> "ZScoreNormaliser":
        cols = self.columns or [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        self.columns = list(cols)
        subset = frame[self.columns]
        self.means = subset.mean()
        self.stds = subset.std().replace(0.0, np.nan)
        self.fitted = True
        if len(frame.index):
            self.fit_span = (frame.index[0], frame.index[-1])
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise NotFittedError(
                "ZScoreNormaliser.transform called before fit. Normalisation "
                "statistics must be fitted on the training window only; fitting "
                "them across the full series leaks out-of-sample data into training."
            )
        out = frame.copy()
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise KeyError(f"columns fitted but absent at transform time: {missing}")
        out[self.columns] = (frame[self.columns] - self.means) / self.stds
        return out

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Only legitimate on a training window. Named explicitly so a call on the
        full series is visible in review."""
        return self.fit(frame).transform(frame)


@dataclass(frozen=True)
class RollingZScore:
    """Backward-looking normalisation. Nothing to fit, nothing to leak."""

    window: int
    min_periods: int | None = None

    def transform(self, frame: pd.DataFrame | pd.Series):
        min_periods = self.min_periods or max(2, self.window // 2)
        rolling = frame.rolling(self.window, min_periods=min_periods)
        mean = rolling.mean()
        std = rolling.std()
        return (frame - mean) / std.replace(0.0, np.nan)
