"""Data quality checks run on every load.

The rule here is "log and quarantine, never silently drop". Anything removed from
the working set is written to the quarantine directory with the reason attached,
so a suspicious backtest can always be traced back to the bars that were taken
out from under it.

There are two tiers, and the split matters:

* **Structural failures are dropped.** Duplicate timestamps, non-positive prices,
  a high below the low, bars stamped when the market is shut. These cannot be
  traded and cannot be right.

* **Suspicious-but-plausible bars are flagged and kept.** Zero volume and large
  sigma jumps. A gold bar can genuinely gap several sigma on a CPI print, and
  deleting exactly the bars that hurt is the most flattering thing a backtester
  can do to itself. Both are configurable, but the default is to keep them and
  say so loudly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data.sessions import SessionCalendar

log = logging.getLogger(__name__)

DROP_REASONS = ("duplicate_timestamp", "nonpositive_price", "ohlc_inconsistent", "outside_session")
FLAG_REASONS = ("zero_volume", "price_spike_sigma")


@dataclass
class QualityReport:
    n_input: int = 0
    n_output: int = 0
    was_unsorted: bool = False
    counts: dict[str, int] = field(default_factory=dict)
    quarantine_path: Path | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_dropped(self) -> int:
        return self.n_input - self.n_output

    @property
    def dropped_fraction(self) -> float:
        return self.n_dropped / self.n_input if self.n_input else 0.0

    def summary_lines(self) -> list[str]:
        lines = [
            f"bars in: {self.n_input}, out: {self.n_output}, "
            f"dropped: {self.n_dropped} ({self.dropped_fraction:.3%})"
        ]
        if self.was_unsorted:
            lines.append("index was not monotonic on load and was sorted")
        for reason, count in sorted(self.counts.items()):
            if count:
                tier = "dropped" if reason in DROP_REASONS else "flagged, retained"
                lines.append(f"  {reason}: {count} ({tier})")
        if self.quarantine_path is not None:
            lines.append(f"quarantined rows written to {self.quarantine_path}")
        lines.extend(f"  note: {n}" for n in self.notes)
        return lines


def flag_bars(
    bars: pd.DataFrame,
    *,
    sigma_threshold: float = 12.0,
    sigma_window: int = 500,
    calendar: SessionCalendar | None = None,
) -> pd.DataFrame:
    """Return a boolean frame of quality flags, one column per reason.

    The spike test uses a rolling standard deviation of the *previous* bars only
    (``shift(1)``). Comparing a bar against a window that includes itself makes a
    lone outlier inflate its own threshold and hide.
    """
    flags = pd.DataFrame(False, index=bars.index, columns=list(DROP_REASONS + FLAG_REASONS))

    flags["duplicate_timestamp"] = bars.index.duplicated(keep="first")

    prices = bars[["open", "high", "low", "close"]]
    flags["nonpositive_price"] = (prices <= 0).any(axis=1) | prices.isna().any(axis=1)

    hi, lo = bars["high"], bars["low"]
    body_max = bars[["open", "close"]].max(axis=1)
    body_min = bars[["open", "close"]].min(axis=1)
    flags["ohlc_inconsistent"] = (hi < lo) | (hi < body_max - 1e-9) | (lo > body_min + 1e-9)

    if calendar is not None:
        flags["outside_session"] = ~calendar.is_open(bars.index)

    flags["zero_volume"] = bars["volume"].fillna(0.0) <= 0

    close = bars["close"].astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.log(close / close.shift(1))
    prior_sigma = log_ret.rolling(sigma_window, min_periods=max(30, sigma_window // 10)).std().shift(1)
    spike = log_ret.abs() > (sigma_threshold * prior_sigma)
    flags["price_spike_sigma"] = spike.fillna(False)

    return flags


def run_quality_checks(
    bars: pd.DataFrame,
    cfg: Config,
    *,
    calendar: SessionCalendar | None = None,
    label: str = "bars",
    pads_outside_session: bool = False,
) -> tuple[pd.DataFrame, QualityReport]:
    """Validate, quarantine, and return the usable bars.

    ``max_flagged_fraction`` exists to abort on a corrupt feed. For most sources
    an out-of-session bar *is* corruption -- most often a timezone mistake, which
    is the failure this project works hardest to catch -- so it counts towards
    that budget and the strict threshold applies.

    A source that pads non-trading hours is different: it emits forward-filled
    weekend rows by design, so a ~38% session drop is the expected outcome of a
    correct load, not evidence of a problem. When the adapter declares
    ``pads_outside_session`` those drops are budgeted separately against
    ``max_outside_session_fraction``. Both budgets stay enforced; a timezone
    error would still blow past the session one.
    """
    q = cfg.section("data").section("quality")
    report = QualityReport(n_input=len(bars))

    if bars.empty:
        return bars, report

    if not bars.index.is_monotonic_increasing:
        report.was_unsorted = True
        bars = bars.sort_index(kind="stable")

    check_session = bool(q.get("drop_outside_session", True))
    flags = flag_bars(
        bars,
        sigma_threshold=float(q.get("sigma_threshold", 12.0)),
        sigma_window=int(q.get("sigma_window", 500)),
        calendar=calendar if check_session else None,
    )

    report.counts = {c: int(flags[c].sum()) for c in flags.columns}

    drop_cols = list(DROP_REASONS)
    if bool(q.get("quarantine_zero_volume", False)):
        drop_cols.append("zero_volume")
    if bool(q.get("quarantine_price_spikes", False)):
        drop_cols.append("price_spike_sigma")

    drop_mask = flags[drop_cols].any(axis=1).to_numpy()

    quarantined = bars.loc[drop_mask].copy()
    if not quarantined.empty:
        reason_matrix = flags.loc[drop_mask, drop_cols].to_numpy()
        quarantined["reasons"] = [
            ",".join(np.asarray(drop_cols)[row]) for row in reason_matrix
        ]
        report.quarantine_path = _write_quarantine(quarantined, cfg, label)

    clean = bars.loc[~drop_mask]

    report.n_output = len(clean)

    max_frac = float(q.get("max_flagged_fraction", 1.0))
    session_dropped = int(flags["outside_session"].sum()) if pads_outside_session else 0
    if pads_outside_session and session_dropped:
        # Budgeted separately below. Everything else still answers to the strict
        # corruption threshold.
        session_fraction = session_dropped / len(bars)
        max_session_frac = float(q.get("max_outside_session_fraction", 0.5))
        if session_fraction > max_session_frac:
            raise ValueError(
                f"data quality: dropped {session_fraction:.2%} of {label} as outside "
                f"session, above max_outside_session_fraction of {max_session_frac:.2%}. "
                "This source pads non-trading hours, so some session drop is expected, "
                "but not this much. Check that the bars are stamped UTC and that the "
                "session calendar matches the instrument before raising the threshold."
            )
        report.notes.append(
            f"{session_dropped} bars fell outside market hours and were dropped. This "
            "source pads non-trading hours with filled rows, so that is expected here "
            "and is not a sign of a broken feed."
        )
        # Counted from the mask rather than by subtraction: a bar can be flagged
        # both outside-session and corrupt, and subtracting would hide it.
        corruption_cols = [c for c in drop_cols if c != "outside_session"]
        corruption_dropped = int(flags[corruption_cols].any(axis=1).sum())
        corruption_fraction = corruption_dropped / len(bars)
    else:
        corruption_fraction = report.dropped_fraction

    if corruption_fraction > max_frac:
        raise ValueError(
            f"data quality: dropped {corruption_fraction:.2%} of {label}, above the "
            f"configured max_flagged_fraction of {max_frac:.2%}. Fix the source rather "
            "than raising the threshold."
        )

    if report.counts.get("price_spike_sigma") and "price_spike_sigma" not in drop_cols:
        report.notes.append(
            f"{report.counts['price_spike_sigma']} bars moved more than "
            f"{q.get('sigma_threshold', 12.0)} sigma and were KEPT. Inspect the "
            "quarantine report before trusting results that depend on them."
        )
    if report.counts.get("zero_volume") and "zero_volume" not in drop_cols:
        report.notes.append(
            f"{report.counts['zero_volume']} zero-volume bars were KEPT; they are "
            "normal in thin gold sessions but cannot be filled at scale."
        )

    for line in report.summary_lines():
        log.info("[quality/%s] %s", label, line)

    return clean, report


def _write_quarantine(rows: pd.DataFrame, cfg: Config, label: str) -> Path:
    from src.config import resolve_path

    out_dir = resolve_path(cfg.get("data.quarantine_dir", "data/quarantine"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"{label}_{stamp}.parquet"
    rows.to_parquet(path)
    return path
