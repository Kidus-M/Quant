"""Probabilistic and deflated Sharpe ratios (Bailey and Lopez de Prado).

The problem this solves: search a hundred parameter combinations on the same data
and the best one will have an impressive Sharpe ratio whether or not any edge
exists. The expected maximum Sharpe under the null grows with the number of
trials, so the raw number from a grid search is not comparable to the raw number
from a single a-priori strategy.

``deflated_sharpe_ratio`` returns the probability that the observed Sharpe exceeds
what the best of ``n_trials`` random strategies would have produced. Below roughly
0.95 the result is indistinguishable from a lucky draw out of the search.

The harness counts every parameter combination it evaluates, across every
walk-forward fold, and feeds that count in. Undercounting trials is the easiest
way to make this test pass, so the trial count is reported alongside the number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.metrics.stats import norm_cdf, norm_ppf, sample_kurtosis, sample_skew

EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class DeflatedSharpeResult:
    sharpe_observed: float
    sharpe_threshold: float
    probabilistic_sharpe: float
    deflated_sharpe: float
    n_trials: int
    n_observations: int
    skew: float
    kurtosis: float

    @property
    def is_significant(self) -> bool:
        return self.deflated_sharpe >= 0.95

    def summary(self) -> str:
        verdict = "clears" if self.is_significant else "does NOT clear"
        return (
            f"deflated Sharpe {self.deflated_sharpe:.3f} after {self.n_trials} trials "
            f"({verdict} the 0.95 bar); raw Sharpe {self.sharpe_observed:.3f} vs a "
            f"selection-bias threshold of {self.sharpe_threshold:.3f}"
        )


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Probability that the true Sharpe exceeds ``benchmark_sharpe``.

    All Sharpe values here are per-observation (not annualised). Annualising them
    first inflates the statistic by sqrt(periods per year) and makes everything
    look significant.
    """
    if n_observations < 2:
        return float("nan")
    denominator = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2
    if denominator <= 0:
        return float("nan")
    z = (sharpe - benchmark_sharpe) * math.sqrt(n_observations - 1) / math.sqrt(denominator)
    return norm_cdf(z)


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` strategies with no real edge.

    This is the bar a grid-search winner has to clear just to be interesting.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials == 1:
        return 0.0
    sigma = math.sqrt(max(sharpe_variance, 0.0))
    if sigma == 0.0:
        return 0.0
    gamma = EULER_MASCHERONI
    term_a = (1.0 - gamma) * norm_ppf(1.0 - 1.0 / n_trials)
    term_b = gamma * norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sigma * (term_a + term_b)


def deflated_sharpe_ratio(
    returns: np.ndarray | None = None,
    *,
    sharpe: float | None = None,
    n_observations: int | None = None,
    n_trials: int = 1,
    trial_sharpes: np.ndarray | None = None,
    skew: float | None = None,
    kurtosis: float | None = None,
) -> DeflatedSharpeResult:
    """Deflate an observed Sharpe for the number of trials that produced it.

    ``trial_sharpes`` is the set of per-observation Sharpe ratios seen during the
    search; its variance is what sets the selection-bias threshold. When it is not
    available the variance is approximated as 1/(T-1), the asymptotic variance of
    a Sharpe estimate under the null, which is the conservative fallback.
    """
    if returns is not None:
        returns = np.asarray(returns, dtype="float64")
        returns = returns[np.isfinite(returns)]
        n_observations = int(returns.size)
        std = returns.std(ddof=1) if returns.size > 1 else 0.0
        sharpe = float(returns.mean() / std) if std > 0 else 0.0
        skew = sample_skew(returns) if skew is None else skew
        kurtosis = sample_kurtosis(returns) if kurtosis is None else kurtosis

    if sharpe is None or n_observations is None:
        raise ValueError("provide either returns, or both sharpe and n_observations")
    skew = 0.0 if skew is None else skew
    kurtosis = 3.0 if kurtosis is None else kurtosis

    if trial_sharpes is not None and len(trial_sharpes) > 1:
        variance = float(np.nanvar(np.asarray(trial_sharpes, dtype="float64"), ddof=1))
    else:
        variance = 1.0 / max(1, n_observations - 1)

    threshold = expected_max_sharpe(max(1, int(n_trials)), variance)
    psr = probabilistic_sharpe_ratio(sharpe, n_observations, skew, kurtosis, 0.0)
    dsr = probabilistic_sharpe_ratio(sharpe, n_observations, skew, kurtosis, threshold)

    return DeflatedSharpeResult(
        sharpe_observed=float(sharpe),
        sharpe_threshold=float(threshold),
        probabilistic_sharpe=float(psr),
        deflated_sharpe=float(dsr),
        n_trials=int(n_trials),
        n_observations=int(n_observations),
        skew=float(skew),
        kurtosis=float(kurtosis),
    )
