"""Normal distribution helpers.

Implemented here rather than pulled in with scipy so the dependency list stays
short. Both are standard: the CDF via ``math.erf``, the inverse via the Acklam
rational approximation, accurate to about 1.15e-9 across the range, which is
several orders of magnitude better than the sampling error on any Sharpe ratio
this project will produce.
"""
from __future__ import annotations

import math

import numpy as np


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


_A = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
_B = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00]
_P_LOW = 0.02425


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF."""
    p = float(p)
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError(f"norm_ppf needs 0 < p < 1, got {p}")

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    if p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
            ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
        (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
    )


def sample_skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    centred = x - x.mean()
    denom = np.sqrt((centred**2).mean()) ** 3
    return float((centred**3).mean() / denom) if denom > 0 else 0.0


def sample_kurtosis(x: np.ndarray) -> float:
    """Non-excess (Pearson) kurtosis: 3.0 for a normal distribution.

    The deflated Sharpe formula is written in terms of this convention, and using
    excess kurtosis instead silently shifts every adjusted number.
    """
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if x.size < 4:
        return 3.0
    centred = x - x.mean()
    denom = ((centred**2).mean()) ** 2
    return float((centred**4).mean() / denom) if denom > 0 else 3.0
