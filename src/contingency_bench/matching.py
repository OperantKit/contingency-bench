"""Generalized Matching Law (GML) parameter estimation.

The GML expresses the relation between behavior allocation and reinforcement
allocation in a concurrent schedule:

.. math::

    \\log\\left(\\frac{B_1}{B_2}\\right) = a \\cdot \\log\\left(\\frac{R_1}{R_2}\\right) + \\log b

where ``a`` is sensitivity (typically 0.5-1.0; 1.0 is strict matching) and
``log b`` is bias.

This module is inlined here from the v0 ``session_analyzer.analytics.matching``
module so that ``contingency-bench`` does not require ``session_analyzer`` as a
runtime dependency. See README "Local dependency notes" for the swap-in path.

References
----------
Baum (1974). On two types of deviation from the matching law: Bias and
undermatching. Journal of the Experimental Analysis of Behavior, 22, 231-242.
https://doi.org/10.1901/jeab.1974.22-231

Davison, M. (2021). Concurrent schedules: How responses per reinforcer
affects estimates of sensitivity to reinforcement. Journal of the
Experimental Analysis of Behavior, 116(1), 114-123.
https://doi.org/10.1002/jeab.694
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class MatchingParameters:
    """Fitted GML parameters."""

    sensitivity: float  # a (slope)
    bias: float  # log b (intercept)
    r_squared: float  # goodness of fit


def generalized_matching_law(
    log_reinforcement_ratio: NDArray[np.float64] | float,
    a: float,
    log_b: float,
) -> NDArray[np.float64] | float:
    """``log(B1/B2) = a * log(R1/R2) + log b``."""
    return a * log_reinforcement_ratio + log_b


def estimate_matching_parameters(
    responses_ratios: Sequence[float],
    reinforcement_ratios: Sequence[float],
) -> MatchingParameters:
    """Fit GML sensitivity and bias from per-condition ratios.

    Parameters
    ----------
    responses_ratios:
        Per-condition ``B_1 / B_2`` values.
    reinforcement_ratios:
        Per-condition ``R_1 / R_2`` values.

    Returns
    -------
    MatchingParameters
        Fitted parameters. Returns zeros if fewer than two valid data points
        survive the positive-ratio filter required for the log transform.
    """
    valid_indices = [
        i
        for i, (b, r) in enumerate(zip(responses_ratios, reinforcement_ratios, strict=True))
        if b > 0 and r > 0
    ]

    if len(valid_indices) < 2:
        return MatchingParameters(sensitivity=0.0, bias=0.0, r_squared=0.0)

    y = np.log(np.asarray([responses_ratios[i] for i in valid_indices], dtype=np.float64))
    x = np.log(np.asarray([reinforcement_ratios[i] for i in valid_indices], dtype=np.float64))

    popt, _pcov = curve_fit(generalized_matching_law, x, y, p0=[1.0, 0.0])
    a, log_b = float(popt[0]), float(popt[1])

    residuals = y - generalized_matching_law(x, a, log_b)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else 0.0

    return MatchingParameters(sensitivity=a, bias=log_b, r_squared=float(r_squared))
