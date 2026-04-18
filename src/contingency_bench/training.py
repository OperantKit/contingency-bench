"""Training-history equalisation for between-model comparisons.

Johnson et al. (2022) showed that the FR vs. VR training schedule type does
not affect the progressive-ratio breakpoint, but total training experience
(trial count) does. For between-model comparisons to be valid, every model
must receive the same number of training steps per condition regardless of
schedule type. The same invariant is enforced for the LLM benchmark below.

References
----------
Johnson, Kangas, Bergman, & Rice (2022). Comparison of fixed- and
variable-ratio schedules of reinforcement on progressive-ratio responding.
Journal of the Experimental Analysis of Behavior, 117, 320-330.
https://doi.org/10.1002/jeab.755
"""

from __future__ import annotations

from collections.abc import Sequence


def equate_training_steps(
    conditions: Sequence[tuple[float, float]],
    steps_per_condition: int,
) -> int:
    """Validate that every condition receives the same step count.

    Parameters
    ----------
    conditions:
        VI-rate tuples to be run.
    steps_per_condition:
        Requested steps per condition. Must be > 0.

    Returns
    -------
    int
        The validated step count (identical for all conditions).
    """
    if steps_per_condition <= 0:
        raise ValueError(
            f"steps_per_condition must be > 0, got {steps_per_condition}. "
            "Training history must be equated across all models and conditions "
            "(Johnson et al., 2022; doi:10.1002/jeab.755)."
        )
    if len(conditions) == 0:
        raise ValueError("conditions list must not be empty.")
    return steps_per_condition
