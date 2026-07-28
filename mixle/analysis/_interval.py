"""The one credible-interval level contract shared by every IC-1 carrier in ``mixle.analysis``.

Each of the analysis surfaces grew its own ``credible_interval(level)`` -- ``RiskQuantity``,
``HabitatModel``, ``BMDResult``, ``TransitionRiskResult``, ``_AFDistribution``, and the
``_SampleDerivedQuantity`` / ``_DeterministicRisk`` carriers -- and they disagreed about what a valid
level is (MXR-080-1580). Some validated it, some let a NaN or a level outside ``(0, 1)`` fall through
to ``np.quantile`` and surface as an incidental "Quantiles must be in the range [0, 1]", and the
deterministic carrier ignored the argument altogether and returned the same point interval for
``level=5.0`` as for ``level=0.9``. A caller sweeping coverage levels across sibling APIs got three
different behaviours for the same mistake.

:func:`validated_level` is that single boundary: finite, strictly inside ``(0, 1)``, non-Boolean, and
applied *before* any quantile or deterministic path -- including the paths that do not otherwise use
the level, since silently ignoring an out-of-range level is what let the mistake through unnoticed.
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np

__all__ = ["validated_level"]


def validated_level(level: Any, *, name: str = "level") -> float:
    """Return ``level`` as a float, or raise if it is not a usable central-interval mass.

    A level is the probability mass the interval is meant to contain, so it must be finite and
    strictly between 0 and 1. Booleans are rejected outright rather than read as ``1``/``0``: a
    Boolean is a flag that leaked into a numeric slot, not a coverage request.
    """
    if isinstance(level, (bool, np.bool_)) or not isinstance(level, Real):
        raise TypeError(f"{name} must be a real scalar probability, not a Boolean or array.")
    value = float(level)
    if not (np.isfinite(value) and 0.0 < value < 1.0):
        raise ValueError(f"{name} must be finite and strictly in (0, 1), got {level!r}.")
    return value
