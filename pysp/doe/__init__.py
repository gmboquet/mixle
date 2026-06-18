"""Design of experiments (DoE) for pysparkplug.

This package builds experiment designs over a bounded input space and (in later additions)
sequential / Bayesian-optimization loops on top of the existing GP and regression machinery.

The first surface is space-filling and classical design generators, all returning a plain
``(n, d)`` numpy matrix of input points scaled into the supplied per-dimension bounds:

    >>> from pysp.doe import latin_hypercube
    >>> x = latin_hypercube([(0.0, 1.0), (-2.0, 2.0)], n=8, seed=0)
    >>> x.shape
    (8, 2)
"""

from __future__ import annotations

from pysp.doe.bayesopt import (
    BayesOptResult,
    available_acquisitions,
    expected_improvement,
    minimize,
    probability_of_improvement,
    propose_batch,
    propose_next,
    register_acquisition,
    upper_confidence_bound,
)
from pysp.doe.designs import (
    Bounds,
    full_factorial,
    latin_hypercube,
    maximin_latin_hypercube,
    random_design,
)

__all__ = [
    "Bounds",
    "full_factorial",
    "latin_hypercube",
    "maximin_latin_hypercube",
    "random_design",
    "BayesOptResult",
    "expected_improvement",
    "probability_of_improvement",
    "upper_confidence_bound",
    "register_acquisition",
    "available_acquisitions",
    "minimize",
    "propose_next",
    "propose_batch",
]
