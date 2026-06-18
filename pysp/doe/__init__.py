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
]
