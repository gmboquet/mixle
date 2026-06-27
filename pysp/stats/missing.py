"""First-class handling of occasional missing entries by MARGINALIZATION (MAR/MCAR), plus the MISSING sentinel.

Two ways to cope with an absent entry:

* **Model** it -- learn a probability that the entry is missing (``OptionalDistribution`` with ``p`` set).
  Appropriate when missingness is informative and you want to estimate its rate.
* **Marginalize** it -- integrate the missing coordinate out of the likelihood so it contributes nothing.
  A missing field then has log-density 0 and yields no sufficient statistics, so EM fits each field from
  its *present* rows only -- the maximum-likelihood estimator under missing-at-random. This is the elegant
  default for "occasional missing entries": you don't model a nuisance missingness rate, and you can still
  read posteriors over the unobserved coordinates given the observed ones.

Marginalization is exactly ``OptionalDistribution(dist, p=None)``; this module surfaces it as a first-class
tool with a canonical sentinel and ergonomic builders (so you don't wrap every field by hand), and pairs
with ``CompositeDistribution.marginal``/``condition`` and ``MixtureDistribution.conditional`` to get the
posterior/imputation over the missing entries (see those methods).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class _Missing:
    """Singleton sentinel marking an absent entry. Test for it with ``x is MISSING``."""

    _instance: _Missing | None = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __reduce__(self):  # pickling preserves the singleton identity
        return (_Missing, ())


MISSING = _Missing()


def marginalized(dist: Any, missing_value: Any = MISSING) -> Any:
    """Wrap ``dist`` so a ``missing_value`` entry is marginalized out (not modeled).

    The wrapped field contributes log-density 0 and no sufficient statistics for missing observations, so
    estimation uses only the present ones. Equivalent to ``OptionalDistribution(dist, p=None,
    missing_value=missing_value)`` -- this is the principled missing-at-random treatment, not a degenerate
    case."""
    from pysp.stats.combinator.optional import OptionalDistribution

    return OptionalDistribution(dist, p=None, missing_value=missing_value)


def composite_with_missing(
    dists: Sequence[Any], missing_value: Any = MISSING
) -> Any:
    """Build a ``CompositeDistribution`` over ``dists`` in which every field tolerates ``missing_value``.

    Each field is wrapped with :func:`marginalized`, so any field of an observation tuple may be
    ``missing_value`` and is integrated out of the likelihood (scoring and EM). Build the composite once
    and pass ``MISSING`` for absent fields in your data -- no per-field bookkeeping."""
    from pysp.stats.combinator.composite import CompositeDistribution

    return CompositeDistribution([marginalized(d, missing_value) for d in dists])


def marginalize_estimator_leaves(estimator: Any, missing_value: Any = MISSING) -> Any:
    """Wrap the leaf estimators of ``estimator`` so they marginalize ``missing_value``.

    Recurses through ``CompositeEstimator`` (per field); every leaf estimator is wrapped in a
    marginalizing ``OptionalEstimator`` (``est_prob=False`` -> no missingness rate is fit, the missing
    value is integrated out). Used by ``pysp.ppl`` to fit a model from data with missing entries without
    imputing them."""
    from pysp.stats.combinator.composite import CompositeEstimator
    from pysp.stats.combinator.optional import OptionalEstimator

    if isinstance(estimator, CompositeEstimator):
        return CompositeEstimator([marginalize_estimator_leaves(c, missing_value) for c in estimator.estimators])
    return OptionalEstimator(estimator, missing_value=missing_value, est_prob=False)


def unwrap_marginalized(dist: Any) -> Any:
    """Inverse of :func:`marginalize_estimator_leaves` on a fitted distribution: strip the marginalizing
    ``OptionalDistribution`` wrappers (recursing through ``CompositeDistribution``) to recover the base
    model whose parameters were fit from the present entries."""
    from pysp.stats.combinator.composite import CompositeDistribution
    from pysp.stats.combinator.optional import OptionalDistribution

    if isinstance(dist, CompositeDistribution):
        return CompositeDistribution([unwrap_marginalized(d) for d in dist.dists])
    if isinstance(dist, OptionalDistribution):
        return dist.dist
    return dist
