"""Tests for the concern-oriented slice: pysp.ops (operations) and pysp.enumeration (the concern)."""

import numpy as np
import pytest

import pysp
from pysp import capability as cap
from pysp import ops


# --------------------------------------------------------------- the headline: quantize
def test_quantize_turns_a_continuous_distribution_enumerable():
    from pysp.stats.base.gaussian import GaussianDistribution

    g = GaussianDistribution(0.0, 1.0)
    # before: continuous, not enumerable
    assert not pysp.supports(g, cap.Enumerable)
    q = ops.quantize(g, bits=6)
    # after: a finite, enumerable, rankable discrete distribution — the operation moved its capabilities
    assert pysp.supports(q, cap.Enumerable)
    assert pysp.supports(q, cap.FiniteSupport)
    assert pysp.supports(q, cap.RankableByIndex)
    assert q.support_size() == 64
    # mass is a proper distribution, peaked near the Gaussian mode
    vals = list(q.pmap.keys())
    probs = np.array([q.pmap[v] for v in vals])
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert abs(float(vals[int(np.argmax(probs))])) < 0.5  # mode near 0


# --------------------------------------------------------------- operations are capability-gated
def test_ops_dispatch_on_capability_not_class():
    from pysp.stats.base.gaussian import GaussianDistribution
    from pysp.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution

    mvn = MultivariateGaussianDistribution(np.zeros(2), np.eye(2))
    assert type(ops.condition(mvn, {0: 1.0})).__name__ == "MultivariateGaussianDistribution"
    assert pysp.supports(ops.mixture([GaussianDistribution(0, 1), GaussianDistribution(3, 1)]), cap.LatentStructured)
    assert pysp.supports(ops.tilt(GaussianDistribution(0, 1), 0.5), cap.Conditionable) is False  # still a dist
    # an operation refuses an object lacking the required capability, early and clearly
    with pytest.raises(cap.CapabilityError):
        ops.condition(GaussianDistribution(0, 1), {0: 1.0})


# --------------------------------------------------------------- the concern module gathers it all
def test_enumeration_module_is_one_home_for_the_concern():
    import pysp.enumeration as enum

    for name in (
        "Enumerable",
        "RankableByIndex",
        "DistributionEnumerator",
        "supports_enumeration",
        "count_budget_index",
        "CountSemiring",
        "top_k",
        "sound_top_k",
    ):
        assert name in enum.__all__ and hasattr(enum, name)


def test_enumeration_spans_distributions_and_relations():
    import pysp.enumeration as enum
    from pysp.relations import Assignment
    from pysp.stats.base.categorical import CategoricalDistribution
    from pysp.stats.compute.pdist import DistributionEnumerator

    # both a distribution and a relation implement the same enumerator() contract and report Enumerable
    cat = CategoricalDistribution({"a": 0.6, "b": 0.4})
    rel = Assignment(np.array([[1.0, 2.0], [3.0, 0.5]]))
    assert enum.supports(cat, enum.Enumerable) and enum.supports(rel, enum.Enumerable)
    assert hasattr(DistributionEnumerator, "__next__") and hasattr(Assignment, "enumerator")
    # the relation enumerates its solutions in best-first order
    first = next(rel.enumerator())
    assert first.objective == pytest.approx(1.5)
