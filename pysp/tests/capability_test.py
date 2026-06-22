"""Tests for pysp.capability — feature detection by what an object supports, not its class."""

import numpy as np
import pytest

from pysp import capability as cap
from pysp.stats.combinator.null_dist import NullDistribution
from pysp.stats.latent.mixture import MixtureDistribution
from pysp.stats.leaf.categorical import CategoricalDistribution
from pysp.stats.leaf.gaussian import GaussianDistribution
from pysp.stats.leaf.poisson import PoissonDistribution
from pysp.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution


def _cat():
    return CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})


# --------------------------------------------------------------- families report capabilities
def test_finite_discrete_is_enumerable_and_rankable():
    caps = cap.capabilities(_cat())
    assert {"Enumerable", "FiniteSupport", "RankableByIndex", "ExponentialFamily"} <= caps
    assert "Conditionable" not in caps and "LatentStructured" not in caps


def test_continuous_is_exponential_family_only():
    caps = cap.capabilities(GaussianDistribution(0.0, 1.0))
    assert "ExponentialFamily" in caps
    assert not cap.supports(GaussianDistribution(0.0, 1.0), cap.Enumerable)
    assert not cap.supports(GaussianDistribution(0.0, 1.0), cap.RankableByIndex)


def test_countable_discrete_is_enumerable_but_not_finite():
    po = PoissonDistribution(2.0)
    assert cap.supports(po, cap.Enumerable)
    assert not cap.supports(po, cap.FiniteSupport)
    assert not cap.supports(po, cap.RankableByIndex)  # the finite ==> rankable edge: not finite => not rankable


def test_elliptical_is_conditionable_and_marginalizable():
    mvn = MultivariateGaussianDistribution(np.zeros(2), np.eye(2))
    assert cap.supports(mvn, cap.Conditionable)
    assert cap.supports(mvn, cap.Marginalizable)


def test_mixture_is_latent_structured():
    mix = MixtureDistribution([GaussianDistribution(0, 1), GaussianDistribution(3, 1)], [0.5, 0.5])
    assert cap.supports(mix, cap.LatentStructured)
    assert cap.supports(mix, cap.PosteriorPredictive)


def test_null_is_neutral():
    assert cap.supports(NullDistribution(), cap.Neutral)
    assert not cap.supports(_cat(), cap.Neutral)


# --------------------------------------------------------------- the implication edge
def test_rankable_implies_enumerable_and_finite():
    c = _cat()
    if cap.supports(c, cap.RankableByIndex):
        assert cap.supports(c, cap.Enumerable) and cap.supports(c, cap.FiniteSupport)


# --------------------------------------------------------------- protocol-flavored capability
def test_engine_resident_estep_protocol():
    class _Resident:
        def seq_update_engine(self, enc, weights, estimate, engine):  # noqa: D401
            return None

    class _Plain:
        pass

    assert cap.supports(_Resident(), cap.EngineResidentEStep)
    assert not cap.supports(_Plain(), cap.EngineResidentEStep)


def test_engine_resident_matches_legacy_getattr_hook():
    # The kernel migration replaced `callable(getattr(acc, "seq_update_engine", None))` with
    # supports(acc, EngineResidentEStep); they must agree for every accumulator.
    for dist in (GaussianDistribution(0, 1), _cat(), PoissonDistribution(2.0)):
        acc = dist.estimator().accumulator_factory().make()
        assert cap.supports(acc, cap.EngineResidentEStep) == callable(getattr(acc, "seq_update_engine", None))


# --------------------------------------------------------------- query surface
def test_require_raises_with_clear_message():
    with pytest.raises(cap.CapabilityError) as ei:
        cap.require(GaussianDistribution(0, 1), cap.Conditionable, "conditioning demo")
    assert "GaussianDistribution" in str(ei.value) and "Conditionable" in str(ei.value)


def test_capabilities_returns_frozenset_of_names():
    caps = cap.capabilities(_cat())
    assert isinstance(caps, frozenset) and all(isinstance(n, str) for n in caps)


# --------------------------------------------------------------- the capability algebra (combinator)
def test_intersect_capabilities_is_facet_preserving():
    cat, po = _cat(), PoissonDistribution(2.0)
    both_finite = cap.intersect_capabilities([cat, cat])
    assert {"Enumerable", "FiniteSupport", "RankableByIndex"} <= both_finite
    # a child with infinite support drops finiteness/rankability but keeps enumerability
    with_infinite = cap.intersect_capabilities([cat, po])
    assert "Enumerable" in with_infinite
    assert "FiniteSupport" not in with_infinite and "RankableByIndex" not in with_infinite
    # no children => the vacuous intersection is everything facet-preserving
    assert cap.intersect_capabilities([]) == frozenset(c.__name__ for c in cap.FACET_PRESERVING)


# --------------------------------------------------------------- generic capability-tiered algorithm
def test_top_k_dispatches_on_capability():
    out = cap.top_k(_cat(), 2)
    assert [v for v, _ in out] == ["a", "b"]  # descending probability
    assert out[0][1] >= out[1][1]
    with pytest.raises(cap.CapabilityError):
        cap.top_k(GaussianDistribution(0, 1), 2)  # continuous => not enumerable => clear failure
