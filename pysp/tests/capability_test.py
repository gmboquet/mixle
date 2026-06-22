"""Tests for pysp.capability — feature detection by what an object supports, not its class."""

import numpy as np
import pytest

from pysp import capability as cap
from pysp.stats.base.categorical import CategoricalDistribution
from pysp.stats.base.gaussian import GaussianDistribution
from pysp.stats.base.poisson import PoissonDistribution
from pysp.stats.combinator.null_dist import NullDistribution
from pysp.stats.latent.mixture import MixtureDistribution
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


# --------------------------------------------------------------- the newly-formalized facets
def test_transform_setvalued_backend_facets():
    from pysp.stats.combinator.transform import AffineTransform
    from pysp.stats.sets.bernoulli_set import BernoulliSetDistribution

    assert cap.supports(AffineTransform(2.0, 1.0), cap.Transform)
    assert not cap.supports(GaussianDistribution(0, 1), cap.Transform)
    assert cap.supports(BernoulliSetDistribution({"a": 0.5, "b": 0.5}), cap.SetValued)
    assert not cap.supports(_cat(), cap.SetValued)
    # exp-family leaves expose engine backend scoring
    assert cap.supports(GaussianDistribution(0, 1), cap.SupportsBackendScoring)


def test_conjugate_updatable_is_the_closed_form_tier():
    from pysp.stats.base.poisson import PoissonDistribution
    from pysp.stats.base.weibull import WeibullDistribution

    # Conjugate families report the capability (closed-form Bayesian update available)...
    assert cap.supports(_cat(), cap.ConjugateUpdatable)
    assert cap.supports(GaussianDistribution(0, 1), cap.ConjugateUpdatable)
    assert cap.supports(PoissonDistribution(2.0), cap.ConjugateUpdatable)
    # ...non-conjugate families don't (Bayesian inference must go numerical: MAP/Laplace/MCMC/VI).
    assert not cap.supports(WeibullDistribution(1.5, 2.0), cap.ConjugateUpdatable)
    # the uniform family-level method agrees with the capability and the registry
    from pysp.stats.bayes.conjugate import is_conjugate_family

    for d in (_cat(), GaussianDistribution(0, 1), WeibullDistribution(1.5, 2.0)):
        assert d.has_conjugate_prior() == cap.supports(d, cap.ConjugateUpdatable) == is_conjugate_family(d)


def test_temporal_point_process_facet():
    from pysp.stats.base.hawkes_process import HawkesProcessDistribution

    h = HawkesProcessDistribution(0.5, 0.3, 1.5, window=10.0)
    assert cap.supports(h, cap.TemporalPointProcess)
    assert not cap.supports(GaussianDistribution(0, 1), cap.TemporalPointProcess)
    # the unified surface is real: intensity + compensator reconstruct the log-density
    t = np.array([0.4, 1.1, 2.0, 3.7])
    loglam = sum(np.log(h.intensity(float(ti), t[:i])) for i, ti in enumerate(t))
    recon = loglam - h.expected_count(0.0, 10.0, t)
    assert recon == pytest.approx(h.log_density(t), abs=1e-9)


def test_core_contracts_are_enforced_abcs():
    # The pdist contracts are real ABCs now: isinstance works, and an incomplete subclass can't
    # be instantiated.
    from pysp.stats.compute.pdist import ParameterEstimator, ProbabilityDistribution

    assert isinstance(_cat(), ProbabilityDistribution)
    assert isinstance(_cat().estimator(), ParameterEstimator)

    class _Incomplete(ProbabilityDistribution):
        pass  # missing log_density/sampler/estimator

    with pytest.raises(TypeError):
        _Incomplete()


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


# --------------------------------------------------------------- combinators compose capabilities
def test_combinators_compose_capabilities_automatically():
    from pysp.stats.combinator.composite import CompositeDistribution
    from pysp.stats.combinator.sequence import SequenceDistribution

    # A composite of two finite-support leaves is itself finite, enumerable, and rankable —
    # support_size composes structurally, so capabilities() reflects it with no class names.
    comp = CompositeDistribution([_cat(), CategoricalDistribution({"x": 0.5, "y": 0.5})])
    assert {"Enumerable", "FiniteSupport", "RankableByIndex"} <= cap.capabilities(comp)
    assert comp.support_size() == 6
    # A sequence with an unbounded (Poisson) length is still enumerable but no longer finite/rankable.
    seq = SequenceDistribution(_cat(), PoissonDistribution(2.0))
    seq_caps = cap.capabilities(seq)
    assert "Enumerable" in seq_caps
    assert "FiniteSupport" not in seq_caps and "RankableByIndex" not in seq_caps


def test_distribution_capabilities_method():
    # dist.capabilities() is the public UX surface; equivalent to pysp.capabilities(dist).
    d = _cat()
    assert d.capabilities() == cap.capabilities(d)
    import pysp

    assert pysp.supports(d, cap.Enumerable) and pysp.capabilities(d) == d.capabilities()


# --------------------------------------------------------------- generic capability-tiered algorithm
def test_top_k_dispatches_on_capability():
    out = cap.top_k(_cat(), 2)
    assert [v for v, _ in out] == ["a", "b"]  # descending probability
    assert out[0][1] >= out[1][1]
    with pytest.raises(cap.CapabilityError):
        cap.top_k(GaussianDistribution(0, 1), 2)  # continuous => not enumerable => clear failure


# --------------------------------------------------------------- the coherence layer
def test_describe_is_plain_english_and_capability_accurate():
    import pysp

    text = pysp.describe(_cat())
    assert "CategoricalDistribution" in text and "can:" in text
    assert "Enumerable" in text and "ConjugateUpdatable" in text  # categorical has both
    mix_text = pysp.describe(MixtureDistribution([GaussianDistribution(0, 1), GaussianDistribution(3, 1)], [0.5, 0.5]))
    assert "latent-variable model" in mix_text and "no closed-form conjugate" in mix_text


def test_describe_is_robust_on_classes_and_non_distributions():
    import pysp

    # passing a class (e.g. a model class) must not crash — it needs a live instance for the rich view
    assert isinstance(pysp.describe(GaussianDistribution), str)  # the class
    assert "RandomForestEstimator" in pysp.describe(pysp.models.RandomForestEstimator)
    assert "can:" in pysp.describe(GaussianDistribution(0, 1))  # the instance is still rich


def test_ws4_distribution_capabilities():
    import numpy as np

    import pysp
    from pysp.capability import Continuous, Discrete, Fittable, HasCDF, Optimizable
    from pysp.relations import Assignment
    from pysp.stats.base.categorical import CategoricalDistribution

    g = GaussianDistribution(0.0, 1.0)
    c = CategoricalDistribution({"a": 0.5, "b": 0.5})
    assert pysp.supports(g, HasCDF) and pysp.supports(g, Continuous) and not pysp.supports(g, Discrete)
    assert pysp.supports(c, Discrete) and not pysp.supports(c, Continuous)
    assert pysp.supports(g, Fittable) and pysp.supports(c, Fittable)
    assert pysp.supports(Assignment(np.array([[1.0, 2.0], [3.0, 4.0]])), Optimizable)
    for d in (g, c):  # Discrete and Continuous are mutually exclusive
        assert not (pysp.supports(d, Discrete) and pysp.supports(d, Continuous))


def test_catalog_is_the_single_vocabulary():
    import pysp

    names = {s.name for s in pysp.catalog()}
    for c in cap.ALL_CAPABILITIES:
        if c.__name__ != "SupportsBackendComponentScoring":  # internal variant
            assert c.__name__ in names, f"{c.__name__} missing from the catalog"
    assert {"Relation", "ComputeEngine", "ForwardOperator", "Surrogate", "EncodedFold"} <= names
    assert cap.render_catalog_markdown().startswith("| Capability |")


def test_what_supports_filters_by_capability():
    import numpy as np

    import pysp
    from pysp.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution

    pool = [GaussianDistribution(0, 1), MultivariateGaussianDistribution(np.zeros(2), np.eye(2)), _cat()]
    assert pysp.what_supports(cap.Conditionable, pool) == ["MultivariateGaussianDistribution"]
