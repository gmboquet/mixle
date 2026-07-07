"""explain(): exact per-part attribution of log p(x). See explain_margin_test.py for the decision-margin
ledger (workstream H1: answers with receipts)."""

import numpy as np

from mixle.inference import explain
from mixle.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
    MixtureDistribution,
)


def test_composite_parts_sum_exactly_and_rank_the_anomalous_field():
    comp = CompositeDistribution((CategoricalDistribution({"a": 0.9, "b": 0.1}), GaussianDistribution(0.0, 1.0)))
    x = ("a", 7.0)  # the Gaussian field is wildly unlikely; the category is typical
    ex = explain(comp, x)
    np.testing.assert_allclose(ex.total, comp.log_density(x), atol=1e-12)
    np.testing.assert_allclose(sum(v for _, v in ex.parts), ex.total, atol=1e-12)
    assert ex.most_anomalous(1)[0][0] == "field[1]"  # the 7-sigma Gaussian, exactly identified


def test_mixture_reports_responsibilities_and_winner_breakdown():
    c0 = CompositeDistribution((CategoricalDistribution({"a": 0.9, "b": 0.1}), GaussianDistribution(-3.0, 1.0)))
    c1 = CompositeDistribution((CategoricalDistribution({"a": 0.1, "b": 0.9}), GaussianDistribution(3.0, 1.0)))
    mix = MixtureDistribution([c0, c1], [0.5, 0.5])
    x = ("b", 3.2)
    ex = explain(mix, x)
    assert ex.component == 1 and ex.responsibilities[1] > 0.95
    np.testing.assert_allclose(ex.total, mix.log_density(x), atol=1e-12)
    assert all(name.startswith("component[1].field") or name == "component[1].prior" for name, _ in ex.parts)
    assert "component posterior" in ex.summary()
    # the winner's own (prior + field) parts do NOT sum to the mixture's true total (that is a logsumexp
    # over every component) -- the gap is the explicit, named correction term, not an omission.
    assert ex.correction != 0.0
    np.testing.assert_allclose(ex.ledger_sum(), ex.total, atol=1e-12)
    assert ex.is_exact()


def test_bayesian_network_parts_are_the_factor_scores():
    from mixle.inference.bayesian_network import (
        HeterogeneousBayesianNetwork,
        _LinearGaussianFactor,
        _MarginalFactor,
    )

    net = HeterogeneousBayesianNetwork(
        [
            _MarginalFactor(0, GaussianDistribution(0.0, 1.0)),
            _LinearGaussianFactor(1, [0], {}, np.array([2.0, 0.0]), 0.5),
        ]
    )
    x = (0.5, 6.0)  # child is ~10 sigma off its conditional mean 1.0 -> the anomalous PART is the EDGE
    ex = explain(net, x)
    np.testing.assert_allclose(ex.total, net.log_density(x), atol=1e-12)
    assert ex.most_anomalous(1)[0][0] == "field[1]|parents(0,)"
