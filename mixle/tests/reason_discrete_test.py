"""reason_discrete: exact Bayes over a hypothesis set with per-source nats attribution."""

import numpy as np

from mixle.inference.belief import CategoricalBelief
from mixle.reason import model_evidence, reason_discrete
from mixle.stats import GaussianDistribution


def test_exact_bayes_and_attribution():
    # two sources; the second is much more decisive
    ans = reason_discrete(3, [("weak", np.log([0.4, 0.35, 0.25])), ("strong", np.log([0.9, 0.05, 0.05]))])
    # exact posterior = normalized product with the uniform prior
    p = np.array([0.4, 0.35, 0.25]) * np.array([0.9, 0.05, 0.05])
    np.testing.assert_allclose(ans.probs, p / p.sum(), atol=1e-12)
    assert ans.map() == 0
    names = [n for n, _ in ans.attribution]
    nats = dict(ans.attribution)
    assert names == ["weak", "strong"] and nats["strong"] > nats["weak"]
    assert "residual entropy" in ans.summary()


def test_fitted_mixle_models_are_evidence():
    # hypothesis k <-> a fitted generative model; observation near mu=4 must select hypothesis 1
    models = [GaussianDistribution(-4.0, 1.0), GaussianDistribution(4.0, 1.0)]
    ans = reason_discrete(["low", "high"], [model_evidence("sensor", models, 3.7)])
    assert ans.map() == "high"
    assert ans.probs[1] > 0.99


def test_prior_and_labels_round_trip():
    prior = CategoricalBelief([0.7, 0.3], labels=["a", "b"])
    ans = reason_discrete(prior, [("even", np.zeros(2))])  # uninformative evidence
    np.testing.assert_allclose(ans.probs, [0.7, 0.3], atol=1e-12)
    assert abs(ans.attribution[0][1]) < 1e-12  # removed ~0 nats
