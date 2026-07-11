"""P10 (experimental) -- PAC-Bayes generalization certificates that compose along the tree.

Receipts: the KL is closed form and composes additively; the McAllester bound holds with the
promised ``1 - delta`` coverage, is non-vacuous, and tightens as ``n`` grows; and the per-node KL
decomposition localizes an overfit subtree. The tightness/coverage are measured, per the card's
kill criterion, not assumed.
"""

from __future__ import annotations

import numpy as np

from mixle.experimental.pac_bayes import (
    bounded_losses,
    certify_generalization,
    gaussian_kl,
    per_component_kl,
    total_kl,
)
from mixle.inference.estimation import optimize
from mixle.stats import (
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
)


def _prior():
    return MixtureDistribution([GaussianDistribution(0.0, 25.0), GaussianDistribution(0.0, 25.0)], [0.5, 0.5])


def _fit(data, seed):
    return optimize(
        data,
        MixtureEstimator([GaussianEstimator(), GaussianEstimator()]),
        out=None,
        rng=np.random.RandomState(seed),
        max_its=100,
    )


def _two_cluster(rng, n, held=False):
    m = 3000 if held else n // 2
    return np.concatenate([rng.normal(-2, 1, m), rng.normal(3, 1.5, m)]).tolist()


def test_gaussian_kl_closed_form() -> None:
    assert gaussian_kl(0.0, 1.0, 0.0, 1.0) == 0.0
    # KL(N(0,1)||N(0,4)) = 0.5(ln 4 + 1/4 - 1).
    assert np.isclose(gaussian_kl(0.0, 1.0, 0.0, 4.0), 0.5 * (np.log(4.0) + 0.25 - 1.0))


def test_kl_composes_additively() -> None:
    model = MixtureDistribution([GaussianDistribution(-2, 1), GaussianDistribution(3, 2)], [0.4, 0.6])
    prior = _prior()
    node = per_component_kl(model, prior)
    # total = sum of per-node Gaussian KL + the mixing-weight KL term.
    from mixle.experimental.pac_bayes import _categorical_kl

    expected = sum(node) + _categorical_kl(np.asarray(model.w), np.asarray(prior.w))
    assert np.isclose(total_kl(model, prior), expected)


def test_bound_holds_and_is_non_vacuous_and_tightens() -> None:
    gaps = []
    for n in (200, 1000, 5000):
        rng = np.random.default_rng(0)
        tr = _two_cluster(rng, n)
        te = _two_cluster(rng, n, held=True)
        model = _fit(tr, 0)
        cert = certify_generalization(model, _prior(), tr, delta=0.05)
        test_loss = float(np.mean(bounded_losses(model, te)))
        assert test_loss <= cert.bound, f"bound violated at n={n}: test {test_loss:.4f} > {cert.bound:.4f}"
        assert not cert.vacuous, f"bound was vacuous at n={n}"
        gaps.append(cert.bound - test_loss)
    # The certificate tightens with more data.
    assert gaps[0] > gaps[-1], f"bound did not tighten with n: gaps={gaps}"


def test_coverage_respects_delta() -> None:
    """Over replications, P(held-out loss > bound) must be <= delta (the PAC-Bayes guarantee)."""
    delta, R, n = 0.1, 150, 500
    violations = 0
    for i in range(R):
        rng = np.random.default_rng(2000 + i)
        tr = _two_cluster(rng, n)
        te = _two_cluster(rng, n, held=True)
        model = _fit(tr, i)
        cert = certify_generalization(model, _prior(), tr, delta=delta)
        if float(np.mean(bounded_losses(model, te))) > cert.bound:
            violations += 1
    rate = violations / R
    assert rate <= delta + 0.03, f"coverage violated: {rate:.3f} > delta={delta}"


def test_per_node_localizes_the_overfit_subtree() -> None:
    """A component collapsed onto outliers carries almost all the KL -- the blame localizes."""
    # Component 1 is tight (var 0.01) and far -- the overfit subtree; component 0 is reasonable.
    overfit = MixtureDistribution([GaussianDistribution(0.0, 4.0), GaussianDistribution(10.0, 0.01)], [0.9, 0.1])
    node = per_component_kl(overfit, _prior())
    cert = certify_generalization(overfit, _prior(), np.random.default_rng(0).normal(0, 2, 500).tolist())
    assert cert.worst_subtree() == 1, f"blame not localized to the overfit component: {node}"
    assert node[1] > 10 * node[0], f"overfit component's KL should dominate: {node}"


def test_bounded_loss_is_in_unit_interval() -> None:
    model = _fit(_two_cluster(np.random.default_rng(0), 400), 0)
    losses = bounded_losses(model, np.random.default_rng(1).normal(0, 3, 500).tolist())
    assert np.all(losses >= 0.0) and np.all(losses < 1.0)


def test_determinism() -> None:
    rng = np.random.default_rng(3)
    tr = _two_cluster(rng, 400)
    model = _fit(tr, 0)
    c1 = certify_generalization(model, _prior(), tr)
    c2 = certify_generalization(model, _prior(), tr)
    assert c1.as_dict() == c2.as_dict()
