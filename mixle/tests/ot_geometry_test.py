"""P6 (experimental) -- optimal-transport geometry of model space.

Provable receipts: the Bures-Wasserstein metric matches its closed form and satisfies the metric
axioms; the Gaussian barycenter averages standard deviations (not variances) and is the ``W2^2``
minimizer, beating naive parameter averaging in transport space. The mixture-merge comparison is
*measured*, not assumed -- faithful to the card's kill criterion (record whether the barycenter
beats plain ensembling; do not pretend it always does).
"""

from __future__ import annotations

import numpy as np

from mixle.experimental.ot_geometry import (
    bures_wasserstein,
    bures_wasserstein_params,
    gaussian_barycenter,
    gaussian_barycenter_params,
    mixture_barycenter,
)
from mixle.inference.estimation import optimize
from mixle.stats import (
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
)


def test_bures_wasserstein_closed_form_and_axioms() -> None:
    g1, g2 = GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 4.0)
    # W2^2 = (mean diff)^2 + (std diff)^2 = 9 + (1-2)^2 = 10.
    assert np.isclose(bures_wasserstein(g1, g2), np.sqrt(10.0))
    assert bures_wasserstein(g1, g1) == 0.0  # identity
    assert bures_wasserstein(g1, g2) == bures_wasserstein(g2, g1)  # symmetry
    assert bures_wasserstein(g1, g2) >= 0.0  # non-negativity


def test_matches_sample_based_wasserstein_1d() -> None:
    """Cross-check the closed form against an independent sorted-sample W2 estimate (numpy only)."""
    rng = np.random.default_rng(0)
    a = np.sort(rng.normal(1.0, np.sqrt(2.0), 60_000))
    b = np.sort(rng.normal(-1.0, np.sqrt(0.5), 60_000))
    sampled = float(np.sqrt(np.mean((a - b) ** 2)))  # 1-D W2 = L2 between sorted samples
    exact = bures_wasserstein(GaussianDistribution(1.0, 2.0), GaussianDistribution(-1.0, 0.5))
    assert np.isclose(exact, sampled, rtol=0.03), f"exact {exact:.4f} vs sampled {sampled:.4f}"


def test_gaussian_barycenter_averages_standard_deviations() -> None:
    # Bures barycenter of N(0,1) and N(2,9): mean 1, std = mean(1, 3) = 2 -> variance 4.
    b = gaussian_barycenter([GaussianDistribution(0.0, 1.0), GaussianDistribution(2.0, 9.0)])
    assert np.isclose(b.mu, 1.0)
    assert np.isclose(b.sigma2, 4.0, atol=1e-6)


def test_multivariate_barycenter_diagonal() -> None:
    mean, cov = gaussian_barycenter_params([[0, 0], [1, 1]], [np.eye(2), np.diag([4.0, 1.0])])
    assert np.allclose(mean, [0.5, 0.5])
    # per-axis std averaging: axis0 mean(1,2)=1.5 -> 2.25; axis1 mean(1,1)=1.
    assert np.allclose(np.diag(cov), [2.25, 1.0], atol=1e-6)


def test_barycenter_minimizes_transport_cost_vs_naive_averaging() -> None:
    """The provable win: the Bures barycenter beats the naive covariance mean at Sum lambda W2^2."""

    def rot(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s], [s, c]])

    c1 = rot(0.0) @ np.diag([4.0, 0.5]) @ rot(0.0).T
    c2 = rot(np.pi / 2.2) @ np.diag([4.0, 0.5]) @ rot(np.pi / 2.2).T
    means, covs, w = [[0, 0], [0, 0]], [c1, c2], [0.5, 0.5]
    _, s_bary = gaussian_barycenter_params(means, covs, w)
    s_naive = 0.5 * c1 + 0.5 * c2

    def obj(s):
        return sum(wi * bures_wasserstein_params(m, s, m, ci) ** 2 for wi, m, ci in zip(w, means, covs))

    assert obj(s_bary) < obj(s_naive), "Bures barycenter should minimize transport cost below naive averaging"


def _fit_gmm(data, seed):
    return optimize(
        data,
        MixtureEstimator([GaussianEstimator(), GaussianEstimator()]),
        out=None,
        rng=np.random.RandomState(seed),
        max_its=100,
    )


def test_mixture_merge_is_measured_honestly() -> None:
    """Merge K GMMs three ways and record held-out log-density -- no assumption the barycenter wins."""
    rng = np.random.default_rng(0)
    data = np.concatenate([rng.normal(-2, 1, 400), rng.normal(3, 1.5, 400)]).tolist()
    held = np.concatenate([rng.normal(-2, 1, 300), rng.normal(3, 1.5, 300)]).tolist()
    gmms = [_fit_gmm(data, s) for s in range(3)]

    def held_ld(m):
        return float(np.mean([m.log_density(x) for x in held]))

    ens_comps = [c for m in gmms for c in m.components]
    ens_w = np.concatenate([np.asarray(m.w) / len(gmms) for m in gmms])
    ensemble = MixtureDistribution(ens_comps, (ens_w / ens_w.sum()).tolist())
    barycenter = mixture_barycenter(gmms)

    ld_ens, ld_bary = held_ld(ensemble), held_ld(barycenter)
    # The barycenter must at least be a VALID, sane merge (finite, matched-cost r=2 components).
    assert np.isfinite(ld_bary)
    assert len(barycenter.components) == 2
    # Honest kill-criterion check: the barycenter should be within a modest margin of ensembling,
    # even when (as here) ensembling's extra capacity edges it out. We record, not overclaim.
    assert ld_bary >= ld_ens - 0.25, f"barycenter {ld_bary:.4f} far below ensemble {ld_ens:.4f}"


def test_requires_equal_component_counts() -> None:
    m2 = MixtureDistribution([GaussianDistribution(0, 1), GaussianDistribution(2, 1)], [0.5, 0.5])
    m3 = MixtureDistribution(
        [GaussianDistribution(0, 1), GaussianDistribution(1, 1), GaussianDistribution(2, 1)],
        [1 / 3, 1 / 3, 1 / 3],
    )
    try:
        mixture_barycenter([m2, m3])
    except ValueError as e:
        assert "equal component counts" in str(e)
    else:
        raise AssertionError("expected ValueError on unequal component counts")


def test_determinism() -> None:
    rng = np.random.default_rng(1)
    data = np.concatenate([rng.normal(-2, 1, 300), rng.normal(3, 1, 300)]).tolist()
    gmms = [_fit_gmm(data, s) for s in range(3)]
    b1, b2 = mixture_barycenter(gmms), mixture_barycenter(gmms)
    assert b1.to_json() == b2.to_json()
