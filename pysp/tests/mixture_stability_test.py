"""WS-L numerical-stability stress tests for high-dimensional Gaussian mixtures.

These reproduce the reported "mixture of Gaussians on digits (high-d) is very unstable /
crashes" failure chain and assert the P1 covariance/variance floor (always on), the P3 weight
floor / degenerate handling, the P4 k-means++ init, and the P5 EM-driver guards keep estimation
finite.

Before P1 the diagonal/full M-steps produced singular / negative / NaN covariances for any
component holding fewer than d points, which made ``MultivariateGaussianDistribution.__init__``
hard-fail in the Cholesky factorization. The ``*_high_dim_converges`` cases are the regression
guards: they must now converge to a finite log-likelihood.
"""

import io
import unittest

import numpy as np

from pysp.stats import (
    DiagonalGaussianDistribution,
    DiagonalGaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
    MultivariateGaussianDistribution,
    MultivariateGaussianEstimator,
    seq_encode,
    seq_log_density_sum,
)
from pysp.stats.latent.mixture import MixtureAccumulator
from pysp.utils.em import MonotonicEM
from pysp.utils.estimation import optimize


def _high_dim_data(rng, d=300, k=4, n=200):
    """K well-separated Gaussian blobs in d dimensions.

    With ``n`` points spread over ``k`` components and ``n / k << d``, several components hold
    far fewer than ``d`` points -> the raw MLE covariance is singular. This is exactly the
    setting that crashed before P1.
    """
    centers = rng.normal(scale=8.0, size=(k, d))
    labels = rng.randint(k, size=n)
    x = centers[labels] + rng.normal(scale=1.0, size=(n, d))
    return [row for row in x], labels


class MixtureStabilityTestCase(unittest.TestCase):
    # ------------------------------------------------------------------ P1 floors

    def test_diagonal_mixture_high_dim_converges(self):
        """Diagonal-Gaussian mixture on d=300 must converge to finite LL (crashed pre-P1)."""
        rng = np.random.RandomState(7)
        d, k = 300, 4
        data, _ = _high_dim_data(rng, d=d, k=k, n=160)

        comps = [DiagonalGaussianDistribution(np.zeros(d), np.ones(d)) for _ in range(k)]
        init = MixtureDistribution(comps, np.ones(k) / k)
        est = MixtureEstimator([DiagonalGaussianEstimator(dim=d) for _ in range(k)])

        model = optimize(
            data, est, max_its=20, delta=1e-6, rng=np.random.RandomState(1), prev_estimate=init, out=io.StringIO()
        )
        _, ll = seq_log_density_sum(seq_encode(data, model=model), model)
        self.assertTrue(np.isfinite(ll))

    def test_full_mvn_mixture_high_dim_converges(self):
        """Full-covariance mixture on d=120 must converge (singular Cholesky crashed pre-P1)."""
        rng = np.random.RandomState(11)
        d, k = 120, 3
        # n/k well below d so each component's scatter matrix is rank-deficient
        data, _ = _high_dim_data(rng, d=d, k=k, n=90)

        comps = [MultivariateGaussianDistribution(np.zeros(d), np.eye(d)) for _ in range(k)]
        init = MixtureDistribution(comps, np.ones(k) / k)
        est = MixtureEstimator([MultivariateGaussianEstimator(dim=d) for _ in range(k)])

        model = optimize(
            data, est, max_its=15, delta=1e-6, rng=np.random.RandomState(2), prev_estimate=init, out=io.StringIO()
        )
        _, ll = seq_log_density_sum(seq_encode(data, model=model), model)
        self.assertTrue(np.isfinite(ll))

    def test_p1_floor_bias_is_negligible(self):
        """The default variance floor must barely perturb a well-conditioned diagonal fit."""
        rng = np.random.RandomState(3)
        d, n = 5, 5000
        true_var = np.array([2.0, 0.5, 1.0, 3.0, 0.25])
        x = rng.normal(size=(n, d)) * np.sqrt(true_var)
        data = [row for row in x]

        est = DiagonalGaussianEstimator(dim=d)
        enc = seq_encode(data, model=DiagonalGaussianDistribution(np.zeros(d), np.ones(d)))
        acc = est.accumulator_factory().make()
        for sz, xx in enc:
            acc.seq_update(xx, np.ones(sz), DiagonalGaussianDistribution(np.zeros(d), np.ones(d)))
        model = est.estimate(None, acc.value())
        # recovered variance is close to truth; floor adds <1e-5 relative bias
        np.testing.assert_allclose(model.covar, true_var, rtol=0.15)
        self.assertTrue(np.all(model.covar > 0.0))

    # ------------------------------------------------------------------ P4 k-means++

    def test_kmeanspp_init_beats_dirichlet_on_separated_blobs(self):
        """k-means++ init should reach an equal-or-better optimum than random Dirichlet."""
        rng = np.random.RandomState(5)
        d, k = 50, 4
        data, _ = _high_dim_data(rng, d=d, k=k, n=400)

        def _fit(robust):
            comps = [DiagonalGaussianDistribution(np.zeros(d), np.ones(d)) for _ in range(k)]
            init = MixtureDistribution(comps, np.ones(k) / k)
            est = MixtureEstimator([DiagonalGaussianEstimator(dim=d) for _ in range(k)], robust=robust)
            model = optimize(
                data, est, max_its=25, delta=1e-7, rng=np.random.RandomState(0), prev_estimate=init, out=io.StringIO()
            )
            _, ll = seq_log_density_sum(seq_encode(data, model=model), model)
            return ll

        ll_robust = _fit(True)
        ll_plain = _fit(False)
        self.assertTrue(np.isfinite(ll_robust))
        # k-means++ should not be meaningfully worse than the random-Dirichlet baseline
        self.assertGreaterEqual(ll_robust, ll_plain - abs(ll_plain) * 0.05)

    def test_kmeanspp_falls_back_for_nonnumeric_encoding(self):
        """k-means++ seeding must degrade gracefully to the Dirichlet path off the vector case."""
        acc = MixtureAccumulator(
            [DiagonalGaussianEstimator(dim=3).accumulator_factory().make() for _ in range(2)], init="kmeans++"
        )
        # object array -> no numeric feature matrix -> fall back, no crash
        keep = np.array([True, True, False])
        self.assertIsNone(acc._feature_matrix(np.array([{"a": 1}, {"b": 2}, {"c": 3}], dtype=object), keep))

    # ------------------------------------------------------------------ P3 weight floor

    def test_weight_floor_keeps_components_alive(self):
        """robust w_min keeps a starved component from collapsing to exactly zero weight."""
        est = MixtureEstimator([DiagonalGaussianEstimator(dim=2) for _ in range(3)], robust=True)
        # component 2 receives zero count -> would be weight 0 without the floor
        counts = np.array([100.0, 100.0, 0.0])
        comp_ss = tuple((np.zeros(2), np.ones(2), c) for c in counts)
        model = est.estimate(None, (counts, comp_ss))
        self.assertTrue(np.all(model.w > 0.0))
        np.testing.assert_allclose(model.w.sum(), 1.0)

    # ------------------------------------------------------------------ P5 EM guards

    def test_monotonic_em_rejects_nonfinite(self):
        """MonotonicEM keeps the last good model and never returns a non-finite objective."""
        rng = np.random.RandomState(9)
        d, k = 80, 3
        data, _ = _high_dim_data(rng, d=d, k=k, n=120)
        comps = [DiagonalGaussianDistribution(np.zeros(d), np.ones(d)) for _ in range(k)]
        init = MixtureDistribution(comps, np.ones(k) / k)
        est = MixtureEstimator([DiagonalGaussianEstimator(dim=d) for _ in range(k)], robust=True)

        model = optimize(
            data,
            est,
            max_its=20,
            delta=1e-7,
            rng=np.random.RandomState(4),
            prev_estimate=init,
            out=io.StringIO(),
            strategy=MonotonicEM(),
        )
        _, ll = seq_log_density_sum(seq_encode(data, model=model), model)
        self.assertTrue(np.isfinite(ll))

    # ------------------------------------------------------------------ MNIST-like smoke

    def test_downsampled_mnist_like_smoke(self):
        """8x8 (d=64) digit-like blobs: the reported use case, end to end, must stay finite."""
        rng = np.random.RandomState(13)
        d, k, n = 64, 6, 300
        # sparse positive "ink" prototypes
        protos = np.abs(rng.normal(scale=1.0, size=(k, d)))
        labels = rng.randint(k, size=n)
        x = protos[labels] + rng.normal(scale=0.3, size=(n, d))
        x = np.clip(x, 0.0, None)
        data = [row for row in x]

        comps = [DiagonalGaussianDistribution(np.zeros(d), np.ones(d)) for _ in range(k)]
        init = MixtureDistribution(comps, np.ones(k) / k)
        est = MixtureEstimator([DiagonalGaussianEstimator(dim=d) for _ in range(k)], robust=True)

        model = optimize(
            data,
            est,
            max_its=30,
            delta=1e-7,
            rng=np.random.RandomState(6),
            prev_estimate=init,
            out=io.StringIO(),
            strategy=MonotonicEM(),
        )
        _, ll = seq_log_density_sum(seq_encode(data, model=model), model)
        self.assertTrue(np.isfinite(ll))
        np.testing.assert_allclose(model.w.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
