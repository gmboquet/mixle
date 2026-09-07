"""k-means++ seeding runs Lloyd iterations before the responsibilities are assigned.

0.8.1 adversarial review P07-F02: the seeding alone picks a far outlier as a center with probability
proportional to its squared distance, the near-hard responsibilities then hand that component a
handful of rows, and EM shrinks it into a spike it never leaves: a 1500-row two-regime panel came
back as weights 0.993 / 0.007 with ``converged=True`` on the seed a shipped notebook uses, while
sklearn's ``GaussianMixture`` (which runs k-means to convergence from the same seeding) and the
Dirichlet initialization both found 0.79 / 0.21. Moving each center to the mean of its rows first
pulls the outlier seed onto its cluster's mass, and a start whose smallest cluster still holds only
a handful of rows is handed to the Dirichlet initialization instead.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from mixle.inference import optimize
from mixle.stats import MixtureEstimator, MultivariateGaussianEstimator
from mixle.stats.latent import mixture as mixture_module


def two_regime_panel(seed=0, days=1500):
    """The stock-portfolio notebook's simulated panel: six assets at their daily volatilities, a
    persistent calm/crisis regime (crisis days 2.5 times as volatile and correlated 0.70 instead of
    0.30), and per-asset Student-t tails (3 to 30 degrees of freedom)."""
    from scipy.stats import chi2

    rng = np.random.RandomState(seed)
    n = 6
    vol = np.array([0.01309, 0.01033, 0.01897, 0.01506, 0.01226, 0.00803])
    mean = np.array([0.00045, 0.00038, 0.00062, 0.00111, 0.00088, 0.00034])
    dof = np.array([3, 30, 30, 8, 30, 30])

    def equicorrelated(rho):
        return rho * np.ones((n, n)) + (1 - rho) * np.eye(n)

    calm, crisis = equicorrelated(0.30), equicorrelated(0.70)
    regime = np.zeros(days, dtype=int)
    for t in range(1, days):
        regime[t] = rng.rand() < (0.06 if regime[t - 1] == 0 else 0.80)
    rows = np.zeros((days, n))
    for t in range(days):
        z = rng.multivariate_normal(np.zeros(n), crisis if regime[t] else calm)
        tail = np.sqrt(dof / chi2.rvs(dof, random_state=rng))
        rows[t] = mean + vol * (2.5 if regime[t] else 1.0) * z * tail
    return rows.tolist(), float(regime.mean())


class LloydIterationsTest(unittest.TestCase):
    def fit_weights(self, rows, seed):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = optimize(
                rows,
                MixtureEstimator([MultivariateGaussianEstimator(dim=6) for _ in range(2)]),
                max_its=200,
                rng=np.random.RandomState(seed),
            )
        return np.sort(np.asarray(model.w))

    def test_no_seed_collapses_a_regime_into_a_spike(self):
        for panel_seed in (0, 1, 3):
            rows, crisis_share = two_regime_panel(panel_seed)
            for seed in range(6):
                with self.subTest(panel=panel_seed, seed=seed):
                    weights = self.fit_weights(rows, seed)
                    self.assertGreater(weights[0], 0.1, weights)
                    self.assertLess(abs(weights[0] - crisis_share), 0.08, weights)

    def test_seeding_alone_is_what_collapsed(self):
        # With the Lloyd loop and the small-cluster guard disabled, some seeds hand a component a
        # handful of rows (on macOS arm64: panel 0 at seeds 3 and 5, panel 1 at seed 1, panel 3 at
        # seeds 0-2; which seeds depends on the BLAS, so the sweep is over the same panels and
        # seeds the test above requires the fix to survive): this pins the mechanism the fix
        # addresses, not a property of the data.
        saved = mixture_module._KMEANS_LLOYD_ITERATIONS, mixture_module._KMEANS_MIN_CLUSTER_FRACTION
        try:
            mixture_module._KMEANS_LLOYD_ITERATIONS = 0
            mixture_module._KMEANS_MIN_CLUSTER_FRACTION = 0.0
            smallest = min(
                self.fit_weights(two_regime_panel(panel_seed)[0], seed)[0]
                for panel_seed in (0, 1, 3)
                for seed in range(6)
            )
        finally:
            mixture_module._KMEANS_LLOYD_ITERATIONS, mixture_module._KMEANS_MIN_CLUSTER_FRACTION = saved
        if smallest >= 0.05:
            # The trajectory is BLAS-dependent; a platform on which no seed in the sweep collapses
            # cannot exhibit the mechanism, and says so rather than failing or passing silently.
            self.skipTest(
                "no seeding-only collapse within the sweep on this platform (smallest weight %.3f)" % smallest
            )
        self.assertLess(smallest, 0.05, smallest)


if __name__ == "__main__":
    unittest.main()
