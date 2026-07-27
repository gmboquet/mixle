"""Sparse mixture scoring (mixle.stats.latent.sparse_mixture): certified top-k tail bounds."""

import math
import unittest

import numpy as np

import mixle.stats as st
from mixle.stats.latent.sparse_mixture import log_density_sup, sparse_mixture_score


def _gmm(rng, k=20):
    comps = [st.GaussianDistribution(float(8 * rng.randn()), float(0.3 + rng.rand())) for _ in range(k)]
    return st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(k))))


class LogDensitySupTest(unittest.TestCase):
    def test_known_family_peaks(self):
        g = st.GaussianDistribution(2.0, 4.0)
        self.assertAlmostEqual(log_density_sup(g), g.log_density(2.0))  # peak is at the mean
        cat = st.CategoricalDistribution({"a": 0.6, "b": 0.4})
        self.assertAlmostEqual(log_density_sup(cat), math.log(0.6))
        pois = st.PoissonDistribution(3.0)
        self.assertAlmostEqual(log_density_sup(pois), pois.log_density(3))  # mode floor(lam)

    def test_unbounded_or_unknown_returns_none(self):
        self.assertIsNone(log_density_sup(st.GammaDistribution(0.5, 1.0)))  # density -> inf at 0

    def test_categorical_default_value_dominant(self):
        # default_value (0.99) exceeds every in-pmap probability (0.005 each): the true peak
        # log-density is achieved by a label OUTSIDE pmap, not by max(pmap.values()). The sup
        # must also apply CategoricalDistribution's -log1p(default_value) normalizer.
        cat = st.CategoricalDistribution(
            {"a": 0.005, "b": 0.005},
            default_value=0.99,
            scoring_only=True,
        )
        sup = log_density_sup(cat)
        outside_label_density = cat.log_density("not_in_pmap")
        self.assertAlmostEqual(sup, outside_label_density, places=10)
        # The bound must actually dominate every achievable log-density, in and out of pmap.
        self.assertGreaterEqual(sup + 1e-12, cat.log_density("a"))
        self.assertGreaterEqual(sup + 1e-12, cat.log_density("b"))
        self.assertGreaterEqual(sup + 1e-12, outside_label_density)

    def test_categorical_default_value_zero_matches_pre_existing_formula(self):
        # default_value=0.0 (the default): log1p(0)=0, so this must reduce exactly to the
        # pre-existing math.log(m) formula already covered by test_known_family_peaks.
        cat = st.CategoricalDistribution({"a": 0.6, "b": 0.4})
        self.assertAlmostEqual(log_density_sup(cat), math.log(0.6))


class SparseScoreTest(unittest.TestCase):
    def test_bracket_contains_exact(self):
        rng = np.random.RandomState(0)
        m = _gmm(rng, 20)
        for _ in range(50):
            x = float(rng.randn() * 10)
            exact = m.log_density(x)
            sc = sparse_mixture_score(m, x, max_components=5)
            self.assertLessEqual(sc.lower, exact + 1e-9)
            self.assertGreaterEqual(sc.upper, exact - 1e-9)  # certified bracket holds

    def test_full_k_is_exact(self):
        rng = np.random.RandomState(1)
        m = _gmm(rng, 12)
        x = 1.3
        sc = sparse_mixture_score(m, x, max_components=12)
        self.assertTrue(sc.exact)
        self.assertAlmostEqual(sc.lower, sc.upper)
        self.assertAlmostEqual(sc.lower, m.log_density(x), places=9)

    def test_bracket_tightens_with_more_components(self):
        rng = np.random.RandomState(2)
        m = _gmm(rng, 30)
        x = 0.5
        widths = [sparse_mixture_score(m, x, k).upper - sparse_mixture_score(m, x, k).lower for k in (2, 8, 30)]
        self.assertGreaterEqual(widths[0] + 1e-12, widths[1])
        self.assertGreaterEqual(widths[1] + 1e-12, widths[2])
        self.assertAlmostEqual(widths[2], 0.0, places=9)  # exact at full k

    def test_bracket_holds_with_sharply_different_categorical_components(self):
        # Scoring-only fallback behavior is tested directly in LogDensitySupTest.
        # Generative mixtures contain normalized probability laws.
        comps = [
            st.CategoricalDistribution({"a": 0.01, "b": 0.99}),
            st.CategoricalDistribution({"a": 0.9, "b": 0.1}),
            st.CategoricalDistribution({"a": 0.5, "b": 0.5}),
        ]
        m = st.MixtureDistribution(comps, [0.2, 0.5, 0.3])
        for x in ("a", "b"):
            exact = m.log_density(x)
            sc = sparse_mixture_score(m, x, max_components=2)
            with self.subTest(x=x):
                self.assertLessEqual(sc.lower, exact + 1e-9)
                self.assertGreaterEqual(sc.upper, exact - 1e-9)  # certified bracket must hold

    def test_unbounded_component_falls_back_to_exact(self):
        # a Gamma(shape<1) leaf has unbounded density -> cannot certify -> exact full scoring
        m = st.MixtureDistribution([st.GammaDistribution(0.5, 1.0), st.GammaDistribution(0.7, 2.0)], [0.5, 0.5])
        sc = sparse_mixture_score(m, 1.0, max_components=1)
        self.assertTrue(sc.exact)
        self.assertAlmostEqual(sc.lower, m.log_density(1.0), places=9)
        self.assertEqual(sc.n_scored, 2)


class CollapseTest(unittest.TestCase):
    def test_collapse_identical_is_exact(self):
        from mixle.stats.latent.sparse_mixture import collapse_identical

        # three identical G(0,1) + one G(5,1): collapse to two components, log p(x) unchanged
        m = st.MixtureDistribution(
            [
                st.GaussianDistribution(0.0, 1.0),
                st.GaussianDistribution(0.0, 1.0),
                st.GaussianDistribution(0.0, 1.0),
                st.GaussianDistribution(5.0, 1.0),
            ],
            [0.2, 0.3, 0.1, 0.4],
        )
        c = collapse_identical(m)
        self.assertEqual(len(c.components), 2)
        self.assertAlmostEqual(sum(c.w), 1.0)
        for x in (-1.0, 0.0, 2.0, 5.0, 7.0):
            self.assertAlmostEqual(c.log_density(x), m.log_density(x), places=9)

    def test_collapse_gaussian_preserves_global_moments(self):
        from mixle.stats.latent.sparse_mixture import collapse_gaussian_mixture

        rng = np.random.RandomState(0)
        comps = [st.GaussianDistribution(float(4 * rng.randn()), float(0.5 + rng.rand())) for _ in range(10)]
        w = list(rng.dirichlet(np.ones(10)))
        m = st.MixtureDistribution(comps, w)

        def moments(mix):
            ws = np.asarray(mix.w)
            mus = np.array([c.mu for c in mix.components])
            s2 = np.array([c.sigma2 for c in mix.components])
            mean = float((ws * mus).sum())
            var = float((ws * (s2 + mus**2)).sum() - mean**2)
            return mean, var

        collapsed = collapse_gaussian_mixture(m, max_components=3)
        self.assertLessEqual(len(collapsed.components), 3)
        m0, v0 = moments(m)
        m1, v1 = moments(collapsed)
        self.assertAlmostEqual(m0, m1, places=6)  # overall mean preserved exactly
        self.assertAlmostEqual(v0, v1, places=6)  # overall variance preserved exactly
        self.assertAlmostEqual(sum(collapsed.w), 1.0)

    def test_collapse_gaussian_rejects_non_gaussian(self):
        from mixle.stats.latent.sparse_mixture import collapse_gaussian_mixture

        m = st.MixtureDistribution([st.GaussianDistribution(0.0, 1.0), st.PoissonDistribution(3.0)], [0.5, 0.5])
        with self.assertRaises(ValueError):
            collapse_gaussian_mixture(m, 1)


if __name__ == "__main__":
    unittest.main()
