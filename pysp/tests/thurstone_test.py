"""Thurstone Gaussian random-utility ranking model: Genz likelihood, sampling, mu recovery."""

import itertools
import unittest

import numpy as np

from pysp.stats import ThurstoneDistribution


class ThurstoneTest(unittest.TestCase):
    def test_genz_likelihood_approximately_normalizes(self):
        for n in (3, 4, 5):
            rng = np.random.RandomState(n)
            d = ThurstoneDistribution(rng.randn(n), n_mc=8000)
            tot = sum(d.density(list(p)) for p in itertools.permutations(range(n)))
            self.assertAlmostEqual(tot, 1.0, delta=0.06)  # Monte-Carlo orthant estimate

    def test_genz_matches_brute_force_simulation(self):
        rng = np.random.RandomState(1)
        mu = np.array([1.5, 0.5, -0.5, -1.5])
        d = ThurstoneDistribution(mu, n_mc=20000)
        u = mu + rng.standard_normal((300000, 4))
        emp = np.argsort(-u, axis=1)
        from collections import Counter

        cnt = Counter(map(tuple, emp))
        for p, c in cnt.most_common(5):
            self.assertAlmostEqual(d.density(list(p)), c / 300000, delta=0.01)

    def test_modal_ordering_is_most_probable(self):
        d = ThurstoneDistribution([2.0, 1.0, 0.0, -1.0], n_mc=8000)
        self.assertGreater(d.log_density([0, 1, 2, 3]), d.log_density([3, 2, 1, 0]))

    def test_sampler_orders_by_utility(self):
        d = ThurstoneDistribution([5.0, 2.0, -2.0, -5.0])  # strongly separated -> near-deterministic
        draws = np.array(d.sampler(seed=0).sample(2000))
        self.assertEqual(int(np.bincount(draws[:, 0]).argmax()), 0)  # item 0 almost always best

    def test_mu_recovery(self):
        true = ThurstoneDistribution([2.0, 1.0, 0.0, -1.0, -2.0])
        samp = true.sampler(seed=2).sample(8000)
        acc = true.estimator().accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(samp), np.ones(len(samp)), None)
        fit = true.estimator().estimate(len(samp), acc.value())
        np.testing.assert_allclose(fit.mu, true.mu, atol=0.2)
        self.assertEqual(list(np.argsort(-fit.mu)), list(np.argsort(-true.mu)))

    def test_validation(self):
        with self.assertRaises(ValueError):
            ThurstoneDistribution([1.0])  # K >= 2


if __name__ == "__main__":
    unittest.main()
