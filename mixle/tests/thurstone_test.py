"""Thurstone Gaussian random-utility ranking approximation contracts."""

import itertools
import unittest

import numpy as np

from mixle.stats import ThurstoneDistribution


class ThurstoneTest(unittest.TestCase):
    def test_genz_likelihood_approximately_normalizes(self):
        for n in (3, 4, 5):
            rng = np.random.RandomState(n)
            d = ThurstoneDistribution(rng.randn(n), n_mc=8000)
            tot = sum(d.density(list(p)) for p in itertools.permutations(range(n)))
            self.assertAlmostEqual(tot, 1.0, places=12)

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
        for n_mc in (0, -1, 1.5, True):
            with self.subTest(n_mc=repr(n_mc)), self.assertRaises((TypeError, ValueError)):
                ThurstoneDistribution([1.0, -1.0], n_mc=n_mc)
        for smoothing in (0.0, -1.0, np.nan):
            with self.subTest(smoothing=repr(smoothing)), self.assertRaises(ValueError):
                ThurstoneDistribution([1.0, -1.0], smoothing=smoothing)

    def test_score_is_batch_invariant_and_support_checked(self):
        d = ThurstoneDistribution([1.0, 0.0, -1.0], n_mc=1000, seed=4)
        target = np.array([[2, 0, 1]])
        alone = d.seq_log_density(target)[0]
        together = d.seq_log_density(np.array([[0, 1, 2], [2, 0, 1], [2, 0, 1]]))
        self.assertEqual(alone, together[1])
        self.assertEqual(together[1], together[2])
        for invalid in ([0, 0, 1], [0, 1], [0, 1, 3], [0.0, 1.5, 2.0]):
            with self.subTest(invalid=repr(invalid)), self.assertRaises((TypeError, ValueError)):
                d.log_density(invalid)

    def test_parameters_are_owned_and_approximation_is_labelled(self):
        mu = np.array([1.0, 0.0, -1.0])
        d = ThurstoneDistribution(mu, n_mc=100, seed=7)
        score = d.log_density([0, 1, 2])
        mu[:] = 100.0
        self.assertEqual(score, d.log_density([0, 1, 2]))
        with self.assertRaises(ValueError):
            d.mu[0] = 2.0
        self.assertEqual(d.approximation_diagnostics.draws, 100)
        self.assertIn("common_random", d.approximation_diagnostics.method)

    def test_accumulator_and_estimator_validate_statistics(self):
        d = ThurstoneDistribution([1.0, 0.0, -1.0], n_mc=100)
        acc = d.estimator(0.5).accumulator_factory().make()
        acc.update([0, 1, 2], 2.0, None)
        snapshot = acc.value()
        snapshot[1][:] = 0.0
        self.assertGreater(acc.value()[1].sum(), 0.0)
        fitted = d.estimator(0.5).estimate(2.0, acc.value())
        self.assertTrue(fitted.fit_diagnostics.regularized)
        self.assertFalse(fitted.fit_diagnostics.exact_mle)
        with self.assertRaises(ValueError):
            d.estimator().estimate(3.0, acc.value())
        with self.assertRaises(ValueError):
            acc.combine((1.0, np.zeros((3, 3))))


if __name__ == "__main__":
    unittest.main()
