"""WS-3: count_dp_seek reports the true marginal log p(value), not the tropical cost (HMM/mixture)."""

import unittest

import numpy as np

import pysp.stats as stats
from pysp.enumeration.density_rank import count_dp_seek


class CountDPSeekLogProbTest(unittest.TestCase):
    def _hmm(self):
        return stats.HiddenMarkovModelDistribution(
            [stats.CategoricalDistribution({"a": 0.7, "b": 0.3}), stats.CategoricalDistribution({"a": 0.2, "b": 0.8})],
            [0.6, 0.4],
            [[0.7, 0.3], [0.4, 0.6]],
            len_dist=stats.CategoricalDistribution({2: 0.5, 3: 0.5}),
            use_numba=False,
        )

    def test_hmm_seek_logprob_is_true_marginal(self):
        # for a marginal (logsumexp-over-paths) family the seek index is the tropical projection, but the
        # reported log_prob must be the true log p(value), not the dominant-path cost
        hmm = self._hmm()
        for i in range(6):
            r = count_dp_seek(hmm, i, oversample=64)
            self.assertAlmostEqual(r.log_prob, hmm.log_density(r.value), places=9)

    def test_mixture_seek_logprob_is_true_marginal(self):
        rng = np.random.RandomState(0)
        mix = stats.MixtureDistribution(
            [stats.IntegerCategoricalDistribution(0, list(rng.dirichlet(np.ones(8)))) for _ in range(3)],
            list(rng.dirichlet(np.ones(3))),
        )
        for i in range(6):
            r = count_dp_seek(mix, i, oversample=64)
            self.assertAlmostEqual(r.log_prob, mix.log_density(r.value), places=9)

    def test_exact_family_unchanged(self):
        comp = stats.CompositeDistribution(
            (stats.CategoricalDistribution({"x": 0.5, "y": 0.5}), stats.CategoricalDistribution({0: 0.3, 1: 0.7}))
        )
        for i in range(4):
            r = count_dp_seek(comp, i, oversample=32)
            self.assertAlmostEqual(r.log_prob, comp.log_density(r.value), places=9)


if __name__ == "__main__":
    unittest.main()
