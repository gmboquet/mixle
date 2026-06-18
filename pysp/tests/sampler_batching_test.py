"""Batched combinator sampling must match the legacy per-draw loop exactly.

The fast (``batched=True``) path draws each child RNG stream in one vectorized call.
Because every child sampler owns an independent ``RandomState``, batching consumes
each stream in the same order as the legacy loop, so the draws are byte-identical to
``batched=False`` for the same seed. These tests pin that guarantee across numeric,
labelled, structured, and nested components.
"""

import unittest

import numpy as np

from pysp.stats.combinator.composite import CompositeDistribution
from pysp.stats.combinator.sequence import SequenceDistribution
from pysp.stats.latent.mixture import MixtureDistribution
from pysp.stats.leaf.categorical import CategoricalDistribution
from pysp.stats.leaf.gaussian import GaussianDistribution
from pysp.stats.leaf.poisson import PoissonDistribution
from pysp.stats.multivariate.mvn import MultivariateGaussianDistribution


class SamplerBatchingParityTestCase(unittest.TestCase):
    @staticmethod
    def _norm(x):
        # normalize numpy scalar wrappers (np.float64 vs float) so equality is value-based;
        # recurse through nested lists/tuples. Batched draws are bit-identical in value but may
        # be wrapped as np.float64 where the legacy per-draw path returned a Python float.
        if isinstance(x, (list, tuple)):
            return [SamplerBatchingParityTestCase._norm(e) for e in x]
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.integer):
            return int(x)
        return x

    def _assert_same(self, a, b):
        # element-wise value equality over arbitrarily nested lists/tuples/scalars (exact: same RNG stream)
        self.assertEqual(self._norm(a), self._norm(b))

    def test_mixture_numeric_parity(self):
        m = MixtureDistribution([GaussianDistribution(-2.0, 1.0), GaussianDistribution(3.0, 0.5)], [0.4, 0.6])
        fast = m.sampler(7).sample(size=5000, batched=True)
        slow = m.sampler(7).sample(size=5000, batched=False)
        self._assert_same(fast, slow)

    def test_mixture_labelled_parity(self):
        m = MixtureDistribution(
            [CategoricalDistribution({"a": 0.7, "b": 0.3}), CategoricalDistribution({"a": 0.1, "c": 0.9})],
            [0.5, 0.5],
        )
        fast = m.sampler(11).sample(size=3000, batched=True)
        slow = m.sampler(11).sample(size=3000, batched=False)
        self._assert_same(fast, slow)

    def test_mixture_structured_parity(self):
        # composite components -> tuple outputs exercise the object-scatter path
        comp = lambda mu: CompositeDistribution((GaussianDistribution(mu, 1.0), PoissonDistribution(mu + 5.0)))
        m = MixtureDistribution([comp(0.0), comp(4.0)], [0.5, 0.5])
        fast = m.sampler(3).sample(size=2000, batched=True)
        slow = m.sampler(3).sample(size=2000, batched=False)
        self._assert_same(fast, slow)

    def test_mixture_multivariate_parity(self):
        # vector-valued (ndarray) leaves exercise the flat-array scatter path, which must
        # carry the trailing sample dimension rather than assume scalar draws
        cov = [[1.0, 0.0], [0.0, 1.0]]
        m = MixtureDistribution(
            [MultivariateGaussianDistribution([0.0, 0.0], cov), MultivariateGaussianDistribution([5.0, 5.0], cov)],
            [0.4, 0.6],
        )
        fast = m.sampler(17).sample(size=2000, batched=True)
        slow = m.sampler(17).sample(size=2000, batched=False)
        self.assertEqual(np.shape(fast[0]), (2,))
        self.assertTrue(np.array_equal(np.asarray(fast), np.asarray(slow)))

    def test_sequence_parity(self):
        s = SequenceDistribution(GaussianDistribution(0.0, 1.0), len_dist=PoissonDistribution(8.0))
        fast = s.sampler(5).sample(size=2000, batched=True)
        slow = s.sampler(5).sample(size=2000, batched=False)
        self._assert_same(fast, slow)

    def test_sequence_single_parity(self):
        s = SequenceDistribution(GaussianDistribution(0.0, 1.0), len_dist=PoissonDistribution(8.0))
        self._assert_same(s.sampler(9).sample(batched=True), s.sampler(9).sample(batched=False))

    def test_nested_mixture_of_sequences_parity(self):
        s0 = SequenceDistribution(GaussianDistribution(-1.0, 1.0), len_dist=PoissonDistribution(5.0))
        s1 = SequenceDistribution(GaussianDistribution(2.0, 0.5), len_dist=PoissonDistribution(9.0))
        m = MixtureDistribution([s0, s1], [0.3, 0.7])
        fast = m.sampler(13).sample(size=1500, batched=True)
        slow = m.sampler(13).sample(size=1500, batched=False)
        self._assert_same(fast, slow)

    def test_sequence_lengths_preserved(self):
        s = SequenceDistribution(GaussianDistribution(0.0, 1.0), len_dist=PoissonDistribution(8.0))
        fast = s.sampler(5).sample(size=500, batched=True)
        slow = s.sampler(5).sample(size=500, batched=False)
        self.assertEqual([len(x) for x in fast], [len(x) for x in slow])


if __name__ == "__main__":
    unittest.main()
