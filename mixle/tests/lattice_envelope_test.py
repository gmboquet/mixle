"""LatticeEnvelopeIndex: the cluster-conditioned (Markov) refinement of the mean-field envelope.

The contract under test: with ``cluster_fn`` the identity, the lattice envelope is EXACT for any model
whose next-token distribution depends only on (depth, last token) -- verified value-for-value against the
exact SeekIndex on such a Markov model, on exactly the counts where the mean-field envelope is measurably
lossy (the interpolation claim: m=1 mean-field < m=V lattice = exact for order-1 models). Deep random
access stays O(L) forwards; returned log-probabilities are always exact.
"""

import math
import unittest

import numpy as np

from mixle.enumeration import AREnvelopeIndex, AutoregressiveEnumerable, LatticeEnvelopeIndex, SeekIndex


def _markov_model(V, L, seed=0, scale=1.5):
    """Next-token distribution depends ONLY on (depth, last token): the lattice's exactness class."""
    W = np.random.RandomState(seed).randn(L, V, V) * scale

    def nlp(prefix):
        d = len(prefix)
        last = prefix[-1] if prefix else 0
        lg = W[d, last]
        m = np.max(lg)
        return np.arange(V), lg - (m + math.log(np.sum(np.exp(lg - m))))

    return nlp


class MarkovExactnessTest(unittest.TestCase):
    def setUp(self):
        self.V, self.L = 5, 4
        self.nlp = _markov_model(self.V, self.L, seed=2)
        self.ar = AutoregressiveEnumerable(self.nlp, max_len=self.L)
        # enough paths that every (depth, last-token) pocket is visited for V=5
        self.lattice = LatticeEnvelopeIndex(self.ar, cluster_fn=lambda t: t, n_paths=400, seed=0)
        self.meanfield = AREnvelopeIndex(self.ar, n_paths=400, seed=0)
        self.exact = SeekIndex(self.ar)
        self.exact.ensure_bits(40.0)

    def test_lattice_exact_where_meanfield_is_lossy(self):
        # compare count(thr) against the exact index at several thresholds: the lattice must be exact
        # (within fp noise), the mean-field must show real error somewhere -- the interpolation claim
        lattice_err = meanfield_err = 0.0
        for rank in (30, 120, 300, 520):
            thr = self.exact.unrank(rank)[1]
            true_n = float(self.exact.count(thr))
            lattice_err = max(lattice_err, abs(self.lattice.count(thr) - true_n))
            meanfield_err = max(meanfield_err, abs(self.meanfield.count(thr) - true_n))
        self.assertLess(lattice_err, 1e-6)
        self.assertGreater(meanfield_err, lattice_err + 0.5)  # mean-field measurably lossy on Markov data

    def test_total_is_exact(self):
        self.assertAlmostEqual(self.lattice.total(), float(self.V**self.L), places=6)

    def test_unrank_lands_in_true_rank_bucket_with_exact_logprobs(self):
        q = self.lattice.quantizer
        for i in (0, 7, 100, 400):
            seq, lp = self.lattice.unrank(i)
            self.assertEqual(len(seq), self.L)
            self.assertAlmostEqual(lp, self.ar.log_density(seq), places=12)
            self.assertEqual(q.fine_bucket(lp), q.fine_bucket(self.exact.unrank(i)[1]), f"rank {i}")

    def test_rank_bracket_is_exact(self):
        for i in (0, 40, 250):
            seq, _lp = self.exact.unrank(i)
            lo, hi = self.lattice.rank_bracket(seq)
            self.assertLessEqual(lo, i + 1e-6)
            self.assertGreaterEqual(hi + 1e-6, i)


class BehaviorTest(unittest.TestCase):
    def test_deep_unrank_is_o_l_forwards(self):
        V, L = 30, 10  # 30**10 ~ 6e14 sequences
        ar = AutoregressiveEnumerable(_markov_model(V, L, seed=3, scale=2.0), max_len=L)
        lattice = LatticeEnvelopeIndex(ar, cluster_fn=lambda t: t, n_paths=64, seed=0, budget_bits=60.0)
        seq, lp = lattice.unrank(10**12)
        self.assertEqual(len(seq), L)
        self.assertAlmostEqual(lp, ar.log_density(seq), places=10)
        lo, hi = lattice.rank_bracket(seq)
        self.assertLessEqual(lo, 1e12)
        self.assertGreaterEqual(hi, 1e12 * 0.2)  # self-consistency of the lattice's own coordinate
        self.assertLess(len(ar._cache), 64 * L * 3)  # calibration-bounded forwards, not count-bounded

    def test_coarse_clusters_still_work(self):
        # m=2 clusters on a Markov model: an estimate between mean-field and exact -- must run and be sane
        ar = AutoregressiveEnumerable(_markov_model(6, 3, seed=4), max_len=3)
        lattice = LatticeEnvelopeIndex(ar, n_clusters=2, n_paths=200, seed=0)
        self.assertAlmostEqual(lattice.total(), 6.0**3, delta=6.0**3 * 0.2)
        seq, lp = lattice.unrank(50)
        self.assertAlmostEqual(lp, ar.log_density(seq), places=12)

    def test_validation(self):
        ar = AutoregressiveEnumerable(_markov_model(4, 2, seed=5), max_len=2)
        with self.assertRaises(ValueError):
            LatticeEnvelopeIndex(ar)  # neither cluster_fn nor n_clusters

        def nlp(prefix):
            return np.arange(3), np.log(np.array([0.5, 0.3, 0.2]))

        with self.assertRaises(ValueError):
            LatticeEnvelopeIndex(AutoregressiveEnumerable(nlp, eos=2), n_clusters=2)  # terminating

    def test_length_one_model(self):
        ar = AutoregressiveEnumerable(_markov_model(5, 1, seed=6), max_len=1)
        lattice = LatticeEnvelopeIndex(ar, cluster_fn=lambda t: t, n_paths=4, seed=0)
        self.assertEqual(int(lattice.total()), 5)
        seqs = {lattice.unrank(i)[0] for i in range(5)}
        self.assertEqual(len(seqs), 5)


if __name__ == "__main__":
    unittest.main()
