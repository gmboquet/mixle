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


class LatticeCalibrationValidationTest(unittest.TestCase):
    """MXR-080-0231: LatticeEnvelopeIndex calibration must reject impossible sampling
    configurations instead of silently indexing an empty prefixes[0], dividing by zero in the
    default cluster function, or crashing deep inside _calibrate with an unclear numpy error."""

    def _ar(self, V=4, L=3, seed=0):
        return AutoregressiveEnumerable(_markov_model(V, L, seed=seed), max_len=L)

    def test_zero_n_clusters_rejected_not_a_delayed_zerodivisionerror(self):
        with self.assertRaises(ValueError):
            LatticeEnvelopeIndex(self._ar(), n_clusters=0, n_paths=4, seed=0)

    def test_negative_n_clusters_rejected(self):
        with self.assertRaises(ValueError):
            LatticeEnvelopeIndex(self._ar(), n_clusters=-1, n_paths=4, seed=0)

    def test_fractional_n_clusters_rejected(self):
        with self.assertRaises(ValueError):
            LatticeEnvelopeIndex(self._ar(), n_clusters=2.5, n_paths=4, seed=0)

    def test_zero_n_paths_rejected_not_a_prefixes_indexerror(self):
        with self.assertRaises(ValueError):
            LatticeEnvelopeIndex(self._ar(), n_clusters=2, n_paths=0, seed=0)

    def test_fractional_n_paths_rejected(self):
        with self.assertRaises(ValueError):
            LatticeEnvelopeIndex(self._ar(), n_clusters=2, n_paths=2.9, seed=0)

    def test_non_finite_budget_bits_rejected(self):
        with self.assertRaises(ValueError):
            LatticeEnvelopeIndex(self._ar(), n_clusters=2, n_paths=4, seed=0, budget_bits=float("inf"))

    def test_non_finite_depth_bits_rejected_by_ensure_bits(self):
        lat = LatticeEnvelopeIndex(self._ar(), n_clusters=2, n_paths=4, seed=0)
        with self.assertRaises(ValueError):
            lat.ensure_bits(float("nan"))

    def test_empty_step_distribution_raises_clear_error(self):
        def empty_branch_model(prefix):
            if len(prefix) > 0 and prefix[-1] == 1:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
            return np.array([0, 1]), np.log(np.array([0.5, 0.5]))

        ar = AutoregressiveEnumerable(empty_branch_model, max_len=3)
        with self.assertRaises(ValueError) as ctx:
            LatticeEnvelopeIndex(ar, n_clusters=2, n_paths=8, seed=0)
        self.assertIn("non-empty step distribution", str(ctx.exception))

    def test_well_formed_construction_still_works(self):
        lat = LatticeEnvelopeIndex(self._ar(), n_clusters=2, n_paths=4, seed=0)
        self.assertGreater(lat.total(), 0.0)


class LatticeRankPrecisionTest(unittest.TestCase):
    """MXR-080-0232: same exact-rank-arithmetic and fixed-length-support requirements as
    AREnvelopeIndex (see envelope_index_test.RankPrecisionTest), checked against
    LatticeEnvelopeIndex's own unrank/threshold/rank_bracket."""

    def _tied_bucket_index(self, V=2, L=60, n_paths=2):
        # scale=0.0: every logit is exactly 0, so the distribution is uniform regardless of
        # context -- every one of the 2**L sequences quantizes into the SAME fine bucket, an exact
        # combinatorial count of 2**60 (verified: lattice.total() == float(2**60) exactly).
        ar = AutoregressiveEnumerable(_markov_model(V, L, seed=0, scale=0.0), max_len=L)
        return LatticeEnvelopeIndex(ar, cluster_fn=lambda t: t, n_paths=n_paths, seed=0, budget_bits=64.0)

    def test_total_and_unrank_agree_past_2_pow_53(self):
        lat = self._tied_bucket_index()
        self.assertEqual(lat.total(), float(2**60))
        seq, lp = lat.unrank(2**60 - 1)
        self.assertEqual(len(seq), 60)
        self.assertTrue(np.isfinite(lp))
        with self.assertRaises(IndexError):
            lat.unrank(2**60)

    def test_adjacent_deep_ranks_never_alias_to_the_same_sequence(self):
        lat = self._tied_bucket_index()
        probes = [2**53 - 2, 2**53 - 1, 2**53, 2**53 + 1, 2**59, 2**60 - 2, 2**60 - 1]
        seqs = [lat.unrank(p)[0] for p in probes]
        self.assertEqual(len(set(seqs)), len(probes), "adjacent deep ranks collapsed onto the same sequence")

    def test_unrank_rejects_non_integer_rank(self):
        lat = LatticeEnvelopeIndex(
            AutoregressiveEnumerable(_markov_model(4, 2, seed=0), max_len=2), n_clusters=2, n_paths=4, seed=0
        )
        for bad in (1.5, "5", None):
            with self.assertRaises(TypeError):
                lat.unrank(bad)

    def test_threshold_rejects_non_integer_rank(self):
        lat = LatticeEnvelopeIndex(
            AutoregressiveEnumerable(_markov_model(4, 2, seed=0), max_len=2), n_clusters=2, n_paths=4, seed=0
        )
        with self.assertRaises(TypeError):
            lat.threshold(1.5)

    def test_rank_bracket_rejects_wrong_length_sequence(self):
        ar = AutoregressiveEnumerable(_markov_model(4, 5, seed=0), max_len=5)
        lat = LatticeEnvelopeIndex(ar, cluster_fn=lambda t: t, n_paths=4, seed=0)
        full_seq, _lp = lat.unrank(0)
        with self.assertRaises(ValueError):
            lat.rank_bracket(full_seq[:2])
        with self.assertRaises(ValueError):
            lat.rank_bracket(full_seq + (0,))
        lo, hi = lat.rank_bracket(full_seq)  # negative control: the right length still works
        self.assertLessEqual(lo, hi)


if __name__ == "__main__":
    unittest.main()
