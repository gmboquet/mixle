"""HMMPathIndex: quantized random-access unranking of HMM state paths (the count-DP companion to A*).

Contract verified against :func:`hmm_best_paths` (exact list-Viterbi) on fully-enumerable models: every
path is unranked exactly once with its exact joint log-probability (completeness), rank order follows the
quantized bucket order up to the documented T-floor smear, rank 0 is Viterbi up to that smear, and counts /
mass brackets bound the brute truth -- including with UNNORMALIZED (positive) emission log-likelihoods,
which the per-position score shift makes well-defined. Deep random access -- rank 1e9 of 6^12 paths -- is
one O(T*K) table walk, no enumeration.
"""

import math
import unittest

import numpy as np

from mixle.enumeration import HMMPathIndex, hmm_best_paths


def _norm_rows(a):
    e = np.exp(a)
    return np.log(e / e.sum(axis=-1, keepdims=True))


def _model(K, T, seed=0, emission_scale=1.5):
    rng = np.random.RandomState(seed)
    return (
        _norm_rows(rng.randn(K)),
        _norm_rows(rng.randn(K, K)),
        rng.randn(T, K) * emission_scale,
    )


def _joint(log_pi, log_A, log_b, path):
    lp = log_pi[path[0]] + log_b[0, path[0]]
    for t in range(1, len(path)):
        lp += log_A[path[t - 1], path[t]] + log_b[t, path[t]]
    return float(lp)


class AgreementWithAStarTest(unittest.TestCase):
    def setUp(self):
        self.K, self.T = 3, 5
        self.log_pi, self.log_A, self.log_b = _model(self.K, self.T, seed=0)
        self.idx = HMMPathIndex(self.log_pi, self.log_A, self.log_b, oversample=64)
        self.astar = list(hmm_best_paths(self.log_pi, self.log_A, self.log_b))

    def test_total_covers_every_path(self):
        total = self.idx.total()
        self.assertEqual(total.value, self.K**self.T)
        self.assertTrue(total.certified)
        self.assertEqual(len(self.astar), self.K**self.T)
        self.assertFalse(self.idx.truncated)

    def test_every_path_unranked_exactly_once_with_exact_logprob(self):
        mine = sorted((self.idx.unrank(i) for i in range(self.idx.total().value)), key=lambda u: u[0])
        exact = sorted(self.astar, key=lambda u: u[0])
        self.assertEqual([p for p, _ in mine], [p for p, _ in exact])
        for (_, lp_m), (_, lp_e) in zip(mine, exact):
            self.assertAlmostEqual(lp_m, lp_e, places=12)

    def test_unranked_logprobs_are_exact(self):
        for i in (0, 37, 121, 242):
            path, lp = self.idx.unrank(i)
            self.assertAlmostEqual(lp, _joint(self.log_pi, self.log_A, self.log_b, path), places=12)

    def test_buckets_nondecreasing_in_rank_up_to_smear(self):
        # the internal walk is ascending in the STRUCTURAL bucket; a true score's bucket_of sits within
        # T-1 fine buckets of it (sum-of-floors vs floor-of-sum), so rank order dips by at most that
        buckets = [self.idx.bucket_of(self.idx.unrank(i)[1]) for i in range(self.idx.total().value)]
        slack = self.T - 1
        self.assertTrue(all(buckets[i + 1] >= buckets[i] - slack for i in range(len(buckets) - 1)))

    def test_rank0_is_viterbi_up_to_smear(self):
        _vp, vlp = self.astar[0]
        _p0, lp0 = self.idx.unrank(0)
        self.assertLessEqual(lp0, vlp + 1e-12)
        self.assertLessEqual(self.idx.bucket_of(lp0) - self.idx.bucket_of(vlp), self.T - 1)

    def test_count_brackets_brute_truth(self):
        # structural buckets under-estimate bits by up to T floors -> count(thr) over-counts by at most
        # the paths within that smear band; verify both sides against the exact enumeration
        thr = self.astar[60][1]
        result = self.idx.count(thr)
        self.assertTrue(result.certified)  # non-truncated index: every query is exact (MXR-080-0230)
        n = result.value
        brute = sum(1 for _p, lp in self.astar if lp >= thr)
        smear = ((self.T + 2) / self.idx.quantizer.fine_per_bit()) * math.log(2.0)
        brute_hi = sum(1 for _p, lp in self.astar if lp >= thr - smear)
        self.assertGreaterEqual(n, brute)
        self.assertLessEqual(n, brute_hi)

    def test_mass_above_brackets_true_mass(self):
        thr = self.astar[40][1]
        true_mass = sum(math.exp(lp) for _p, lp in self.astar if lp >= thr)
        bound = self.idx.mass_above(thr)
        self.assertTrue(bound.certified)
        self.assertLessEqual(bound.lower, true_mass + 1e-12)
        # upper covers at least the true mass of the counted set (over-count only inflates it)
        self.assertGreaterEqual(bound.upper, true_mass - 1e-12)

    def test_threshold_and_iter(self):
        lp5 = self.idx.threshold(5)
        self.assertAlmostEqual(lp5, self.idx.unrank(4)[1])
        head = list(self.idx.iter_paths())[:4]
        self.assertEqual([p for p, _ in head], [self.idx.unrank(i)[0] for i in range(4)])


class DeepAccessTest(unittest.TestCase):
    def test_unrank_1e9_is_one_table_walk(self):
        K, T = 6, 12  # 6**12 ~ 2.18e9 paths: A* would need 1e9 expansions to reach this rank
        log_pi, log_A, log_b = _model(K, T, seed=1, emission_scale=1.0)
        idx = HMMPathIndex(log_pi, log_A, log_b, oversample=16)
        self.assertEqual(idx.total().value, K**T)
        path, lp = idx.unrank(10**9)
        self.assertEqual(len(path), T)
        self.assertAlmostEqual(lp, _joint(log_pi, log_A, log_b, path), places=10)
        # deeper rank -> deeper bucket (up to the T-floor smear, negligible at these separations)
        probes = [0, 10**3, 10**6, 10**9, 2 * 10**9]
        buckets = [idx.bucket_of(idx.unrank(i)[1]) for i in probes]
        self.assertTrue(all(buckets[i + 1] >= buckets[i] - (T - 1) for i in range(len(buckets) - 1)))

    def test_out_of_range(self):
        log_pi, log_A, log_b = _model(2, 3, seed=2)
        idx = HMMPathIndex(log_pi, log_A, log_b)
        with self.assertRaises(IndexError):
            idx.unrank(2**3)
        with self.assertRaises(IndexError):
            idx.unrank(-1)


class StructureEdgeCasesTest(unittest.TestCase):
    def test_impossible_transitions_are_excluded(self):
        K, T = 3, 4
        log_pi, log_A, log_b = _model(K, T, seed=3)
        log_A = log_A.copy()
        log_A[0, 1] = -np.inf  # forbid 0 -> 1
        idx = HMMPathIndex(log_pi, log_A, log_b, oversample=32)
        total = idx.total()
        self.assertTrue(total.certified)
        n = total.value
        self.assertLess(n, K**T)
        for i in range(n):
            path, lp = idx.unrank(i)
            self.assertTrue(np.isfinite(lp))
            for t in range(1, T):
                self.assertFalse(path[t - 1] == 0 and path[t] == 1)

    def test_budget_truncation_flag(self):
        log_pi, log_A, log_b = _model(3, 6, seed=4)
        idx = HMMPathIndex(log_pi, log_A, log_b, budget_bits=6.0)  # far too shallow for 3**6 paths
        self.assertTrue(idx.truncated)
        result = idx.total()
        # MXR-080-0230: truncated -> total() is only a certified LOWER bound, not the exact answer.
        self.assertFalse(result.certified)
        self.assertLess(result.value, 3**6)
        full = HMMPathIndex(log_pi, log_A, log_b)  # default budget covers everything
        self.assertFalse(full.truncated)
        full_result = full.total()
        self.assertTrue(full_result.certified)
        self.assertEqual(full_result.value, 3**6)

    def test_single_position_model(self):
        log_pi, log_A, log_b = _model(4, 1, seed=5)
        idx = HMMPathIndex(log_pi, log_A, log_b, oversample=64)
        self.assertEqual(idx.total().value, 4)
        paths = {idx.unrank(i)[0] for i in range(4)}
        self.assertEqual(paths, {(s,) for s in range(4)})


class RankValidationTestCase(unittest.TestCase):
    """MXR-080-0208: unrank/threshold/iter_paths must reject booleans, fractional numbers, and
    non-finite values instead of comparing the original value and then silently truncating it
    with int()."""

    def setUp(self):
        self.log_pi, self.log_A, self.log_b = _model(3, 4, seed=7)
        self.idx = HMMPathIndex(self.log_pi, self.log_A, self.log_b, oversample=32)

    def test_unrank_rejects_fractional_rank(self):
        # The audit's exact example: unrank(0.5) must not silently return rank zero (i.e. behave
        # as though int(0.5) == 0 were a legitimate request for "the item at rank 0").
        with self.assertRaises(TypeError):
            self.idx.unrank(0.5)
        self.idx.unrank(0)  # rank 0 itself is, of course, valid -- proves this is real validation

    def test_unrank_rejects_whole_valued_float_rank(self):
        with self.assertRaises(TypeError):
            self.idx.unrank(1.0)

    def test_unrank_rejects_bool_rank(self):
        with self.assertRaises(TypeError):
            self.idx.unrank(True)
        with self.assertRaises(TypeError):
            self.idx.unrank(False)

    def test_unrank_rejects_non_finite_rank(self):
        with self.assertRaises(TypeError):
            self.idx.unrank(float("nan"))
        with self.assertRaises(TypeError):
            self.idx.unrank(float("inf"))

    def test_threshold_rejects_fractional_rank(self):
        with self.assertRaises(TypeError):
            self.idx.threshold(1.5)

    def test_threshold_rejects_bool_rank(self):
        with self.assertRaises(TypeError):
            self.idx.threshold(True)

    def test_iter_paths_rejects_fractional_start(self):
        with self.assertRaises(TypeError):
            list(self.idx.iter_paths(0.5))

    def test_iter_paths_rejects_bool_start(self):
        with self.assertRaises(TypeError):
            list(self.idx.iter_paths(True))

    def test_valid_integer_ranks_still_work(self):
        # Negative control: legitimate int and numpy-int ranks are unaffected.
        p0, lp0 = self.idx.unrank(0)
        p1, lp1 = self.idx.unrank(np.int64(1))
        self.assertNotEqual(p0, p1)
        self.assertAlmostEqual(self.idx.threshold(1), lp0)
        head = list(self.idx.iter_paths(0))[:2]
        self.assertEqual([p for p, _ in head], [p0, p1])


class ModelValidationTest(unittest.TestCase):
    """MXR-080-0228: hmm_best_paths/HMMPathIndex must validate a coherent model -- shapes, the
    finite-or-impossible score contract, and the normalized-probability contract -- before running
    any dynamic program, instead of failing with an opaque broadcast error (mismatched shapes) or
    crashing on a zero-size numpy reduction (an all-impossible initial state)."""

    def setUp(self):
        self.K, self.T = 2, 3
        self.log_pi = np.log(np.array([0.5, 0.5]))
        self.log_A = np.log(np.array([[0.5, 0.5], [0.5, 0.5]]))
        self.log_b = np.zeros((self.T, self.K))

    # -- shape mismatches: the audit's "opaque broadcast error" --------------------------------------

    def test_hmm_path_index_rejects_mismatched_log_pi_length(self):
        # numpy's own broadcast error is ALSO a ValueError, so pin the message too: this must be
        # MY clean "shapes disagree" validation, not an opaque broadcast failure deeper in the
        # pipeline (the audit's literal complaint).
        bad_log_pi = np.zeros(self.K + 1)
        with self.assertRaises(ValueError) as cm:
            HMMPathIndex(bad_log_pi, self.log_A, self.log_b)
        self.assertIn("disagree", str(cm.exception))
        self.assertNotIn("broadcast", str(cm.exception))

    def test_hmm_best_paths_rejects_mismatched_log_pi_length(self):
        bad_log_pi = np.zeros(self.K + 1)
        with self.assertRaises(ValueError):
            list(hmm_best_paths(bad_log_pi, self.log_A, self.log_b))

    def test_hmm_best_paths_rejects_oversized_log_pi_not_just_undersized(self):
        # A log_pi LONGER than K must be rejected outright, not silently accepted while the extra
        # state is ignored (the pre-fix behavior for this specific direction of mismatch: n_states
        # came from log_b's shape, so successors() indexed only the first K entries of log_pi).
        bad_log_pi = np.concatenate([self.log_pi, [-np.inf]])
        with self.assertRaises(ValueError):
            list(hmm_best_paths(bad_log_pi, self.log_A, self.log_b))

    def test_hmm_path_index_rejects_non_square_log_A(self):
        bad_log_A = np.zeros((self.K, self.K + 1))
        with self.assertRaises(ValueError) as cm:
            HMMPathIndex(self.log_pi, bad_log_A, self.log_b)
        self.assertIn("square", str(cm.exception))
        self.assertNotIn("broadcast", str(cm.exception))

    def test_hmm_path_index_rejects_log_A_wrong_K(self):
        bad_log_A = _norm_rows(np.random.RandomState(9).randn(self.K + 1, self.K + 1))
        with self.assertRaises(ValueError) as cm:
            HMMPathIndex(self.log_pi, bad_log_A, self.log_b)
        self.assertIn("disagree", str(cm.exception))
        self.assertNotIn("broadcast", str(cm.exception))

    def test_hmm_path_index_rejects_wrong_log_b_ndim(self):
        with self.assertRaises(ValueError) as cm:
            HMMPathIndex(self.log_pi, self.log_A, self.log_b.ravel())
        self.assertIn("2-D", str(cm.exception))
        self.assertNotIn("unpack", str(cm.exception))

    def test_shape_validation_does_not_depend_on_multi_position_loops_running(self):
        # T=1 bypasses _backward_viterbi's loop entirely (range(t_len - 2, -1, -1) is empty), so the
        # shape check must not be smuggled in only via code that a single-position model skips.
        bad_log_pi = np.zeros(self.K + 2)
        log_b_1 = np.zeros((1, self.K))
        with self.assertRaises(ValueError):
            HMMPathIndex(bad_log_pi, self.log_A, log_b_1)
        with self.assertRaises(ValueError):
            list(hmm_best_paths(bad_log_pi, self.log_A, log_b_1))

    # -- finite-or-impossible numeric contract --------------------------------------------------------

    def test_rejects_nan_in_log_pi(self):
        bad = self.log_pi.copy()
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            HMMPathIndex(bad, self.log_A, self.log_b)

    def test_rejects_nan_in_log_A(self):
        bad = self.log_A.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            HMMPathIndex(self.log_pi, bad, self.log_b)

    def test_rejects_nan_in_log_b(self):
        bad = self.log_b.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            HMMPathIndex(self.log_pi, self.log_A, bad)

    def test_rejects_positive_infinity_in_log_b(self):
        # +inf is not finite-or-impossible even though log_b may otherwise be positive (a density).
        bad = self.log_b.copy()
        bad[0, 0] = np.inf
        with self.assertRaises(ValueError):
            HMMPathIndex(self.log_pi, self.log_A, bad)

    def test_rejects_positive_log_pi_entry(self):
        bad = np.array([0.5, -1.0])  # exp(0.5) > 1: not a valid probability
        with self.assertRaises(ValueError):
            HMMPathIndex(bad, self.log_A, self.log_b)

    def test_rejects_super_stochastic_log_A_row(self):
        bad = np.zeros((self.K, self.K))  # every row sums to K units of probability, not <= 1
        with self.assertRaises(ValueError):
            HMMPathIndex(self.log_pi, bad, self.log_b)

    def test_positive_log_b_entry_is_allowed(self):
        # log_b is an emission LOG-LIKELIHOOD (may be an unnormalized/continuous density > 1) --
        # this must NOT be rejected the way a positive log_pi/log_A entry is.
        ok = self.log_b.copy()
        ok[0, 0] = 3.0
        HMMPathIndex(self.log_pi, self.log_A, ok)  # must not raise

    def test_sub_stochastic_log_A_row_is_allowed(self):
        # The established "forbid one transition without renormalizing the row" pattern (see
        # StructureEdgeCasesTest.test_impossible_transitions_are_excluded) must keep working: a row
        # summing to LESS than 1 is not the same violation as summing to MORE than 1.
        ok = self.log_A.copy()
        ok[0, 1] = -np.inf
        idx = HMMPathIndex(self.log_pi, ok, self.log_b)  # must not raise
        self.assertTrue(idx.total().certified)

    # -- empty support: represented explicitly, never crashed ------------------------------------------

    def test_all_impossible_initial_state_is_represented_not_crashed(self):
        bad_log_pi = np.array([-np.inf, -np.inf])
        idx = HMMPathIndex(bad_log_pi, self.log_A, self.log_b)  # must not raise/crash
        self.assertTrue(idx.empty_support)
        result = idx.total()
        self.assertEqual(result.value, 0)
        self.assertTrue(result.certified)  # provably zero paths, not merely "none found in budget"
        with self.assertRaises(IndexError):
            idx.unrank(0)

    def test_all_impossible_position_mid_sequence_is_represented_not_crashed(self):
        # Not just the INITIAL position: any position with no finite score at all (every state's
        # emission likelihood is -inf there) makes the whole model empty-support the same way.
        bad_log_b = self.log_b.copy()
        bad_log_b[1] = -np.inf
        idx = HMMPathIndex(self.log_pi, self.log_A, bad_log_b)  # must not raise/crash
        self.assertTrue(idx.empty_support)
        self.assertEqual(idx.total().value, 0)

    def test_normal_model_has_no_empty_support(self):
        idx = HMMPathIndex(self.log_pi, self.log_A, self.log_b)
        self.assertFalse(idx.empty_support)
        self.assertEqual(idx.total().value, self.K**self.T)


class ExactCountPrecisionTest(unittest.TestCase):
    """MXR-080-0229: path counts and cumulative offsets must stay exact past float64's 2**53
    exact-integer range, or total()/unrank() silently disagree about the support -- a rank total()
    claims exists gets rejected -- and adjacent deep ranks can alias to the same stored offset."""

    def test_total_and_unrank_agree_past_2_pow_53(self):
        # The audit's own example: 2 states, 54 positions -> 2**54 paths, comfortably past 2**53.
        K, T = 2, 54
        log_pi = np.log(np.array([0.5, 0.5]))
        log_A = np.log(np.array([[0.5, 0.5], [0.5, 0.5]]))
        log_b = np.zeros((T, K))
        idx = HMMPathIndex(log_pi, log_A, log_b)
        result = idx.total()
        self.assertEqual(result.value, 2**54)
        self.assertIsInstance(result.value, int)
        self.assertTrue(result.certified)
        self.assertFalse(idx.truncated)
        # the audit's exact failure mode: unrank(total - 1) must succeed, not be rejected as out of range
        path, lp = idx.unrank(2**54 - 1)
        self.assertEqual(len(path), T)
        self.assertTrue(np.isfinite(lp))
        # one rank past the true end must still be correctly rejected
        with self.assertRaises(IndexError):
            idx.unrank(2**54)

    def test_adjacent_deep_ranks_never_alias_to_the_same_path(self):
        # Every one of the 2**54 paths ties exactly in this symmetric model, so all of them land in
        # the SAME quantized bucket -- the entire burden of distinguishing ranks falls on the
        # internal offset-consumption walk, exactly what float64 accumulation could silently corrupt
        # past 2**53 (two distinct integer ranks rounding to the same float64 offset/count).
        K, T = 2, 54
        log_pi = np.log(np.array([0.5, 0.5]))
        log_A = np.log(np.array([[0.5, 0.5], [0.5, 0.5]]))
        log_b = np.zeros((T, K))
        idx = HMMPathIndex(log_pi, log_A, log_b)
        probes = [
            2**53 - 2,
            2**53 - 1,
            2**53,
            2**53 + 1,
            2**53 + 2,
            2**54 - 5,
            2**54 - 4,
            2**54 - 3,
            2**54 - 2,
            2**54 - 1,
        ]
        paths = [idx.unrank(i)[0] for i in probes]
        self.assertEqual(len(set(paths)), len(probes), "adjacent deep ranks collapsed onto the same path")

    def test_total_value_is_exact_python_int_for_a_different_base_too(self):
        K, T = 3, 40  # 3**40 ~ 1.2e19, also far past 2**53
        log_pi = np.log(np.full(K, 1.0 / K))
        log_A = np.log(np.full((K, K), 1.0 / K))
        log_b = np.zeros((T, K))
        idx = HMMPathIndex(log_pi, log_A, log_b)
        result = idx.total()
        self.assertEqual(result.value, K**T)
        self.assertIsInstance(result.value, int)

    def test_count_is_exact_past_2_pow_53(self):
        K, T = 2, 54
        log_pi = np.log(np.array([0.5, 0.5]))
        log_A = np.log(np.array([[0.5, 0.5], [0.5, 0.5]]))
        log_b = np.zeros((T, K))
        idx = HMMPathIndex(log_pi, log_A, log_b)
        # an arbitrarily permissive threshold must count every single path exactly, not a
        # float64-rounded approximation of 2**54.
        result = idx.count(-1e12)
        self.assertEqual(result.value, 2**54)
        self.assertTrue(result.certified)


class TruncatedCertificationTest(unittest.TestCase):
    """MXR-080-0230: total()/count()/mass_above() on a truncated index must carry an explicit
    certified/uncertified marker instead of returning a plain number indistinguishable from a
    complete answer."""

    def setUp(self):
        # Asymmetric 2-state, 2-position model: 4 total paths spread across multiple quantized
        # buckets. A perfectly symmetric model ties every path into bucket 0 and never truncates
        # under a shallow budget, which would defeat the point of this test.
        rng = np.random.RandomState(0)
        self.K, self.T = 2, 2
        self.log_pi = _norm_rows(rng.randn(self.K) * 3.0)
        self.log_A = _norm_rows(rng.randn(self.K, self.K) * 3.0)
        self.log_b = np.zeros((self.T, self.K))
        self.brute = list(hmm_best_paths(self.log_pi, self.log_A, self.log_b))
        self.assertEqual(len(self.brute), 4)  # sanity: K**T == 4 valid paths

    def test_truncated_total_is_uncertified_lower_bound(self):
        idx = HMMPathIndex(self.log_pi, self.log_A, self.log_b, budget_bits=0.0)
        self.assertTrue(idx.truncated)
        result = idx.total()
        self.assertFalse(result.certified)
        self.assertLessEqual(result.value, len(self.brute))
        self.assertGreaterEqual(result.value, 1)  # the Viterbi path itself is always within budget

    def test_arbitrarily_low_threshold_does_not_silently_report_the_complete_count(self):
        # The audit's exact scenario: a truncated index queried with a threshold low enough that
        # EVERY path qualifies must not silently return the in-budget total as if it answered the
        # full question -- it must say so via certified=False, and value must not be mistaken for
        # the true count of qualifying paths.
        idx = HMMPathIndex(self.log_pi, self.log_A, self.log_b, budget_bits=0.0)
        self.assertTrue(idx.truncated)
        result = idx.count(-1e9)
        self.assertFalse(result.certified)
        self.assertLess(result.value, len(self.brute))  # under-counts vs. the true 4 valid paths

    def test_certified_when_query_stays_within_the_built_budget(self):
        # A shallow query (the Viterbi path's own threshold) on a truncated index is still exactly
        # answerable -- certification tracks the QUERY's own reach, not merely the global flag.
        idx = HMMPathIndex(self.log_pi, self.log_A, self.log_b, budget_bits=0.1, oversample=8)
        self.assertTrue(idx.truncated)
        best_lp = max(lp for _p, lp in self.brute)
        result = idx.count(best_lp)
        self.assertTrue(result.certified)
        self.assertEqual(result.value, 1)

    def test_mass_above_upper_becomes_inf_when_uncertified(self):
        idx = HMMPathIndex(self.log_pi, self.log_A, self.log_b, budget_bits=0.0)
        self.assertTrue(idx.truncated)
        bound = idx.mass_above(-1e9)
        self.assertFalse(bound.certified)
        self.assertEqual(bound.upper, math.inf)
        # lower remains a sound (if loose) bound: it must not exceed the true total mass
        true_mass = sum(math.exp(lp) for _p, lp in self.brute)
        self.assertLessEqual(bound.lower, true_mass + 1e-9)

    def test_non_truncated_index_is_always_certified(self):
        idx = HMMPathIndex(self.log_pi, self.log_A, self.log_b)  # default budget: no truncation
        self.assertFalse(idx.truncated)
        self.assertTrue(idx.total().certified)
        self.assertTrue(idx.count(-1e9).certified)
        mass = idx.mass_above(-1e9)
        self.assertTrue(mass.certified)
        self.assertNotEqual(mass.upper, math.inf)


if __name__ == "__main__":
    unittest.main()
