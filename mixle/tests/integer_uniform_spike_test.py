"""Regression test: IntegerUniformSpikeEstimator.estimate() computed each candidate spike
location's log-likelihood as (count - count_vec[i]) * log1p(-p_vec[i]) + count_vec[i] * log(p_vec[i]).
Whenever a candidate had zero observations (count_vec[i] == 0) or captured every observation
(count_vec[i] == count), one of the two terms multiplied a real zero against log(0) == -inf,
producing NaN. np.argmax does not skip NaN -- it latches onto the first NaN it encounters and
never releases it, so estimate() would silently return the first unobserved candidate as the
fitted spike location instead of the true maximum-likelihood one, with no error raised.
"""

import unittest
import warnings

import numpy as np

from mixle.stats.univariate.discrete.integer_uniform_spike import IntegerUniformSpikeEstimator


def _fit(min_val, max_val, observations, weight=1.0, pseudo_count=None, suff_stat=None):
    est = IntegerUniformSpikeEstimator(min_val=min_val, max_val=max_val, pseudo_count=pseudo_count, suff_stat=suff_stat)
    acc = est.accumulator_factory().make()
    for x in observations:
        acc.update(x, weight, None)
    return est.estimate(len(observations) * weight, acc.value())


class IntegerUniformSpikeEstimateArgmaxTestCase(unittest.TestCase):
    def test_all_mass_on_a_candidate_with_unobserved_neighbors(self):
        # 5 candidates (values 0..4), all 5 observations land on value 1: an unambiguous MLE fit
        # of k=1, p=1.0. Values 0, 2, 3, 4 are unobserved (count_vec[i] == 0), the exact condition
        # that used to poison ll[i] with NaN for i < 1 and make argmax stop there.
        fit = _fit(min_val=0, max_val=4, observations=[1, 1, 1, 1, 1])
        self.assertEqual(fit.k, 1)
        self.assertEqual(fit.p, 1.0)

    def test_no_runtime_warning_when_some_candidates_are_unobserved(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _fit(min_val=0, max_val=4, observations=[1, 1, 1, 1, 1])

    def test_correct_argmax_with_partial_zero_counts(self):
        # value 0 gets 5 observations, value 2 gets 3, values 1/3/4 unobserved. The true winner
        # (index 0) must beat both the unobserved candidates and the runner-up (index 2); verify
        # against an independently computed reference log-likelihood rather than a hand pick.
        fit = _fit(min_val=0, max_val=4, observations=[0, 0, 0, 0, 0, 2, 2, 2])
        count_vec = np.array([5.0, 0.0, 3.0, 0.0, 0.0])
        n = count_vec.sum()
        m = len(count_vec)

        def reference_ll(i):
            c = count_vec[i]
            non_spike = n - c
            term_a = 0.0 if non_spike == 0.0 else non_spike * (np.log1p(-c / n) - np.log(m - 1))
            term_b = 0.0 if c == 0.0 else c * np.log(c / n)
            return term_a + term_b

        expected_k = int(np.argmax([reference_ll(i) for i in range(m)]))
        self.assertEqual(fit.k, expected_k)
        self.assertEqual(expected_k, 0)

    def test_pseudo_count_branch_suff_stat_partial(self):
        # self.suff_stat = (k_pseudo, None) branch: pseudo_count only touches slot k_pseudo=0, so
        # slots 2/3/4 stay at raw zero count and still trigger the NaN unless the fix holds. Old
        # code latches onto the first NaN (index 2, verified via negative control) instead of the
        # true winner (index 1, which holds all 5 real observations).
        fit = _fit(min_val=0, max_val=4, observations=[1, 1, 1, 1, 1], pseudo_count=0.5, suff_stat=(0, None))
        self.assertEqual(fit.k, 1)

    def test_pseudo_count_branch_suff_stat_full(self):
        # self.suff_stat = (k_pseudo, weight) branch: same reasoning as the partial-suff_stat case.
        fit = _fit(min_val=0, max_val=4, observations=[1, 1, 1, 1, 1], pseudo_count=0.5, suff_stat=(0, 2.0))
        self.assertEqual(fit.k, 1)

    def test_pseudo_count_branch_broadcast(self):
        # self.suff_stat is None (default): pseudo_count broadcasts to every slot, which pushes
        # every raw-zero slot away from zero and masks the bug for pseudo_count > 0. Use 0.0 (a
        # legal but degenerate value) to keep this a genuine trigger of the same code path.
        fit = _fit(min_val=0, max_val=4, observations=[1, 1, 1, 1, 1], pseudo_count=0.0)
        self.assertEqual(fit.k, 1)

    def test_uniform_counts_tie_breaks_to_first_index(self):
        # sanity check: with no zero counts at all, behavior is unaffected by the fix.
        fit = _fit(min_val=0, max_val=3, observations=[0, 1, 2, 3])
        self.assertEqual(fit.k, 0)
        self.assertEqual(fit.p, 0.25)


if __name__ == "__main__":
    unittest.main()
