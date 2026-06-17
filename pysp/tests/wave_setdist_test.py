"""Tests for the Bernoulli set distribution enumerator and the set/association module imports.

Covers BernoulliSetEnumerator (pysp/stats/setdist.py) against brute force on a 4-element
universe: non-increasing order, exact de-duplication, log_prob == log_density, top-k
agreement with the brute-force top-k, and fail-fast EnumerationError when a membership
probability lies outside [0, 1]. Also smoke-tests imports of the five set/association
modules: setdist, int_edit_setdist, int_edit_stepsetdist, hidden_association, and
int_hidden_association.
"""

import importlib
import itertools
import unittest

import numpy as np

from pysp.stats.compute.pdist import EnumerationError
from pysp.stats.sets.setdist import BernoulliSetDistribution, BernoulliSetEnumerator
from pysp.stats.sets.int_edit_stepsetdist import IntegerStepBernoulliEditEstimator

TOL = 1e-9


def all_subsets(universe):
    """All subsets of universe as tuples, in no particular order."""
    out = []
    for n in range(len(universe) + 1):
        out.extend(itertools.combinations(sorted(universe), n))
    return out


def canon(value):
    """Order-invariant canonical key for a set-valued observation."""
    return frozenset(value)


def tiers(pairs):
    """Map rounded log_prob tiers to the set of canonical values in each tier (tie-safe compare)."""
    out = {}
    for v, lp in pairs:
        out.setdefault(round(lp, 8), set()).add(canon(v))
    return out


def expected_two_level(successes, trials, min_prob=0.0):
    """Independent brute-force implementation of the two-level threshold fit."""
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    obs = np.flatnonzero(trials > 0)
    if len(obs) == 0:
        return np.full(len(successes), 0.5)

    def clip(x):
        if min_prob <= 0.0:
            return float(x)
        return float(np.clip(x, min_prob, 1.0 - min_prob))

    sidx = obs[np.argsort(-(successes[obs] / trials[obs]))]
    cs = np.cumsum(successes[sidx])
    ct = np.cumsum(trials[sidx])
    tot_s = cs[-1]
    tot_t = ct[-1]

    best_ll = -np.inf
    best = None
    for i in range(len(obs)):
        sh, th = cs[i], ct[i]
        p = clip(sh / th)
        ll = (sh * np.log(p) if sh > 0 else 0.0) + ((th - sh) * np.log1p(-p) if th > sh else 0.0)
        if i + 1 < len(obs):
            sl, tl = tot_s - sh, tot_t - th
            q = clip(sl / tl)
            ll += (sl * np.log(q) if sl > 0 else 0.0) + ((tl - sl) * np.log1p(-q) if tl > sl else 0.0)
        else:
            q = 0.0
        if ll > best_ll:
            best_ll = ll
            best = (p, q, i)

    p, q, k = best
    out = np.full(len(successes), clip(tot_s / tot_t))
    out[sidx[: k + 1]] = p
    out[sidx[k + 1 :]] = q
    return out


class BernoulliSetEnumeratorTestCase(unittest.TestCase):
    """BernoulliSetEnumerator enumeration against brute force on a 4-element universe."""

    def setUp(self):
        self.pmap = {"a": 0.7, "b": 0.4, "c": 0.1, "d": 0.55}
        self.dist = BernoulliSetDistribution(self.pmap, min_prob=0)
        self.brute = sorted([(v, self.dist.log_density(list(v))) for v in all_subsets(self.pmap)], key=lambda t: -t[1])

    def test_enumerates_full_support_in_order(self):
        items = list(self.dist.enumerator())

        self.assertEqual(len(items), 16)

        lps = [lp for _, lp in items]
        for i in range(len(lps) - 1):
            self.assertGreaterEqual(lps[i], lps[i + 1] - TOL)

        seen = set(canon(v) for v, _ in items)
        self.assertEqual(len(seen), len(items))
        self.assertEqual(seen, set(canon(v) for v, _ in self.brute))

        self.assertAlmostEqual(sum(np.exp(lp) for lp in lps), 1.0, places=10)

    def test_log_prob_matches_log_density(self):
        for v, lp in self.dist.enumerator():
            self.assertAlmostEqual(lp, self.dist.log_density(v), delta=TOL)

    def test_top_k_matches_brute_force(self):
        for k in (1, 3, 7, 16):
            top = self.dist.enumerator().top_k(k)
            self.assertEqual(len(top), min(k, 16))
            # Tie-safe comparison: identical log_prob tiers must contain identical value sets,
            # except possibly the last (cut) tier, where enumeration may pick any tie subset.
            top_tiers = tiers(top)
            brute_tiers = tiers(self.brute[: len(top)])
            self.assertEqual(set(top_tiers), set(brute_tiers))
            cut = min(top_tiers)
            for lp in top_tiers:
                if lp == cut:
                    self.assertEqual(len(top_tiers[lp]), len(brute_tiers[lp]))
                else:
                    self.assertEqual(top_tiers[lp], brute_tiers[lp])

    def test_required_and_zero_elements(self):
        # p=1 with min_prob=0 makes 'a' required (include-only); p=0 makes 'b' exclude-only.
        dist = BernoulliSetDistribution({"a": 1.0, "b": 0.0, "c": 0.5}, min_prob=0)
        items = list(dist.enumerator())

        self.assertEqual(len(items), 2)
        for v, lp in items:
            self.assertIn("a", v)
            self.assertNotIn("b", v)
            self.assertAlmostEqual(lp, dist.log_density(v), delta=TOL)
        self.assertEqual(set(canon(v) for v, _ in items), {frozenset(["a"]), frozenset(["a", "c"])})

    def test_min_prob_smoothing_consistent(self):
        # With the default min_prob, p=1 and p=0 are smoothed instead of hard constraints;
        # the enumerator must still agree with log_density on every subset.
        dist = BernoulliSetDistribution({"a": 1.0, "b": 0.0, "c": 0.5}, min_prob=1.0e-128)
        items = dist.enumerator().top_k(8)
        self.assertEqual(len(items), 8)
        lps = [lp for _, lp in items]
        for i in range(len(lps) - 1):
            self.assertGreaterEqual(lps[i], lps[i + 1] - TOL)
        for v, lp in items:
            self.assertAlmostEqual(lp, dist.log_density(v), delta=TOL)
        self.assertEqual(len(set(canon(v) for v, _ in items)), 8)

    def test_empty_support(self):
        dist = BernoulliSetDistribution({}, min_prob=0)
        items = list(dist.enumerator())
        self.assertEqual(len(items), 1)
        self.assertEqual(list(items[0][0]), [])
        self.assertAlmostEqual(items[0][1], 0.0, delta=TOL)

    def test_invalid_probability_fails_fast(self):
        with np.errstate(invalid="ignore"):
            dist = BernoulliSetDistribution({"a": 1.5, "b": 0.2}, min_prob=0)
        with self.assertRaises(EnumerationError) as cm:
            dist.enumerator()
        self.assertIn("does not support enumeration", str(cm.exception))
        self.assertTrue(cm.exception.reason)

    def test_enumerator_type(self):
        self.assertIsInstance(self.dist.enumerator(), BernoulliSetEnumerator)


class StepBernoulliEditEstimatorTestCase(unittest.TestCase):
    def test_pseudo_count_enters_threshold_fit(self):
        count_mat = np.array(
            [
                [0.0, 0.0, 10.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
            ]
        )
        tot_sum = 10.0
        est = IntegerStepBernoulliEditEstimator(num_vals=3, pseudo_count=4.0, min_prob=0.0)
        dist = est.estimate(None, (count_mat, tot_sum, None))

        s1 = count_mat[:, 0] + count_mat[:, 2]
        s0 = tot_sum - s1
        expected_removal = expected_two_level(count_mat[:, 0] + 1.0, s1 + 2.0)
        expected_addition = expected_two_level(count_mat[:, 1] + 1.0, s0 + 2.0)

        np.testing.assert_allclose(np.exp(dist.log_edit_pmat[:, 1]), expected_removal, atol=1.0e-12)
        np.testing.assert_allclose(np.exp(dist.log_edit_pmat[:, 2]), expected_addition, atol=1.0e-12)
        self.assertTrue(np.all(np.exp(dist.log_edit_pmat[:, 1]) > 0.0))
        self.assertTrue(np.all(np.exp(dist.log_edit_pmat[:, 2]) > 0.0))

    def test_reference_pseudo_count_enters_threshold_fit(self):
        count_mat = np.array(
            [
                [1.0, 0.0, 9.0],
                [8.0, 1.0, 1.0],
                [0.0, 7.0, 0.0],
            ]
        )
        reference = np.array(
            [
                [0.8, 0.2, 0.2, 0.8],
                [0.3, 0.7, 0.7, 0.3],
                [0.6, 0.4, 0.4, 0.6],
            ]
        )
        tot_sum = 10.0
        alpha = 6.0
        est = IntegerStepBernoulliEditEstimator(
            num_vals=3, pseudo_count=alpha, suff_stat=reference, min_prob=0.0
        )
        dist = est.estimate(None, (count_mat, tot_sum, None))

        s1 = count_mat[:, 0] + count_mat[:, 2]
        s0 = tot_sum - s1
        expected_removal = expected_two_level(
            count_mat[:, 0] + alpha * reference[:, 1],
            s1 + alpha * (reference[:, 1] + reference[:, 3]),
        )
        expected_addition = expected_two_level(
            count_mat[:, 1] + alpha * reference[:, 2],
            s0 + alpha * (reference[:, 0] + reference[:, 2]),
        )

        np.testing.assert_allclose(np.exp(dist.log_edit_pmat[:, 1]), expected_removal, atol=1.0e-12)
        np.testing.assert_allclose(np.exp(dist.log_edit_pmat[:, 2]), expected_addition, atol=1.0e-12)

    def test_min_probability_floor_enters_threshold_fit(self):
        count_mat = np.array(
            [
                [0.0, 0.0, 10.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
            ]
        )
        est = IntegerStepBernoulliEditEstimator(num_vals=3, pseudo_count=None, min_prob=0.05)
        dist = est.estimate(None, (count_mat, 10.0, None))

        probs = np.exp(dist.log_edit_pmat)
        self.assertGreaterEqual(float(np.min(probs)), 0.05 - 1.0e-12)
        self.assertLessEqual(float(np.max(probs)), 0.95 + 1.0e-12)
        np.testing.assert_allclose(probs[:, 0] + probs[:, 2], 1.0, atol=1.0e-12)
        np.testing.assert_allclose(probs[:, 1] + probs[:, 3], 1.0, atol=1.0e-12)

    def test_invalid_step_estimator_regularization_raises(self):
        with self.assertRaises(ValueError):
            IntegerStepBernoulliEditEstimator(num_vals=3, pseudo_count=-1.0)
        with self.assertRaises(ValueError):
            IntegerStepBernoulliEditEstimator(num_vals=3, min_prob=0.75)


class SetDistImportSmokeTestCase(unittest.TestCase):
    """Import smoke test for the five set/association modules and their five-part protocol classes."""

    MODULES = {
        "pysp.stats.sets.setdist": [
            "BernoulliSetDistribution",
            "BernoulliSetSampler",
            "BernoulliSetEstimator",
            "BernoulliSetAccumulator",
            "BernoulliSetAccumulatorFactory",
            "BernoulliSetDataEncoder",
            "BernoulliSetEnumerator",
        ],
        "pysp.stats.sets.int_edit_setdist": [
            "IntegerBernoulliEditDistribution",
            "IntegerBernoulliEditSampler",
            "IntegerBernoulliEditEstimator",
            "IntegerBernoulliEditAccumulator",
            "IntegerBernoulliEditAccumulatorFactory",
            "IntegerBernoulliEditDataEncoder",
        ],
        "pysp.stats.sets.int_edit_stepsetdist": [
            "IntegerStepBernoulliEditDistribution",
            "IntegerStepBernoulliEditSampler",
            "IntegerStepBernoulliEditEstimator",
            "IntegerStepBernoulliEditAccumulator",
            "IntegerStepBernoulliEditAccumulatorFactory",
            "IntegerStepBernoulliEditDataEncoder",
        ],
        "pysp.stats.latent.hidden_association": [
            "HiddenAssociationDistribution",
            "HiddenAssociationSampler",
            "HiddenAssociationEstimator",
            "HiddenAssociationAccumulator",
            "HiddenAssociationAccumulatorFactory",
            "HiddenAssociationDataEncoder",
        ],
        "pysp.stats.latent.int_hidden_association": [
            "IntegerHiddenAssociationDistribution",
            "IntegerHiddenAssociationSampler",
            "IntegerHiddenAssociationEstimator",
            "IntegerHiddenAssociationAccumulator",
            "IntegerHiddenAssociationAccumulatorFactory",
            "IntegerHiddenAssociationDataEncoder",
        ],
    }

    def test_imports_and_protocol_classes(self):
        for mod_name, class_names in self.MODULES.items():
            mod = importlib.import_module(mod_name)
            self.assertTrue((mod.__doc__ or "").strip(), "%s lacks a module docstring" % mod_name)
            for cls_name in class_names:
                cls = getattr(mod, cls_name, None)
                self.assertIsNotNone(cls, "%s.%s missing" % (mod_name, cls_name))
                self.assertTrue((cls.__doc__ or "").strip(), "%s.%s lacks a docstring" % (mod_name, cls_name))


if __name__ == "__main__":
    unittest.main()
