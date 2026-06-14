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

from pysp.stats.pdist import EnumerationError
from pysp.stats.setdist import BernoulliSetDistribution, BernoulliSetEnumerator

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


class SetDistImportSmokeTestCase(unittest.TestCase):
    """Import smoke test for the five set/association modules and their five-part protocol classes."""

    MODULES = {
        "pysp.stats.setdist": [
            "BernoulliSetDistribution",
            "BernoulliSetSampler",
            "BernoulliSetEstimator",
            "BernoulliSetAccumulator",
            "BernoulliSetAccumulatorFactory",
            "BernoulliSetDataEncoder",
            "BernoulliSetEnumerator",
        ],
        "pysp.stats.int_edit_setdist": [
            "IntegerBernoulliEditDistribution",
            "IntegerBernoulliEditSampler",
            "IntegerBernoulliEditEstimator",
            "IntegerBernoulliEditAccumulator",
            "IntegerBernoulliEditAccumulatorFactory",
            "IntegerBernoulliEditDataEncoder",
        ],
        "pysp.stats.int_edit_stepsetdist": [
            "IntegerStepBernoulliEditDistribution",
            "IntegerStepBernoulliEditSampler",
            "IntegerStepBernoulliEditEstimator",
            "IntegerStepBernoulliEditAccumulator",
            "IntegerStepBernoulliEditAccumulatorFactory",
            "IntegerStepBernoulliEditDataEncoder",
        ],
        "pysp.stats.hidden_association": [
            "HiddenAssociationDistribution",
            "HiddenAssociationSampler",
            "HiddenAssociationEstimator",
            "HiddenAssociationAccumulator",
            "HiddenAssociationAccumulatorFactory",
            "HiddenAssociationDataEncoder",
        ],
        "pysp.stats.int_hidden_association": [
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
