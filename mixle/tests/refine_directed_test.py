"""REFINE-a: diagnosis-directed correction vs blind structure search, on the planted-fault benchmark.

Extends DIAGNOSE-a's exact planted fault (mixle/tests/diagnose_test.py's two-field missing-edge model)
to four fields -- two independent noise fields alongside the one true (moderate, not near-deterministic)
dependency -- so blind search has a real O(n^2) candidate-edge space to search blindly, instead of the
trivial single-possible-edge case a bare two-field model would give it.

Kill criterion (stated up front, per the card): if directed correction does not reach the SAME held-out
target in FEWER trials than blind search, this is a negative result to record in
notes/refine-directed-negative.md, not to paper over.
"""

import unittest
from unittest import mock

import numpy as np

import mixle.inference.refine as refine_module
from mixle.inference.bayesian_network import HeterogeneousBayesianNetwork, _LinearGaussianFactor, _MarginalFactor
from mixle.inference.explain import diagnose
from mixle.inference.refine import (
    apply_add_edge_fix,
    blind_search_trials_to_target,
    directed_correction,
)
from mixle.stats import CategoricalDistribution, GaussianDistribution


def _buggy_net():
    """The planted fault: field1 = 0.6*field0 + noise, but modeled as fully independent; field2/field3 are
    genuinely independent noise fields, correctly modeled -- diagnose() must not flag either of them."""
    return HeterogeneousBayesianNetwork(
        [
            _MarginalFactor(0, GaussianDistribution(0.0, 1.0)),
            _MarginalFactor(1, GaussianDistribution(0.0, 1.0)),
            _MarginalFactor(2, GaussianDistribution(0.0, 1.0)),
            _MarginalFactor(3, GaussianDistribution(0.0, 1.0)),
        ]
    )


def _make_rows(n, seed):
    rng = np.random.RandomState(seed)
    f0 = rng.normal(0.0, 1.0, size=n)
    f1 = 0.6 * f0 + rng.normal(0.0, 0.6, size=n)
    f2 = rng.normal(0.0, 1.0, size=n)
    f3 = rng.normal(0.0, 1.0, size=n)
    return [(float(a), float(b), float(c), float(d)) for a, b, c, d in zip(f0, f1, f2, f3)]


def _diagnose_probe_cases():
    # a tight deterministic-ish grid probing the SAME true relationship (field1 ~ 0.6*field0), used only
    # to feed diagnose() -- separate from the noisier i.i.d. `data` blind search fits on
    a_grid = np.linspace(-3.0, 3.0, 41)
    rows = [(float(a), float(0.6 * a), 0.0, 0.0) for a in a_grid]
    background = [r for r in rows if abs(r[0]) <= 1.5]
    failing = [r for r in rows if abs(r[0]) > 2.2]
    return background, failing


class ApplyAddEdgeFixTest(unittest.TestCase):
    def test_planted_fault_is_named_dominant_and_the_fix_applies(self):
        background, failing = _diagnose_probe_cases()
        fault = diagnose(_buggy_net(), failing, background=background)
        self.assertEqual(fault.suggested_fix, "add_edge")

        data = _make_rows(300, seed=1)
        fixed = apply_add_edge_fix(_buggy_net(), fault, data)
        self.assertIsNotNone(fixed)
        # the two untouched noise fields (2 and 3) still have zero parents; only field1 gained one
        edges = fixed.edges()
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0][1], 1)  # child is field 1

    def test_non_add_edge_fix_returns_none_not_a_guess(self):
        from mixle.inference.explain import FaultReport

        fault = FaultReport(dominant="field[0]|x+field[1]|y", suggested_fix="upgrade_leaf")
        result = apply_add_edge_fix(_buggy_net(), fault, _make_rows(50, seed=2))
        self.assertIsNone(result)

    def test_direction_is_chosen_by_fit_quality_not_field_index_order(self):
        # The true relationship here is field1 -> field0 (field0 = 0.6*field1 + noise) -- the
        # OPPOSITE of this module's other fixture (field0 -> field1, where field0 happens to have
        # the smaller index and IS the true parent, so that fixture alone can't catch this bug).
        # `parent, child = sorted(idx)` always picked the smaller field index as parent regardless
        # of which orientation the data actually supports; this fixture is specifically built so
        # that "smaller index = parent" gives the WRONG (reversed) answer.
        from mixle.inference.explain import FaultReport

        rng = np.random.RandomState(5)
        n = 400
        f1 = rng.normal(0.0, 1.0, size=n)
        f0 = 0.6 * f1 + rng.normal(0.0, 0.6, size=n)
        f2 = rng.normal(0.0, 1.0, size=n)
        f3 = rng.normal(0.0, 1.0, size=n)
        data = [(float(a), float(b), float(c), float(d)) for a, b, c, d in zip(f0, f1, f2, f3)]

        fault = FaultReport(dominant="field[0]|x+field[1]|y", suggested_fix="add_edge")
        fixed = apply_add_edge_fix(_buggy_net(), fault, data)
        self.assertIsNotNone(fixed)
        edges = fixed.edges()
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0], (1, 0))  # parent=field1, child=field0: the true generating direction


class RefineVsBlindSearchTest(unittest.TestCase):
    def test_directed_correction_beats_blind_search_in_trials(self):
        background, failing = _diagnose_probe_cases()
        train = _make_rows(200, seed=10)
        held_out = _make_rows(200, seed=11)

        directed = directed_correction(_buggy_net(), failing, train, held_out, background=background)
        self.assertEqual(directed.n_trials, 1)
        target_score = directed.history[-1]

        blind = blind_search_trials_to_target(train, held_out, target_score, round_size=10, max_rounds=20, seed=0)

        # KILL CRITERION: directed must reach the target in fewer trials than blind search, or record
        # the negative result in notes/refine-directed-negative.md and keep blind search.
        directed_trials = directed.n_trials
        blind_trials = blind.n_trials if blind.n_trials is not None else float("inf")
        self.assertLess(
            directed_trials,
            blind_trials,
            f"REFINE-a kill criterion failed: directed={directed_trials} trial(s), "
            f"blind={blind.n_trials} round(s) (None = never reached target within max_rounds); "
            "record the negative result in notes/refine-directed-negative.md",
        )

    def test_directed_correction_verifiably_improves_held_out_score(self):
        background, failing = _diagnose_probe_cases()
        train = _make_rows(200, seed=20)
        held_out = _make_rows(200, seed=21)
        result = directed_correction(_buggy_net(), failing, train, held_out, background=background)
        self.assertEqual(result.n_trials, 1)
        before, after = result.history
        self.assertGreater(after, before)

    def test_a_well_specified_model_has_no_actionable_fix(self):
        from mixle.inference.bayesian_network import _LinearGaussianFactor

        well_specified = HeterogeneousBayesianNetwork(
            [
                _MarginalFactor(0, GaussianDistribution(0.0, 1.0)),
                _LinearGaussianFactor(1, [0], {}, np.array([0.6, 0.0]), 0.6),
                _MarginalFactor(2, GaussianDistribution(0.0, 1.0)),
                _MarginalFactor(3, GaussianDistribution(0.0, 1.0)),
            ]
        )
        background, failing = _diagnose_probe_cases()
        train = _make_rows(200, seed=30)
        held_out = _make_rows(200, seed=31)
        result = directed_correction(well_specified, failing, train, held_out, background=background)
        self.assertIsNone(result.n_trials)  # nothing dominant -> no fix applied -> correctly unreached


class AddEdgePreservesTheModelTest(unittest.TestCase):
    """MXR-080-1627: 'add edge' must ADD, not rewrite the child's factor."""

    @staticmethod
    def _two_driver_rows(n, seed):
        """field1 is driven by BOTH field0 and field2, so 2 -> 1 is decisively the better orientation."""
        rng = np.random.RandomState(seed)
        f0 = rng.normal(0.0, 1.0, size=n)
        f2 = rng.normal(0.0, 1.0, size=n)
        f1 = 0.9 * f0 + 0.9 * f2 + rng.normal(0.0, 0.05, size=n)
        f3 = rng.normal(0.0, 1.0, size=n)
        return [(float(a), float(b), float(c), float(d)) for a, b, c, d in zip(f0, f1, f2, f3)]

    def _net_with_existing_edge(self, data):
        """field1 already depends on field0; field2/field3 stay independent roots."""
        cols = [[row[i] for row in data] for i in range(4)]
        return HeterogeneousBayesianNetwork(
            [
                _MarginalFactor(0, GaussianDistribution(0.0, 1.0)),
                _LinearGaussianFactor.fit(1, [0], cols, {}),
                _MarginalFactor(2, GaussianDistribution(0.0, 1.0)),
                _MarginalFactor(3, GaussianDistribution(0.0, 1.0)),
            ]
        )

    def test_existing_parents_are_kept_when_a_new_one_is_added(self):
        from mixle.inference.explain import FaultReport

        data = self._two_driver_rows(300, seed=41)
        net = self._net_with_existing_edge(data)
        self.assertEqual(net.edges(), [(0, 1)])
        fault = FaultReport(dominant="field[1]|x+field[2]|y", suggested_fix="add_edge")
        fixed = apply_add_edge_fix(net, fault, data)
        self.assertIsNotNone(fixed)
        child1 = next(f for f in fixed.factors if f.child == 1)
        # Before the fix the child's factor was rebuilt from [parent] alone, so the pre-existing
        # 0 -> 1 dependency was silently deleted by an operation named "add edge".
        self.assertEqual(sorted(child1.parents), [0, 2])
        self.assertIn((0, 1), fixed.edges())

    def test_a_non_linear_gaussian_child_is_reported_unsupported_not_rewritten(self):
        from mixle.inference.explain import FaultReport

        data = [(float(a), "yes" if a > 0 else "no", 0.0, 0.0) for a in np.linspace(-3, 3, 60)]
        net = HeterogeneousBayesianNetwork(
            [
                _MarginalFactor(0, GaussianDistribution(0.0, 1.0)),
                _MarginalFactor(1, CategoricalDistribution({"yes": 0.5, "no": 0.5})),
                _MarginalFactor(2, GaussianDistribution(0.0, 1.0)),
                _MarginalFactor(3, GaussianDistribution(0.0, 1.0)),
            ]
        )
        fault = FaultReport(dominant="field[0]|x+field[1]|y", suggested_fix="add_edge")
        fixed = apply_add_edge_fix(net, fault, data)
        # The 1 -> 0 orientation is expressible, so an edit may still be returned -- but the
        # categorical child must never have been turned into a scalar linear-Gaussian regression,
        # and the raw "could not convert string to float" from inside fit() must not escape.
        if fixed is not None:
            child1 = next(f for f in fixed.factors if f.child == 1)
            self.assertNotIsInstance(child1, _LinearGaussianFactor)

    def test_the_valid_orientation_survives_a_cyclic_one(self):
        from mixle.inference.explain import FaultReport

        data = self._two_driver_rows(300, seed=42)
        net = self._net_with_existing_edge(data)
        # 0 -> 1 already exists, so orienting 1 -> 0 would close a cycle. The other orientation
        # (adding 0 as a parent it already has) is a no-op, so this must report unsupported rather
        # than raise out of the constructor before the alternative was considered.
        fault = FaultReport(dominant="field[0]|x+field[1]|y", suggested_fix="add_edge")
        self.assertIsNone(apply_add_edge_fix(net, fault, data))


class NonFiniteScoreTest(unittest.TestCase):
    """MXR-080-1628: a candidate that could not be scored is not a verified improvement."""

    def test_nan_after_score_is_not_a_successful_correction(self):
        background, failing = _diagnose_probe_cases()
        train, held_out = _make_rows(200, seed=50), _make_rows(200, seed=51)
        real = refine_module.held_out_log_likelihood

        def scoring_that_fails_on_the_candidate(model, data):
            value = real(model, data)
            return float("nan") if getattr(model, "_candidate", False) else value

        original = _buggy_net()
        with mock.patch.object(refine_module, "held_out_log_likelihood", scoring_that_fails_on_the_candidate):
            with mock.patch.object(refine_module, "apply_add_edge_fix") as fake_fix:
                candidate = _buggy_net()
                candidate._candidate = True
                fake_fix.return_value = candidate
                result = refine_module.directed_correction(original, failing, train, held_out, background=background)
        # `after <= before` is FALSE for NaN, so this used to be returned as a 1-trial success.
        self.assertIsNone(result.n_trials)
        self.assertIs(result.final_model, original)

    def test_empty_held_out_is_rejected(self):
        background, failing = _diagnose_probe_cases()
        with self.assertRaises(ValueError):
            directed_correction(_buggy_net(), failing, _make_rows(50, seed=52), [], background=background)

    def test_a_genuine_improvement_is_still_accepted(self):
        background, failing = _diagnose_probe_cases()
        train, held_out = _make_rows(200, seed=53), _make_rows(200, seed=54)
        result = directed_correction(_buggy_net(), failing, train, held_out, background=background)
        self.assertEqual(result.n_trials, 1)


if __name__ == "__main__":
    unittest.main()
