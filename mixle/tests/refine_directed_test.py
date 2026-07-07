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

import numpy as np

from mixle.inference.bayesian_network import HeterogeneousBayesianNetwork, _MarginalFactor
from mixle.inference.explain import diagnose
from mixle.inference.refine import (
    apply_add_edge_fix,
    blind_search_trials_to_target,
    directed_correction,
)
from mixle.stats import GaussianDistribution


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


if __name__ == "__main__":
    unittest.main()
