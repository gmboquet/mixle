"""Fast, network-free coverage for the temporal flagship's inspection helpers (worklist F10.2).

``examples/flagship_temporal_sunspots.py`` adds ``describe_emissions`` (emission inspection) and
``check_impossible_observation`` (impossible-observation handling). Both are pure functions of a
fitted :class:`~mixle.stats.latent.hidden_markov.HiddenMarkovModelDistribution`, so their logic is
exercised here against small, hand-built models with known parameters -- no network fetch, no EM fit,
no ``hmmlearn`` dependency, so this file runs unconditionally in the default fast gate. The real,
network-gated end-to-end exercise of these same functions against the actual fitted sunspots model
lives in ``flagship_temporal_sunspots_smoke_test.py``.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

from mixle.stats import CategoricalDistribution, HiddenMarkovModelDistribution

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from flagship_temporal_sunspots import check_impossible_observation, describe_emissions  # noqa: E402


def _two_state_model(default_value: float = 0.0) -> HiddenMarkovModelDistribution:
    """A hand-built 2-state, 3-symbol HMM with known emission/transition parameters."""
    return HiddenMarkovModelDistribution(
        topics=[
            CategoricalDistribution(pmap={0: 0.7, 1: 0.2, 2: 0.1}, default_value=default_value),
            CategoricalDistribution(pmap={0: 0.1, 1: 0.1, 2: 0.8}, default_value=default_value),
        ],
        w=[0.5, 0.5],
        transitions=[[0.9, 0.1], [0.25, 0.75]],
    )


class DescribeEmissionsTest(unittest.TestCase):
    def test_top_symbols_sorted_descending_by_probability(self):
        model = _two_state_model()
        out = describe_emissions(model, top_k=2)
        self.assertEqual(len(out), 2)
        # state 0's two most probable symbols, in descending-probability order
        self.assertEqual(out[0]["state"], 0)
        self.assertEqual(out[0]["top_symbols"], [(0, 0.7), (1, 0.2)])
        # state 1's two most probable symbols
        self.assertEqual(out[1]["top_symbols"], [(2, 0.8), (0, 0.1)])

    def test_top_symbols_ties_broken_by_symbol_ascending(self):
        model = HiddenMarkovModelDistribution(
            topics=[CategoricalDistribution(pmap={5: 0.5, 2: 0.5}, default_value=0.0)],
            w=[1.0],
            transitions=[[1.0]],
        )
        out = describe_emissions(model, top_k=2)
        # equal probability -> lower symbol index sorts first
        self.assertEqual(out[0]["top_symbols"], [(2, 0.5), (5, 0.5)])

    def test_entropy_matches_hand_computed_shannon_entropy(self):
        model = _two_state_model()
        out = describe_emissions(model)
        expected_state0 = -(0.7 * math.log(0.7) + 0.2 * math.log(0.2) + 0.1 * math.log(0.1))
        expected_state1 = -(0.1 * math.log(0.1) + 0.1 * math.log(0.1) + 0.8 * math.log(0.8))
        self.assertAlmostEqual(out[0]["entropy_nats"], expected_state0, places=10)
        self.assertAlmostEqual(out[1]["entropy_nats"], expected_state1, places=10)

    def test_uniform_emission_hits_max_entropy(self):
        model = HiddenMarkovModelDistribution(
            topics=[CategoricalDistribution(pmap={0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}, default_value=0.0)],
            w=[1.0],
            transitions=[[1.0]],
        )
        out = describe_emissions(model)
        self.assertAlmostEqual(out[0]["entropy_nats"], math.log(4), places=10)

    def test_self_transition_and_expected_sojourn(self):
        model = _two_state_model()
        out = describe_emissions(model)
        self.assertAlmostEqual(out[0]["self_transition"], 0.9, places=12)
        self.assertAlmostEqual(out[0]["expected_sojourn_steps"], 1.0 / (1.0 - 0.9), places=10)
        self.assertAlmostEqual(out[1]["self_transition"], 0.75, places=12)
        self.assertAlmostEqual(out[1]["expected_sojourn_steps"], 1.0 / (1.0 - 0.75), places=10)

    def test_zero_self_transition_gives_unit_sojourn(self):
        model = HiddenMarkovModelDistribution(
            topics=[
                CategoricalDistribution(pmap={0: 1.0}, default_value=0.0),
                CategoricalDistribution(pmap={1: 1.0}, default_value=0.0),
            ],
            w=[0.5, 0.5],
            transitions=[[0.0, 1.0], [1.0, 0.0]],
        )
        out = describe_emissions(model)
        self.assertEqual(out[0]["self_transition"], 0.0)
        self.assertEqual(out[0]["expected_sojourn_steps"], 1.0)

    def test_absorbing_state_gives_infinite_sojourn(self):
        model = HiddenMarkovModelDistribution(
            topics=[CategoricalDistribution(pmap={0: 1.0}, default_value=0.0)],
            w=[1.0],
            transitions=[[1.0]],
        )
        out = describe_emissions(model)
        self.assertEqual(out[0]["self_transition"], 1.0)
        self.assertEqual(out[0]["expected_sojourn_steps"], float("inf"))


class CheckImpossibleObservationTest(unittest.TestCase):
    def test_valid_sequence_scores_finite(self):
        model = _two_state_model()
        result = check_impossible_observation(model, [0, 1, 2, 0, 1], n_symbols=3)
        self.assertTrue(result["valid_is_finite"])
        self.assertTrue(math.isfinite(result["valid_ll"]))

    def test_out_of_support_symbol_is_flagged_as_exactly_negative_infinity(self):
        model = _two_state_model()
        result = check_impossible_observation(model, [0, 1, 2, 0, 1, 2, 0, 1], n_symbols=3)
        self.assertEqual(result["impossible_symbol"], 3)
        self.assertTrue(result["correctly_flagged"])
        self.assertEqual(result["impossible_ll"], float("-inf"))
        # not NaN, not some other non-finite value masquerading as -inf
        self.assertFalse(math.isnan(result["impossible_ll"]))

    def test_does_not_raise_for_out_of_support_symbol(self):
        # a KeyError/IndexError here would be the "silently wrong" failure mode this guards against;
        # unittest turns an uncaught exception into a test error, so simply calling it is the check.
        model = _two_state_model()
        check_impossible_observation(model, [0, 1, 2], n_symbols=3)

    def test_negative_control_nonzero_default_value_is_not_flagged(self):
        # if the emission model assigns nonzero mass OUTSIDE its pmap (default_value > 0), the
        # "impossible" sentinel is not actually impossible for this model, and the checker must say
        # so -- proving the assertion above is a real behavioral check, not a vacuous True.
        model = _two_state_model(default_value=0.05)
        result = check_impossible_observation(model, [0, 1, 2, 0, 1], n_symbols=3)
        self.assertFalse(result["correctly_flagged"])
        self.assertTrue(math.isfinite(result["impossible_ll"]))

    def test_multi_state_model_flags_impossible_regardless_of_which_state_is_active(self):
        # every state's pmap excludes the sentinel, so the sequence is impossible under ANY hidden
        # path, not just the most likely one -- the whole point of scoring via seq_log_density.
        model = HiddenMarkovModelDistribution(
            topics=[
                CategoricalDistribution(pmap={0: 0.9, 1: 0.1}, default_value=0.0),
                CategoricalDistribution(pmap={0: 0.1, 1: 0.9}, default_value=0.0),
                CategoricalDistribution(pmap={0: 0.5, 1: 0.5}, default_value=0.0),
            ],
            w=[1 / 3, 1 / 3, 1 / 3],
            transitions=[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
        )
        result = check_impossible_observation(model, [0, 1, 0, 1, 0, 1], n_symbols=2)
        self.assertTrue(result["correctly_flagged"])


if __name__ == "__main__":
    unittest.main()
