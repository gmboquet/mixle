"""Fast failure-path checks for verified belief-walk composition."""

import unittest

import numpy as np

from mixle.reason.belief_walk import HopTransport, WalkResult, belief_walk
from mixle.reason.transport_edge import PremiseReceipt, PremiseStatus


def _receipt(name: str, status: PremiseStatus = PremiseStatus.PASS) -> PremiseReceipt:
    return PremiseReceipt(
        edge_name=name,
        verifier="belief-walk-contract-fixture@1",
        dataset_digest=f"sha256:{'1' * 64}",
        metric="fixture-contract",
        evaluated_at="2026-07-25T00:00:00+00:00",
        status=status,
    )


class _Sampler:
    def __init__(self, transform):
        self.transform = transform

    def sample_given_batch(self, values):
        return self.transform(np.asarray(values, dtype=float))


class _Fit:
    def __init__(self, transform):
        self.transform = transform

    def sampler(self, seed=None):
        del seed
        return _Sampler(self.transform)


class BeliefWalkContractTest(unittest.TestCase):
    def test_affirmative_receipt_and_declared_dimensions_compose(self):
        hop = HopTransport("double", _Fit(lambda values: 2.0 * values), _receipt("double"), 1, 1)
        result = belief_walk([hop], np.array([2.0]), n_draws=4, seed=0)
        np.testing.assert_array_equal(result.samples, np.full((4, 1), 4.0))

    def test_non_passing_receipt_is_refused(self):
        hop = HopTransport(
            "failed",
            _Fit(lambda values: values),
            _receipt("failed", PremiseStatus.INCONCLUSIVE),
            1,
            1,
        )
        with self.assertRaisesRegex(ValueError, "affirmative premise receipt"):
            belief_walk([hop], np.array([1.0]), n_draws=4)

    def test_each_hop_must_preserve_rows_and_declared_output_width(self):
        hop = HopTransport("short", _Fit(lambda values: values[:-1]), _receipt("short"), 1, 1)
        with self.assertRaisesRegex(ValueError, "shape"):
            belief_walk([hop], np.array([1.0]), n_draws=4)

    def test_each_hop_must_return_finite_samples(self):
        hop = HopTransport("nan", _Fit(lambda values: values * np.nan), _receipt("nan"), 1, 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            belief_walk([hop], np.array([1.0]), n_draws=4)

    def test_draw_count_and_initial_value_are_validated(self):
        hop = HopTransport("identity", _Fit(lambda values: values), _receipt("identity"), 1, 1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            belief_walk([hop], np.array([1.0]), n_draws=0)
        with self.assertRaisesRegex(ValueError, "finite vector"):
            belief_walk([hop], np.array([np.nan]), n_draws=4)

    def test_simultaneous_interval_uses_joint_alpha_budget(self):
        samples = np.arange(200, dtype=float).reshape(100, 2)
        result = WalkResult([], samples)
        joint_lo, joint_hi = result.simultaneous_credible_interval(alpha=0.1)
        marginal_lo, marginal_hi = result.credible_interval(alpha=0.1)
        self.assertTrue(np.all(joint_lo <= marginal_lo))
        self.assertTrue(np.all(joint_hi >= marginal_hi))


if __name__ == "__main__":
    unittest.main()
