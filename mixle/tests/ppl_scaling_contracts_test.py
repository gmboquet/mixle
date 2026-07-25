"""Fast contract tests for scaling-law generation, prediction, and allocation."""

import unittest

import numpy as np

from mixle.ppl.scaling_laws import (
    ScalingLawAllocationController,
    ScalingLawDiagnostics,
    ScalingLawFit,
    allocate_compute,
    allocate_compute_learned,
    allocate_fixed_heuristic,
    fit_scaling_law,
    generate_synthetic_chinchilla_data,
)


class _FakePosterior:
    def __init__(self):
        self.values = {
            "E": np.array([1.0, 2.0, 3.0]),
            "log_A": np.log([1.0, 4.0, 9.0]),
            "log_alpha": np.log([0.2, 0.5, 1.0]),
            "log_B": np.log([2.0, 3.0, 5.0]),
            "log_beta": np.log([0.3, 0.6, 0.9]),
            "log_sigma": np.log([0.1, 0.1, 0.2]),
        }

    def posterior(self, name):
        return self.values[name]


def _usable_fit():
    return ScalingLawFit(
        fitted=_FakePosterior(),
        n0=10.0,
        d0=20.0,
        diagnostics=ScalingLawDiagnostics(
            usable=True,
            status="usable_with_caveat",
            acceptance_rate=0.25,
            posterior_draws=3,
            burn=20,
            n_chains=1,
            posterior_finite=True,
            caveats=("test_fixture",),
        ),
    )


class _Design:
    def __init__(self):
        self.rows = []

    def __len__(self):
        return len(self.rows)

    def add(self, point, reward, constraints, fingerprint):
        self.rows.append((point, reward, constraints, fingerprint))

    def propose(self, bounds, seed, fingerprint):
        return [0.5 * (bounds[0][0] + bounds[0][1])]


class SyntheticGenerationContractTest(unittest.TestCase):
    def test_configuration_is_attached_to_list_compatible_output(self):
        records = generate_synthetic_chinchilla_data(n_points=4, seed=7, noise_sd=0.0)
        self.assertIsInstance(records, list)
        self.assertEqual(len(records), 4)
        self.assertEqual(records.configuration["seed"], 7)
        self.assertEqual(records.configuration["n_points"], 4)
        self.assertEqual(records.configuration["noise_sd"], 0.0)

    def test_invalid_designs_are_rejected_before_sampling(self):
        invalid = (
            {"n_points": 2.5},
            {"n_points": 0},
            {"noise_sd": -1.0},
            {"noise_sd": float("nan")},
            {"n_range": (10.0, 1.0)},
            {"d_range": (0.0, 1.0)},
            {"a": -1.0},
            {"alpha": 0.0},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_synthetic_chinchilla_data(**kwargs)


class ScalingFitContractTest(unittest.TestCase):
    def test_predict_mean_is_joint_draw_expectation(self):
        fit = _usable_fit()
        expected = float(np.mean(fit.predict_samples(100.0, 200.0)))
        self.assertAlmostEqual(fit.predict_mean(100.0, 200.0), expected)
        plug_in = (
            fit.mean("E")
            + fit.mean("A") * 10.0 ** (-fit.mean("alpha"))
            + fit.mean("B") * 10.0 ** (-fit.mean("beta"))
        )
        self.assertNotAlmostEqual(fit.predict_mean(100.0, 200.0), plug_in)

    def test_malformed_records_and_controls_are_rejected_before_mcmc(self):
        valid = [(1.0 + i, 2.0 + i, 3.0) for i in range(6)]
        cases = (
            (valid[:5], {}),
            (valid[:-1] + [(1.0, 2.0)], {}),
            (valid[:-1] + [(1.0, 2.0, float("nan"))], {}),
            (valid[:-1] + [(0.0, 2.0, 3.0)], {}),
            (valid, {"draws": 2.5}),
            (valid, {"draws": 10}),
            (valid, {"burn": 0}),
            (valid, {"scale": -0.1}),
            (valid, {"seed": 1, "rng": np.random.RandomState(1)}),
        )
        for records, kwargs in cases:
            with self.subTest(records=records, kwargs=kwargs), self.assertRaises(ValueError):
                fit_scaling_law(records, **kwargs)

    def test_unusable_fit_cannot_drive_decisions(self):
        fit = ScalingLawFit(_FakePosterior(), 10.0, 20.0)
        with self.assertRaises(RuntimeError):
            allocate_compute(fit, 1.0e12)


class AllocationContractTest(unittest.TestCase):
    def test_invalid_fixed_and_bayesian_allocation_controls_are_rejected(self):
        for kwargs in (
            {"compute_budget": float("nan")},
            {"compute_budget": 1.0, "ratio": 0.0},
            {"compute_budget": 1.0, "flops_per_token_param": -1.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                allocate_fixed_heuristic(**kwargs)

        fit = _usable_fit()
        for kwargs in (
            {"compute_budget": float("inf")},
            {"compute_budget": 1.0e12, "n_bounds": (10.0, 1.0)},
            {"compute_budget": 1.0e12, "n_init": 1.5},
            {"compute_budget": 1.0e12, "n_iter": 0},
            {"compute_budget": 1.0e12, "flops_per_token_param": 0.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                allocate_compute(fit, **kwargs)

    def test_learned_allocation_remains_pending_until_measured(self):
        fit = _usable_fit()
        design = _Design()
        controller = ScalingLawAllocationController(n_bounds=(10.0, 1000.0), design=design, seed=1)
        n, d, returned, proposal = allocate_compute_learned(
            fit, 6.0e6, controller=controller, flops_per_token_param=6.0
        )
        self.assertIs(returned, controller)
        self.assertAlmostEqual(6.0 * n * d, 6.0e6)
        self.assertEqual(len(design), 0)
        self.assertIn(proposal.proposal_id, controller.pending)

        rejected = controller.record_outcome(
            proposal,
            realized_gain=float("nan"),
            realized_cost=0.0,
            provenance="failed-run",
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.submitted_gain, "nan")
        self.assertEqual(rejected.submitted_cost, "0.0")
        self.assertEqual(len(design), 0)
        self.assertIn(proposal.proposal_id, controller.pending)

        accepted = controller.record_outcome(
            proposal,
            realized_gain=-0.2,
            realized_cost=120.0,
            provenance="training-run:sha256:abc",
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(len(design), 1)
        self.assertNotIn(proposal.proposal_id, controller.pending)

        duplicate = controller.record_outcome(
            proposal,
            realized_gain=1.0,
            realized_cost=100.0,
            provenance="duplicate",
        )
        self.assertFalse(duplicate.accepted)
        self.assertEqual(len(design), 1)


if __name__ == "__main__":
    unittest.main()
