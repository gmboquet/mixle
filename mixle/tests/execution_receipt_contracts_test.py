"""Contracts distinguishing placement/precision/JIT plans from observed execution."""

import importlib.util
import unittest

import numpy as np

import mixle.stats as stats
from mixle.inference import JITExecutionResult, jit_em_mixture, jit_seq_log_density, optimize
from mixle.inference.placement import PoolSpec, plan_placement
from mixle.inference.planning import BlockPlan, EstimationCertificate, Guarantee
from mixle.inference.precision_plan import recommend_compute_precision


class PlacementReceiptContractTest(unittest.TestCase):
    def test_plan_is_bound_but_never_claims_execution(self):
        block = BlockPlan(
            "neural",
            "NeuralDensity",
            "gradient",
            Guarantee.HEURISTIC,
            True,
            "pool_eligible",
            "gradient residual ~8.0 TFLOP",
        )
        certificate = EstimationCertificate(Guarantee.HEURISTIC, [block])
        plan = plan_placement(
            certificate,
            PoolSpec(available=True),
            model_digest="model-v1",
            data_digest="data-v1",
            version_digest="release-0.8.0",
        )
        self.assertEqual(plan.execution_status, "not_executed")
        self.assertEqual(plan.pool_blocks[0].execution_status, "not_executed")
        self.assertIsNone(plan.pool_blocks[0].observed_placement)
        self.assertEqual(len(plan.context_digest), 64)
        self.assertEqual(plan.bindings["data"], "data-v1")

    def test_unknown_work_proxy_fails_closed_to_local(self):
        block = BlockPlan(
            "unknown",
            "NeuralDensity",
            "gradient",
            Guarantee.HEURISTIC,
            True,
            "pool_eligible",
            "gradient block without a measurement",
        )
        plan = plan_placement(
            EstimationCertificate(Guarantee.HEURISTIC, [block]),
            PoolSpec(available=True),
        )
        self.assertEqual(plan.placements[0].placement, "local")
        self.assertEqual(plan.placements[0].estimate_source, "unavailable")


class PrecisionReceiptContractTest(unittest.TestCase):
    @staticmethod
    def _model():
        return stats.MixtureDistribution(
            [stats.GaussianDistribution(-1.0, 1.0), stats.GaussianDistribution(1.0, 1.0)],
            [0.5, 0.5],
        )

    def test_reduced_precision_is_measured_against_the_requested_error_target(self):
        model = self._model()
        data = list(np.linspace(-2.0, 2.0, 50))
        plan = recommend_compute_precision(model, data, target_rel_error=1.0e-4, sample_size=20)
        self.assertEqual(plan.execution_status, "not_executed")
        self.assertEqual(plan.validation_count, 20)
        self.assertEqual(len(plan.context_digest), 64)
        if plan.reduced():
            self.assertLessEqual(plan.observed_rel_error, plan.target_rel_error)

    def test_optimize_records_the_dtype_that_entered_execution(self):
        model = self._model()
        estimator = model.estimator()
        optimize(
            list(np.linspace(-2.0, 2.0, 50)),
            estimator,
            prev_estimate=model,
            max_its=1,
            out=None,
            precision="minimal",
        )
        plan = estimator.last_precision_plan
        self.assertEqual(plan.execution_status, "executed")
        self.assertIsNotNone(plan.executed_dtype)

    def test_invalid_precision_validation_controls_are_rejected(self):
        with self.assertRaises(ValueError):
            recommend_compute_precision(self._model(), [0.0], target_rel_error=0.0)
        with self.assertRaises(TypeError):
            recommend_compute_precision(self._model(), [0.0], sample_size=1.5)


@unittest.skipUnless(importlib.util.find_spec("jax"), "jax is not installed")
class JITReceiptContractTest(unittest.TestCase):
    def test_scorer_receipt_changes_only_after_observed_execution(self):
        model = stats.GaussianDistribution(0.0, 1.0)
        scorer = jit_seq_log_density(model)
        self.assertEqual(scorer.receipt["execution_status"], "not_executed")
        values = scorer([0.0, 1.0])
        self.assertEqual(values.shape, (2,))
        self.assertEqual(scorer.receipt["execution_status"], "executed")
        self.assertTrue(scorer.receipt["compiled"])
        self.assertEqual(scorer.receipt["output_count"], 2)

    def test_jit_em_can_return_an_observed_execution_receipt(self):
        model = stats.MixtureDistribution(
            [stats.GaussianDistribution(-1.0, 1.0), stats.GaussianDistribution(1.0, 1.0)],
            [0.5, 0.5],
        )
        result = jit_em_mixture(model, [-1.0, -0.5, 0.5, 1.0], max_its=2, return_receipt=True)
        self.assertIsInstance(result, JITExecutionResult)
        self.assertEqual(result.receipt["execution_status"], "executed")
        self.assertEqual(result.receipt["iterations"], 2)
        self.assertGreater(result.receipt["target_evaluations"], 0)

    def test_invalid_jit_controls_fail_before_execution(self):
        with self.assertRaises(ValueError):
            jit_em_mixture(object(), [0.0], max_its=0)


if __name__ == "__main__":
    unittest.main()
