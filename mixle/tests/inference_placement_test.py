"""Placement planning (A4): the local-vs-pool axis -- closed forms stay local, gradients may offload."""

import unittest

from mixle.inference import PoolSpec, plan_placement
from mixle.inference.planning import BlockPlan, EstimationCertificate, Guarantee
from mixle.telemetry import Telemetry


def _cert(blocks):
    agg = min((b.guarantee for b in blocks), default=Guarantee.HEURISTIC)
    return EstimationCertificate(guarantee=agg, blocks=blocks)


def _closed(name):
    return BlockPlan(name, "Gaussian", "closed_form_mle", Guarantee.GLOBAL_UNIQUE, False, "local", "exp family")


def _grad(name, tflop_reason="gradient residual ~8.0 TFLOP"):
    return BlockPlan(name, "NeuralDensity", "gradient", Guarantee.HEURISTIC, True, "pool_eligible", tflop_reason)


class NoPoolTest(unittest.TestCase):
    def test_everything_local_without_a_pool(self):
        plan = plan_placement(_cert([_closed("a"), _grad("b")]), PoolSpec(available=False))
        self.assertEqual(len(plan.pool_blocks), 0)
        self.assertEqual(len(plan.local_blocks), 2)
        self.assertEqual(plan.est_pool_cost, 0.0)

    def test_default_spec_is_no_pool(self):
        plan = plan_placement(_cert([_grad("b")]))
        self.assertEqual(len(plan.pool_blocks), 0)


class WithPoolTest(unittest.TestCase):
    def test_only_heavy_gradient_blocks_offload(self):
        cert = _cert([_closed("mix"), _grad("neural"), _closed("gauss")])
        plan = plan_placement(cert, PoolSpec(available=True, flop_threshold_tflop=1.0))
        self.assertEqual(len(plan.pool_blocks), 1)
        self.assertEqual(plan.pool_blocks[0].name, "neural")
        self.assertTrue(all(p.placement == "local" for p in plan.placements if p.name != "neural"))

    def test_closed_form_never_offloads_even_with_a_pool(self):
        plan = plan_placement(_cert([_closed("a"), _closed("b")]), PoolSpec(available=True))
        self.assertEqual(len(plan.pool_blocks), 0)

    def test_small_gradient_block_stays_local_below_threshold(self):
        cert = _cert([_grad("tiny", "gradient residual ~0.2 TFLOP")])
        plan = plan_placement(cert, PoolSpec(available=True, flop_threshold_tflop=1.0))
        self.assertEqual(len(plan.pool_blocks), 0)  # not worth the round-trip
        self.assertIn("below", plan.placements[0].reason)

    def test_pool_block_is_priced(self):
        plan = plan_placement(_cert([_grad("neural")]), PoolSpec(available=True, cost_per_hour=1.0))
        self.assertGreater(plan.pool_blocks[0].est_cost, 0.0)
        self.assertEqual(plan.est_pool_cost, plan.pool_blocks[0].est_cost)


class TelemetryTest(unittest.TestCase):
    def test_emits_a_placement_event_per_block(self):
        tel = Telemetry()
        plan_placement(_cert([_closed("a"), _grad("b")]), PoolSpec(available=True), telemetry=tel)
        events = list(tel.events(kind="placement"))
        self.assertEqual(len(events), 2)
        self.assertEqual({e.choice for e in events}, {"local", "pool"})


class ReportTest(unittest.TestCase):
    def test_report_and_as_dict(self):
        plan = plan_placement(_cert([_closed("a"), _grad("b")]), PoolSpec(available=True))
        self.assertIn("PlacementPlan", plan.report())
        d = plan.as_dict()
        self.assertEqual(d["n_blocks"], 2)
        self.assertEqual(d["n_pool"], 1)


if __name__ == "__main__":
    unittest.main()
