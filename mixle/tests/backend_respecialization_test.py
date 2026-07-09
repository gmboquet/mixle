"""Tests for the D6 backend re-specialization module (mixle.inference.backend_respecialization).

Acceptance criteria per the ConditionalJIT roadmap (D6):
  1. Tolerance-equal output across a backend swap (re-specializing changes speed, not correctness).
  2. A measured, real speedup from at least one re-specialization decision.
  3. Compile-economics decision correctness on a synthetic many-vs-few-remaining-calls boundary.
"""

from __future__ import annotations

import importlib.util
import time
import unittest

import numpy as np

from mixle.inference.backend_respecialization import (
    DensityTable,
    NodeBackend,
    RespecializationAction,
    compile_forward,
    decide_density_table,
    decide_frozen_precision_drop,
    decide_hot_compile,
    estimate_compile_benefit,
    estimate_compile_cost,
)
from mixle.inference.node_report import node_report
from mixle.stats import ExponentialDistribution, GaussianDistribution, MixtureDistribution

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from mixle.engines.torch_engine import TorchEngine


def _hot_mixture_and_forward(n_components: int = 64, n_obs: int = 50_000):
    """A mixture big enough that a fused torch.compile graph genuinely beats eager per-call
    overhead, but small enough to keep the test fast (see calibration in the PR description)."""
    components = [GaussianDistribution(float(i) * 0.1, 1.0 + 0.01 * i) for i in range(n_components)]
    weights = [1.0 / n_components] * n_components
    dist = MixtureDistribution(components, weights)
    data = dist.sampler(seed=1).sample(size=n_obs)
    enc = dist.dist_to_encoder().seq_encode(data)
    engine = TorchEngine(dtype=torch.float64, compile=True)

    def forward(enc):
        return dist.kernel(engine=engine).score(enc)

    return dist, engine, enc, forward


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class ToleranceEqualAcrossSwapTest(unittest.TestCase):
    """Acceptance (1): re-specializing must not change WHAT is computed."""

    def test_compiled_forward_matches_eager_forward(self):
        dist, engine, enc, forward = _hot_mixture_and_forward()
        eager_out = forward(enc)
        compiled = compile_forward(forward, engine=engine)
        compiled_out = compiled(enc)

        max_diff = float(torch.max(torch.abs(compiled_out - eager_out)))
        print("tolerance-equal (compile): max |eager - compiled| = %.3e" % max_diff)
        self.assertLess(max_diff, 1.0e-9)

    def test_node_backend_output_identical_before_and_after_apply(self):
        dist, engine, enc, forward = _hot_mixture_and_forward(n_components=8, n_obs=2000)
        backend = NodeBackend(dist, forward=forward, engine=engine)
        before = backend(enc)

        report = node_report(dist, field_path="root", nobs=float(2000))
        decision = decide_hot_compile(report, activation_ratio=1.0, expected_remaining_calls=1000.0)
        self.assertEqual(decision.action, RespecializationAction.COMPILE)
        backend.apply(decision)
        after = backend(enc)

        max_diff = float(torch.max(torch.abs(after - before)))
        print("tolerance-equal (NodeBackend swap): max |before - after| = %.3e" % max_diff)
        self.assertLess(max_diff, 1.0e-9)

    def test_density_table_matches_direct_call(self):
        dist = ExponentialDistribution(1.5)

        def forward(x):
            return dist.log_density(x)

        seed_points = [0.1, 0.5, 1.0, 2.0, 3.5]
        table = DensityTable(forward, seed_points)
        for x in seed_points + [0.1, 1.0]:  # repeat two points to exercise real cache hits
            direct = forward(x)
            cached = table.lookup(x)
            self.assertAlmostEqual(direct, cached, places=12)
        # New point outside the seed set: falls back to the function AND gets cached.
        self.assertEqual(len(table), len(seed_points))
        table.lookup(9.0)
        self.assertEqual(len(table), len(seed_points) + 1)
        self.assertAlmostEqual(table.lookup(9.0), forward(9.0), places=12)


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class MeasuredSpeedupTest(unittest.TestCase):
    """Acceptance (2): a measured, real speedup from re-specialization."""

    def test_compiled_hot_leaf_is_faster_than_eager_after_warmup(self):
        _, engine, enc, forward = _hot_mixture_and_forward()
        compiled = compile_forward(forward, engine=engine)

        n_warmup_calls = 3
        n_timed_calls = 15

        # torch.compile pays a real, expected upfront tracing/codegen cost on its first call(s) --
        # document rather than hide this: warm it up before timing, exactly as the module docstring
        # says a caller must for compilation's benefit to actually materialize.
        warmup_start = time.perf_counter()
        for _ in range(n_warmup_calls):
            compiled(enc)
        warmup_elapsed = time.perf_counter() - warmup_start

        t0 = time.perf_counter()
        for _ in range(n_timed_calls):
            forward(enc)
        eager_elapsed = time.perf_counter() - t0
        eager_per_call = eager_elapsed / n_timed_calls

        t0 = time.perf_counter()
        for _ in range(n_timed_calls):
            compiled(enc)
        compiled_elapsed = time.perf_counter() - t0
        compiled_per_call = compiled_elapsed / n_timed_calls

        speedup = eager_per_call / compiled_per_call
        print(
            "measured speedup: eager=%.5fs/call compiled=%.5fs/call (post-warmup, warmup=%.3fs over %d calls) "
            "-> %.2fx" % (eager_per_call, compiled_per_call, warmup_elapsed, n_warmup_calls, speedup)
        )
        self.assertGreater(
            speedup,
            1.05,
            "expected a real (>5%%) speedup from a compiled hot leaf after warmup; measured %.2fx" % speedup,
        )


class CompileEconomicsDecisionTest(unittest.TestCase):
    """Acceptance (3): the decision-boundary test -- many vs. few expected remaining calls."""

    def _hot_report(self):
        # nobs=1.0 keeps the per-call cost proxy small (a few units), so the compile-vs-don't
        # boundary is driven by "how many more times will this run", not swamped by a single
        # round's dataset size -- the point of this test.
        dist = GaussianDistribution(0.0, 1.0)
        return node_report(dist, field_path="root", nobs=1.0)

    def test_recommends_compile_when_many_calls_remain(self):
        report = self._hot_report()
        cost = estimate_compile_cost(report)
        many_calls = cost * 1000.0  # comfortably enough calls to amortize the fixed overhead
        benefit = estimate_compile_benefit(report, many_calls)
        self.assertGreater(benefit, cost)

        decision = decide_hot_compile(report, activation_ratio=1.0, expected_remaining_calls=many_calls)
        self.assertEqual(decision.action, RespecializationAction.COMPILE)
        self.assertTrue(decision.worth_it)
        self.assertGreater(decision.net_benefit, 0.0)

    def test_does_not_recommend_compile_when_few_calls_remain(self):
        report = self._hot_report()
        few_calls = 1.0  # a handful more executions -- nowhere near enough to pay back a compile
        cost = estimate_compile_cost(report)
        benefit = estimate_compile_benefit(report, few_calls)
        self.assertLess(benefit, cost)  # sanity-check the economics before checking the decision

        decision = decide_hot_compile(report, activation_ratio=1.0, expected_remaining_calls=few_calls)
        self.assertEqual(decision.action, RespecializationAction.NONE)
        self.assertFalse(decision.worth_it)
        self.assertLessEqual(decision.net_benefit, 0.0)

    def test_cold_node_is_never_a_compile_candidate_regardless_of_calls(self):
        report = self._hot_report()
        decision = decide_hot_compile(report, activation_ratio=0.1, expected_remaining_calls=1.0e9)
        self.assertEqual(decision.action, RespecializationAction.NONE)
        self.assertIn("not hot enough", decision.rationale)

    def test_frozen_node_is_never_a_compile_candidate(self):
        from mixle.stats import NullDistribution

        dist = NullDistribution()
        report = node_report(dist, field_path="root")
        self.assertEqual(report.update_kind, "frozen")
        decision = decide_hot_compile(report, activation_ratio=1.0, expected_remaining_calls=1.0e9)
        self.assertEqual(decision.action, RespecializationAction.NONE)
        self.assertIn("frozen", decision.rationale)

    def test_already_compiled_node_is_left_alone(self):
        report = self._hot_report()
        decision = decide_hot_compile(
            report, activation_ratio=1.0, expected_remaining_calls=1.0e9, already_compiled=True
        )
        self.assertEqual(decision.action, RespecializationAction.NONE)


class FrozenPrecisionDropTest(unittest.TestCase):
    def test_frozen_node_recommends_reduced_precision(self):
        from mixle.stats import NullDistribution

        report = node_report(NullDistribution(), field_path="root")
        decision = decide_frozen_precision_drop(report)
        self.assertEqual(decision.action, RespecializationAction.REDUCE_PRECISION)

    def test_converged_q_gain_recommends_reduced_precision(self):
        dist = GaussianDistribution(0.0, 1.0)
        report = node_report(dist, field_path="root", prev_residual=None)
        # Manually construct a converged report: q_gain within tolerance of zero.
        import dataclasses

        converged = dataclasses.replace(report, update_kind="closed_form", q_gain=1.0e-9)
        decision = decide_frozen_precision_drop(converged, q_gain_tol=1.0e-6)
        self.assertEqual(decision.action, RespecializationAction.REDUCE_PRECISION)

    def test_still_moving_node_keeps_full_precision(self):
        dist = GaussianDistribution(0.0, 1.0)
        report = node_report(dist, field_path="root")
        import dataclasses

        moving = dataclasses.replace(report, update_kind="closed_form", q_gain=5.0)
        decision = decide_frozen_precision_drop(moving, q_gain_tol=1.0e-6)
        self.assertEqual(decision.action, RespecializationAction.NONE)

    def test_already_reduced_is_left_alone(self):
        report = node_report(GaussianDistribution(0.0, 1.0), field_path="root")
        decision = decide_frozen_precision_drop(report, already_reduced=True)
        self.assertEqual(decision.action, RespecializationAction.NONE)


class DensityTableDecisionTest(unittest.TestCase):
    def test_stable_closed_form_recommends_density_table(self):
        dist = ExponentialDistribution(1.5)
        report = node_report(dist, field_path="root", nobs=1.0)
        # A one-parameter conjugate-exponential-family leaf: fully closed-form, no gradient loop.
        self.assertIn(report.update_kind, ("closed_form", "conjugate_closed_form"))
        decision = decide_density_table(report, expected_remaining_calls=1.0e6, n_query_points=8, structure_stable=True)
        self.assertEqual(decision.action, RespecializationAction.DENSITY_TABLE)

    def test_unstable_structure_declines_density_table(self):
        dist = ExponentialDistribution(1.5)
        report = node_report(dist, field_path="root", nobs=1.0)
        decision = decide_density_table(
            report, expected_remaining_calls=1.0e6, n_query_points=8, structure_stable=False
        )
        self.assertEqual(decision.action, RespecializationAction.NONE)

    def test_gradient_node_has_no_closed_form_to_table_cache(self):
        dist = ExponentialDistribution(1.5)
        report = node_report(dist, field_path="root", nobs=1.0)
        import dataclasses

        gradient_report = dataclasses.replace(report, update_kind="gradient")
        decision = decide_density_table(
            gradient_report, expected_remaining_calls=1.0e6, n_query_points=8, structure_stable=True
        )
        self.assertEqual(decision.action, RespecializationAction.NONE)

    def test_too_few_remaining_calls_declines_density_table(self):
        dist = ExponentialDistribution(1.5)
        report = node_report(dist, field_path="root", nobs=1.0)
        decision = decide_density_table(report, expected_remaining_calls=1.0, n_query_points=8, structure_stable=True)
        self.assertEqual(decision.action, RespecializationAction.NONE)


class NodeBackendMechanismTest(unittest.TestCase):
    """Exercises NodeBackend/DensityTable end-to-end on plain numpy nodes (no torch required)."""

    def test_density_table_backend_swap_matches_eager_and_populates_cache(self):
        dist = ExponentialDistribution(1.5)
        backend = NodeBackend(dist, forward=lambda x: dist.log_density(x))
        query_points = [0.1, 0.5, 1.0, 2.0]
        before = [backend(x) for x in query_points]

        report = node_report(dist, field_path="root", nobs=1000.0)
        decision = decide_density_table(
            report, expected_remaining_calls=1.0e6, n_query_points=len(query_points), structure_stable=True
        )
        self.assertEqual(decision.action, RespecializationAction.DENSITY_TABLE)
        backend.apply(decision, table_seed_points=query_points)
        self.assertEqual(backend.action, RespecializationAction.DENSITY_TABLE)

        after = [backend(x) for x in query_points]
        np.testing.assert_allclose(before, after, atol=1.0e-12, rtol=1.0e-12)

        # A repeat query is a genuine cache hit (table doesn't grow past the seed set).
        table_len_before = len(backend._table)
        backend(query_points[0])
        self.assertEqual(len(backend._table), table_len_before)

    def test_none_action_leaves_backend_eager(self):
        dist = ExponentialDistribution(1.5)
        backend = NodeBackend(dist, forward=lambda x: dist.log_density(x))
        report = node_report(dist, field_path="root", nobs=1.0)
        decision = decide_hot_compile(report, activation_ratio=0.0, expected_remaining_calls=1.0)
        backend.apply(decision)
        self.assertEqual(backend.action, RespecializationAction.NONE)
        self.assertAlmostEqual(backend(1.0), dist.log_density(1.0), places=12)


if __name__ == "__main__":
    unittest.main()
