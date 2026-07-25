"""Per-NODE precision planning (mixle.inference.node_precision_plan).

Generalizes precision_plan's model-global allocator to every node of a composed tree: each node gets
its own safety verdict (reusing the identical per-leaf check), and a mixed-precision EM fit can run
each top-level child of the root combinator at its own assigned precision. See node_precision_plan.py
for the exact execution-scope boundary (why "top-level child of the root" is the granularity this
codebase can genuinely execute differently, not deeper nesting).
"""

import unittest
from unittest.mock import patch

import numpy as np
import pytest

import mixle.stats as st
from mixle.inference.node_precision_plan import (
    FUSED_FP32_ABS_LOG_TOLERANCE,
    _data_magnitude_safe,
    mixed_precision_fit,
    recommend_tree_precision,
)
from mixle.utils.optional_deps import HAS_NUMBA

pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="mixed-precision execution uses the fused (numba) kernel")


def _mixed_tree():
    """A MixtureDistribution with two well-conditioned Composite components (safe for float32) and
    one near-degenerate component (unsafe -> must stay float64)."""
    rng = np.random.RandomState(0)
    safe_a = st.CompositeDistribution((st.GaussianDistribution(-3.0, 1.0), st.GaussianDistribution(2.0, 0.8)))
    safe_b = st.CompositeDistribution((st.GaussianDistribution(4.0, 1.2), st.GaussianDistribution(-1.0, 0.6)))
    unsafe = st.CompositeDistribution((st.GaussianDistribution(0.0, 1e-8), st.GaussianDistribution(0.0, 1e-8)))
    m = st.MixtureDistribution([safe_a, safe_b, unsafe], [0.4, 0.4, 0.2])
    data = m.sampler(1).sample(20000)
    return m, data, rng


class RecommendTreePrecisionTest(unittest.TestCase):
    def test_walks_every_node(self):
        m, data, _ = _mixed_tree()
        plan = recommend_tree_precision(m, data)
        # root + 3 components + 2 factors each = 1 + 3 + 6 = 10 nodes
        self.assertEqual(len(plan.nodes), 10)
        self.assertIn((), plan.nodes)
        self.assertIn(("components", "0"), plan.nodes)
        self.assertIn(("components", "0", "dists", "0"), plan.nodes)

    def test_picker_matches_analytic_ground_truth_per_node(self):
        # Ground truth: components 0 and 1 are well-conditioned Gaussians (variance >> min_variance,
        # bounded magnitude) -> float32-safe. Component 2 is near-zero-variance (1e-8 << 1e-6 default
        # floor) -> NOT safe, must stay float64. This is known analytically from construction, not
        # just "close enough": assert the picker's choice matches exactly, node by node.
        m, data, _ = _mixed_tree()
        plan = recommend_tree_precision(m, data)

        self.assertEqual(np.dtype(plan.dtype_for(("components", "0"))), np.float32)
        self.assertEqual(np.dtype(plan.dtype_for(("components", "1"))), np.float32)
        self.assertEqual(np.dtype(plan.dtype_for(("components", "2"))), np.float64)

        # every leaf under components 0/1 is float32; every leaf under component 2 is float64
        for i in (0, 1):
            for j in (0, 1):
                self.assertEqual(np.dtype(plan.dtype_for(("components", str(i), "dists", str(j)))), np.float32)
        for j in (0, 1):
            self.assertEqual(np.dtype(plan.dtype_for(("components", "2", "dists", str(j)))), np.float64)

        # Child scores are always combined at the root in float64.
        self.assertEqual(np.dtype(plan.dtype_for(())), np.float64)
        self.assertEqual(
            plan.top_level_child_paths(),
            [("components", "0"), ("components", "1"), ("components", "2")],
        )

        self.assertIn(plan.dtype_for(("components", "2")), (np.float64,))
        self.assertIn("degenerate", plan.nodes[("components", "2", "dists", "0")].rationale)

    def test_advertised_bound_is_an_absolute_executed_subtree_enclosure(self):
        m, data, _ = _mixed_tree()
        plan = recommend_tree_precision(m, data)
        self.assertLessEqual(
            plan.advertised_bound(("components", "0")),
            len(data) * FUSED_FP32_ABS_LOG_TOLERANCE,
        )
        self.assertEqual(plan.advertised_bound(("components", "2")), 0.0)
        self.assertLessEqual(plan.advertised_bound(()), len(data) * FUSED_FP32_ABS_LOG_TOLERANCE)
        with self.assertRaisesRegex(ValueError, "does not execute independently"):
            plan.advertised_bound(("components", "0", "dists", "0"))

    def test_all_safe_tree_executes_children_in_float32_and_combines_in_float64(self):
        rng = np.random.RandomState(1)
        comps = [
            st.CompositeDistribution(
                tuple(st.GaussianDistribution(float(rng.randn()), float(0.5 + rng.rand())) for _ in range(3))
            )
            for _ in range(3)
        ]
        m = st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(3))))
        data = m.sampler(2).sample(20000)
        plan = recommend_tree_precision(m, data)
        self.assertEqual(np.dtype(plan.dtype_for(())), np.float64)
        self.assertTrue(all(plan.nodes[path].reduced() for path in plan.top_level_child_paths()))

    def test_none_model(self):
        plan = recommend_tree_precision(None, [1.0, 2.0])
        self.assertEqual(np.dtype(plan.dtype_for(())), np.float64)

    def test_nan_in_data_falls_back_every_node_to_float64(self):
        # Regression (MXR-080-0145 sibling): NaN in the sample makes np.max return NaN, and IEEE-754
        # defines every comparison against NaN as False -- so _data_magnitude_safe's
        # `amax > max_magnitude` was False and fell through to the SAFE-looking `return True` instead
        # of the correct float64 verdict, silently promoting otherwise-safe leaves (components 0/1) to
        # float32 even though the data feeding them contains NaN. The data-magnitude check gates every
        # node UNIFORMLY (see recommend_tree_precision's docstring), so a NaN anywhere in the sample
        # must fall every node -- not just the ones near the near-degenerate component 2 -- back to
        # float64.
        m, data, _ = _mixed_tree()
        data = list(data)
        data[0] = (float("nan"),) + tuple(data[0][1:])
        plan = recommend_tree_precision(m, data)
        self.assertTrue(all(not n.reduced() for n in plan.nodes.values()))
        self.assertIn("non-finite", plan.nodes[()].rationale)
        self.assertIn("non-finite", plan.nodes[("components", "0")].rationale)

    def test_non_fusible_subtree_is_reconciled_to_float64(self):
        model = st.CompositeDistribution((st.GaussianDistribution(0.0, 1.0), st.GaussianDistribution(1.0, 2.0)))
        data = model.sampler(0).sample(100)
        with patch("mixle.stats.compute.fused_codegen.fusible", return_value=False):
            plan = recommend_tree_precision(model, data)
        self.assertTrue(all(np.dtype(plan.dtype_for(path)) == np.float64 for path in plan.top_level_child_paths()))


class DataMagnitudeSafeTest(unittest.TestCase):
    """Direct unit coverage of node_precision_plan._data_magnitude_safe's NaN/Inf guard -- the shared
    data-magnitude check recommend_tree_precision gates every node of the tree with uniformly."""

    def test_nan_in_data_is_unsafe(self):
        data = list(np.random.RandomState(11).randn(2000) * 2.0 + 1.0)
        data[500] = float("nan")
        safe, rationale, amax = _data_magnitude_safe(data, 1e6, 4096)
        self.assertFalse(safe)
        self.assertIn("non-finite", rationale)
        self.assertIsNone(amax)

    def test_all_nan_data_is_unsafe(self):
        safe, rationale, amax = _data_magnitude_safe([float("nan")] * 50, 1e6, 4096)
        self.assertFalse(safe)
        self.assertIn("non-finite", rationale)
        self.assertIsNone(amax)

    def test_inf_in_data_is_unsafe(self):
        # Analogous to the NaN case above. +inf/-inf happen to already be caught by the `amax`
        # magnitude check (abs(inf) > 1e6 is True), but the guard must be explicit rather than relying
        # on that incidental comparison behavior.
        pos_inf = list(np.random.RandomState(12).randn(2000) * 2.0 + 1.0)
        pos_inf[500] = float("inf")
        safe, _, _ = _data_magnitude_safe(pos_inf, 1e6, 4096)
        self.assertFalse(safe)

        neg_inf = list(np.random.RandomState(13).randn(2000) * 2.0 + 1.0)
        neg_inf[500] = float("-inf")
        safe, _, _ = _data_magnitude_safe(neg_inf, 1e6, 4096)
        self.assertFalse(safe)

    def test_finite_well_conditioned_data_is_safe(self):
        # Negative control for the NaN/Inf guard: finite, well-conditioned data must be unaffected.
        data = list(np.random.RandomState(14).randn(2000) * 2.0 + 1.0)
        safe, rationale, amax = _data_magnitude_safe(data, 1e6, 4096)
        self.assertTrue(safe)
        self.assertIsNotNone(amax)


class MixedPrecisionFitTest(unittest.TestCase):
    def test_mixed_precision_fit_uses_runtime_validated_scores(self):
        m, data, _ = _mixed_tree()
        plan = recommend_tree_precision(m, data)

        from mixle.inference import optimize

        f64_fit = optimize(data, m.estimator(), prev_estimate=m, max_its=15, out=None)
        mixed_fit = mixed_precision_fit(m, data, plan=plan, max_its=15)

        total_ll_f64 = float(f64_fit.seq_log_density(f64_fit.dist_to_encoder().seq_encode(data)).sum())
        total_ll_mixed = float(mixed_fit.seq_log_density(mixed_fit.dist_to_encoder().seq_encode(data)).sum())

        self.assertTrue(np.isfinite(total_ll_mixed))
        self.assertLess(abs(total_ll_mixed - total_ll_f64) / abs(total_ll_f64), 1e-3)
        self.assertLessEqual(plan.advertised_bound(()), len(data) * FUSED_FP32_ABS_LOG_TOLERANCE)

    def test_unsafe_component_is_byte_identical_to_float64(self):
        # Regression: when EVERY node is unsafe (all float64), the mixed-precision driver must
        # reproduce the ordinary float64 fit exactly (no precision change actually applied).
        rng = np.random.RandomState(3)
        comps = [
            st.CompositeDistribution((st.GaussianDistribution(0.0, 1e-8), st.GaussianDistribution(1.0, 1e-8)))
            for _ in range(2)
        ]
        m = st.MixtureDistribution(comps, [0.5, 0.5])
        data = m.sampler(4).sample(4000)
        plan = recommend_tree_precision(m, data)
        self.assertTrue(all(not n.reduced() for n in plan.nodes.values()))

        from mixle.inference import optimize

        f64_fit = optimize(data, m.estimator(), prev_estimate=m, max_its=8, out=None)
        mixed_fit = mixed_precision_fit(m, data, plan=plan, max_its=8)

        self.assertTrue(np.allclose(sorted(f64_fit.w), sorted(mixed_fit.w), atol=1e-10))

    def test_plan_is_bound_to_its_validation_data(self):
        m, data, _ = _mixed_tree()
        plan = recommend_tree_precision(m, data)
        changed = list(data)
        changed[0] = changed[1]
        with self.assertRaisesRegex(ValueError, "different data"):
            mixed_precision_fit(m, changed, plan=plan, max_its=1)

    def test_impossible_mixture_rows_are_rejected_not_replaced_by_priors(self):
        model = st.MixtureDistribution(
            [st.CategoricalDistribution({"a": 1.0}), st.CategoricalDistribution({"b": 1.0})],
            [0.5, 0.5],
        )
        data = ["c"]
        plan = recommend_tree_precision(model, data)
        with self.assertRaisesRegex(ValueError, "zero probability"):
            mixed_precision_fit(model, data, plan=plan, max_its=1)


if __name__ == "__main__":
    unittest.main()
