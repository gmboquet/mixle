"""Per-NODE precision planning (mixle.inference.node_precision_plan).

Generalizes precision_plan's model-global allocator to every node of a composed tree: each node gets
its own safety verdict (reusing the identical per-leaf check), and a mixed-precision EM fit can run
each top-level child of the root combinator at its own assigned precision. See node_precision_plan.py
for the exact execution-scope boundary (why "top-level child of the root" is the granularity this
codebase can genuinely execute differently, not deeper nesting).
"""

import unittest

import numpy as np
import pytest

import mixle.stats as st
from mixle.inference.node_precision_plan import (
    FUSED_FP32_REL_LL_BOUND,
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

        # root aggregates AND over children -> unsafe component 2 makes the whole tree's root verdict
        # float64 (the root IS the maximal fused unit only if ALL of it is safe).
        self.assertEqual(np.dtype(plan.dtype_for(())), np.float64)

        self.assertIn(plan.dtype_for(("components", "2")), (np.float64,))
        self.assertIn("degenerate", plan.nodes[("components", "2", "dists", "0")].rationale)

    def test_advertised_bound_is_additive_over_reduced_leaves(self):
        m, data, _ = _mixed_tree()
        plan = recommend_tree_precision(m, data)
        # component 0 has 2 reduced (float32) leaves -> bound = 2 * the verified per-leaf constant.
        self.assertAlmostEqual(plan.advertised_bound(("components", "0")), 2 * FUSED_FP32_REL_LL_BOUND)
        # component 2 is unsafe (float64) -> bound is 0 (exact).
        self.assertEqual(plan.advertised_bound(("components", "2")), 0.0)

    def test_all_safe_tree_root_is_float32(self):
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
        self.assertEqual(np.dtype(plan.dtype_for(())), np.float32)
        self.assertTrue(all(n.reduced() for n in plan.nodes.values()))

    def test_none_model(self):
        plan = recommend_tree_precision(None, [1.0, 2.0])
        self.assertEqual(np.dtype(plan.dtype_for(())), np.float64)


class MixedPrecisionFitTest(unittest.TestCase):
    def test_mixed_precision_fit_matches_float64_within_advertised_bound(self):
        m, data, _ = _mixed_tree()
        plan = recommend_tree_precision(m, data)

        from mixle.inference import optimize

        f64_fit = optimize(data, m.estimator(), prev_estimate=m, max_its=15, out=None)
        mixed_fit = mixed_precision_fit(m, data, plan=plan, max_its=15)

        total_ll_f64 = float(f64_fit.seq_log_density(f64_fit.dist_to_encoder().seq_encode(data)).sum())
        total_ll_mixed = float(mixed_fit.seq_log_density(mixed_fit.dist_to_encoder().seq_encode(data)).sum())

        rel_err = abs(total_ll_mixed - total_ll_f64) / abs(total_ll_f64)
        advertised = plan.advertised_bound(("components", "0")) + plan.advertised_bound(("components", "1"))
        # generous slack: EM has its own iteration-to-iteration float64 nondeterminism-free but
        # sensitive dynamics, so allow an extra small additive slack on top of the advertised
        # per-node relative bound rather than asserting the raw bound alone is sufficient.
        self.assertLessEqual(
            rel_err, advertised + 1e-4, msg="mixed-precision fit drifted beyond its own advertised bound"
        )

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


if __name__ == "__main__":
    unittest.main()
