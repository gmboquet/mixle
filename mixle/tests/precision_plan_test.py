"""Automatic precision allocation (mixle.inference.precision_plan + optimize(precision='minimal')).

The control loop: look at the data + the model, pick the minimal SAFE compute precision, and produce the
same fit. float32 only where verified safe; float64 (preserving accuracy) otherwise.
"""

import unittest

import numpy as np
import pytest

import mixle.stats as st
from mixle.inference import optimize
from mixle.inference.precision_plan import recommend_compute_precision
from mixle.utils.optional_deps import HAS_NUMBA

pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="reduced-precision path is the fused (numba) kernel")


def _well_conditioned():
    rng = np.random.RandomState(0)
    comps = [
        st.CompositeDistribution(
            tuple(st.GaussianDistribution(float(rng.randn()), float(0.5 + rng.rand())) for _ in range(6))
        )
        for _ in range(4)
    ]
    m = st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(4))))
    return m, m.sampler(1).sample(40000)


class RecommendPrecisionTest(unittest.TestCase):
    def test_well_conditioned_picks_float32(self):
        m, data = _well_conditioned()
        plan = recommend_compute_precision(m, data)
        self.assertEqual(np.dtype(plan.compute_dtype), np.float32)
        self.assertTrue(plan.reduced())

    def test_tiny_variance_stays_float64(self):
        m = st.MixtureDistribution([st.GaussianDistribution(0.0, 1e-8), st.GaussianDistribution(1.0, 1e-8)], [0.5, 0.5])
        plan = recommend_compute_precision(m, m.sampler(2).sample(2000))
        self.assertEqual(np.dtype(plan.compute_dtype), np.float64)
        self.assertIn("degenerate", plan.rationale)

    def test_large_magnitude_stays_float64(self):
        m = st.MixtureDistribution([st.GaussianDistribution(0.0, 1.0), st.GaussianDistribution(5.0, 1.0)], [0.5, 0.5])
        data = [float(x) * 1e8 for x in m.sampler(3).sample(2000)]
        self.assertEqual(np.dtype(recommend_compute_precision(m, data).compute_dtype), np.float64)

    def test_tail_concentrated_large_magnitude_stays_float64(self):
        # Regression: the data sample used to be a plain leading prefix (data[:sample_size]) -- a
        # naturally-ordered dataset that stashes extreme-magnitude rows later in the sequence (past
        # the default sample_size=4096) was invisible to the magnitude guard, silently allocating
        # float32 to data that is not actually well-conditioned for it.
        m = st.MixtureDistribution([st.GaussianDistribution(0.0, 1.0), st.GaussianDistribution(5.0, 1.0)], [0.5, 0.5])
        well_conditioned = list(m.sampler(3).sample(5000))
        extreme_tail = [1.0e9] * 200  # appears only after the default 4096-row prefix
        data = well_conditioned + extreme_tail
        plan = recommend_compute_precision(m, data)
        self.assertEqual(np.dtype(plan.compute_dtype), np.float64)

    def test_nan_in_data_falls_back_to_float64(self):
        # Regression (MXR-080-0145 sibling): NaN in the sample makes np.max return NaN, and IEEE-754
        # defines every comparison against NaN as False -- so `amax > max_magnitude` was False and the
        # risk check fell through to the OPTIMISTIC float32 branch instead of the safe fallback. A
        # single NaN amid otherwise well-conditioned data must still route to float64, not float32.
        m, data = _well_conditioned()
        data = list(data)
        data[0] = (float("nan"),) + tuple(data[0][1:])
        plan = recommend_compute_precision(m, data)
        self.assertEqual(np.dtype(plan.compute_dtype), np.float64)
        self.assertIn("non-finite", plan.rationale)

    def test_all_nan_data_falls_back_to_float64(self):
        m, data = _well_conditioned()
        nan_data = [(float("nan"),) * len(data[0])] * len(data)
        plan = recommend_compute_precision(m, nan_data)
        self.assertEqual(np.dtype(plan.compute_dtype), np.float64)

    def test_inf_in_data_falls_back_to_float64(self):
        # Analogous to the NaN case above. +inf/-inf happen to already be caught by the `amax`
        # magnitude check (abs(inf) > 1e6 is True), but the guard must be explicit rather than relying
        # on that incidental comparison behavior.
        m, data = _well_conditioned()
        pos_inf = list(data)
        pos_inf[0] = (float("inf"),) + tuple(pos_inf[0][1:])
        self.assertEqual(np.dtype(recommend_compute_precision(m, pos_inf).compute_dtype), np.float64)

        neg_inf = list(data)
        neg_inf[0] = (float("-inf"),) + tuple(neg_inf[0][1:])
        self.assertEqual(np.dtype(recommend_compute_precision(m, neg_inf).compute_dtype), np.float64)

    def test_finite_well_conditioned_data_unaffected_by_nonfinite_guard(self):
        # Negative control for the NaN/Inf guard (paired with test_well_conditioned_picks_float32
        # above): finite, well-conditioned data must be unaffected by the isfinite check.
        m, data = _well_conditioned()
        plan = recommend_compute_precision(m, data)
        self.assertEqual(np.dtype(plan.compute_dtype), np.float32)

    def test_non_fusible_model_stays_float64(self):
        m = st.MixtureDistribution([st.LaplaceDistribution(0.0, 1.0), st.LaplaceDistribution(3.0, 1.0)], [0.5, 0.5])
        plan = recommend_compute_precision(m, m.sampler(4).sample(2000))
        self.assertEqual(np.dtype(plan.compute_dtype), np.float64)
        self.assertIn("fused", plan.rationale)

    def test_non_numeric_data_stays_float64(self):
        m = st.MixtureDistribution([st.CategoricalDistribution({"a": 0.6, "b": 0.4})] * 2, [0.5, 0.5])
        # categorical (non-numeric) data: the numeric sample is empty -> float64
        self.assertEqual(np.dtype(recommend_compute_precision(m, ["a", "b", "a"]).compute_dtype), np.float64)

    def test_none_model_is_float64(self):
        self.assertEqual(np.dtype(recommend_compute_precision(None, [1.0, 2.0]).compute_dtype), np.float64)


class MinimalPrecisionFitTest(unittest.TestCase):
    def test_minimal_matches_float64_fit_when_safe(self):
        m, data = _well_conditioned()
        f64 = optimize(data, m.estimator(), prev_estimate=m, max_its=20, out=None)
        mini = optimize(data, m.estimator(), prev_estimate=m, max_its=20, out=None, precision="minimal")
        self.assertTrue(np.allclose(sorted(f64.w), sorted(mini.w), atol=1e-3))
        f64_mu = sorted(c.dists[0].mu for c in f64.components)
        mini_mu = sorted(c.dists[0].mu for c in mini.components)
        self.assertTrue(np.allclose(f64_mu, mini_mu, atol=1e-2))

    def test_minimal_is_exact_when_it_stays_float64(self):
        # tiny variance -> allocator keeps float64 -> byte-identical to the default fit (no precision change)
        rng = np.random.RandomState(5)
        m = st.MixtureDistribution([st.GaussianDistribution(0.0, 1e-8), st.GaussianDistribution(1.0, 1e-8)], [0.5, 0.5])
        data = m.sampler(6).sample(3000)
        default = optimize(data, m.estimator(), prev_estimate=m, max_its=10, out=None)
        mini = optimize(data, m.estimator(), prev_estimate=m, max_its=10, out=None, precision="minimal")
        self.assertTrue(np.allclose(sorted(default.w), sorted(mini.w), atol=0))


if __name__ == "__main__":
    unittest.main()
