import importlib
import unittest

import numpy as np
import pytest

from pysp.inference.gradient_fit import (
    _gradient_log_prior_state,
    _gradient_raw_state,
    _torch_for_gradient_fit,
)
from pysp.inference.priors import GammaPrior
from pysp.stats import PoissonDistribution, UniformDistribution

pytestmark = [pytest.mark.torch, pytest.mark.optional]

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class GammaPriorClampAuditTestCase(unittest.TestCase):
    """Regression: the gamma MAP prior log-density must not poison the objective when a
    saturated Adam tail drives a positive-constrained parameter's raw value so negative that
    ``torch.exp(raw)`` underflows to exactly 0.0 (making ``torch.log(value) == -inf``)."""

    def _build_state(self, dist):
        torch, engine = _torch_for_gradient_fit(None)
        leaves = []
        state = _gradient_raw_state(dist, engine, torch, leaves)
        return torch, engine, state

    def _saturate(self, torch, raw, raw_name):
        # Drive the raw log-parameter so negative that exp() underflows to *exactly* 0.0,
        # reproducing the saturated-tail failure mode the clamp guards against.
        raw[raw_name] = torch.tensor(-1.0e3, dtype=raw[raw_name].dtype, requires_grad=True)
        assert float(torch.exp(raw[raw_name])) == 0.0

    def test_positive_gamma_branch_finite_at_underflow_shape_one(self):
        # shape == 1.0 (the common default): (shape - 1) * log(value) == 0.0 * (-inf) == NaN
        # before the clamp.
        torch, engine, state = self._build_state(PoissonDistribution(2.0))
        self._saturate(torch, state[3], "log_lam")
        priors = GammaPrior(shape=1.0, rate=2.0, parameter="lam").as_dict()
        lp = _gradient_log_prior_state(state, priors, 0.0, torch, engine, {})
        self.assertTrue(np.isfinite(float(lp)), f"gamma log-prior was non-finite: {float(lp)!r}")

    def test_positive_gamma_branch_finite_at_underflow_sparse_shape(self):
        # shape < 1.0 (sparsity prior): (shape - 1) * log(value) diverges to +inf before the clamp.
        torch, engine, state = self._build_state(PoissonDistribution(2.0))
        self._saturate(torch, state[3], "log_lam")
        priors = GammaPrior(shape=0.5, rate=2.0, parameter="lam").as_dict()
        lp = _gradient_log_prior_state(state, priors, 0.0, torch, engine, {})
        self.assertTrue(np.isfinite(float(lp)), f"gamma log-prior was non-finite: {float(lp)!r}")

    def test_ordered_bound_gamma_branch_finite_at_underflow(self):
        # The coupled (ordered-bound delta) gamma branch shares the same unclamped log.
        dist = UniformDistribution(-5.0, 5.0)
        torch, engine, state = self._build_state(dist)
        raw = state[3]
        delta_raw = next(k for k in raw if "high" in k or "minus" in k or "delta" in k)
        self._saturate(torch, raw, delta_raw)
        priors = GammaPrior(shape=1.0, rate=10.0, parameter="high_minus_low").as_dict()
        lp = _gradient_log_prior_state(state, priors, 0.0, torch, engine, {})
        self.assertTrue(np.isfinite(float(lp)), f"ordered-bound gamma log-prior was non-finite: {float(lp)!r}")


if __name__ == "__main__":
    unittest.main()
