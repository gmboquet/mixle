"""The typed update graph drives execution: plan_execution turns contracts into optimize() knobs.

This is the checker-to-planner step: compute band -> precision, convergence certificate -> monotone
gate, compile failures still raising before any fitting, and the (narrower) typed-adapter limits
surfaced as notes rather than silently ignored. The end-to-end test actually RUNS optimize with the
planned kwargs and checks the receipts agree with the plan.
"""

import numpy as np
import pytest

# float32 eligibility exists only where the fused numba kernel does; without numba every band is
# float64 and the discriminating assertions would be vacuous (same gate as typed_compute_band_test).
pytest.importorskip("numba")

from mixle.experimental.typed_runtime import plan_execution  # noqa: E402
from mixle.inference import optimize  # noqa: E402
from mixle.stats import GaussianDistribution, LaplaceDistribution, MixtureDistribution  # noqa: E402


def _gauss_mixture():
    return MixtureDistribution([GaussianDistribution(float(m), 1.0) for m in (-4.0, 0.0, 4.0)], [1 / 3] * 3)


class PlanDerivationTest:
    def test_fp32_safe_monotone_tree_plans_minimal_precision_and_the_strict_gate(self):
        model = _gauss_mixture()
        plan = plan_execution(model, model.estimator(), nobs=500)
        assert plan.precision == "minimal"
        assert plan.monotone is True
        assert plan.blockers == ()
        assert plan.optimize_kwargs == {"monotone": True, "precision": "minimal"}
        assert "float32" in plan.explain() or "minimal" in plan.explain()

    def test_unvalidated_family_plans_float64_and_names_the_weakest_link(self):
        model = MixtureDistribution([GaussianDistribution(-4.0, 1.0), LaplaceDistribution(4.0, 2.0)], [0.5, 0.5])
        plan = plan_execution(model, model.estimator(), nobs=500)
        assert plan.precision is None
        assert "precision" not in plan.optimize_kwargs
        assert any("weakest link" in n for n in plan.notes)

    def test_mutable_leaf_plans_best_visited(self):
        torch = pytest.importorskip("torch")
        from mixle.models import GradLeaf

        class DiagGauss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mu = torch.nn.Parameter(torch.zeros(1))
                self.log_sigma = torch.nn.Parameter(torch.zeros(1))

            def log_density(self, x):
                d = torch.distributions.Normal(self.mu, torch.exp(self.log_sigma))
                return d.log_prob(x if x.dim() > 1 else x.unsqueeze(-1)).sum(-1)

        torch.manual_seed(0)
        model = MixtureDistribution(
            [GradLeaf(DiagGauss(), m_steps=3, lr=0.05), GaussianDistribution(1.0, 1.0)], [0.5, 0.5]
        )
        plan = plan_execution(model, model.estimator(), nobs=500)
        assert plan.monotone is False, "a mutable leaf's certificate must plan best-visited selection"

    def test_shared_components_surface_the_adapter_refusal_as_a_note(self):
        shared = GaussianDistribution(0.0, 1.0)
        model = MixtureDistribution([shared, shared], [0.5, 0.5])
        plan = plan_execution(model, model.estimator(), nobs=100)
        assert any("shared components" in n for n in plan.adapter_notes)
        kwargs = plan.optimize_kwargs  # adapter notes are NOT blockers: optimize's full-tree path handles this
        assert "monotone" in kwargs


class PlanExecutionTest:
    def test_optimize_runs_with_the_planned_kwargs_and_receipts_agree(self):
        rng = np.random.RandomState(0)
        data = [float(v) for v in np.concatenate([rng.normal(-4, 1, 800), rng.normal(4, 1, 800)])]
        model = MixtureDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.5, 0.5])
        est = model.estimator()
        plan = plan_execution(model, est, nobs=len(data))
        fit = optimize(data, estimator=est, prev_estimate=model, max_its=6, delta=None, **plan.optimize_kwargs)
        assert np.isfinite(sum(fit.log_density(x) for x in data[:10]))
        recorded = getattr(est, "last_precision_plan", None)
        assert recorded is not None, "precision='minimal' must engage the runtime precision planner"
        assert np.dtype(recorded.compute_dtype) in (np.float32, np.float64)

    def test_blockers_make_optimize_kwargs_refuse(self):
        from mixle.experimental.typed_runtime.planner import ExecutionPlan

        plan = ExecutionPlan(precision=None, monotone=None, blockers=("example blocker",))
        with pytest.raises(RuntimeError, match="example blocker"):
            _ = plan.optimize_kwargs
