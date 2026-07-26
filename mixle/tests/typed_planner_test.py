"""The typed update graph drives execution only when contracts carry validated evidence.

Unknown compute bands and convergence certificates leave ``optimize`` settings unset rather than
fabricating them from model names or method presence. Compile failures still raise before fitting,
and the narrower typed-adapter limits surface as notes rather than being silently ignored.
"""

import numpy as np
import pytest

from mixle.experimental.typed_runtime import plan_execution
from mixle.inference import optimize
from mixle.stats import GaussianDistribution, LaplaceDistribution, MixtureDistribution


def _gauss_mixture():
    return MixtureDistribution([GaussianDistribution(float(m), 1.0) for m in (-4.0, 0.0, 4.0)], [1 / 3] * 3)


class PlanDerivationTest:
    def test_builtin_tree_defers_precision_and_gate_without_acceptance_evidence(self):
        model = _gauss_mixture()
        plan = plan_execution(model, model.estimator(), nobs=500)
        assert plan.precision is None
        assert plan.monotone is None
        assert plan.blockers == ()
        assert plan.optimize_kwargs == {"monotone": None}
        assert "float64" in plan.explain()
        assert "unknown" in plan.explain()

    def test_unvalidated_family_plans_float64_and_names_the_weakest_link(self):
        model = MixtureDistribution([GaussianDistribution(-4.0, 1.0), LaplaceDistribution(4.0, 2.0)], [0.5, 0.5])
        plan = plan_execution(model, model.estimator(), nobs=500)
        assert plan.precision is None
        assert "precision" not in plan.optimize_kwargs
        assert any("weakest link" in n for n in plan.notes)

    def test_mutable_leaf_is_not_assigned_a_certificate_from_structure(self):
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
        assert plan.monotone is None
        assert any("certificate is unknown" in note for note in plan.notes)

    def test_shared_components_surface_the_adapter_refusal_as_a_note(self):
        shared = GaussianDistribution(0.0, 1.0)
        model = MixtureDistribution([shared, shared], [0.5, 0.5])
        plan = plan_execution(model, model.estimator(), nobs=100)
        assert any("shared components" in n for n in plan.adapter_notes)
        kwargs = plan.optimize_kwargs  # adapter notes are NOT blockers: optimize's full-tree path handles this
        assert "monotone" in kwargs


class PlanExecutionTest:
    def test_optimize_runs_without_claiming_an_unplanned_precision_receipt(self):
        rng = np.random.RandomState(0)
        data = [float(v) for v in np.concatenate([rng.normal(-4, 1, 800), rng.normal(4, 1, 800)])]
        model = MixtureDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.5, 0.5])
        est = model.estimator()
        plan = plan_execution(model, est, nobs=len(data))
        fit = optimize(data, estimator=est, prev_estimate=model, max_its=6, delta=None, **plan.optimize_kwargs)
        assert np.isfinite(sum(fit.log_density(x) for x in data[:10]))
        recorded = getattr(est, "last_precision_plan", None)
        assert plan.precision is None
        assert recorded is None

    def test_blockers_make_optimize_kwargs_refuse(self):
        from mixle.experimental.typed_runtime.planner import ExecutionPlan

        plan = ExecutionPlan(precision=None, monotone=None, blockers=("example blocker",))
        with pytest.raises(RuntimeError, match="example blocker"):
            _ = plan.optimize_kwargs
