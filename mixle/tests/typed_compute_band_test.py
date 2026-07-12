"""Compute-band axis of the typed update graph: what precision the model STRUCTURE permits.

Float32 eligibility is declared from the same validated family set the runtime planner uses
(single source of truth), the tree-level band composes by weakest link, and the data-side
safety checks remain a runtime decision -- this axis answers "could this subtree ever take the
validated reduced-precision kernel," not "should this fit."
"""

import pytest

# float32 eligibility exists only where the fused numba kernel does; without numba every band is
# correctly float64 and the discriminating assertions below would be vacuous.
pytest.importorskip("numba")

from mixle.experimental.typed_runtime import compile_update_graph
from mixle.experimental.typed_runtime.contracts import ComputeBand, weakest_band
from mixle.stats import GaussianDistribution, LaplaceDistribution, MixtureDistribution


def _graph(model):
    return compile_update_graph(model, model.estimator(), nobs=500)


class ComputeBandTest:
    def test_fp32_safe_classical_tree_is_eligible_end_to_end(self):
        model = MixtureDistribution([GaussianDistribution(float(m), 1.0) for m in (-4.0, 0.0, 4.0)], [1 / 3] * 3)
        graph = _graph(model)
        assert graph.compute_band is ComputeBand.FLOAT32_ELIGIBLE
        assert all(n.contract.compute_band is ComputeBand.FLOAT32_ELIGIBLE for n in graph.nodes)

    def test_one_unvalidated_family_drops_the_tree_to_float64(self):
        model = MixtureDistribution([GaussianDistribution(-4.0, 1.0), LaplaceDistribution(4.0, 2.0)], [0.5, 0.5])
        graph = _graph(model)
        bands = {n.node_id: n.contract.compute_band for n in graph.nodes}
        assert ComputeBand.FLOAT32_ELIGIBLE in bands.values()  # the Gaussian sibling stays eligible
        assert graph.compute_band is ComputeBand.FLOAT64

    def test_neural_leaf_is_float64(self):
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
            [GradLeaf(DiagGauss(), m_steps=5, lr=0.05), GaussianDistribution(1.0, 1.0)], [0.5, 0.5]
        )
        graph = _graph(model)
        assert graph.compute_band is ComputeBand.FLOAT64

    def test_explain_and_as_dict_carry_the_band(self):
        model = MixtureDistribution([GaussianDistribution(-4.0, 1.0), GaussianDistribution(4.0, 1.0)], [0.5, 0.5])
        graph = _graph(model)
        text = graph.explain()
        assert "compute band (weakest link): float32_eligible" in text
        assert "band=float32_eligible" in text
        assert all(n.contract.as_dict()["compute_band"] == "float32_eligible" for n in graph.nodes)

    def test_weakest_band_composition(self):
        b = ComputeBand
        assert weakest_band([b.FLOAT32_ELIGIBLE, b.FLOAT32_ELIGIBLE]) is b.FLOAT32_ELIGIBLE
        assert weakest_band([b.FLOAT32_ELIGIBLE, b.FLOAT64]) is b.FLOAT64
        assert weakest_band([]) is b.FLOAT64
