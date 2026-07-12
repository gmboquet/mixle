"""Convergence-certificate axis of the typed update graph (worklist Q5.4 reporting).

The compiler types update MECHANICS (exact / generalized-EM / first-order); this axis types the
GUARANTEE class each mechanism carries, and the graph reports the tree-level certificate as the
weakest node's -- guarantees compose by minimum along the update path.
"""

import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.typed_runtime import compile_update_graph  # noqa: E402
from mixle.experimental.typed_runtime.contracts import (  # noqa: E402
    ConvergenceCertificate,
    weakest_certificate,
)
from mixle.models import GradLeaf  # noqa: E402
from mixle.stats import GaussianDistribution, MixtureDistribution  # noqa: E402


class DiagGauss(torch.nn.Module):
    def __init__(self, mu0=0.0):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor([float(mu0)]))
        self.log_sigma = torch.nn.Parameter(torch.zeros(1))

    def log_density(self, x):
        d = torch.distributions.Normal(self.mu, torch.exp(self.log_sigma))
        return d.log_prob(x if x.dim() > 1 else x.unsqueeze(-1)).sum(-1)


def _graph(model):
    return compile_update_graph(model, model.estimator(), nobs=500)


def _neural_mixture(lr_decay=None):
    torch.manual_seed(0)
    return MixtureDistribution(
        [
            GradLeaf(DiagGauss(0.5), m_steps=10, lr=0.05, lr_decay=lr_decay),
            GaussianDistribution(-1.0, 3.0),
        ],
        [0.5, 0.5],
    )


class TreeCertificateTest:
    def test_classical_tree_is_monotone_certified(self):
        model = MixtureDistribution([GaussianDistribution(float(m), 1.0) for m in (-4.0, 0.0, 4.0)], [1 / 3] * 3)
        graph = _graph(model)
        assert graph.convergence_certificate is ConvergenceCertificate.MONOTONE_CERTIFIED
        assert all(
            node.contract.convergence_certificate is ConvergenceCertificate.MONOTONE_CERTIFIED for node in graph.nodes
        )

    def test_constant_lr_neural_leaf_drops_tree_to_best_visited(self):
        graph = _graph(_neural_mixture(lr_decay=None))
        certs = {node.node_id: node.contract.convergence_certificate for node in graph.nodes}
        assert ConvergenceCertificate.BEST_VISITED in certs.values()
        assert ConvergenceCertificate.MONOTONE_CERTIFIED in certs.values()  # the classical sibling
        assert graph.convergence_certificate is ConvergenceCertificate.BEST_VISITED

    def test_saem_window_schedule_upgrades_to_robbins_monro(self):
        graph = _graph(_neural_mixture(lr_decay=0.75))
        assert graph.convergence_certificate is ConvergenceCertificate.ROBBINS_MONRO_SCHEDULE

    def test_schedule_outside_the_window_stays_best_visited(self):
        graph = _graph(_neural_mixture(lr_decay=0.4))
        assert graph.convergence_certificate is ConvergenceCertificate.BEST_VISITED

    def test_explain_reports_certificates_per_node_and_tree_level(self):
        text = _graph(_neural_mixture(lr_decay=0.75)).explain()
        assert "convergence certificate (weakest link): robbins_monro_schedule" in text
        assert "cert=robbins_monro_schedule" in text
        assert "cert=monotone_certified" in text

    def test_as_dict_carries_the_certificate(self):
        graph = _graph(_neural_mixture(lr_decay=0.75))
        payloads = [node.contract.as_dict() for node in graph.nodes]
        assert any(p["convergence_certificate"] == "robbins_monro_schedule" for p in payloads)

    def test_weakest_certificate_ordering_and_empty_case(self):
        c = ConvergenceCertificate
        assert weakest_certificate([c.MONOTONE_CERTIFIED, c.ROBBINS_MONRO_SCHEDULE]) is c.ROBBINS_MONRO_SCHEDULE
        assert weakest_certificate([c.ROBBINS_MONRO_SCHEDULE, c.BEST_VISITED]) is c.BEST_VISITED
        assert weakest_certificate([c.BEST_VISITED, c.UNKNOWN]) is c.UNKNOWN
        assert weakest_certificate([]) is c.UNKNOWN
