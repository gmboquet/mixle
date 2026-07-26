"""Convergence certificates require explicit acceptance evidence and compose by weakest link."""

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
    def test_classical_tree_does_not_infer_monotonicity_from_update_names(self):
        model = MixtureDistribution([GaussianDistribution(float(m), 1.0) for m in (-4.0, 0.0, 4.0)], [1 / 3] * 3)
        graph = _graph(model)
        assert graph.convergence_certificate is ConvergenceCertificate.UNKNOWN
        assert all(node.contract.convergence_certificate is ConvergenceCertificate.UNKNOWN for node in graph.nodes)

    def test_neural_method_and_schedule_names_do_not_fabricate_certificates(self):
        graph = _graph(_neural_mixture(lr_decay=None))
        certs = {node.node_id: node.contract.convergence_certificate for node in graph.nodes}
        assert set(certs.values()) == {ConvergenceCertificate.UNKNOWN}
        assert graph.convergence_certificate is ConvergenceCertificate.UNKNOWN

        for decay in (0.4, 0.75):
            scheduled = _graph(_neural_mixture(lr_decay=decay))
            assert scheduled.convergence_certificate is ConvergenceCertificate.UNKNOWN

    def test_explain_reports_certificates_per_node_and_tree_level(self):
        text = _graph(_neural_mixture(lr_decay=0.75)).explain()
        assert "convergence certificate (weakest link): unknown" in text
        assert "cert=unknown" in text

    def test_as_dict_carries_the_certificate(self):
        graph = _graph(_neural_mixture(lr_decay=0.75))
        payloads = [node.contract.as_dict() for node in graph.nodes]
        assert all(p["convergence_certificate"] == "unknown" for p in payloads)

    def test_weakest_certificate_ordering_and_empty_case(self):
        c = ConvergenceCertificate
        assert weakest_certificate([c.MONOTONE_CERTIFIED, c.ROBBINS_MONRO_SCHEDULE]) is c.ROBBINS_MONRO_SCHEDULE
        assert weakest_certificate([c.ROBBINS_MONRO_SCHEDULE, c.BEST_VISITED]) is c.BEST_VISITED
        assert weakest_certificate([c.BEST_VISITED, c.UNKNOWN]) is c.UNKNOWN
        assert weakest_certificate([]) is c.UNKNOWN
