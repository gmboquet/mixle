"""The gradient bridge (mixle.models.grad_leaf): a torch module IS the model.

The load-bearing claims: a bare nn.Module fits via ``optimize(x, module)`` with zero contract code
and matches the explicit wrapper bitwise; gradient leaves compose with classical families in
mixture EM; and the control story holds -- frozen parameters stay frozen (a fully frozen module is
a fixed distribution), custom losses and optimizers are hooks, and the raw module is always
reachable on the fitted leaf.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.inference.estimation import optimize  # noqa: E402
from mixle.models import GradLeaf, NeuralDensity  # noqa: E402
from mixle.stats import GaussianDistribution, MixtureDistribution  # noqa: E402


class DiagGauss(torch.nn.Module):
    """The smallest honest density module: a learnable diagonal Gaussian."""

    def __init__(self, dim: int = 1, mu0: float = 0.0):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.full((dim,), float(mu0)))
        self.log_sigma = torch.nn.Parameter(torch.zeros(dim))

    def _dist(self):
        return torch.distributions.Normal(self.mu, torch.exp(self.log_sigma))

    def log_density(self, x):
        return self._dist().log_prob(x).sum(-1)

    def sample(self, n: int):
        return self._dist().sample((n,))


def _data(mu, sigma, n, seed):
    rng = np.random.RandomState(seed)
    return [float(v) for v in rng.normal(mu, sigma, n)]


class BareModuleTest(unittest.TestCase):
    def test_optimize_accepts_a_bare_module(self):
        # the whole point: no wrapper, no estimator, no contract -- optimize(x, module)
        data = _data(3.0, 0.5, 300, seed=0)
        fitted = optimize(data, DiagGauss(1), max_its=25, out=None)  # EM iterations are the outer budget
        self.assertIsInstance(fitted, GradLeaf)
        self.assertAlmostEqual(float(fitted.module.mu.detach()[0]), 3.0, delta=0.3)
        self.assertGreater(fitted.log_density(3.0), fitted.log_density(0.0))

    def test_bare_module_matches_the_explicit_wrapper_bitwise(self):
        # coercion is a spelling, not a different code path: identical module state, identical fit
        data = _data(1.5, 1.0, 200, seed=1)
        m1, m2 = DiagGauss(1), DiagGauss(1)
        m2.load_state_dict(m1.state_dict())
        a = optimize(data, m1, max_its=3, out=None)
        b = optimize(data, GradLeaf(m2), max_its=3, out=None)
        np.testing.assert_array_equal(
            a.seq_log_density(np.asarray(data)[:, None]), b.seq_log_density(np.asarray(data)[:, None])
        )

    def test_neural_density_is_the_same_machinery(self):
        # the historical wrapper is now a thin name over the bridge -- same fit, same numbers
        data = _data(-2.0, 0.7, 200, seed=2)
        m1, m2 = DiagGauss(1), DiagGauss(1)
        m2.load_state_dict(m1.state_dict())
        a = optimize(data, GradLeaf(m1), max_its=3, out=None)
        b = optimize(data, NeuralDensity(m2), max_its=3, out=None)
        self.assertIsInstance(b, NeuralDensity)
        np.testing.assert_array_equal(
            a.seq_log_density(np.asarray(data)[:, None]), b.seq_log_density(np.asarray(data)[:, None])
        )

    def test_sampling_and_the_escape_hatch(self):
        fitted = optimize(_data(0.0, 1.0, 150, seed=3), DiagGauss(1), max_its=2, out=None)
        s = fitted.sampler(seed=0).sample(size=25)
        self.assertEqual(s.shape, (25, 1))
        self.assertIsInstance(fitted.module, torch.nn.Module)  # nothing is trapped

        class NoSample(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mu = torch.nn.Parameter(torch.zeros(1))

            def log_density(self, x):
                return -0.5 * ((x - self.mu) ** 2).sum(-1)

        with self.assertRaises(TypeError):  # scoring-only modules refuse to sample, with a real message
            GradLeaf(NoSample()).sampler(seed=0).sample(size=3)


class CompositionTest(unittest.TestCase):
    def test_gradient_leaves_mix_with_classical_families(self):
        # a neural leaf and a classical Gaussian in ONE mixture, fit by ONE EM call -- warm-started
        # from the composed prototype (build the structure, fit it), so the E-step splits the two
        # regimes between a torch module and a closed-form family from iteration one.
        rng = np.random.RandomState(4)
        data = [float(v) for v in np.concatenate([rng.normal(-4.0, 1.0, 150), rng.normal(4.0, 1.0, 150)])]
        proto = MixtureDistribution(
            [GradLeaf(DiagGauss(1, mu0=-1.0), m_steps=60, lr=0.05), GaussianDistribution(1.0, 1.0)], [0.5, 0.5]
        )
        fitted = optimize(data, proto.estimator(), prev_estimate=proto, max_its=10, out=None)
        self.assertIsInstance(fitted, MixtureDistribution)
        comp_means = sorted([float(fitted.components[0].module.mu.detach()[0]), float(fitted.components[1].mu)])
        self.assertAlmostEqual(comp_means[0], -4.0, delta=0.6)
        self.assertAlmostEqual(comp_means[1], 4.0, delta=0.6)


class ControlStoryTest(unittest.TestCase):
    def test_frozen_parameters_stay_frozen(self):
        # LLaVA-style: freeze one part, train the other -- requires_grad_(False) is the whole API
        module = DiagGauss(1)
        module.log_sigma.requires_grad_(False)
        before = module.log_sigma.detach().clone()
        fitted = optimize(_data(5.0, 2.0, 250, seed=5), module, max_its=25, out=None)
        np.testing.assert_array_equal(fitted.module.log_sigma.detach().numpy(), before.numpy())
        self.assertAlmostEqual(float(fitted.module.mu.detach()[0]), 5.0, delta=0.5)

    def test_a_fully_frozen_module_is_a_fixed_distribution(self):
        module = DiagGauss(1, mu0=1.0)
        for p in module.parameters():
            p.requires_grad_(False)
        before = {k: v.clone() for k, v in module.state_dict().items()}
        fitted = optimize(_data(9.0, 1.0, 100, seed=6), module, max_its=2, out=None)
        for k, v in fitted.module.state_dict().items():
            np.testing.assert_array_equal(v.numpy(), before[k].numpy())

    def test_custom_loss_hook_owns_the_objective(self):
        # an L2 anchor toward zero in the loss visibly shrinks the fitted mean vs the default NLL
        data = _data(4.0, 0.5, 200, seed=7)

        def anchored(module, x, w):
            return -(w * module.log_density(x)).sum() + 50.0 * (module.mu**2).sum()

        m1, m2 = DiagGauss(1), DiagGauss(1)
        m2.load_state_dict(m1.state_dict())
        free = optimize(data, GradLeaf(m1), max_its=3, out=None)
        pulled = optimize(data, GradLeaf(m2, loss=anchored), max_its=3, out=None)
        self.assertLess(abs(float(pulled.module.mu.detach()[0])), abs(float(free.module.mu.detach()[0])))

    def test_custom_optimizer_hook(self):
        fitted = optimize(
            _data(2.0, 1.0, 200, seed=8),
            GradLeaf(DiagGauss(1), optimizer=lambda params: torch.optim.SGD(params, lr=0.05)),
            max_its=3,
            out=None,
        )
        self.assertAlmostEqual(float(fitted.module.mu.detach()[0]), 2.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
