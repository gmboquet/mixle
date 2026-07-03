"""Neural density adapter (mixle.models.neural_density): wrap ANY torch density as a composable mixle leaf.

The point is the wrapper, not the architecture: a module exposing log_density/sample becomes a five-piece
Distribution that trains by responsibility-weighted MLE and composes into a mixture with classical families.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import mixle.stats as st  # noqa: E402
from mixle.inference import optimize  # noqa: E402
from mixle.models.neural_density import NeuralDensity, build_coupling_flow, build_vae  # noqa: E402


def _two_modes(seed, n=500):
    r = np.random.RandomState(seed)
    hi = r.rand(n) < 0.5
    x = np.where(hi[:, None], r.randn(n, 2) * 0.3 + [3, 3], r.randn(n, 2) * 0.3 + [-3, -3])
    return [row for row in x]


def _ll(model, data):
    return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))


class FlowLeafTest(unittest.TestCase):
    def test_flow_leaf_beats_gaussian_on_multimodal_density(self):
        train, test = _two_modes(0), _two_modes(1)
        flow = NeuralDensity(build_coupling_flow(2, hidden=32, layers=6), m_steps=80, lr=5e-3)
        fit = optimize(train, flow.estimator(), prev_estimate=flow, max_its=8, out=None)
        gauss = optimize(train, st.MultivariateGaussianEstimator(dim=2), max_its=20, out=None)
        # a flexible neural density models the two modes; a single Gaussian cannot
        self.assertGreater(_ll(fit, test) - _ll(gauss, test), 200.0)

    def test_samples_are_bimodal(self):
        train = _two_modes(2)
        flow = NeuralDensity(build_coupling_flow(2, layers=6), m_steps=80, lr=5e-3)
        fit = optimize(train, flow.estimator(), prev_estimate=flow, max_its=8, out=None)
        s = np.asarray(fit.sampler(0).sample(400))
        self.assertEqual(s.shape, (400, 2))
        # samples land near both modes (+3,+3) and (-3,-3)
        self.assertTrue(np.any(s[:, 0] > 1.0) and np.any(s[:, 0] < -1.0))


class CompositionTest(unittest.TestCase):
    def test_flow_composes_in_a_mixture_with_a_gaussian(self):
        train = _two_modes(3)
        est = st.MixtureEstimator(
            [NeuralDensity(build_coupling_flow(2, layers=6)).estimator(), st.MultivariateGaussianEstimator(dim=2)]
        )
        init = st.MixtureDistribution(
            [
                NeuralDensity(build_coupling_flow(2, layers=6)),
                st.MultivariateGaussianDistribution(np.zeros(2), np.eye(2)),
            ],
            [0.5, 0.5],
        )
        mix = optimize(train, est, prev_estimate=init, max_its=8, out=None)
        self.assertEqual(len(mix.components), 2)  # the flow trained as a component alongside the Gaussian
        self.assertTrue(np.isfinite(mix.log_density(train[0])))
        self.assertGreater(
            _ll(mix, train), _ll(optimize(train, st.MultivariateGaussianEstimator(dim=2), max_its=20, out=None), train)
        )


class VAETest(unittest.TestCase):
    def test_vae_elbo_beats_gaussian_on_multimodal_density(self):
        # the VAE's log_density is the ELBO (a LOWER bound). If the bound already beats the Gaussian's EXACT
        # log-likelihood, the VAE's true likelihood beats it by at least that margin -- a valid one-sided claim.
        train, test = _two_modes(0), _two_modes(1)
        vae = NeuralDensity(build_vae(2, latent=2, hidden=64), m_steps=150, lr=5e-3)
        fit = optimize(train, vae.estimator(), prev_estimate=vae, max_its=15, out=None)
        gauss = optimize(train, st.MultivariateGaussianEstimator(dim=2), max_its=20, out=None)
        self.assertGreater(_ll(fit, test) - _ll(gauss, test), 200.0)

    def test_vae_samples_are_bimodal(self):
        train = _two_modes(2)
        vae = NeuralDensity(build_vae(2, latent=2, hidden=64), m_steps=150, lr=5e-3)
        fit = optimize(train, vae.estimator(), prev_estimate=vae, max_its=15, out=None)
        s = np.asarray(fit.sampler(0).sample(400))
        self.assertEqual(s.shape, (400, 2))
        # the decoder learned both modes at (+3,+3) and (-3,-3), not a single blob between them
        self.assertTrue(np.any(s[:, 0] > 1.0) and np.any(s[:, 0] < -1.0))


class GeneralityTest(unittest.TestCase):
    def test_wraps_any_module_exposing_log_density(self):
        # the wrapper is not flow-specific: any module with log_density(x)->(n,) works
        class StdNormalModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
                self._p = torch.nn.Parameter(torch.zeros(1))  # a (trivial) parameter so it's a real module

            def log_density(self, x):
                return -0.5 * (x**2).sum(dim=1) - 0.5 * self.dim * float(np.log(2 * np.pi))

            def sample(self, n):
                return torch.randn(n, self.dim)

        leaf = NeuralDensity(StdNormalModule(2))
        x = np.array([[0.0, 0.0], [1.0, -1.0]])
        got = leaf.seq_log_density(x)
        want = -0.5 * (x**2).sum(axis=1) - 0.5 * 2 * np.log(2 * np.pi)
        self.assertTrue(np.allclose(got, want, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
