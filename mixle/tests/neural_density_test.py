"""Neural density adapter (mixle.models.neural_density): wrap ANY torch density as a composable mixle leaf.

The point is the wrapper, not the architecture: a module exposing log_density/sample becomes a five-piece
Distribution that trains by responsibility-weighted MLE and composes into a mixture with classical families.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import itertools  # noqa: E402

import mixle.stats as st  # noqa: E402
from mixle.inference import optimize  # noqa: E402
from mixle.models.neural_density import (  # noqa: E402
    NeuralDensity,
    build_autoregressive_categorical,
    build_coupling_flow,
    build_maf,
    build_vae,
)


def _two_modes(seed, n=500):
    r = np.random.RandomState(seed)
    hi = r.rand(n) < 0.5
    x = np.where(hi[:, None], r.randn(n, 2) * 0.3 + [3, 3], r.randn(n, 2) * 0.3 + [-3, -3])
    return [row for row in x]


def _ll(model, data):
    return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))


def _seed(s=0):
    """Torch-model tests must be order-independent: pin the global RNG that drives module init and Adam."""
    torch.manual_seed(s)
    np.random.seed(s)


class FlowLeafTest(unittest.TestCase):
    def test_flow_leaf_beats_gaussian_on_multimodal_density(self):
        _seed()
        train, test = _two_modes(0), _two_modes(1)
        flow = NeuralDensity(build_coupling_flow(2, hidden=32, layers=6), m_steps=80, lr=5e-3)
        fit = optimize(train, flow.estimator(), prev_estimate=flow, max_its=8, out=None)
        gauss = optimize(train, st.MultivariateGaussianEstimator(dim=2), max_its=20, out=None)
        # a flexible neural density models the two modes; a single Gaussian cannot
        self.assertGreater(_ll(fit, test) - _ll(gauss, test), 200.0)

    def test_samples_are_bimodal(self):
        _seed()
        train = _two_modes(2)
        flow = NeuralDensity(build_coupling_flow(2, layers=6), m_steps=80, lr=5e-3)
        fit = optimize(train, flow.estimator(), prev_estimate=flow, max_its=8, out=None)
        s = np.asarray(fit.sampler(0).sample(400))
        self.assertEqual(s.shape, (400, 2))
        # samples land near both modes (+3,+3) and (-3,-3)
        self.assertTrue(np.any(s[:, 0] > 1.0) and np.any(s[:, 0] < -1.0))


class CompositionTest(unittest.TestCase):
    def test_flow_composes_in_a_mixture_with_a_gaussian(self):
        _seed()
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
        _seed()
        # the VAE's log_density is the ELBO (a LOWER bound). If the bound already beats the Gaussian's EXACT
        # log-likelihood, the VAE's true likelihood beats it by at least that margin -- a valid one-sided claim.
        train, test = _two_modes(0), _two_modes(1)
        vae = NeuralDensity(build_vae(2, latent=2, hidden=64), m_steps=150, lr=5e-3)
        fit = optimize(train, vae.estimator(), prev_estimate=vae, max_its=15, out=None)
        gauss = optimize(train, st.MultivariateGaussianEstimator(dim=2), max_its=20, out=None)
        self.assertGreater(_ll(fit, test) - _ll(gauss, test), 200.0)

    def test_vae_samples_are_bimodal(self):
        _seed()
        train = _two_modes(2)
        vae = NeuralDensity(build_vae(2, latent=2, hidden=64), m_steps=150, lr=5e-3)
        fit = optimize(train, vae.estimator(), prev_estimate=vae, max_its=15, out=None)
        s = np.asarray(fit.sampler(0).sample(400))
        self.assertEqual(s.shape, (400, 2))
        # the decoder learned both modes at (+3,+3) and (-3,-3), not a single blob between them
        self.assertTrue(np.any(s[:, 0] > 1.0) and np.any(s[:, 0] < -1.0))


def _curved_pair(seed, n=800):
    """x1 ~ N(0, 1.5^2); x2 | x1 ~ N(0.5 x1^2 - 1, 0.3^2) -- a curved autoregressive dependence a Gaussian can't fit."""
    r = np.random.RandomState(seed)
    x1 = 1.5 * r.randn(n)
    x2 = 0.5 * x1**2 - 1.0 + 0.3 * r.randn(n)
    return [np.array([x1[i], x2[i]]) for i in range(n)]


class MAFTest(unittest.TestCase):
    def test_density_integrates_to_one(self):
        _seed()
        # the exactness claim, made crisp: a 1-D flow is a proper normalized density, so int p(x) dx = 1.
        flow = build_maf(1, hidden=16, blocks=2)
        leaf = NeuralDensity(flow)
        grid = np.linspace(-8.0, 8.0, 4001)
        dens = np.exp(leaf.seq_log_density(grid.reshape(-1, 1)))
        integral = np.trapezoid(dens, grid)
        self.assertAlmostEqual(integral, 1.0, delta=0.02)

    def test_maf_beats_gaussian_on_curved_dependence(self):
        _seed()
        train, test = _curved_pair(0), _curved_pair(1)
        maf = NeuralDensity(build_maf(2, hidden=64, blocks=3), m_steps=80, lr=5e-3)
        fit = optimize(train, maf.estimator(), prev_estimate=maf, max_its=8, out=None)
        gauss = optimize(train, st.MultivariateGaussianEstimator(dim=2), max_its=20, out=None)
        self.assertGreater(_ll(fit, test) - _ll(gauss, test), 100.0)

    def test_composes_honestly_in_a_mixture(self):
        _seed()
        # exact density => mixing with a Gaussian is a fair comparison of two exact leaves (no bound bias)
        train = _curved_pair(2)
        est = st.MixtureEstimator(
            [NeuralDensity(build_maf(2, blocks=3)).estimator(), st.MultivariateGaussianEstimator(dim=2)]
        )
        init = st.MixtureDistribution(
            [NeuralDensity(build_maf(2, blocks=3)), st.MultivariateGaussianDistribution(np.zeros(2), np.eye(2))],
            [0.5, 0.5],
        )
        mix = optimize(train, est, prev_estimate=init, max_its=6, out=None)
        self.assertEqual(len(mix.components), 2)
        self.assertTrue(np.isfinite(mix.log_density(train[0])))


def _markov_discrete(seed, n=1000, dim=3, cats=4):
    """A discrete chain: x0 uniform, then x_i = (x_{i-1} + step) % cats -- strong nearest-neighbor dependence."""
    r = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        x = [r.randint(cats)]
        for _ in range(dim - 1):
            x.append((x[-1] + r.randint(0, 2)) % cats)  # mostly repeat or step by one
        out.append(np.array(x, dtype=float))
    return out


class AutoregressiveCategoricalTest(unittest.TestCase):
    def test_density_sums_to_one_over_the_finite_space(self):
        _seed()
        # exactness for a discrete density: sum over ALL C^dim configurations must be 1.
        D, C = 2, 3
        leaf = NeuralDensity(build_autoregressive_categorical(D, C, hidden=16))
        configs = np.array(list(itertools.product(range(C), repeat=D)), dtype=float)
        total = float(np.exp(leaf.seq_log_density(configs)).sum())
        self.assertAlmostEqual(total, 1.0, delta=1e-4)

    def test_beats_independent_categorical_on_a_chain(self):
        _seed()
        train, test = _markov_discrete(0), _markov_discrete(1)
        ar = NeuralDensity(build_autoregressive_categorical(3, 4, hidden=64), m_steps=120, lr=5e-3)
        fit = optimize(train, ar.estimator(), prev_estimate=ar, max_its=8, out=None)
        # independent baseline: per-coordinate empirical categoricals (blind to the nearest-neighbor dependence)
        arr = np.array(train, dtype=int)
        marg = [np.bincount(arr[:, d], minlength=4) / len(arr) for d in range(3)]
        indep_ll = float(sum(np.log(marg[d][int(row[d])] + 1e-12) for row in test for d in range(3)))
        self.assertGreater(_ll(fit, test) - indep_ll, 100.0)


class GeneralityTest(unittest.TestCase):
    def test_wraps_any_module_exposing_log_density(self):
        _seed()

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
