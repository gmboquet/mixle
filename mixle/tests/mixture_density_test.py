"""Conditional neural density adapter (mixle.models.mixture_density): wrap ANY torch p(y|x) as a mixle leaf.

The point is the wrapper; build_mdn is the reference instance. The claim that earns it: a mixture density network
captures a MULTIMODAL, HETEROSCEDASTIC conditional that a single-Gaussian NeuralLeaf structurally cannot -- and it
still composes and fits under the same EM M-step.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.inference import optimize  # noqa: E402
from mixle.models.mixture_density import NeuralConditionalDensity, build_conditional_flow, build_mdn  # noqa: E402
from mixle.models.neural import make_mlp  # noqa: E402
from mixle.models.neural_leaf import NeuralLeaf  # noqa: E402


def _inverse_problem(seed, n=800):
    """t = x + 0.3 sin(2 pi x) + noise, observed as (t, x): p(x | t) is multimodal (the forward map isn't 1-1)."""
    r = np.random.RandomState(seed)
    x = r.rand(n)
    t = x + 0.3 * np.sin(2.0 * np.pi * x) + 0.02 * r.randn(n)
    return [((float(t[i]),), (float(x[i]),)) for i in range(n)]


def _ll(model, data):
    return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))


class ConditionalDensityTest(unittest.TestCase):
    def test_mdn_beats_single_gaussian_on_multimodal_conditional(self):
        train, test = _inverse_problem(0), _inverse_problem(1)
        mdn = NeuralConditionalDensity(build_mdn(1, 1, k=5, hidden=32), m_steps=120, lr=5e-3)
        fit = optimize(train, mdn.estimator(), prev_estimate=mdn, max_its=8, out=None)
        # a single-Gaussian conditional leaf: one mean per t, cannot represent the multiple valid x
        gauss = optimize(train, NeuralLeaf(make_mlp(1, [32, 32], 1), lr=1e-2).estimator(), max_its=40, out=None)
        self.assertGreater(_ll(fit, test) - _ll(gauss, test), 100.0)

    def test_samples_given_are_multimodal(self):
        train = _inverse_problem(2)
        mdn = NeuralConditionalDensity(build_mdn(1, 1, k=5, hidden=32), m_steps=120, lr=5e-3)
        fit = optimize(train, mdn.estimator(), prev_estimate=mdn, max_its=8, out=None)
        # at t ~ 0.5 the inverse has branches on both sides; repeated draws should not collapse to one point
        s = np.array([fit.sampler(i).sample_given((0.5,)) for i in range(200)]).reshape(-1)
        self.assertGreater(s.std(), 0.1)
        self.assertTrue(np.isfinite(_ll(fit, train)))


def _within_y_curve(seed, n=800):
    """y1 | x ~ N(x, 0.4), y2 | y1 ~ N(y1^2, 0.1): y2 depends on y1 (WITHIN y), not just on x.

    A single-Gaussian NeuralLeaf gives an isotropic mean f(x) -- it cannot represent the y2 = y1^2 correlation.
    """
    r = np.random.RandomState(seed)
    x = 1.2 * r.randn(n)
    y1 = x + 0.4 * r.randn(n)
    y2 = y1**2 + 0.1 * r.randn(n)
    return [((float(x[i]),), (float(y1[i]), float(y2[i]))) for i in range(n)]


class ConditionalFlowTest(unittest.TestCase):
    def test_conditional_flow_beats_single_gaussian_on_within_y_structure(self):
        train, test = _within_y_curve(0), _within_y_curve(1)
        cf = NeuralConditionalDensity(build_conditional_flow(1, 2, hidden=32, layers=6), m_steps=100, lr=5e-3)
        fit = optimize(train, cf.estimator(), prev_estimate=cf, max_its=8, out=None)
        # NeuralLeaf: p(y|x) = N(y; mlp(x), sigma^2 I) -- isotropic, mean-only, blind to the y2=y1^2 coupling
        gauss = optimize(train, NeuralLeaf(make_mlp(1, [32, 32], 2), lr=1e-2).estimator(), max_its=40, out=None)
        self.assertGreater(_ll(fit, test) - _ll(gauss, test), 100.0)

    def test_samples_reproduce_the_within_y_relation(self):
        train = _within_y_curve(2)
        cf = NeuralConditionalDensity(build_conditional_flow(1, 2, hidden=32, layers=6), m_steps=100, lr=5e-3)
        fit = optimize(train, cf.estimator(), prev_estimate=cf, max_its=8, out=None)
        s = np.array([fit.sampler(i).sample_given((0.7,)) for i in range(300)])
        # at x=0.7, y1 ~ 0.7 and y2 ~ y1^2: the sampled (y1, y2) track the parabola, not an axis-aligned blob
        self.assertGreater(np.corrcoef(s[:, 0] ** 2, s[:, 1])[0, 1], 0.5)


class GeneralityTest(unittest.TestCase):
    def test_wraps_any_module_exposing_conditional_log_density(self):
        # the adapter is not MDN-specific: any module with log_density(x, y)->(n,) works
        class LinearGaussian(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.tensor([[2.0]]))

            def log_density(self, x, y):
                mean = x @ self.w
                return (-0.5 * ((y - mean) ** 2).sum(1) - 0.5 * float(np.log(2 * np.pi))).reshape(-1)

            def sample_given(self, x):
                return x @ self.w + torch.randn(x.shape[0], 1)

        leaf = NeuralConditionalDensity(LinearGaussian())
        x = np.array([[1.0], [2.0]])
        y = np.array([[2.0], [4.0]])  # exactly on the mean => log N = -0.5*log(2pi)
        got = leaf.seq_log_density((x, y))
        self.assertTrue(np.allclose(got, -0.5 * np.log(2 * np.pi), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
