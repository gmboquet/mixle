"""Conditional neural density adapter (mixle.models.mixture_density): wrap ANY torch p(y|x) as a mixle leaf.

The point is the wrapper; build_mdn is the reference instance. The claim that earns it: a mixture density network
captures a MULTIMODAL, HETEROSCEDASTIC conditional that a single-Gaussian NeuralGaussian structurally cannot -- and it
still composes and fits under the same EM M-step.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import itertools  # noqa: E402

from mixle.inference import optimize  # noqa: E402
from mixle.models.mixture_density import (  # noqa: E402
    NeuralConditionalDensity,
    build_conditional_autoregressive_categorical,
    build_conditional_flow,
    build_mdn,
)
from mixle.models.neural import make_mlp  # noqa: E402
from mixle.models.neural_leaf import NeuralGaussian  # noqa: E402


def _inverse_problem(seed, n=800):
    """t = x + 0.3 sin(2 pi x) + noise, observed as (t, x): p(x | t) is multimodal (the forward map isn't 1-1)."""
    r = np.random.RandomState(seed)
    x = r.rand(n)
    t = x + 0.3 * np.sin(2.0 * np.pi * x) + 0.02 * r.randn(n)
    return [((float(t[i]),), (float(x[i]),)) for i in range(n)]


def _ll(model, data):
    return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))


def _seed(s=0):
    """Torch-model tests must be order-independent: pin the global RNG that drives module init and Adam."""
    torch.manual_seed(s)
    np.random.seed(s)


class ConditionalDensityTest(unittest.TestCase):
    def test_mdn_beats_single_gaussian_on_multimodal_conditional(self):
        _seed()
        train, test = _inverse_problem(0), _inverse_problem(1)
        mdn = NeuralConditionalDensity(build_mdn(1, 1, k=5, hidden=32), m_steps=120, lr=5e-3)
        fit = optimize(train, mdn.estimator(), prev_estimate=mdn, max_its=8, out=None)
        # a single-Gaussian conditional leaf: one mean per t, cannot represent the multiple valid x
        gauss = optimize(train, NeuralGaussian(make_mlp(1, [32, 32], 1), lr=1e-2).estimator(), max_its=40, out=None)
        self.assertGreater(_ll(fit, test) - _ll(gauss, test), 100.0)

    def test_samples_given_are_multimodal(self):
        _seed()
        train = _inverse_problem(2)
        mdn = NeuralConditionalDensity(build_mdn(1, 1, k=5, hidden=32), m_steps=120, lr=5e-3)
        fit = optimize(train, mdn.estimator(), prev_estimate=mdn, max_its=8, out=None)
        # at t ~ 0.5 the inverse has branches on both sides; repeated draws should not collapse to one point
        s = np.array([fit.sampler(i).sample_given((0.5,)) for i in range(200)]).reshape(-1)
        self.assertGreater(s.std(), 0.1)
        self.assertTrue(np.isfinite(_ll(fit, train)))


def _within_y_curve(seed, n=800):
    """y1 | x ~ N(x, 0.4), y2 | y1 ~ N(y1^2, 0.1): y2 depends on y1 (WITHIN y), not just on x.

    A single-Gaussian NeuralGaussian gives an isotropic mean f(x) -- it cannot represent the y2 = y1^2 correlation.
    """
    r = np.random.RandomState(seed)
    x = 1.2 * r.randn(n)
    y1 = x + 0.4 * r.randn(n)
    y2 = y1**2 + 0.1 * r.randn(n)
    return [((float(x[i]),), (float(y1[i]), float(y2[i]))) for i in range(n)]


class ConditionalFlowTest(unittest.TestCase):
    def test_conditional_flow_beats_single_gaussian_on_within_y_structure(self):
        _seed()
        train, test = _within_y_curve(0), _within_y_curve(1)
        cf = NeuralConditionalDensity(build_conditional_flow(1, 2, hidden=32, layers=6), m_steps=100, lr=5e-3)
        fit = optimize(train, cf.estimator(), prev_estimate=cf, max_its=8, out=None)
        # NeuralGaussian: p(y|x) = N(y; mlp(x), sigma^2 I) -- isotropic, mean-only, blind to the y2=y1^2 coupling
        gauss = optimize(train, NeuralGaussian(make_mlp(1, [32, 32], 2), lr=1e-2).estimator(), max_its=40, out=None)
        self.assertGreater(_ll(fit, test) - _ll(gauss, test), 100.0)

    def test_samples_reproduce_the_within_y_relation(self):
        _seed()
        train = _within_y_curve(2)
        cf = NeuralConditionalDensity(build_conditional_flow(1, 2, hidden=32, layers=6), m_steps=100, lr=5e-3)
        fit = optimize(train, cf.estimator(), prev_estimate=cf, max_its=8, out=None)
        s = np.array([fit.sampler(i).sample_given((0.7,)) for i in range(300)])
        # at x=0.7, y1 ~ 0.7 and y2 ~ y1^2: the sampled (y1, y2) track the parabola, not an axis-aligned blob
        self.assertGreater(np.corrcoef(s[:, 0] ** 2, s[:, 1])[0, 1], 0.5)


def _cond_discrete(seed, n=1200, xcat=3, C=4):
    """x one-hot (xcat); y0 depends on x; y1=(y0+1)%C; y2=(y1+step)%C -- WITHIN-y coupling x alone can't predict."""
    r = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        xi = r.randint(xcat)
        onehot = np.eye(xcat)[xi]
        y0 = (xi + r.randint(0, 2)) % C
        y1 = (y0 + 1) % C
        y2 = (y1 + r.randint(0, 2)) % C
        out.append((onehot, np.array([y0, y1, y2], dtype=float)))
    return out


class ConditionalAutoregressiveCategoricalTest(unittest.TestCase):
    def test_conditional_density_sums_to_one_over_y_space(self):
        _seed()
        # exactness of the conditional: for a fixed x, sum over ALL C^y_dim configs of y must be 1.
        X, D, C = 3, 2, 3
        leaf = NeuralConditionalDensity(build_conditional_autoregressive_categorical(X, D, C, hidden=16))
        x = np.eye(X)[1]
        ys = np.array(list(itertools.product(range(C), repeat=D)), dtype=float)
        xs = np.repeat(x[None, :], len(ys), axis=0)
        total = float(np.exp(leaf.seq_log_density((xs, ys))).sum())
        self.assertAlmostEqual(total, 1.0, delta=1e-4)

    def test_beats_conditional_independent_baseline(self):
        _seed()
        train, test = _cond_discrete(0), _cond_discrete(1)
        car = NeuralConditionalDensity(
            build_conditional_autoregressive_categorical(3, 3, 4, hidden=64), m_steps=120, lr=5e-3
        )
        fit = optimize(train, car.estimator(), prev_estimate=car, max_its=8, out=None)
        # baseline: empirical p(y_d | x) per coordinate independently -- blind to the y2|y1, y1|y0 coupling
        xtr = np.array([np.argmax(xy[0]) for xy in train])
        ytr = np.array([xy[1] for xy in train], dtype=int)
        cond = {}  # (x_cat, d) -> categorical over C
        for xc in range(3):
            for d in range(3):
                cond[(xc, d)] = np.bincount(ytr[xtr == xc, d], minlength=4) / max((xtr == xc).sum(), 1)
        indep_ll = 0.0
        for onehot, y in test:
            xc = int(np.argmax(onehot))
            indep_ll += float(sum(np.log(cond[(xc, d)][int(y[d])] + 1e-12) for d in range(3)))
        self.assertGreater(_ll(fit, test) - indep_ll, 100.0)


class GeneralityTest(unittest.TestCase):
    def test_wraps_any_module_exposing_conditional_log_density(self):
        _seed()

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
