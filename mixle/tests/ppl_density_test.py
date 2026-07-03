"""Neural densities through the mixle.ppl surface (mixle.ppl.density).

The declarative wiring: Flow/MAF/VAE/DiscreteAR are p(x); MDN/CondFlow/CondDiscreteAR are p(y|x). Each .fit()s
through the standard optimize loop (no training loop in user code) and returns a bound RandomVariable whose .dist
is the composable leaf. These tests exercise the PPL entry points, not the leaves directly.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.ppl import MDN, DiscreteAR, Flow, lower  # noqa: E402


def _seed(s=0):
    torch.manual_seed(s)
    np.random.seed(s)


def _two_modes(seed, n=400):
    r = np.random.RandomState(seed)
    hi = r.rand(n) < 0.5
    x = np.where(hi[:, None], r.randn(n, 2) * 0.3 + [3, 3], r.randn(n, 2) * 0.3 + [-3, -3])
    return [row for row in x]


def _gaussian_ll(train, test):
    """Full-Gaussian (MLE mean/cov) log-likelihood on test -- the baseline a flexible density must beat."""
    tr = np.asarray(train)
    mu, cov = tr.mean(0), np.cov(tr.T) + 1e-6 * np.eye(2)
    te = np.asarray(test)
    d = te - mu
    inv = np.linalg.inv(cov)
    quad = np.einsum("ni,ij,nj->n", d, inv, d)
    return float(np.sum(-0.5 * quad - 0.5 * np.log((2 * np.pi) ** 2 * np.linalg.det(cov))))


class UnconditionalPPLTest(unittest.TestCase):
    def test_flow_fit_beats_a_gaussian(self):
        _seed()
        train, test = _two_modes(0), _two_modes(1)
        m = Flow(2, layers=6).fit(train, its=8)
        self.assertEqual(m._kind, "bound")  # fit returns a bound RandomVariable
        self.assertGreater(m.log_likelihood(test) - _gaussian_ll(train, test), 150.0)

    def test_discrete_ar_density_is_normalized_after_fit(self):
        _seed()
        import itertools

        data = [np.array([r % 3, (r // 3) % 3], dtype=float) for r in np.random.randint(0, 9, size=300)]
        m = DiscreteAR(2, 3, hidden=16).fit(data, its=4)
        configs = np.array(list(itertools.product(range(3), repeat=2)), dtype=float)
        total = float(np.exp(m.dist.seq_log_density(m.dist.dist_to_encoder().seq_encode(list(configs)))).sum())
        self.assertAlmostEqual(total, 1.0, delta=1e-4)

    def test_density_rv_lowers_and_composes(self):
        _seed()
        rv = Flow(2)
        from mixle.models.neural_density import NeuralDensity

        self.assertIsInstance(lower(rv, target="dist"), NeuralDensity)  # composes as a distribution
        self.assertTrue(hasattr(lower(rv, target="estimator"), "accumulator_factory"))  # ...and as an estimator


def _inverse_problem(seed, n=700):
    """observe t = x + 0.3 sin(2 pi x) + noise; recover x | t (multimodal). data = x, covariate = t."""
    r = np.random.RandomState(seed)
    x = r.rand(n)
    t = x + 0.3 * np.sin(2.0 * np.pi * x) + 0.02 * r.randn(n)
    return [(float(xi),) for xi in x], [(float(ti),) for ti in t]


class ConditionalPPLTest(unittest.TestCase):
    def test_mdn_conditional_fit_beats_marginal(self):
        _seed()
        xtr, ttr = _inverse_problem(0)
        xte, tte = _inverse_problem(1)
        m = MDN(1, 1, k=5).fit(xtr, given={"x": ttr}, its=8)
        self.assertEqual(type(m.dist).__name__, "NeuralConditionalDensity")
        cond_ll = float(np.sum(m.dist.seq_log_density(m.dist.dist_to_encoder().seq_encode(list(zip(tte, xte))))))
        # a marginal Gaussian on x that ignores the covariate t -- what "no conditional structure" would score
        xarr = np.array([v[0] for v in xtr])
        mu, sd = xarr.mean(), xarr.std()
        marg_ll = float(
            np.sum(-0.5 * ((np.array([v[0] for v in xte]) - mu) / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi)))
        )
        self.assertGreater(cond_ll - marg_ll, 100.0)

    def test_conditional_fit_requires_covariates(self):
        _seed()
        xtr, _ = _inverse_problem(0)
        with self.assertRaises(ValueError):
            MDN(1, 1).fit(xtr, its=2)  # no given={"x": ...}


if __name__ == "__main__":
    unittest.main()
