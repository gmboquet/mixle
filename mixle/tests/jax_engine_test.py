"""JAX compute engine: op-surface parity with NumPy, array registration, and functional autograd.

Skipped unless ``jax`` is installed (it is an optional extra). Exercised in CI's optional-extras job.
"""

import importlib.util
import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE, JaxEngine, engine_of, to_numpy
from mixle.engines.base import ComputeEngine

_HAS_JAX = importlib.util.find_spec("jax") is not None


def _cases():
    """name -> callable(engine) -> result, built from the engine's own asarray so dtype policy holds."""
    a, b = [1.0, 4.0, 9.0], [2.0, 1.0, 0.5]
    mat, vec = [[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0]
    idx = [0, 0, 2]
    return {
        "log": lambda e: e.log(e.asarray(a)),
        "exp": lambda e: e.exp(e.asarray(b)),
        "sqrt": lambda e: e.sqrt(e.asarray(a)),
        "abs": lambda e: e.abs(e.asarray([-1.0, 2.0, -3.0])),
        "where": lambda e: e.where(e.asarray([True, False, True]), e.asarray(a), e.asarray(b)),
        "maximum": lambda e: e.maximum(e.asarray(a), e.asarray(b)),
        "clip": lambda e: e.clip(e.asarray(a), 2.0, 5.0),
        "floor": lambda e: e.floor(e.asarray([1.7, 2.2])),
        "isnan": lambda e: e.isnan(e.asarray([1.0, float("nan")])),
        "isinf": lambda e: e.isinf(e.asarray([1.0, float("inf")])),
        "sum": lambda e: e.sum(e.asarray(a)),
        "sum-axis": lambda e: e.sum(e.asarray(mat), axis=1),
        "max": lambda e: e.max(e.asarray(a)),
        "max-axis": lambda e: e.max(e.asarray(mat), axis=1),
        "dot": lambda e: e.dot(e.asarray(vec), e.asarray(vec)),
        "matmul": lambda e: e.matmul(e.asarray(mat), e.asarray(mat)),
        "cumsum": lambda e: e.cumsum(e.asarray(a)),
        "logsumexp": lambda e: e.logsumexp(e.asarray(a)),
        "bincount": lambda e: e.bincount(e.asarray(idx)),
        "searchsorted": lambda e: e.searchsorted(e.asarray(a), e.asarray([3.0, 8.0])),
        "gammaln": lambda e: e.gammaln(e.asarray([1.5, 2.5, 3.5])),
        "digamma": lambda e: e.digamma(e.asarray([1.5, 2.5, 3.5])),
        "betaln": lambda e: e.betaln(e.asarray(2.0), e.asarray(3.0)),
        "erf": lambda e: e.erf(e.asarray([0.0, 0.5, 1.0])),
    }


@unittest.skipUnless(_HAS_JAX, "jax is not installed")
class JaxEngineTest(unittest.TestCase):
    def setUp(self):
        self.je = JaxEngine()

    def test_required_ops_present(self):
        missing = [op for op in ComputeEngine.REQUIRED_OPS if getattr(JaxEngine, op, None) is None]
        self.assertEqual(missing, [])

    def test_op_parity_with_numpy(self):
        for name, fn in _cases().items():
            got = np.asarray(self.je.to_numpy(fn(self.je)), dtype=np.float64)
            exp = np.asarray(NUMPY_ENGINE.to_numpy(fn(NUMPY_ENGINE)), dtype=np.float64)
            self.assertTrue(np.allclose(got, exp, equal_nan=True), f"op {name} diverged: {got} vs {exp}")

    def test_float64_and_accumulator(self):
        self.assertEqual(np.asarray(self.je.asarray([1.0, 2.0])).dtype, np.float64)  # x64 enabled
        self.assertEqual(self.je.accumulator_dtype, np.float64)

    def test_index_add_is_functional(self):
        out = self.je.zeros(3)
        out = self.je.index_add(out, self.je.asarray([0, 0, 2]), self.je.asarray([1.0, 2.0, 5.0]))
        self.assertTrue(np.allclose(self.je.to_numpy(out), [3.0, 0.0, 5.0]))

    def test_array_registration_and_to_numpy(self):
        arr = self.je.asarray([1.0, 2.0, 3.0])
        self.assertEqual(engine_of(arr).name, "jax")
        self.assertEqual(engine_of(np.array([1.0, 2.0])).name, "numpy")
        self.assertIsInstance(to_numpy(arr), np.ndarray)

    def test_portable_kernel_matches_numpy(self):
        # the same backend-neutral kernel, run on each engine, agrees (and matches scipy)
        def gauss_logpdf(e, x, mu, sigma2):
            x, mu, sigma2 = e.asarray(x), e.asarray(mu), e.asarray(sigma2)
            return -0.5 * (e.log(e.asarray(2 * np.pi)) + e.log(sigma2) + (x - mu) ** 2 / sigma2)

        from scipy.stats import norm

        data = np.random.RandomState(0).randn(500)
        jx = np.asarray(self.je.to_numpy(gauss_logpdf(self.je, data, 0.3, 1.7)))
        self.assertTrue(np.allclose(jx, norm(0.3, np.sqrt(1.7)).logpdf(data)))

    def test_functional_autograd(self):
        import jax

        grad = jax.grad(lambda x: self.je.sum(x**2))(self.je.asarray([1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(self.je.to_numpy(grad), [2.0, 4.0, 6.0]))


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class JaxLeafFittingParityTest(unittest.TestCase):
    """The leaf families that declare jax (engine_ready=(...,'jax')) fit on JaxEngine with a result
    identical to the host numpy fit: scoring runs on jax (jit-able), the E-step accumulation round-trips
    through host numpy, so parity is exact. Guards both the declarations and the parity contract."""

    def _parity(self, proto, data, grid):
        from mixle.inference import optimize

        jf = optimize(data, proto.estimator(), engine=JaxEngine(), max_its=15, out=None)
        nf = optimize(data, proto.estimator(), max_its=15, out=None)
        je = jf.dist_to_encoder().seq_encode(grid)
        ne = nf.dist_to_encoder().seq_encode(grid)
        diff = float(np.max(np.abs(np.asarray(jf.seq_log_density(je)) - np.asarray(nf.seq_log_density(ne)))))
        self.assertLess(diff, 1e-6)

    def test_declared_leaf_families_fit_on_jax_with_parity(self):
        import mixle.stats as S

        rng = np.random.RandomState(0)
        self._parity(S.GaussianDistribution(0, 1), list(rng.normal(3, 2, 4000)), list(np.linspace(-2, 8, 40)))
        self._parity(S.PoissonDistribution(1.0), list(rng.poisson(4, 4000).astype(float)), list(np.arange(0, 15.0)))
        self._parity(S.ExponentialDistribution(1.0), list(rng.exponential(2, 4000)), list(np.linspace(0.1, 10, 40)))
        self._parity(S.GammaDistribution(1, 1), list(rng.gamma(3, 2, 4000)), list(np.linspace(0.1, 20, 40)))
        self._parity(S.BernoulliDistribution(0.5), list(rng.binomial(1, 0.3, 4000).astype(float)), [0.0, 1.0])
        self._parity(S.LogGaussianDistribution(0, 1), list(rng.lognormal(1, 0.5, 4000)), list(np.linspace(0.1, 20, 40)))

    def test_jit_scoring_matches_numpy(self):
        # JaxEngine(compile=True) jit-compiles the scoring kernel; the jit'd result must match numpy.
        import mixle.stats as S

        eng = JaxEngine(compile=True)
        x = np.random.RandomState(1).randn(2000)
        f = eng.compile(
            lambda xx: S.GaussianDistribution.backend_log_density_from_params(
                xx, eng.asarray(0.5), eng.asarray(2.0), eng
            )
        )
        got = np.asarray(eng.to_numpy(f(eng.asarray(x))))
        ref = np.asarray(
            NUMPY_ENGINE.to_numpy(
                S.GaussianDistribution.backend_log_density_from_params(
                    NUMPY_ENGINE.asarray(x), NUMPY_ENGINE.asarray(0.5), NUMPY_ENGINE.asarray(2.0), NUMPY_ENGINE
                )
            )
        )
        self.assertTrue(np.allclose(got, ref, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
