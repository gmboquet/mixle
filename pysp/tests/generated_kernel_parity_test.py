"""Parity (and a soft benchmark) for declaration-generated numba leaf kernels.

This is the harness for the "real engine support" conversion work: for every leaf family that
advertises a generated numba kernel, the generated scorer must match the scalar ``log_density`` and
the legacy ``seq_log_density`` exactly (up to float noise). Adding a family to ``CASES`` guards its
conversion against regressions.

It also records a rough generated-vs-legacy timing as an informational benchmark (never asserted,
so it cannot make CI flaky).
"""
import time
import unittest

import numpy as np

import pysp.stats as s
from pysp.stats import (
    BetaDistribution, BinomialDistribution, GaussianDistribution, PoissonDistribution,
    ExponentialDistribution, GammaDistribution, GeometricDistribution,
    LaplaceDistribution, LogisticDistribution, StudentTDistribution, WeibullDistribution,
    ParetoDistribution, UniformDistribution,
)


def _samples(rng, kind):
    return {
        'beta': lambda: list(rng.beta(2.0, 3.0, size=400)),
        'binomial': lambda: [int(v) for v in rng.binomial(10, 0.3, size=400)],
        'gaussian': lambda: list(rng.normal(0.0, 1.0, size=400)),
        'poisson': lambda: [int(v) for v in rng.poisson(3.0, size=400)],
        'exponential': lambda: list(rng.exponential(2.0, size=400)),
        'gamma': lambda: list(rng.gamma(2.0, 2.0, size=400)),
        'geometric': lambda: [int(v) + 1 for v in rng.geometric(0.3, size=400)],
        'laplace': lambda: list(rng.laplace(0.5, 1.3, size=400)),
        'logistic': lambda: list(rng.logistic(0.0, 1.0, size=400)),
        'studentt': lambda: list(rng.standard_t(5.0, size=400)),
        'weibull': lambda: [abs(v) + 1.0e-3 for v in rng.weibull(1.5, size=400) * 2.0],
        'pareto': lambda: [(v + 1.0) for v in rng.pareto(2.5, size=400)],
        'uniform': lambda: list(rng.uniform(-1.0, 2.0, size=400)),
    }[kind]()


# (label, distribution, sample-kind)
CASES = [
    ('beta', BetaDistribution(2.0, 3.0), 'beta'),
    ('binomial', BinomialDistribution(0.3, 10), 'binomial'),
    ('gaussian', GaussianDistribution(0.0, 1.0), 'gaussian'),
    ('poisson', PoissonDistribution(3.0), 'poisson'),
    ('exponential', ExponentialDistribution(2.0), 'exponential'),
    ('gamma', GammaDistribution(2.0, 2.0), 'gamma'),
    ('geometric', GeometricDistribution(0.3), 'geometric'),
    # non-exp-family leaves lit up by the generic symbolic->numba compiler
    ('laplace', LaplaceDistribution(0.5, 1.3), 'laplace'),
    ('logistic', LogisticDistribution(0.0, 1.0), 'logistic'),
    ('studentt', StudentTDistribution(5.0, 0.0, 1.0), 'studentt'),
    ('weibull', WeibullDistribution(1.5, 2.0), 'weibull'),
    ('pareto', ParetoDistribution(1.0, 2.5), 'pareto'),
    ('uniform', UniformDistribution(-1.0, 2.0), 'uniform'),
]


class GeneratedKernelParityTestCase(unittest.TestCase):

    def test_generated_numba_matches_scalar_and_legacy(self):
        rng = np.random.RandomState(7)
        for label, dist, kind in CASES:
            with self.subTest(family=label):
                if not s.generated_numba_log_density_available(dist):
                    self.skipTest('%s has no generated numba kernel' % label)
                data = _samples(rng, kind)
                enc = dist.dist_to_encoder().seq_encode(data)
                ref = np.asarray([dist.log_density(x) for x in data], dtype=np.float64)
                legacy = np.asarray(dist.seq_log_density(enc), dtype=np.float64)
                generated = np.asarray(s.generated_numba_log_density(dist, enc), dtype=np.float64)
                self.assertTrue(np.allclose(legacy, ref, atol=1.0e-9, rtol=1.0e-9),
                                '%s legacy seq_log_density disagrees with scalar' % label)
                self.assertTrue(np.allclose(generated, ref, atol=1.0e-9, rtol=1.0e-9),
                                '%s generated kernel disagrees with scalar' % label)

    def test_beta_uses_exp_family_strategy(self):
        # Regression: Beta previously fell back to backend_log_density_from_params because its
        # exp-family sufficient-statistics unpacked the full encoder tuple instead of the two
        # scoring stats. It must now use the optimal exp_family strategy and advertise numba.
        dist = BetaDistribution(2.0, 3.0)
        diag = s.generated_log_density_diagnostics(dist)
        self.assertEqual(diag.get('strategy'), 'exp_family')
        self.assertIsNone(diag.get('fallback_reason'))
        self.assertEqual(s.capabilities_for(type(dist)).kernel_status, 'numba_adapter')

    def test_non_exp_family_leaves_have_generic_numba_kernel(self):
        # Regression for the generic symbolic->numba compiler: these leaves are not exponential
        # families, so they only get a generated kernel through the lowered backend formula.
        non_exp = [
            LaplaceDistribution(0.5, 1.3), LogisticDistribution(0.0, 1.0),
            StudentTDistribution(5.0, 0.0, 1.0), WeibullDistribution(1.5, 2.0),
            ParetoDistribution(1.0, 2.5), UniformDistribution(-1.0, 2.0),
        ]
        for dist in non_exp:
            with self.subTest(family=type(dist).__name__):
                self.assertTrue(s.generated_numba_log_density_available(dist))
                self.assertEqual(s.capabilities_for(type(dist)).kernel_status, 'numba_adapter')

    def test_generated_benchmark_informational(self):
        # Not asserted - just records generated-vs-legacy timing so regressions in the harness
        # surface in -v output without making CI flaky.
        rng = np.random.RandomState(1)
        for label, dist, kind in CASES:
            if not s.generated_numba_log_density_available(dist):
                continue
            data = _samples(rng, kind)
            enc = dist.dist_to_encoder().seq_encode(data)
            s.generated_numba_log_density(dist, enc)      # warm JIT
            t0 = time.time()
            for _ in range(50):
                s.generated_numba_log_density(dist, enc)
            t_gen = time.time() - t0
            t0 = time.time()
            for _ in range(50):
                dist.seq_log_density(enc)
            t_legacy = time.time() - t0
            print('  [bench] %-12s generated=%.4fs legacy=%.4fs' % (label, t_gen, t_legacy))
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
