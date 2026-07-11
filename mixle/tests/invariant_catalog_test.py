"""A parametrized invariant catalog for stable distribution families (worklist Q5.1).

Q5.1 asks for one place that states the invariants every stable family must satisfy and checks them, rather
than the invariants living implicitly in scattered tests. This is that catalog. For each family it verifies
the minimum invariants:

  * **normalized density/mass** where tractable -- a discrete family's pmf sums to 1 over its support; a
    continuous family's pdf integrates to ~1 over a wide range;
  * **scalar/vectorized score agreement** -- ``log_density(x)`` equals the ``seq_log_density`` route;
  * **finite score after a valid fit** -- fitting the family to its own samples yields finite log-densities
    (a proxy for finite fitted parameters);
  * **serialization score equivalence** -- a family round-tripped through the type-tagged registry scores
    identically.

Adding a stable family to ``_CATALOG`` is how its invariants get enforced; the catalog is the machine-
readable contract.
"""

import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np

import mixle.stats as st
from mixle.inference.estimation import optimize
from mixle.utils.serialization import ensure_pysp_serialization_registry, from_serializable, to_serializable


@dataclass
class Case:
    name: str
    dist: Any
    estimator: Any
    kind: str  # "discrete" or "continuous"
    probes: list  # points to score for scalar/vectorized/serialization checks
    support: Any = None  # discrete: iterable of support points to sum; continuous: (lo, hi) integration range
    integral_points: int = 20001  # continuous normalization grid


_CATALOG = [
    Case("Gaussian", st.GaussianDistribution(1.0, 2.0), st.GaussianEstimator(), "continuous",
         probes=[-1.0, 0.5, 1.0, 3.0], support=(-40.0, 40.0)),
    Case("Gamma", st.GammaDistribution(2.5, 1.5), st.GammaEstimator(), "continuous",
         probes=[0.2, 1.0, 3.0, 6.0], support=(1e-6, 80.0)),
    Case("Exponential", st.ExponentialDistribution(0.7), st.ExponentialEstimator(), "continuous",
         probes=[0.1, 1.0, 3.0], support=(0.0, 120.0)),
    Case("LogGaussian", st.LogGaussianDistribution(0.0, 0.5), st.LogGaussianEstimator(), "continuous",
         probes=[0.3, 1.0, 2.0], support=(1e-6, 60.0)),
    Case("Poisson", st.PoissonDistribution(4.0), st.PoissonEstimator(), "discrete",
         probes=[0, 2, 4, 9], support=range(0, 80)),
    Case("Geometric", st.GeometricDistribution(0.3), st.GeometricEstimator(), "discrete",
         probes=[1, 2, 5], support=range(1, 400)),
    Case("Categorical", st.CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2}),
         st.CategoricalEstimator(), "discrete", probes=["a", "b", "c"], support=("a", "b", "c")),
]  # fmt: skip


class InvariantCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_pysp_serialization_registry()

    def test_normalized_density_or_mass(self):
        for case in _CATALOG:
            with self.subTest(family=case.name):
                if case.kind == "discrete":
                    mass = sum(float(np.exp(case.dist.log_density(k))) for k in case.support)
                    self.assertAlmostEqual(mass, 1.0, places=4, msg=f"{case.name}: pmf sums to {mass}")
                else:
                    lo, hi = case.support
                    xs = np.linspace(lo, hi, case.integral_points)
                    pdf = np.exp(np.array([case.dist.log_density(float(x)) for x in xs]))
                    integral = float(np.trapezoid(pdf, xs))
                    self.assertAlmostEqual(integral, 1.0, places=2, msg=f"{case.name}: pdf integrates to {integral}")

    def test_scalar_vectorized_agreement(self):
        for case in _CATALOG:
            with self.subTest(family=case.name):
                data = list(case.dist.sampler(seed=0).sample(64))
                enc = case.dist.dist_to_encoder().seq_encode(data)
                seq = np.asarray(case.dist.seq_log_density(enc), dtype=float)
                scalar = np.array([float(case.dist.log_density(x)) for x in data])
                np.testing.assert_allclose(seq, scalar, atol=1e-8, err_msg=f"{case.name}: scalar != vectorized")

    def test_finite_score_after_fit(self):
        for case in _CATALOG:
            with self.subTest(family=case.name):
                data = list(case.dist.sampler(seed=1).sample(400))
                fitted = optimize(data, case.estimator, max_its=25, out=None)
                lls = np.array([float(fitted.log_density(x)) for x in data])
                self.assertTrue(np.all(np.isfinite(lls)), f"{case.name}: non-finite log-density after fit")

    def test_serialization_score_equivalence(self):
        for case in _CATALOG:
            with self.subTest(family=case.name):
                back = from_serializable(to_serializable(case.dist))
                for x in case.probes:
                    self.assertAlmostEqual(
                        float(case.dist.log_density(x)),
                        float(back.log_density(x)),
                        places=10,
                        msg=f"{case.name}: serialized score differs at {x!r}",
                    )


if __name__ == "__main__":
    unittest.main()
