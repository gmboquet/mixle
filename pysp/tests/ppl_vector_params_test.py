"""pysp.ppl axis 2: vector / matrix-valued distribution parameters as inference targets.

An MVN's mean *vector* and full *covariance matrix* are inferable. The covariance uses a
Cholesky parameterization (Sigma = L Lᵀ, diagonal positive, off-diagonal real, assembled in
rebuild) so it is symmetric positive-definite by construction — reusing the scalar slot system.
"""

import unittest

import numpy as np

from pysp.ppl import MVN, Categorical, DiagGaussian, Dirichlet, free, increasing, ode_residual, ordered, param


class ParamHandleConstraintTestCase(unittest.TestCase):
    """param(...) handles let constraints reference a vector PARAMETER during inference."""

    def setUp(self):
        rng = np.random.RandomState(1)
        self.mu = np.array([-1.0, 0.5, 2.0])  # ordered
        self.X = [list(x) for x in (self.mu + rng.standard_normal((3000, 3)))]

    def test_shape_constraint_on_vector_parameter(self):
        m = param("mu", 3)
        fit = MVN(3, mean=m, cov=free).fit(
            self.X,
            how="ensemble",
            constraints=increasing(m),
            draws=800,
            burn=300,
            walkers=24,
            rng=np.random.RandomState(2),
        )
        mm = np.asarray(fit.params["mean"])
        self.assertTrue(np.all(np.diff(mm) > 0))  # increasing enforced on the inferred mean

    def test_entry_constraints_on_vector_parameter(self):
        m = param("mu", 3)
        fit = MVN(3, mean=m, cov=free).fit(
            self.X,
            how="ensemble",
            constraints=(m[0] < m[1]) & (m[1] < m[2]),
            draws=800,
            burn=300,
            walkers=24,
            rng=np.random.RandomState(3),
        )
        mm = np.asarray(fit.params["mean"])
        self.assertTrue(np.all(np.diff(mm) > 0))

    def test_param_model_auto_routes_to_inference(self):
        m = param("mu", 3)
        fit = MVN(3, mean=m, cov=free).fit(self.X)  # auto must not pick EM (which ignores the param)
        self.assertTrue(np.allclose(np.asarray(fit.params["mean"]), self.mu, atol=0.3))


class ODEResidualTestCase(unittest.TestCase):
    def test_residual_small_on_true_solution(self):
        t = np.arange(0, 2, 0.1)
        y = param("y", len(t))
        c = ode_residual(y, lambda yy: -0.5 * yy, dt=0.1)  # dy/dt = -0.5 y
        r_true = np.abs(c.residual({y: np.exp(-0.5 * t)}))
        r_bad = np.abs(c.residual({y: np.sin(3 * t)}))
        self.assertLess(float(r_true.max()), 0.05)  # ~ forward-Euler discretization error
        self.assertGreater(float(r_bad.max()), 1.0)


class LeafVectorParameterTestCase(unittest.TestCase):
    """Leaf distributions whose parameter is a vector: Categorical probs (simplex), Dirichlet
    alpha (positive vector)."""

    def test_categorical_probs_simplex(self):
        rng = np.random.RandomState(0)
        cats = list(rng.choice(3, size=4000, p=[0.5, 0.3, 0.2]))
        m = Categorical(free, dim=3).fit(
            cats, how="ensemble", draws=800, burn=300, walkers=16, rng=np.random.RandomState(1)
        )
        p = np.array([m.dist.pmap.get(i, 0.0) for i in range(3)])
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)  # on the simplex
        self.assertTrue(np.allclose(p, [0.5, 0.3, 0.2], atol=0.05))

    def test_categorical_free_needs_dim(self):
        with self.assertRaises(ValueError):
            Categorical(free)

    def test_dirichlet_alpha_vector(self):
        rng = np.random.RandomState(0)
        data = [list(x) for x in rng.dirichlet([2.0, 3.0, 5.0], size=3000)]
        m = Dirichlet(free, dim=3).fit(data, how="map")
        alpha = np.asarray(m.params["alpha"])
        self.assertTrue(np.all(alpha > 0))  # positive vector
        self.assertTrue(np.allclose(alpha, [2.0, 3.0, 5.0], atol=0.6))


class MVNParameterInferenceTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.true_mu = np.array([1.0, -2.0])
        self.true_cov = np.array([[2.0, 0.8], [0.8, 1.0]])
        self.X = [list(x) for x in rng.multivariate_normal(self.true_mu, self.true_cov, 3000)]

    def test_em_default_still_works(self):
        m = MVN(2).fit(self.X)
        self.assertTrue(np.allclose(m.params["mean"], self.true_mu, atol=0.2))

    def test_mean_and_covariance_map(self):
        m = MVN(2, mean=free, cov=free).fit(self.X, how="map")
        self.assertTrue(np.allclose(m.params["mean"], self.true_mu, atol=0.2))
        cov = np.asarray(m.params["cov"])
        self.assertTrue(np.allclose(cov, self.true_cov, atol=0.3))
        self.assertTrue(np.allclose(cov, cov.T))  # symmetric
        self.assertTrue(np.all(np.linalg.eigvalsh(cov) > 0))  # positive-definite (Cholesky)

    def test_mean_and_covariance_ensemble(self):
        m = MVN(2, mean=free, cov=free).fit(
            self.X, how="ensemble", draws=800, burn=300, walkers=24, rng=np.random.RandomState(1)
        )
        self.assertTrue(np.allclose(m.params["mean"], self.true_mu, atol=0.25))
        cov = np.asarray(m.params["cov"])
        self.assertTrue(np.all(np.linalg.eigvalsh(cov) > 0))

    def test_mean_only(self):
        m = MVN(2, mean=free).fit(self.X, how="map")  # covariance fixed at I
        self.assertTrue(np.allclose(m.params["mean"], self.true_mu, atol=0.2))

    def test_ordered_mean_is_increasing(self):
        rng = np.random.RandomState(1)
        mu = np.array([-1.0, 0.5, 2.0])  # genuinely ordered
        X = [list(x) for x in (mu + rng.standard_normal((3000, 3)))]
        m = MVN(3, mean=ordered, cov=free).fit(
            X, how="ensemble", draws=1000, burn=400, walkers=30, rng=np.random.RandomState(2)
        )
        mm = np.asarray(m.params["mean"])
        self.assertTrue(np.all(np.diff(mm) > 0))  # increasing by construction
        self.assertTrue(np.allclose(mm, mu, atol=0.3))


class DiagGaussianParameterTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.mu = np.array([2.0, -1.0, 0.5])
        self.sd = np.array([1.0, 2.0, 0.5])
        self.X = [list(x) for x in (self.mu + self.sd * rng.standard_normal((3000, 3)))]

    def test_mean_and_variance(self):
        m = DiagGaussian(3, mean=free, var=free).fit(self.X, how="map")
        self.assertTrue(np.allclose(m.params["mean"], self.mu, atol=0.2))
        var = np.asarray(m.params["var"])
        self.assertTrue(np.allclose(var, self.sd**2, atol=0.4))
        self.assertTrue(np.all(var > 0))  # positive variances by construction

    def test_em_default_still_works(self):
        m = DiagGaussian(3).fit(self.X)
        self.assertTrue(np.allclose(m.params["mean"], self.mu, atol=0.2))


if __name__ == "__main__":
    unittest.main()
