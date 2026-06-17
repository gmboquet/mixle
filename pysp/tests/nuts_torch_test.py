"""Torch-native NUTS: posterior recovery, multi-chain R-hat, and agreement with numpy NUTS."""

import importlib.util
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class NutsTorchTestCase(unittest.TestCase):
    def setUp(self):
        import torch

        rng = np.random.RandomState(0)
        d = 8
        a = rng.standard_normal((d, d))
        self.cov = a @ a.T + d * np.eye(d)  # SPD covariance
        self.mu = rng.standard_normal(d) * 2.0
        self.prec = np.linalg.inv(self.cov)
        self.d = d

        mu_t = torch.tensor(self.mu, dtype=torch.float64)
        prec_t = torch.tensor(self.prec, dtype=torch.float64)

        def logp(theta):
            diff = theta - mu_t
            return -0.5 * diff @ (prec_t @ diff)

        self.logp = logp

    def test_recovers_gaussian(self):
        from pysp.infer import nuts_torch

        res = nuts_torch(self.logp, dim=self.d, num_samples=1500, warmup=600, chains=1, rng=1, compile=False)
        mean = res.samples.mean(axis=0)
        np.testing.assert_allclose(mean, self.mu, atol=0.35)
        var = res.samples.var(axis=0)
        true_var = np.diag(self.cov)
        self.assertTrue(np.all(np.abs(var - true_var) / true_var < 0.5))  # marginal variances within 50%

    def test_compiled_path_runs(self):
        from pysp.infer import nuts_torch

        res = nuts_torch(self.logp, dim=self.d, num_samples=400, warmup=300, chains=1, rng=2, compile=True)
        self.assertEqual(res.samples.shape[1], self.d)
        self.assertTrue(np.all(np.isfinite(res.samples)))
        np.testing.assert_allclose(res.samples.mean(axis=0), self.mu, atol=0.6)  # rough recovery; exercises compile

    def test_multichain_rhat(self):
        from pysp.infer import nuts_torch

        res = nuts_torch(self.logp, dim=self.d, num_samples=800, warmup=400, chains=4, rng=3, compile=False)
        self.assertTrue(np.all(res.rhat < 1.05))
        self.assertTrue(np.all(res.ess > 40))

    def test_agreement_with_numpy_nuts(self):
        from pysp.infer import nuts as nuts_np
        from pysp.infer import nuts_torch

        mu, prec = self.mu, self.prec

        def vg(theta):  # numpy fused value_and_grad of the same Gaussian
            theta = np.asarray(theta, dtype=float)
            diff = theta - mu
            return -0.5 * diff @ (prec @ diff), -(prec @ diff)

        r_np = nuts_np(vg, dim=self.d, num_samples=1500, warmup=600, chains=2, rng=5)
        r_t = nuts_torch(self.logp, dim=self.d, num_samples=1500, warmup=600, chains=2, rng=5, compile=False)
        se = np.sqrt(np.diag(self.cov) / np.clip(r_t.ess, 1.0, None))
        # posterior means consistent within a few MC standard errors (not bit-identical: different RNG substrate)
        self.assertTrue(np.all(np.abs(r_np.samples.mean(axis=0) - r_t.samples.mean(axis=0)) < 6.0 * se + 0.2))


if __name__ == "__main__":
    unittest.main()
