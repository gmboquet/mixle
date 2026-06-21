"""FieldPosterior.sample: joint Gaussian draws from the Laplace/VI posterior (Phase B field UQ)."""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from pysp.ppl.field import GaussianField, GaussianProxy, RandomWalk, fit_field


def _fit(how):
    n = 8
    rng = np.random.RandomState(0)
    truth = np.cumsum(rng.randn(n)) * 0.4
    y = 2.0 * truth + 0.5 + rng.randn(n) * 0.4
    field = GaussianField(np.arange(n), RandomWalk(scale=0.4, ridge=3.0), name="T")
    return fit_field(field, [GaussianProxy(y, slope=2.0, intercept=0.5, scale=0.4)], how=how)


@unittest.skipUnless(HAS_TORCH, "fit_field requires PyTorch")
class FieldSampleTest(unittest.TestCase):
    def test_laplace_draws_match_posterior_moments(self):
        """Empirical mean/covariance of the draws must recover the Laplace posterior mean/cov."""
        post = _fit("laplace")
        t = post.sample(40000, rng=1)["T"]
        self.assertEqual(t.shape, (40000, 8))
        np.testing.assert_allclose(t.mean(0), post.mean("T"), atol=0.02)
        np.testing.assert_allclose(np.cov(t.T), post.cov("T"), atol=0.01)
        np.testing.assert_allclose(t.std(0), post.sd("T"), atol=0.02)

    def test_vi_mean_field_draws_match_marginals(self):
        post = _fit("vi")
        t = post.sample(20000, rng=2)["T"]
        np.testing.assert_allclose(t.std(0), post.sd("T"), atol=0.02)
        np.testing.assert_allclose(t.mean(0), post.mean("T"), atol=0.02)

    def test_seed_is_deterministic(self):
        post = _fit("laplace")
        np.testing.assert_array_equal(post.sample(10, rng=7)["T"], post.sample(10, rng=7)["T"])

    def test_map_posterior_has_no_covariance_to_sample(self):
        with self.assertRaises(ValueError):
            _fit("map").sample(5)

    def test_nodes_filter_returns_subset(self):
        post = _fit("laplace")
        out = post.sample(5, rng=0, nodes=["T"])
        self.assertEqual(set(out), {"T"})


if __name__ == "__main__":
    unittest.main()
