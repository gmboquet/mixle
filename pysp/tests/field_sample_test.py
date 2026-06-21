"""FieldPosterior.sample: joint Gaussian draws from the Laplace/VI posterior (Phase B field UQ)."""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from pysp.ppl import free
    from pysp.ppl.field import GaussianField, GaussianProxy, RandomWalk, fit_field


def _fit(how, intercept=0.5):
    n = 8
    rng = np.random.RandomState(0)
    truth = np.cumsum(rng.randn(n)) * 0.4
    y = 2.0 * truth + 0.5 + rng.randn(n) * 0.4
    field = GaussianField(np.arange(n), RandomWalk(scale=0.4, ridge=3.0), name="T")
    return fit_field(field, [GaussianProxy(y, slope=2.0, intercept=intercept, scale=0.4)], how=how)


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


@unittest.skipUnless(HAS_TORCH, "fit_field requires PyTorch")
class FieldConditionalSampleTest(unittest.TestCase):
    def setUp(self):
        self.post = _fit("laplace", intercept=free)  # a free intercept node beside the field "T"
        self.pnode = next(n for n in self.post._layout if n != "T")

    def test_conditioning_on_a_nodes_map_leaves_others_at_their_map(self):
        """Conditioning a Gaussian on a coordinate at its mean leaves the other means unchanged."""
        a_map = self.post.mean(self.pnode)
        s = self.post.sample(60000, rng=1, given={self.pnode: a_map})
        np.testing.assert_allclose(s[self.pnode], a_map)  # fixed node returns its given value
        np.testing.assert_allclose(s["T"].mean(0), self.post.mean("T"), atol=0.02)

    def test_conditioning_shrinks_variance(self):
        a_map = self.post.mean(self.pnode)
        s = self.post.sample(60000, rng=2, given={self.pnode: a_map})
        self.assertTrue(np.all(s["T"].std(0) <= self.post.sd("T") + 1e-9))

    def test_different_given_value_shifts_the_field(self):
        a_map = self.post.mean(self.pnode)
        base = self.post.sample(40000, rng=3, given={self.pnode: a_map})["T"].mean(0)
        shifted = self.post.sample(40000, rng=3, given={self.pnode: a_map + 1.0})["T"].mean(0)
        self.assertFalse(np.allclose(base, shifted, atol=0.02))

    def test_conditioning_requires_full_covariance(self):
        with self.assertRaises(ValueError):  # VI exposes only marginals, no cross-node coupling
            _fit("vi", intercept=free).sample(2, given={self.pnode: 0.0})

    def test_fixing_every_node_raises(self):
        with self.assertRaises(ValueError):
            self.post.sample(2, given={self.pnode: 0.0, "T": self.post.mean("T")})


if __name__ == "__main__":
    unittest.main()
