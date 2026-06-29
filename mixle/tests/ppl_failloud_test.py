"""C3 fail-loud: how='auto' never silently degrades an inference claim.

- When auto falls back to MAP because a prior has no closed-form posterior, it warns (a point estimate,
  not a posterior) -- and the conjugate-bridge routing is *sound* (a non-conjugate prior family is not
  routed to the conjugate path and does not crash).
- When a binary GLMM is fit by PQL with few observations per group, it warns that the estimate is
  approximate.
"""

import unittest
import warnings

import numpy as np

from mixle.ppl import Bernoulli, Beta, Field, Gamma, Group, Normal, Poisson, free
from mixle.ppl.inference import stats_conjugate_supported


class AutoMapFallbackTest(unittest.TestCase):
    def test_non_conjugate_prior_is_not_routed_to_conjugate(self):
        # a Beta prior on a Gaussian mean is NOT conjugate -> must not be claimed by the bridge
        self.assertFalse(stats_conjugate_supported(Normal(Beta(2, 2, name="m"), 1.0)))
        self.assertFalse(stats_conjugate_supported(Poisson(Beta(2, 2, name="p"))))  # wrong prior family
        # genuine pairs still supported
        self.assertTrue(stats_conjugate_supported(Poisson(Gamma(2, 1, name="lam"))))

    def test_auto_map_fallback_warns_and_fits(self):
        rng = np.random.RandomState(0)
        data = list(rng.normal(0.4, 1.0, 400))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = Normal(Beta(2, 2, name="m"), 1.0).fit(data)  # prior, no closed form -> MAP
            warned = any("point estimate, not a posterior" in str(x.message) for x in w)
        self.assertTrue(warned)
        self.assertAlmostEqual(float(m.dist.mu), 0.4, delta=0.2)  # MAP still produced a sensible fit

    def test_conjugate_does_not_warn(self):
        rng = np.random.RandomState(1)
        data = list(rng.poisson(3.0, 400).astype(float))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Poisson(Gamma(2, 1, name="lam")).fit(data)
            self.assertFalse(any("point estimate" in str(x.message) for x in w))


class PQLBiasWarningTest(unittest.TestCase):
    def test_binary_glmm_small_groups_warns(self):
        rng = np.random.RandomState(2)
        # many groups, very few binary observations per group -> PQL's biased regime
        n_groups, per = 40, 3
        xs, gs, ys = [], [], []
        for gi in range(n_groups):
            b = rng.normal(0, 1.0)
            for _ in range(per):
                x = rng.normal(0, 1)
                xs.append(x)
                gs.append(gi)
                ys.append(int(rng.rand() < 1.0 / (1.0 + np.exp(-(0.5 * x + b)))))
        model = Bernoulli(free * Field("x") + free + Group("g"))  # binary GLMM with a random intercept
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(ys, given={"x": xs, "g": gs})
            warned = any("penalized quasi-likelihood" in str(x.message).lower() for x in w)
        self.assertTrue(warned)


if __name__ == "__main__":
    unittest.main()
