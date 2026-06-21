"""Mixture.conditional + MVN/MVT marginal: the conditional of a mixture is itself a mixture (Phase B)."""

import unittest

import numpy as np
from scipy.special import logsumexp

from pysp.stats import MixtureDistribution
from pysp.stats import MultivariateGaussianDistribution as MVN
from pysp.stats import MultivariateStudentTDistribution as MVT


def _rcov(rng, d):
    chol = rng.randn(d, d)
    return chol @ chol.T + np.eye(d)


class MarginalTest(unittest.TestCase):
    def test_mvn_marginal_matches_subdimension_density(self):
        rng = np.random.RandomState(0)
        d = MVN(np.array([0.0, 1.0, 2.0]), _rcov(rng, 3))
        m = d.marginal([0, 2])
        np.testing.assert_allclose(m.mu, [0.0, 2.0])
        # marginal density == integral of joint; check it equals N(mu[[0,2]], cov[[0,2],[0,2]])
        ref = MVN(d.mu[[0, 2]], np.asarray(d.covar)[np.ix_([0, 2], [0, 2])])
        for _ in range(5):
            x = rng.randn(2)
            self.assertAlmostEqual(m.log_density(x), ref.log_density(x), places=10)

    def test_mvt_marginal_keeps_dof(self):
        rng = np.random.RandomState(1)
        d = MVT(7.0, np.array([0.0, 1.0, 2.0]), _rcov(rng, 3))
        m = d.marginal([1, 2])
        self.assertEqual(m.dof, 7.0)
        np.testing.assert_allclose(m.mu, [1.0, 2.0])


class MixtureConditionalTest(unittest.TestCase):
    def _check_joint_over_marginal(self, mix, comps, obs):
        obs_idx = sorted(obs)
        x_o = np.array([obs[i] for i in obs_idx])
        unobs = [i for i in range(comps[0].dim) if i not in obs]
        cond = mix.conditional(obs)
        self.assertEqual(cond.num_components, len(comps))
        log_marg = logsumexp([np.log(w) + c.marginal(obs_idx).log_density(x_o) for w, c in zip(mix.w, comps)])
        rng = np.random.RandomState(3)
        for _ in range(6):
            x_u = rng.randn(len(unobs))
            full = np.empty(comps[0].dim)
            full[obs_idx] = x_o
            full[unobs] = x_u
            self.assertAlmostEqual(cond.log_density(x_u), mix.log_density(full) - log_marg, places=9)
        return cond

    def test_gaussian_mixture_conditional_equals_joint_over_marginal(self):
        rng = np.random.RandomState(0)
        comps = [MVN(np.array([0.0, 1.0, 2.0]), _rcov(rng, 3)), MVN(np.array([3.0, -1.0, 0.0]), _rcov(rng, 3))]
        mix = MixtureDistribution(comps, [0.4, 0.6])
        cond = self._check_joint_over_marginal(mix, comps, {0: 1.5})
        # responsibilities are reweighted by the observed-coordinate fit
        self.assertFalse(np.allclose(cond.w, mix.w))
        self.assertAlmostEqual(float(cond.w.sum()), 1.0, places=10)

    def test_studentt_mixture_conditional_equals_joint_over_marginal(self):
        rng = np.random.RandomState(2)
        comps = [MVT(8.0, np.array([0.0, 1.0, 2.0]), _rcov(rng, 3)), MVT(8.0, np.array([2.0, 0.0, -1.0]), _rcov(rng, 3))]
        mix = MixtureDistribution(comps, [0.5, 0.5])
        self._check_joint_over_marginal(mix, comps, {2: -0.5})

    def test_conditional_mixture_samples_unobserved_dimensions(self):
        rng = np.random.RandomState(0)
        comps = [MVN(np.array([0.0, 1.0, 2.0]), _rcov(rng, 3)), MVN(np.array([3.0, -1.0, 0.0]), _rcov(rng, 3))]
        cond = MixtureDistribution(comps, [0.4, 0.6]).conditional({0: 1.5})
        s = np.asarray(cond.sampler(seed=2).sample(40000))
        self.assertEqual(s.shape, (40000, 2))  # the two unobserved dims
        analytic = cond.w[0] * cond.components[0].mu + cond.w[1] * cond.components[1].mu
        np.testing.assert_allclose(s.mean(0), analytic, atol=0.05)


if __name__ == "__main__":
    unittest.main()
