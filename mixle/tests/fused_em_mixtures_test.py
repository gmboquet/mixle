"""Fused-EM (optimize(reuse_estep_ll=True)) support for the mixture-style families.

For each family this verifies two things:
  * PARITY: with _track_ll enabled the accumulator's reported batch log-likelihood (_seq_ll)
    equals seq_log_density_sum() for the same data and distribution.
  * FUSED: fixed-iteration EM with reuse_estep_ll=True reaches the same optimum as the standard
    loop (same init/seed -> essentially identical fit).

Families covered: HierarchicalMixture, JointMixture, SemiSupervisedMixture, DiracLengthMixture,
HeterogeneousMixture, GaussianMixture (mvn).
"""

import io
import unittest

import numpy as np

from mixle.inference.estimation import optimize
from mixle.stats import (
    CategoricalDistribution,
    DiracLengthMixtureDistribution,
    GammaDistribution,
    GammaEstimator,
    GaussianDistribution,
    GaussianEstimator,
    HeterogeneousMixtureDistribution,
    HierarchicalMixtureDistribution,
    JointMixtureDistribution,
    LogGaussianDistribution,
    LogGaussianEstimator,
    MultivariateGaussianEstimator,
    SemiSupervisedMixtureDistribution,
    seq_encode,
    seq_log_density_sum,
)
from mixle.stats.latent.gaussian_mixture import GaussianMixtureDistribution, GaussianMixtureEstimator
from mixle.stats.latent.heterogeneous_mixture import HeterogeneousMixtureEstimator
from mixle.stats.latent.hierarchical_mixture import HierarchicalMixtureEstimator
from mixle.stats.latent.joint_mixture import JointMixtureEstimator
from mixle.stats.latent.semi_supervised_mixture import SemiSupervisedMixtureEstimator


class FusedEMMixturesTestCase(unittest.TestCase):
    def _parity(self, dist, est, data):
        enc = seq_encode(data, model=dist)
        _, ref = seq_log_density_sum(enc, dist)
        acc = est.accumulator_factory().make()
        acc._track_ll = True
        for sz, x in enc:
            acc.seq_update(x, np.ones(sz), dist)
        self.assertAlmostEqual(acc._seq_ll, ref, places=5)

    def _fused(self, est, data):
        std = optimize(data, est, max_its=12, delta=None, rng=np.random.RandomState(1), out=io.StringIO())
        fused = optimize(
            data, est, max_its=12, delta=None, rng=np.random.RandomState(1), out=io.StringIO(), reuse_estep_ll=True
        )
        _, ls = seq_log_density_sum(seq_encode(data, model=std), std)
        _, lf = seq_log_density_sum(seq_encode(data, model=fused), fused)
        self.assertAlmostEqual(ls, lf, places=6)

    def _default_off(self, est):
        acc = est.accumulator_factory().make()
        self.assertFalse(acc._track_ll)
        self.assertEqual(acc._seq_ll, 0.0)

    # ------------------------------------------------------------------ hmixture

    def _hmixture(self):
        dist = HierarchicalMixtureDistribution(
            [GaussianDistribution(-2.0, 1.0), GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 1.0)],
            [0.5, 0.5],
            [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]],
            len_dist=CategoricalDistribution({2: 0.5, 3: 0.5}),
        )
        est = HierarchicalMixtureEstimator(
            [GaussianEstimator(), GaussianEstimator(), GaussianEstimator()],
            num_mixtures=2,
            len_estimator=CategoricalDistribution({2: 0.5}).estimator(),
        )
        return dist, est, dist.sampler(seed=1).sample(60)

    def test_hmixture(self):
        dist, est, data = self._hmixture()
        self._parity(dist, est, data)
        self._fused(est, data)
        self._default_off(est)

    # ------------------------------------------------------------------ jmixture

    def _jmixture(self):
        taus12 = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]])
        taus21 = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
        dist = JointMixtureDistribution(
            [GaussianDistribution(-2.0, 1.0), GaussianDistribution(2.0, 1.0)],
            [GaussianDistribution(0.0, 1.0), GaussianDistribution(5.0, 1.0), GaussianDistribution(-5.0, 1.0)],
            [0.5, 0.5],
            [0.4, 0.3, 0.3],
            taus12,
            taus21,
        )
        est = JointMixtureEstimator(
            [GaussianEstimator(), GaussianEstimator()], [GaussianEstimator(), GaussianEstimator(), GaussianEstimator()]
        )
        return dist, est, dist.sampler(seed=1).sample(60)

    def test_jmixture(self):
        dist, est, data = self._jmixture()
        self._parity(dist, est, data)
        self._fused(est, data)
        self._default_off(est)

    # --------------------------------------------------------------- ss_mixture

    def _ss_mixture(self):
        dist = SemiSupervisedMixtureDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.5, 0.5]
        )
        est = SemiSupervisedMixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        rng = np.random.RandomState(0)
        data = []
        for _ in range(60):
            v = float(rng.randn() * 2)
            r = rng.rand()
            if r < 0.4:
                data.append((v, None))
            elif r < 0.7:
                data.append((v, [(0, 1.0)]))
            else:
                data.append((v, [(0, 0.3), (1, 0.7)]))
        return dist, est, data

    def test_ss_mixture(self):
        dist, est, data = self._ss_mixture()
        self._parity(dist, est, data)
        self._fused(est, data)
        self._default_off(est)

    # ------------------------------------------------------------ dirac_length

    def _dirac_length(self):
        dist = DiracLengthMixtureDistribution(CategoricalDistribution({2: 0.3, 3: 0.3, 4: 0.4}), p=0.6, v=0)
        return dist, dist.estimator(), dist.sampler(seed=1).sample(80)

    def test_dirac_length(self):
        dist, est, data = self._dirac_length()
        self._parity(dist, est, data)
        self._fused(est, data)
        self._default_off(est)

    def test_dirac_length_no_valid_rows(self):
        # Exercises the len(idx_v)==0 branch of seq_update (no Dirac-valued observations): the
        # tracked LL must still match seq_log_density_sum.
        dist = DiracLengthMixtureDistribution(CategoricalDistribution({2: 0.3, 3: 0.3, 4: 0.4}), p=0.6, v=0)
        data = [2, 3, 4, 3, 2, 4, 3, 2]  # none equal v=0
        self._parity(dist, dist.estimator(), data)

    # ------------------------------------------------------- heterogeneous_mixture

    def _heterogeneous(self):
        dist = HeterogeneousMixtureDistribution(
            [GammaDistribution(2.0, 2.0), LogGaussianDistribution(1.0, 0.25)], [0.5, 0.5]
        )
        est = HeterogeneousMixtureEstimator([GammaEstimator(), LogGaussianEstimator()])
        return dist, est, dist.sampler(1).sample(200)

    def test_heterogeneous_mixture(self):
        dist, est, data = self._heterogeneous()
        self._parity(dist, est, data)
        self._fused(est, data)
        self._default_off(est)

    # ----------------------------------------------------------------- mvnmixture

    def _mvnmixture(self):
        dist = GaussianMixtureDistribution(mu=[[-2.0, -2.0], [2.0, 2.0]], sig2=[[1.0, 1.0], [1.0, 1.0]], w=[0.5, 0.5])
        est = GaussianMixtureEstimator([MultivariateGaussianEstimator(dim=2), MultivariateGaussianEstimator(dim=2)])
        return dist, est, dist.sampler(seed=1).sample(80)

    def test_mvnmixture(self):
        dist, est, data = self._mvnmixture()
        self._parity(dist, est, data)
        self._fused(est, data)
        self._default_off(est)


if __name__ == "__main__":
    unittest.main()
