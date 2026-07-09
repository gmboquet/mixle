"""WS-2: Bingham distribution on S^2; normalizer verified exactly via mpmath sphere integration."""

import unittest

import numpy as np

import mixle
from mixle.capability import Fittable
from mixle.stats import BinghamDistribution as B


class BinghamTest(unittest.TestCase):
    def test_normalizer_integrates_to_one_mpmath(self):
        import mpmath as mp

        # dps=10 keeps a large margin below the assertion tolerance (1e-9): the true numerical
        # error at this precision is ~1.5e-11 (verified against a correct formula) vs the ~4e-16
        # floor at dps=20, and an injected 0.1% normalization bug is still caught cleanly at this
        # precision. This cuts the double-nested sphere quadrature's runtime by roughly 3.5x.
        mp.mp.dps = 10
        m = np.eye(3)
        for z in [[-4.0, -2.0, 0.0], [-8.0, -1.0, 0.0], [-1.0, -1.0, 0.0]]:
            d = B(m, z)
            zz = d.z
            integ = mp.e ** mp.mpf(-d._log_c) * mp.quad(
                lambda th, zz=zz: mp.quad(
                    lambda ph: (
                        mp.e
                        ** (
                            zz[0] * (mp.sin(th) * mp.cos(ph)) ** 2
                            + zz[1] * (mp.sin(th) * mp.sin(ph)) ** 2
                            + zz[2] * mp.cos(th) ** 2
                        )
                        * mp.sin(th)
                    ),
                    [0, 2 * mp.pi],
                ),
                [0, mp.pi],
            )
            with self.subTest(z=z):
                self.assertTrue(mp.almosteq(integ, 1, 1e-9))

    def test_antipodal_symmetry(self):
        d = B(np.eye(3), [-5.0, -2.0, 0.0])
        xs = d.sampler(seed=0).sample(6)
        self.assertTrue(np.allclose([d.log_density(x) for x in xs], [d.log_density(-x) for x in xs]))

    def test_moment_identity(self):
        # E[(m_i.x)^2] == d log c / d z_i jointly validates the normalizer and the ACG-rejection sampler
        from mixle.stats.directional.bingham import _bingham_norm

        d = B(np.eye(3), [-6.0, -2.0, 0.0])
        xs = np.array(d.sampler(seed=1).sample(12000))
        for i in (0, 1):
            zp = d.z.copy()
            zp[i] += 1e-3
            zm = d.z.copy()
            zm[i] -= 1e-3
            grad = (np.log(_bingham_norm(zp)) - np.log(_bingham_norm(zm))) / 2e-3
            with self.subTest(i=i):
                self.assertAlmostEqual((xs[:, i] ** 2).mean(), grad, delta=0.01)

    def test_seq_matches_scalar(self):
        d = B(np.eye(3), [-5.0, -2.0, 0.0])
        xs = d.sampler(seed=0).sample(6)
        self.assertTrue(
            np.allclose([d.log_density(x) for x in xs], d.seq_log_density(d.dist_to_encoder().seq_encode(xs)))
        )

    def test_estimator_recovers_params(self):
        rng = np.random.RandomState(2)
        g, _ = np.linalg.qr(rng.randn(3, 3))
        true = B(g, [-6.0, -2.0, 0.0])
        data = np.array(true.sampler(seed=1).sample(12000))
        est = true.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(true.dist_to_encoder().seq_encode(data), np.ones(len(data)), None)
        m = est.estimate(len(data), acc.value())
        self.assertTrue(np.allclose(np.sort(m.z), [-6.0, -2.0, 0.0], atol=0.6))
        self.assertAlmostEqual(abs(m.m[:, int(np.argmax(m.z))] @ g[:, 2]), 1.0, delta=1e-2)

    def test_uniform_limit(self):
        # equal concentrations -> uniform on the sphere (constant density 1/(4 pi))
        d = B(np.eye(3), [0.0, 0.0, 0.0])
        self.assertAlmostEqual(d.density(np.array([0.3, 0.4, np.sqrt(1 - 0.25)])), 1.0 / (4 * np.pi), places=6)

    def test_capabilities(self):
        self.assertTrue(mixle.supports(B(np.eye(3), [-3.0, -1.0, 0.0]), Fittable))


if __name__ == "__main__":
    unittest.main()
