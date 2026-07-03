"""Energy-based density (mixle.models.energy): p(x) ∝ exp(-E(x)), trained by NCE, sampled by Langevin.

The one neural density with an intractable normalizer, so trained and scored APPROXIMATELY (the energy-model
analogue of the VAE's ELBO caveat). The tests assert what an honest EBM must deliver: NCE learns an
approximately-normalized density (integrates to ~1), it models structure a single Gaussian can't, and Langevin
sampling reaches the modes.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import mixle.stats as st  # noqa: E402
from mixle.inference import optimize  # noqa: E402
from mixle.models.energy import EnergyModel, build_energy_net  # noqa: E402


def _seed(s=0):
    torch.manual_seed(s)
    np.random.seed(s)


def _two_modes(seed, n=500):
    r = np.random.RandomState(seed)
    hi = r.rand(n) < 0.5
    x = np.where(hi[:, None], r.randn(n, 2) * 0.3 + [3, 3], r.randn(n, 2) * 0.3 + [-3, -3])
    return [row for row in x]


def _ll(model, data):
    return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))


class EnergyModelTest(unittest.TestCase):
    def test_nce_gives_an_approximately_normalized_density(self):
        _seed()
        # NCE is consistent: -E(x) + log_norm approximates a NORMALIZED log-density, so a 1-D model's density
        # integrates to ~1. It is approximate (unlike an exact flow), so the tolerance is loose but real.
        data = [
            np.array([v])
            for v in np.r_[np.random.RandomState(0).randn(300) * 0.4 - 2, np.random.RandomState(1).randn(300) * 0.4 + 2]
        ]
        em = EnergyModel(build_energy_net(1, hidden=32), m_steps=300, lr=5e-3, noise_ratio=2)
        fit = optimize(data, em.estimator(), prev_estimate=em, max_its=6, out=None)
        grid = np.linspace(-8.0, 8.0, 4001)
        integral = float(np.trapezoid(np.exp(fit.seq_log_density(grid.reshape(-1, 1))), grid))
        self.assertAlmostEqual(integral, 1.0, delta=0.3)

    def test_beats_a_gaussian_on_a_multimodal_density(self):
        _seed()
        train, test = _two_modes(0), _two_modes(1)
        em = EnergyModel(build_energy_net(2, hidden=64), m_steps=300, lr=5e-3, noise_ratio=2)
        fit = optimize(train, em.estimator(), prev_estimate=em, max_its=6, out=None)
        gauss = optimize(train, st.MultivariateGaussianEstimator(dim=2), max_its=20, out=None)
        self.assertGreater(_ll(fit, test) - _ll(gauss, test), 200.0)

    def test_langevin_samples_are_bimodal(self):
        _seed()
        train = _two_modes(2)
        em = EnergyModel(build_energy_net(2, hidden=64), m_steps=300, lr=5e-3, noise_ratio=2, langevin_steps=60)
        fit = optimize(train, em.estimator(), prev_estimate=em, max_its=6, out=None)
        s = np.asarray(fit.sampler(0).sample(300))
        self.assertEqual(s.shape, (300, 2))
        self.assertTrue(np.any(s[:, 0] > 1.0) and np.any(s[:, 0] < -1.0))  # Langevin reaches both modes


if __name__ == "__main__":
    unittest.main()
