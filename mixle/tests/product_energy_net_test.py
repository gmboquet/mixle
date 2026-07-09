"""build_product_energy_net (mixle.models.energy): the energy-based product of experts -- multiply arbitrary
expert densities and renormalize by NCE. This is the general-continuous complement to the closed-form
mixle.ops.product_of_experts (which handles only Categorical/Gaussian and raises otherwise).

Correctness is checked analytically with quadratic (Gaussian) energy experts whose product has a known
closed form, so the combinator is verified independently of NCE fitting quality; one test fits the whole
product by NCE on easy data and checks it integrates to ~1."""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from mixle.inference import optimize  # noqa: E402
from mixle.models.energy import EnergyModel, build_energy_net, build_product_energy_net  # noqa: E402


class _GaussianEnergy(nn.Module):
    """E(x) = ||x - mu||^2 / (2 s^2), the energy of N(mu, s^2 I) -- exact and fixed, for analytic checks."""

    def __init__(self, mu, s=1.0):
        super().__init__()
        self.dim = len(mu)
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.s = float(s)
        self.log_norm = nn.Parameter(torch.zeros(()))

    def energy(self, x):
        return ((x - self.mu) ** 2).sum(-1) / (2.0 * self.s**2)


class ProductEnergyNetTest(unittest.TestCase):
    def test_product_energy_is_the_sum_of_expert_energies(self):
        a, b = _GaussianEnergy([-2.0, 0.0]), _GaussianEnergy([2.0, 1.0])
        prod = build_product_energy_net([a, b])
        x = torch.randn(20, 2)
        with torch.no_grad():
            total = prod.energy(x).numpy()
            summed = (a.energy(x) + b.energy(x)).numpy()
        np.testing.assert_allclose(total, summed, atol=1e-6)

    def test_expert_energies_decomposition_sums_to_total(self):
        a, b, c = _GaussianEnergy([0.0, 0.0]), _GaussianEnergy([1.0, 1.0]), _GaussianEnergy([-1.0, 2.0])
        prod = build_product_energy_net([a, b, c])
        x = torch.randn(15, 2)
        with torch.no_grad():
            per = prod.expert_energies(x)
            total = prod.energy(x)
        self.assertEqual(tuple(per.shape), (15, 3))
        np.testing.assert_allclose(per.sum(-1).numpy(), total.numpy(), atol=1e-6)

    def test_gaussian_product_has_the_analytic_precision_weighted_mode_and_shape(self):
        # N(-2,1) x N(2,1) along x0 => N(0, 1/2): mode at 0, and the unnormalized log-density drop from
        # x0=0 to x0=1 is exactly 1.0 (the summed quadratics give -x0^2 - const along x0).
        poe = EnergyModel(build_product_energy_net([_GaussianEnergy([-2.0, 0.0]), _GaussianEnergy([2.0, 0.0])]))
        grid = np.linspace(-4.0, 4.0, 17)
        pts = np.c_[grid, np.zeros_like(grid)]
        ld = poe.seq_log_density(pts)
        self.assertAlmostEqual(float(grid[int(np.argmax(ld))]), 0.0, places=6)
        self.assertAlmostEqual(float(ld[8] - ld[10]), 1.0, places=4)  # ld(x0=0) - ld(x0=1)

    def test_product_concentrates_at_the_intersection(self):
        # two experts sharp around the origin; their product (conjunction) peaks at the intersection
        a, b = _GaussianEnergy([0.0, 0.0], s=0.5), _GaussianEnergy([0.0, 0.0], s=0.5)
        poe = EnergyModel(build_product_energy_net([a, b]))
        ld = poe.seq_log_density(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        self.assertEqual(int(np.argmax(ld)), 0)

    def test_wraps_into_energy_model_and_normalizes_after_nce(self):
        np.random.seed(0)
        torch.manual_seed(0)
        data = [row for row in (np.random.randn(500, 1) * 0.7 + 1.0)]
        poe = EnergyModel(
            build_product_energy_net([build_energy_net(1, hidden=16), build_energy_net(1, hidden=16)]), m_steps=250
        )
        fit = optimize(data, poe.estimator(), prev_estimate=poe, max_its=6, out=None)
        grid = np.linspace(-6.0, 8.0, 3001)
        integral = float(np.trapezoid(np.exp(fit.seq_log_density(grid.reshape(-1, 1))), grid))
        self.assertAlmostEqual(integral, 1.0, delta=0.3)

    def test_requires_at_least_two_experts(self):
        with self.assertRaises(ValueError):
            build_product_energy_net([_GaussianEnergy([0.0])])

    def test_experts_must_share_a_dim(self):
        with self.assertRaises(ValueError):
            build_product_energy_net([_GaussianEnergy([0.0, 0.0]), _GaussianEnergy([0.0])])


if __name__ == "__main__":
    unittest.main()
