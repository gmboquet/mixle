"""HamiltonianNet (mixle.models.hamiltonian): a learned dynamical system whose flow conserves its own
energy by construction (the symplectic gradient of a learned scalar H), not by penalty or by luck."""

import unittest

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class HamiltonianNetTest(unittest.TestCase):
    def test_conserves_energy_at_random_untrained_init(self):
        # the conservation guarantee comes from reading dynamics off H's symplectic gradient, not from
        # training -- checked here at several random, UNTRAINED initializations.
        from mixle.models.hamiltonian import HamiltonianNet, leapfrog_rollout

        for seed in range(4):
            torch.manual_seed(seed)
            net = HamiltonianNet(dim=1, hidden=[32, 32])
            q0, p0 = torch.tensor([1.0]), torch.tensor([0.5])
            qs, ps = leapfrog_rollout(net, q0, p0, dt=0.05, n_steps=200)
            with torch.no_grad():
                hs = net.hamiltonian(qs, ps)
            self.assertLess(float((hs - hs[0]).abs().max()), 1e-3)

    def test_conserves_energy_in_higher_dimensions_too(self):
        from mixle.models.hamiltonian import HamiltonianNet, leapfrog_rollout

        torch.manual_seed(1)
        net = HamiltonianNet(dim=3, hidden=[32, 32])
        q0, p0 = torch.randn(3), torch.randn(3)
        qs, ps = leapfrog_rollout(net, q0, p0, dt=0.03, n_steps=150)
        with torch.no_grad():
            hs = net.hamiltonian(qs, ps)
        self.assertLess(float((hs - hs[0]).abs().max()), 1e-2)

    def test_invalid_dim_raises(self):
        from mixle.models.hamiltonian import HamiltonianNet

        with self.assertRaises(ValueError):
            HamiltonianNet(dim=0)

    def test_learns_the_harmonic_oscillator_and_the_learned_flow_stays_bounded(self):
        # H = 0.5(q^2 + p^2): dq/dt = p, dp/dt = -q -- a textbook conservative system. Fit the derivative-
        # matching data, then confirm (a) the learned derivatives match the true ones and (b) a rollout on
        # the TRAINED net's own learned H still conserves that H (the structural guarantee survives training).
        from mixle.models.hamiltonian import HamiltonianNet, leapfrog_rollout

        torch.manual_seed(0)
        net = HamiltonianNet(dim=1, hidden=[32, 32])
        opt = torch.optim.Adam(net.module.parameters(), lr=0.01)

        rng = np.random.RandomState(0)
        n = 500
        q = torch.as_tensor(rng.uniform(-2, 2, (n, 1)).astype("float32"))
        p = torch.as_tensor(rng.uniform(-2, 2, (n, 1)).astype("float32"))
        dq_true, dp_true = p.clone(), -q.clone()

        for _ in range(600):
            opt.zero_grad()
            dq_pred, dp_pred = net.time_derivative(q.clone().requires_grad_(True), p.clone().requires_grad_(True))
            loss = ((dq_pred - dq_true) ** 2).mean() + ((dp_pred - dp_true) ** 2).mean()
            loss.backward()
            opt.step()
        self.assertLess(float(loss.detach()), 0.01)

        q0, p0 = torch.tensor([1.5]), torch.tensor([0.0])
        qs, ps = leapfrog_rollout(net, q0, p0, dt=0.05, n_steps=200)
        with torch.no_grad():
            hs = net.hamiltonian(qs, ps)
        self.assertLess(float((hs - hs[0]).abs().max()), 0.01)
        # a true SHO with q0=1.5, p0=0 has amplitude 1.5 -- the learned flow should stay in that ballpark,
        # not diverge or collapse to zero
        self.assertGreater(float(qs.max()), 0.5)
        self.assertLess(float(qs.max()), 3.0)


if __name__ == "__main__":
    unittest.main()
