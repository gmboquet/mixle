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
        # not diverge or collapse to zero. `qs` now carries a live graph back to `net`'s parameters (the
        # rollout is no longer detached step to step), so detach before pulling out a plain float.
        self.assertGreater(float(qs.detach().max()), 0.5)
        self.assertLess(float(qs.detach().max()), 3.0)

    def test_rollout_loss_backprops_to_net_parameters_and_initial_state(self):
        # Regression test: every per-step derivative in `leapfrog_rollout` used to be `.detach()`-ed before
        # being folded back into q/p, severing the autograd graph at each step boundary. A rollout-loss
        # `.backward()` -- the standard training recipe for Hamiltonian NNs (fit the network by matching a
        # simulated multi-step trajectory against observed data) -- then updated NOTHING: every
        # `net.module` parameter's `.grad` was silently `None`, with no error or warning.
        from mixle.models.hamiltonian import HamiltonianNet, leapfrog_rollout

        torch.manual_seed(0)
        net = HamiltonianNet(dim=1, hidden=[16, 16])
        q0 = torch.tensor([1.0], requires_grad=True)
        p0 = torch.tensor([0.5], requires_grad=True)
        qs, ps = leapfrog_rollout(net, q0, p0, dt=0.1, n_steps=5)
        loss = (qs**2 + ps**2).sum()
        loss.backward()

        # every parameter that can possibly influence dq/dt, dp/dt = dH/dp, -dH/dq should get a gradient --
        # EXCEPT the final layer's bias, which only ever adds a q,p-independent CONSTANT to the scalar H
        # and so is provably invisible to any derivative-based loss (a "gauge freedom" inherent to this
        # architecture, not a symptom of the bug above: it is also always None for a direct, un-rolled
        # `net.time_derivative(...)`-based loss, with no rollout involved at all).
        final_linear = [m for m in net.module if isinstance(m, torch.nn.Linear)][-1]
        for name, param in net.module.named_parameters():
            if param is final_linear.bias:
                continue
            self.assertIsNotNone(param.grad, f"expected a gradient for {name}")
            self.assertGreater(float(param.grad.abs().sum()), 0.0, f"expected a nonzero gradient for {name}")
        self.assertIsNone(final_linear.bias.grad)

        self.assertIsNotNone(q0.grad)
        self.assertIsNotNone(p0.grad)

    def test_rollout_loss_gradient_matches_finite_difference(self):
        # The real correctness bar for the fix above: not just "grad is not None" but that q0.grad is the
        # TRUE trajectory sensitivity through the rollout, not the trivial, dynamics-blind
        # sum_t(2*q_t)/sum_t(2*p_t) artifact that `(qs**2 + ps**2).sum().backward()` collapsed to when every
        # step was detached. Verified two ways: against a central finite difference on q0/p0 (float64
        # throughout, so the comparison isn't dominated by float32 rounding noise), and against that trivial
        # artifact directly.
        from mixle.models.hamiltonian import HamiltonianNet, leapfrog_rollout

        torch.manual_seed(0)
        net = HamiltonianNet(dim=1, hidden=[16, 16])
        net.module = net.module.double()

        def rollout_loss(q0_val, p0_val, requires_grad):
            q0 = torch.tensor([q0_val], dtype=torch.float64, requires_grad=requires_grad)
            p0 = torch.tensor([p0_val], dtype=torch.float64, requires_grad=requires_grad)
            qs, ps = leapfrog_rollout(net, q0, p0, dt=0.1, n_steps=5)
            return (qs**2 + ps**2).sum(), q0, p0

        loss, q0, p0 = rollout_loss(1.0, 0.5, requires_grad=True)
        loss.backward()

        eps = 1e-6
        loss_plus, _, _ = rollout_loss(1.0 + eps, 0.5, requires_grad=False)
        loss_minus, _, _ = rollout_loss(1.0 - eps, 0.5, requires_grad=False)
        fd_q0_grad = (float(loss_plus.detach()) - float(loss_minus.detach())) / (2 * eps)

        loss_plus_p, _, _ = rollout_loss(1.0, 0.5 + eps, requires_grad=False)
        loss_minus_p, _, _ = rollout_loss(1.0, 0.5 - eps, requires_grad=False)
        fd_p0_grad = (float(loss_plus_p.detach()) - float(loss_minus_p.detach())) / (2 * eps)

        self.assertAlmostEqual(float(q0.grad), fd_q0_grad, places=5)
        self.assertAlmostEqual(float(p0.grad), fd_p0_grad, places=5)

        # and confirm it's nowhere near the old trivial, dynamics-blind artifact
        qs_check, _ = leapfrog_rollout(
            net,
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            dt=0.1,
            n_steps=5,
        )
        trivial_q0_grad = float((2 * qs_check).sum().detach())
        self.assertGreater(abs(float(q0.grad) - trivial_q0_grad), 1e-3)


if __name__ == "__main__":
    unittest.main()
