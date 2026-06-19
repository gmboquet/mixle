"""Tests for the 2D wave-equation forward and full-waveform-inversion recovery (phase 2 completion)."""

import unittest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from pysp.ppl import Differential, WaveEquation2D, free, joint
    from pysp.ppl.ops import make_ops


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class SpongeTestCase(unittest.TestCase):
    def test_absorbing_layer_is_edge_localized(self):
        n = 30
        wave = WaveEquation2D(n, dt=0.01, absorb_width=5, absorb_strength=2.0)
        g = wave._gamma.reshape(n, n)
        self.assertGreater(g[0, n // 2], 0.0)  # damping near the edge
        self.assertEqual(g[n // 2, n // 2], 0.0)  # none in the interior
        self.assertEqual(WaveEquation2D(n, dt=0.01, absorb_width=0)._gamma.max(), 0.0)


def _fwi_problem(n=40, nt=140, amp_true=0.5):
    h = 1.0 / (n - 1)
    c0 = 1.0
    dt = 0.3 * h / c0
    wave = WaveEquation2D(n, dt=dt, spacing=h, absorb_width=6, absorb_strength=0.6 / dt)
    xx, yy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    bump = np.exp(-((xx - 0.6) ** 2 + (yy - 0.5) ** 2) / 0.01).ravel()
    src_node = n * 20 + 8
    f0 = 5.0
    tg = np.arange(nt + 1) * dt
    a = (np.pi * f0 * (tg - 0.25)) ** 2
    ricker = 400.0 * (1 - 2 * a) * np.exp(-a)
    recv = torch.as_tensor(np.array([n * 10 + 32, n * 20 + 32, n * 30 + 32]))
    ops = make_ops()
    state0 = wave.pack(torch.zeros(n * n), torch.zeros(n * n))

    def make_step(c2):
        def step(state, i):
            s = ops.zeros(n * n).clone()
            s[src_node] = float(ricker[i])
            return wave.step(state, c2, ops, source=s)

        return step

    def record(state, i):
        return wave.displacement(state)[recv]

    def c2_of(amp):
        return c0**2 * (1.0 + amp * torch.as_tensor(bump))

    return ops, wave, state0, make_step, record, c2_of, nt


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class WaveForwardTestCase(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_forward_is_stable(self):
        ops, wave, state0, make_step, record, c2_of, nt = _fwi_problem()
        rec = ops.integrate_record(make_step(c2_of(0.5)), state0, nt, record, checkpoint=12)
        self.assertTrue(torch.isfinite(rec).all())
        self.assertLess(float(rec.abs().max()), 100.0)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class FullWaveformInversionTestCase(unittest.TestCase):
    """Recover a velocity perturbation from recorded waveforms (the FWI inverse problem)."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def test_recovers_velocity_perturbation(self):
        ops, wave, state0, make_step, record, c2_of, nt = _fwi_problem(amp_true=0.5)

        class _P:
            amp = torch.tensor(0.5)

        rec_true = ops.integrate_record(make_step(c2_of(_P.amp)), state0, nt, record, checkpoint=12).detach().numpy()
        sig = 0.01 * np.abs(rec_true).max()
        y_obs = rec_true.reshape(-1) + sig * np.random.RandomState(0).randn(rec_true.size)

        amp = free(1, name="amp", support="positive")

        def forward(p, o):
            return o.integrate_record(make_step(c2_of(p.amp)), state0, nt, record, checkpoint=12).reshape(-1)

        am, asd = (
            joint([Differential(y_obs, drivers=[amp], scale=sig, forward=forward)])
            .fit(how="gauss_newton")
            .posterior("amp")
        )
        self.assertLess(abs(am - 0.5), 2 * asd)
        self.assertGreater(asd, 0.0)


if __name__ == "__main__":
    unittest.main()
