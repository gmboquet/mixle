"""Acoustic wave with a perfectly-matched-layer absorbing boundary (Phase 5)."""

import unittest

import numpy as np

from pysp.ppl.wave_pml import solve_wave_pml


class WavePMLTest(unittest.TestCase):
    def test_pml_absorbs_outgoing_waves(self):
        _, e_pml = solve_wave_pml((120, 120), c=1.0, n_steps=700, absorb=True)
        _, e_hard = solve_wave_pml((120, 120), c=1.0, n_steps=700, absorb=False)
        peak = e_hard.max()
        self.assertLess(e_pml[-1], 0.02 * peak)  # PML drains the domain
        self.assertGreater(e_hard[-1], 0.5 * peak)  # a hard wall traps the energy
        self.assertGreater(e_hard[-1] / max(e_pml[-1], 1e-12), 20)  # PML >> hard wall

    def test_stability(self):
        field, e = solve_wave_pml((60, 60), c=1.0, n_steps=400)
        self.assertTrue(np.all(np.isfinite(e)))
        self.assertTrue(np.all(np.isfinite(field)))

    def test_energy_rises_then_falls(self):
        _, e = solve_wave_pml((100, 100), c=1.0, n_steps=600, absorb=True)
        self.assertGreater(e.max(), 5 * e[-1])  # injected, then absorbed away
        self.assertGreater(np.argmax(e), 0)


if __name__ == "__main__":
    unittest.main()
