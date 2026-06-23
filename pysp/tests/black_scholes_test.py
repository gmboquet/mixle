"""WS-14: Black-Scholes option-pricing PDE, validated against the closed-form Black-Scholes-Merton price."""

import unittest

import numpy as np
from scipy.stats import norm

from pysp.ppl.dynamics import black_scholes_rhs, integrate_adaptive


def _bs_call(s, k, r, sig, t):
    d1 = (np.log(s / k) + (r + sig * sig / 2) * t) / (sig * np.sqrt(t))
    return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d1 - sig * np.sqrt(t))


class BlackScholesTest(unittest.TestCase):
    def setUp(self):
        self.k, self.r, self.sig, self.t = 100.0, 0.05, 0.2, 1.0
        self.s = np.linspace(0.0, 400.0, 801)

    def test_call_matches_closed_form(self):
        v = integrate_adaptive(black_scholes_rhs(self.sig, self.r, self.s), np.maximum(self.s - self.k, 0.0),
                               [self.t], rtol=1e-8, atol=1e-8)[-1]
        for s0 in (80.0, 100.0, 120.0, 150.0):
            i = int(np.argmin(np.abs(self.s - s0)))
            with self.subTest(s0=s0):
                self.assertAlmostEqual(v[i], _bs_call(s0, self.k, self.r, self.sig, self.t), delta=0.01)

    def test_put_call_parity(self):
        # the PDE put should satisfy P = C - S + K e^{-rT}
        vc = integrate_adaptive(black_scholes_rhs(self.sig, self.r, self.s), np.maximum(self.s - self.k, 0.0),
                                [self.t], rtol=1e-8, atol=1e-8)[-1]
        vp = integrate_adaptive(black_scholes_rhs(self.sig, self.r, self.s), np.maximum(self.k - self.s, 0.0),
                                [self.t], rtol=1e-8, atol=1e-8)[-1]
        for s0 in (90.0, 110.0):
            i = int(np.argmin(np.abs(self.s - s0)))
            parity = vc[i] - s0 + self.k * np.exp(-self.r * self.t)
            with self.subTest(s0=s0):
                self.assertAlmostEqual(vp[i], parity, delta=0.05)


if __name__ == "__main__":
    unittest.main()
