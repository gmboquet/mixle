"""Exponential-family canonical-map test for the Pareto distribution (WS-J).

Pareto (with the scale ``xm`` held fixed) is an exponential family in the shape ``alpha``:
``T(x) = log x``, ``eta = -alpha``, ``A = -log(alpha) - alpha*log(xm)``, and the base
``log h(x) = -log x`` on the support ``[xm, inf)`` (``-inf`` below ``xm``). This validates the
reconstruction ``log h(x) + <eta, T(x)> - A == log_density`` against the family's own
``seq_log_density``. Kept standalone (not folded into the shared ``exp_family_test`` catalog) so
it does not conflict with other in-flight exp-family-spec additions.
"""

import unittest

import numpy as np

from pysp.stats.compute.exp_family import ExponentialFamilyForm, is_exponential_family, to_exponential_family
from pysp.stats.univariate.continuous.pareto import ParetoDistribution


class ParetoExponentialFamilyTest(unittest.TestCase):
    def _draws(self, xm, alpha, n=64, seed=0):
        rng = np.random.RandomState(seed)
        # numpy's pareto is Lomax (xm=1); (lomax + 1) * xm gives classical Pareto on [xm, inf).
        return (rng.pareto(alpha, n) + 1.0) * xm

    def test_reconstruction(self):
        for xm, alpha in [(1.0, 2.5), (2.0, 1.3), (0.5, 4.0)]:
            with self.subTest(xm=xm, alpha=alpha):
                d = ParetoDistribution(xm, alpha)
                self.assertTrue(is_exponential_family(d))
                form = to_exponential_family(d)
                self.assertIsInstance(form, ExponentialFamilyForm)

                x = self._draws(xm, alpha)
                eta = form.natural_parameters()
                t = np.asarray(form.sufficient_statistics(x), dtype=np.float64)
                a = float(form.log_partition())
                h = np.asarray(form.log_base_measure(x), dtype=np.float64)
                self.assertEqual(t.shape[1], eta.shape[0])
                self.assertEqual(form.dim, eta.shape[0])

                recon = h + t @ eta - a
                enc = d.dist_to_encoder().seq_encode(list(x))
                ref = np.asarray(d.seq_log_density(enc), dtype=np.float64)
                np.testing.assert_allclose(recon, ref, atol=1e-9)
                np.testing.assert_allclose(np.asarray(form.log_density(x)), ref, atol=1e-9)

    def test_below_support_is_neg_inf(self):
        """The base measure sends observations below xm to -inf (matching log_density)."""
        d = ParetoDistribution(2.0, 3.0)
        x = np.array([0.5, 1.0, 2.5, 5.0])  # first two are below xm=2
        form = to_exponential_family(d)
        recon = np.asarray(form.log_density(x), dtype=np.float64)
        ref = np.asarray(d.seq_log_density(d.dist_to_encoder().seq_encode(list(x))), dtype=np.float64)
        self.assertTrue(np.isneginf(recon[0]) and np.isneginf(recon[1]))
        np.testing.assert_allclose(recon[2:], ref[2:], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
