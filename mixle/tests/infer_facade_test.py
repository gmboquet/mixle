"""Tests for the bring-your-own-target inference facade (``mixle.inference``).

Covers the fast fused-``value_and_grad`` NUTS path, the public R-hat/ESS diagnostics over plain
arrays, and (when Torch is present) ADVI on an external batched target. Also asserts the F0 fused
path issues strictly fewer target evaluations than the historical split-callable path.
"""

import unittest

import numpy as np

import mixle.inference as infer
from mixle.inference.diagnostics import ess, rhat
from mixle.inference.mcmc import nuts as nuts_sampler
from mixle.ppl.autograd import torch_available

HAS_TORCH = torch_available()


def _gaussian_target(mu, cov):
    """Return (value_and_grad, log_target, grad) for N(mu, cov)."""
    prec = np.linalg.inv(cov)

    def vg(x):
        x = np.asarray(x, dtype=float)
        d = x - mu
        return float(-0.5 * d @ prec @ d), -prec @ d

    def lt(x):
        x = np.asarray(x, dtype=float)
        d = x - mu
        return float(-0.5 * d @ prec @ d)

    def gr(x):
        x = np.asarray(x, dtype=float)
        return -prec @ (x - mu)

    return vg, lt, gr


class InferNutsTest(unittest.TestCase):
    def test_recovers_gaussian_posterior(self):
        mu = np.array([1.0, -2.0, 0.5])
        cov = np.array([[1.0, 0.4, 0.0], [0.4, 2.0, 0.3], [0.0, 0.3, 0.5]])
        vg, _, _ = _gaussian_target(mu, cov)
        res = infer.nuts(vg, dim=3, num_samples=4000, warmup=1000, chains=1, rng=0)
        self.assertEqual(res.samples.shape, (4000, 3))
        np.testing.assert_allclose(res.samples.mean(axis=0), mu, atol=0.1)
        np.testing.assert_allclose(np.cov(res.samples.T), cov, atol=0.25)

    def test_multichain_rhat_near_one(self):
        mu = np.array([0.0, 3.0])
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        vg, _, _ = _gaussian_target(mu, cov)
        res = infer.nuts(vg, dim=2, num_samples=2000, warmup=1000, chains=4, rng=1)
        self.assertEqual(res.chains.shape, (4, 2000, 2))
        self.assertTrue(np.all(res.rhat < 1.05), msg=f"rhat={res.rhat}")
        self.assertTrue(np.all(res.ess > 200), msg=f"ess={res.ess}")

    def test_fused_path_fewer_evals_than_split(self):
        mu = np.array([0.5, -1.0])
        cov = np.array([[1.0, 0.3], [0.3, 1.5]])
        vg, lt, gr = _gaussian_target(mu, cov)

        split_counter = {"lt": 0, "gr": 0}

        def lt_c(x):
            split_counter["lt"] += 1
            return lt(x)

        def gr_c(x):
            split_counter["gr"] += 1
            return gr(x)

        split = nuts_sampler(lt_c, gr_c, np.zeros(2), num_samples=500, warmup=500, rng=np.random.RandomState(7))
        fused = nuts_sampler(
            value_and_grad=vg, initial=np.zeros(2), num_samples=500, warmup=500, rng=np.random.RandomState(7)
        )

        # Identical RNG -> identical trajectory; the fused path runs the SAME iterations but caches
        # endpoint gradients and fuses logp+grad, so it issues far fewer underlying evaluations.
        np.testing.assert_allclose(np.asarray(split.samples), np.asarray(fused.samples))
        # Both paths now cache endpoint gradients, so they run the same number of leapfrog steps.
        # The split path must call BOTH callables per step (a forward for logp + a forward for the
        # gradient), while the fused path needs exactly one fused call per step. So the split path
        # issues exactly 2x the underlying forwards the fused path does -- and the fused call count
        # equals each split callable's count.
        fused_forwards = fused.num_target_evals
        self.assertEqual(split_counter["lt"], fused_forwards)
        self.assertEqual(split_counter["gr"], fused_forwards)
        split_forwards = split_counter["lt"] + split_counter["gr"]
        self.assertEqual(split_forwards, 2 * fused_forwards)


class DiagnosticsTest(unittest.TestCase):
    def test_rhat_on_handbuilt_array(self):
        rng = np.random.RandomState(0)
        # Four well-mixed chains from the same target -> rhat ~ 1.
        chains = rng.standard_normal((4, 500, 3))
        r = rhat(chains)
        self.assertEqual(r.shape, (3,))
        self.assertTrue(np.all(np.abs(r - 1.0) < 0.1), msg=f"rhat={r}")

    def test_rhat_flags_divergent_chains(self):
        rng = np.random.RandomState(0)
        chains = rng.standard_normal((4, 500, 1))
        chains[0] += 10.0  # one chain stuck far away
        r = rhat(chains)
        self.assertGreater(float(r[0]), 1.5)

    def test_rhat_scalar_param_2d_input(self):
        rng = np.random.RandomState(1)
        chains = rng.standard_normal((3, 400))  # (n_chains, n_draws) scalar param
        r = rhat(chains)
        self.assertEqual(r.shape, (1,))
        self.assertLess(abs(float(r[0]) - 1.0), 0.1)

    def test_ess_iid_near_n(self):
        rng = np.random.RandomState(2)
        draws = rng.standard_normal((2000, 2))  # iid -> ess ~ n
        e = ess(draws)
        self.assertEqual(e.shape, (2,))
        self.assertTrue(np.all(e > 1500), msg=f"ess={e}")

    def test_ess_autocorrelated_below_n(self):
        rng = np.random.RandomState(3)
        n = 2000
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.9 * x[t - 1] + rng.standard_normal()  # AR(1), strong autocorrelation
        e = ess(x)
        self.assertEqual(e.shape, (1,))
        self.assertLess(float(e[0]), 0.5 * n)

    def test_ess_pools_chains(self):
        rng = np.random.RandomState(4)
        chains = rng.standard_normal((3, 1000, 1))
        e = ess(chains)
        self.assertGreater(float(e[0]), 2000)  # pooled over 3 iid chains of 1000


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class AdviExternalTargetTest(unittest.TestCase):
    def test_advi_recovers_gaussian(self):
        import torch

        mu = np.array([1.0, -1.0])
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        prec_t = torch.tensor(np.linalg.inv(cov))
        mu_t = torch.tensor(mu)

        def target_batch(U):
            d = U - mu_t
            return -0.5 * (d @ prec_t * d).sum(dim=1)

        res = infer.advi(
            target_batch,
            u0=np.zeros(2),
            s0=np.ones(2),
            samples=4000,
            mc=32,
            steps=1500,
            lr=0.05,
            family="fullrank",
            rng=0,
        )
        self.assertEqual(res.samples.shape, (4000, 2))
        np.testing.assert_allclose(res.mean, mu, atol=0.15)
        np.testing.assert_allclose(np.cov(res.samples.T), cov, atol=0.4)

    def test_advi_meanfield_runs(self):
        import torch

        mu_t = torch.tensor([2.0, 0.0])

        def target_batch(U):
            d = U - mu_t
            return -0.5 * (d * d).sum(dim=1)

        res = infer.advi(target_batch, u0=np.zeros(2), s0=np.ones(2), samples=1000, steps=800, rng=1)
        np.testing.assert_allclose(res.mean, [2.0, 0.0], atol=0.15)
        np.testing.assert_allclose(res.scale, [1.0, 1.0], atol=0.2)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class AdviControlValidationTest(unittest.TestCase):
    """MXR-080-1643: an AdviResult may not come back from a run that never optimized anything."""

    @staticmethod
    def _target(U):
        return -0.5 * (U * U).sum(dim=1)

    def _advi(self, **kw):
        base = dict(u0=np.zeros(2), s0=np.ones(2), samples=16, mc=4, steps=3, lr=0.05, rng=0)
        base.update(kw)
        return infer.advi(self._target, **base)

    def test_non_positive_or_fractional_counts_are_rejected(self):
        for field in ("samples", "mc", "steps"):
            for bad, exc in ((0, ValueError), (-1, ValueError), (True, TypeError), (2.5, TypeError)):
                with self.subTest(field=repr(field), bad=repr(bad)), self.assertRaises(exc):
                    self._advi(**{field: bad})

    def test_learning_rate_must_be_finite_and_positive(self):
        for bad in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(lr=repr(bad)), self.assertRaisesRegex(ValueError, "lr must be"):
                self._advi(lr=bad)

    def test_alpha_domain_is_declared_and_enforced(self):
        for bad in (float("nan"), float("inf"), -0.5):
            with self.subTest(alpha=repr(bad)), self.assertRaisesRegex(ValueError, "alpha must be"):
                self._advi(alpha=bad)
        self.assertIsNotNone(self._advi(alpha=0.0))  # IWAE bound stays supported

    def test_initialization_must_be_aligned_finite_and_positively_scaled(self):
        bad_inits = [
            (np.zeros(2), np.zeros(2)),  # zero scale -> scale 0 and objective -inf
            (np.zeros(2), -np.ones(2)),  # negative scale -> all-NaN parameters
            (np.zeros(2), np.ones(3)),  # misaligned
            (np.array([0.0, np.nan]), np.ones(2)),  # non-finite mean
            (np.zeros(2), np.array([1.0, np.inf])),  # non-finite scale
            (np.zeros((2, 2)), np.ones((2, 2))),  # not one-dimensional
            (np.zeros(0), np.zeros(0)),  # nothing to infer
        ]
        for u0, s0 in bad_inits:
            with self.subTest(u0=repr(u0.shape), s0=repr(s0.shape)), self.assertRaises(ValueError):
                self._advi(u0=u0, s0=s0)

    def test_a_valid_run_still_returns_finite_fitted_parameters(self):
        res = self._advi(steps=50, samples=32)
        self.assertEqual(res.samples.shape, (32, 2))
        self.assertTrue(np.isfinite(res.mean).all())
        self.assertTrue(np.all(res.scale > 0.0))
        self.assertTrue(np.isfinite(res.objective))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class AdviTargetRankTest(unittest.TestCase):
    """MXR-080-1644: the target must honour its documented ``(mc,)`` per-draw output contract."""

    def _advi(self, target, **kw):
        base = dict(u0=np.zeros(2), s0=np.ones(2), samples=8, mc=4, steps=2, lr=0.05, rng=0)
        base.update(kw)
        return infer.advi(target, **base)

    def test_scalar_sum_over_the_whole_batch_is_rejected(self):
        def scalar_target(U):
            return -0.5 * (U * U).sum()  # one number for the whole Monte Carlo batch

        with self.assertRaisesRegex(ValueError, "one log-density per Monte Carlo draw"):
            self._advi(scalar_target)

    def test_matrix_output_is_rejected(self):
        def matrix_target(U):
            return -0.5 * (U * U)  # (mc, d), not (mc,)

        with self.assertRaisesRegex(ValueError, "one log-density per Monte Carlo draw"):
            self._advi(matrix_target)

    def test_broadcastable_wrong_width_output_is_rejected(self):
        def wrong_width(U):
            return -0.5 * (U * U).sum(dim=1)[:1]  # (1,) broadcasts against (mc,) but is not it

        with self.assertRaisesRegex(ValueError, "one log-density per Monte Carlo draw"):
            self._advi(wrong_width)

    def test_detached_target_is_rejected_before_backprop(self):
        import torch

        def detached(U):
            return torch.zeros(U.shape[0], dtype=U.dtype)  # no gradient path to the variational params

        with self.assertRaisesRegex(ValueError, "detached"):
            self._advi(detached)

    def test_the_documented_shape_is_accepted(self):
        def good(U):
            return -0.5 * (U * U).sum(dim=1)

        res = self._advi(good, steps=20)
        self.assertEqual(res.samples.shape, (8, 2))


class AdviObjectiveHonestyTest(unittest.TestCase):
    """Audit A-1/A-2/A-3: alpha-near-1 band, K disclosure, finiteness-not-convergence wording."""

    @staticmethod
    def _fit(alpha, seed=0):
        def target_batch(u):
            return -0.5 * ((u - 2.0) ** 2).sum(dim=1)

        return infer.advi(
            target_batch, u0=np.zeros(1), s0=np.ones(1), samples=64, mc=32, steps=400, lr=0.05, alpha=alpha, rng=seed
        )

    def test_alpha_just_below_one_routes_to_the_elbo_branch(self):
        # An exact float test sent alpha = 1 - 1e-9 down the Renyi branch, where the objective
        # divides logsumexp round-off by (1 - alpha); inside the band both runs must be identical.
        edge, exact = self._fit(1.0 - 1e-9), self._fit(1.0)
        self.assertEqual(edge.objective, exact.objective)
        np.testing.assert_allclose(edge.mean, exact.mean)
        self.assertLess(abs(edge.mean[0] - 2.0), 0.25)

    def test_objective_reports_its_evaluation_draw_count(self):
        res = self._fit(0.5)
        self.assertEqual(res.objective_n_eval, 256)  # max(mc=32, 256): the K the bound is at
        self.assertTrue(np.isfinite(res.objective))


if __name__ == "__main__":
    unittest.main()
