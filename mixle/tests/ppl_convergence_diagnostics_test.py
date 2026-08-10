"""Rank-normalized convergence diagnostics + NUTS divergence tracking (Vehtari et al. 2021)."""

import unittest
import warnings

import numpy as np

from mixle.ppl.diagnostics import bulk_ess, convergence_diagnostics, split_rhat, tail_ess


class RankNormalizedDiagnosticsTest(unittest.TestCase):
    def test_iid_draws(self):
        x = np.random.RandomState(0).standard_normal((4, 1000))
        self.assertAlmostEqual(split_rhat(x), 1.0, delta=0.02)
        self.assertGreater(bulk_ess(x), 3000)  # near the 4000 independent draws
        self.assertGreater(tail_ess(x), 2500)

    def test_autocorrelation_lowers_ess(self):
        rng = np.random.RandomState(0)
        ar = np.zeros((4, 1000))
        for c in range(4):
            for t in range(1, 1000):
                ar[c, t] = 0.9 * ar[c, t - 1] + rng.standard_normal()
        self.assertLess(bulk_ess(ar), 600)  # AR(0.9): theoretical ESS ~ n*(1-phi)/(1+phi) ~ 210

    def test_nonconverged_chains_flag_high_rhat(self):
        rng = np.random.RandomState(0)
        bad = rng.standard_normal((4, 1000)) + np.array([[-5], [0], [5], [10]])
        self.assertGreater(split_rhat(bad), 1.5)  # chains in different places

    def test_scale_mismatch_is_detected_by_folded_rhat(self):
        pattern = np.tile([-1.0, 1.0], 500)
        mismatched = np.vstack([pattern, pattern * 100.0])
        self.assertGreater(split_rhat(mismatched), 1.1)

    def test_malformed_inputs_are_rejected(self):
        # Shape errors are shape errors for every diagnostic: a flat vector and a 3-D array are
        # neither (n_chains, n_draws).
        for draws in (np.arange(20.0), np.zeros((2, 2, 5))):
            for diagnostic in (split_rhat, bulk_ess, tail_ess):
                with self.assertRaises(ValueError):
                    diagnostic(draws)

    def test_single_chain_is_rejected_for_rhat_but_estimable_for_ess(self):
        # R-hat compares independent chains and is meaningless with one, so it still rejects.
        # The ESS estimators split each chain into halves before estimating, which is exactly the
        # standard split-ESS construction -- a single long chain is an ordinary thing to want an
        # effective sample size for, and refusing it left every single-chain fit with ess=None.
        one_chain = np.linspace(0.0, 1.0, 64).reshape(1, -1)
        with self.assertRaises(ValueError):
            split_rhat(one_chain)
        self.assertTrue(np.isfinite(bulk_ess(one_chain)))
        self.assertTrue(np.isfinite(tail_ess(one_chain)))

    def test_single_chain_ess_still_needs_enough_draws_to_split(self):
        # Halving one chain must leave each half above the four-draw floor, so the floor doubles.
        with self.assertRaises(ValueError):
            bulk_ess(np.linspace(0.0, 1.0, 6).reshape(1, -1))
        nonfinite = np.ones((2, 20))
        nonfinite[0, 0] = np.nan
        with self.assertRaises(ValueError):
            convergence_diagnostics(nonfinite)

    def test_constant_chains_are_explicitly_unavailable(self):
        result = convergence_diagnostics(np.ones((4, 100)))
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["available"])
        self.assertEqual(set(result["unavailable"]), {"split_rhat", "bulk_ess", "tail_ess"})
        self.assertTrue(np.isnan(result["split_rhat"]))
        self.assertTrue(np.isnan(result["bulk_ess"]))
        self.assertTrue(np.isnan(result["tail_ess"]))


class NutsDivergenceTest(unittest.TestCase):
    def test_funnel_produces_divergences(self):
        from mixle.inference.mcmc import nuts

        def logp(z):
            v, x = z
            return -0.5 * (v / 3.0) ** 2 - 0.5 * (x * x) * np.exp(-v) - 0.5 * v

        def grad(z):
            v, x = z
            return np.array([-(v / 9.0) + 0.5 * (x * x) * np.exp(-v) - 0.5, -x * np.exp(-v)])

        res = nuts(logp, grad, np.array([0.0, 0.0]), num_samples=500, warmup=500, rng=np.random.RandomState(1))
        self.assertEqual(res.divergences.shape, (500,))
        self.assertGreater(int(res.divergences.sum()), 0)  # the funnel neck forces divergences

    def test_well_conditioned_target_has_few_divergences(self):
        from mixle.inference.mcmc import nuts

        res = nuts(
            lambda z: -0.5 * float(z @ z),
            lambda z: -z,
            np.zeros(3),
            num_samples=400,
            warmup=400,
            rng=np.random.RandomState(0),
        )
        self.assertLess(int(res.divergences.sum()), 20)  # a Gaussian rarely diverges


class SummaryExposesDiagnosticsTest(unittest.TestCase):
    def test_nuts_summary_carries_new_diagnostics(self):
        from mixle.ppl import Normal

        rng = np.random.RandomState(0)
        data = rng.normal(2.0, 1.5, 300)
        model = Normal(Normal(0, 10, name="mu"), Normal(0, 10, name="sig"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(data, how="nuts", draws=400, burn=400, chains=4, rng=np.random.RandomState(1))
        s = fit.summary()
        for key in ("_split_rhat", "_bulk_ess", "_tail_ess"):
            self.assertIn(key, s)
            self.assertIn("mu", s[key])
        self.assertLess(s["_split_rhat"]["mu"], 1.05)
        self.assertGreater(s["_bulk_ess"]["mu"], 100)


class EnsembleWalkerChainDiagnosticsTest(unittest.TestCase):
    """STAT-RR21-14: sweep-major pooled walker states read as one serial chain made the
    autocorrelation look like white noise -- median bulk ESS was the full state count, MCSE was
    understated 3.97x, and only 33% of analytic posterior means fell inside +/-1.96 MCSE. Each
    walker is now its own chain; replaying the reviewer's protocol measures SD/MCSE 0.88 and 96%
    coverage."""

    def test_mcse_covers_the_analytic_posterior_mean(self):
        import warnings

        import mixle.ppl as P
        from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

        sigma, prior_sd, n_obs = 2.0, 10.0, 40
        data = GaussianDistribution(1.0, sigma**2).sampler(seed=0).sample(n_obs)
        analytic_mean = (np.sum(data) / sigma**2) / (n_obs / sigma**2 + 1.0 / prior_sd**2)
        means, mcses, esses = [], [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(30):
                fit = P.Normal(P.Normal(0, prior_sd), sigma).fit(
                    data,
                    how="ensemble",
                    draws=100,
                    burn=50,
                    walkers=8,
                    chains=2,
                    rng=np.random.RandomState(seed),
                )
                row = fit.summary()["arg0"]
                means.append(row["mean"])
                mcses.append(row["mcse"])
                esses.append(row["ess_bulk"])
        means = np.asarray(means)
        mcses = np.asarray(mcses)
        ratio = means.std(ddof=1) / mcses.mean()
        self.assertLess(ratio, 2.0)  # was 3.97 flattened
        self.assertGreater(ratio, 0.4)
        coverage = np.mean(np.abs(means - analytic_mean) <= 1.959963984540054 * mcses)
        self.assertGreaterEqual(coverage, 0.80)  # was 0.33
        # walker serial correlation is REAL: the honest ESS is far below the raw state count
        self.assertLess(np.median(esses), 800)


class BimodalCertificationTest(unittest.TestCase):
    """STAT-RR22-12/-13: two same-cloud ensembles fell into ONE mode of a symmetric bimodal
    posterior and certified it -- R-hat 1.0095, ESS 1,683, status ok, mean wrong by 4,054 MCSEs.
    Prior-drawn walker inits + ensemble-level R-hat + real ok-thresholds close both halves."""

    def test_one_mode_is_never_certified_ok(self):
        import warnings

        import mixle.ppl as P
        from mixle.ppl.summarize import posterior_summary

        y = np.random.RandomState(7).normal(4.0, 0.5, size=40)
        for seed in range(2):
            mu = P.Normal(0, 5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = P.Normal(mu**2, 0.5).fit(
                    y,
                    how="ensemble",
                    draws=1500,
                    burn=500,
                    walkers=8,
                    chains=2,
                    rng=np.random.RandomState(seed),
                )
            summ = fit.summary()
            name = next(k for k in summ if not k.startswith("_"))
            mean = summ[name]["mean"]
            row = next(v for k, v in posterior_summary(fit).items() if not k.startswith("_"))
            certified_one_mode = row["diagnostic_status"] == "ok" and abs(mean) > 0.5
            self.assertFalse(certified_one_mode, f"seed {seed}: certified a one-mode mean {mean:+.3f}")

    def test_ok_enforces_thresholds_and_healthy_fits_keep_it(self):
        import warnings

        import mixle.ppl as P
        from mixle.ppl.summarize import posterior_summary
        from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

        data = GaussianDistribution(1.0, 4.0).sampler(seed=0).sample(300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bad = P.Normal(P.Normal(0, 10), P.HalfNormal(5)).fit(data, how="mcmc", draws=60, burn=5, chains=2)
            good = P.Normal(P.Normal(0, 10), P.HalfNormal(5)).fit(data, how="mcmc", draws=2000, burn=1000, chains=4)
        bad_statuses = [v["diagnostic_status"] for k, v in posterior_summary(bad).items() if not k.startswith("_")]
        self.assertIn("unconverged-by-diagnostics", bad_statuses)  # R-hat 1.24 / ESS 9 was "ok"
        good_statuses = [v["diagnostic_status"] for k, v in posterior_summary(good).items() if not k.startswith("_")]
        self.assertTrue(all(s == "ok" for s in good_statuses))

    def test_raw_ensemble_ess_is_walker_aware(self):
        import warnings

        import mixle.ppl as P
        from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

        data = GaussianDistribution(1.0, 4.0).sampler(seed=0).sample(40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = P.Normal(P.Normal(0, 10), 2.0).fit(
                data,
                how="ensemble",
                draws=100,
                burn=50,
                walkers=8,
                chains=2,
                rng=np.random.RandomState(0),
            )
        raw_ess = float(np.min(np.atleast_1d(fit.result.raw.effective_sample_size())))
        self.assertLess(raw_ess, 400.0)  # the flattened pseudo-chain read 1,143 (30.7x the truth)


if __name__ == "__main__":
    unittest.main()
