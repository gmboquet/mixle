"""Wave-3 diagnostics/introspection regressions (t5/t2/t4 external-review sweep).

Covers: ``supports``/``require`` accepting the capability-name strings the library itself emits;
``describe`` self-describing the structure-learned network ``optimize()`` returns by default;
``summarize`` deriving genuinely closed-form moments (finite mixtures, finite enumerable supports)
and receipting what it cannot produce; ``compare`` labelling rows distinguishably; ``fit`` naming
missing data (NaN) as the actual problem with the ``missing='marginalize'`` remedy; and the
``AutoregressiveEnumerable.unrank`` ordering contract (quantized between-bucket order, exact
log-probabilities, exact order at fine resolution).
"""

import unittest

import numpy as np

import mixle
from mixle.capability import CapabilityError
from mixle.stats import (
    BetaBinomialDistribution,
    CategoricalDistribution,
    IntegerCategoricalDistribution,
    MixtureDistribution,
    OptionalDistribution,
)
from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


class SupportsAcceptsOwnVocabularyTest(unittest.TestCase):
    """t2: supports() leaked a raw isinstance() TypeError on capabilities()' own name strings."""

    def test_every_emitted_capability_name_round_trips(self):
        g = GaussianDistribution(0.0, 1.0)
        names = mixle.capabilities(g)
        self.assertIn("HasMoments", names)
        for name in names:
            self.assertIs(mixle.supports(g, name), True, name)

    def test_name_string_matches_type_query(self):
        from mixle.capability import Enumerable, HasMoments

        g = GaussianDistribution(0.0, 1.0)
        self.assertEqual(mixle.supports(g, "HasMoments"), mixle.supports(g, HasMoments))
        self.assertEqual(mixle.supports(g, "Enumerable"), mixle.supports(g, Enumerable))
        self.assertFalse(mixle.supports(g, "Enumerable"))  # resolvable name, honestly unsupported

    def test_unknown_name_is_named_in_the_error(self):
        g = GaussianDistribution(0.0, 1.0)
        with self.assertRaises(TypeError) as ctx:
            mixle.supports(g, "NotACapability")
        message = str(ctx.exception)
        self.assertIn("NotACapability", message)
        self.assertIn("HasMoments", message)  # says what to pass instead
        self.assertNotIn("isinstance", message)  # no leaked internal machinery

    def test_catalogued_role_without_a_class_points_home(self):
        g = GaussianDistribution(0.0, 1.0)
        with self.assertRaises(TypeError) as ctx:
            mixle.supports(g, "Relation")
        self.assertIn("Relation", str(ctx.exception))
        self.assertIn("mixle.relations", str(ctx.exception))

    def test_require_accepts_name_strings(self):
        from mixle.capability import require

        g = GaussianDistribution(0.0, 1.0)
        require(g, "HasMoments")  # must not raise
        with self.assertRaises(CapabilityError) as ctx:
            require(g, "Enumerable", "top_k")
        self.assertIn("Enumerable", str(ctx.exception))

    def test_non_type_non_string_is_rejected_clearly(self):
        with self.assertRaises(TypeError) as ctx:
            mixle.supports(GaussianDistribution(0.0, 1.0), 3.14)
        self.assertIn("capability", str(ctx.exception))


class DescribeSelfDescribesOptimizeDefaultTest(unittest.TestCase):
    """t5: describe() reported 'no catalogued capability detected' for optimize()'s default result."""

    @staticmethod
    def _records(n=200, seed=7):
        rs = np.random.RandomState(seed)
        a = rs.choice(["x", "y", "z"], n, p=[0.5, 0.3, 0.2])
        b = np.where(rs.rand(n) < 0.8, a, rs.choice(["x", "y", "z"], n))
        c = rs.choice(["M", "F"], n)
        return list(zip(a.tolist(), b.tolist(), c.tolist()))

    def test_structure_learned_network_describes_itself(self):
        from mixle.inference import optimize

        bn = optimize(self._records(), seed=0)
        text = mixle.describe(bn)
        self.assertIn(type(bn).__name__, text)
        self.assertNotIn("no catalogued capability detected", text)
        self.assertIn("can:", text)
        self.assertIn("score", text)
        self.assertIn("sample", text)
        # no estimator() -> must not advertise "estimate" or the MAP/MCMC refit routes it cannot run
        self.assertNotIn("estimate", text.split("can:")[1].splitlines()[0])
        self.assertIn(".describe()", text)  # points at the object's own structured report

    def test_score_and_sample_object_without_estimator_is_rich(self):
        class Scorer:
            def log_density(self, x):
                return -1.0

            def sampler(self, seed=None):
                return None

        text = mixle.describe(Scorer())
        self.assertIn("Scorer", text)
        self.assertNotIn("no catalogued capability detected", text)
        self.assertIn("can:", text)

    def test_plain_distribution_view_is_unchanged(self):
        text = mixle.describe(GaussianDistribution(0.0, 1.0))
        self.assertIn("can:", text)
        self.assertIn("estimate", text)
        self.assertIn("sample", text)


class SummarizeClosedFormsTest(unittest.TestCase):
    """t5/t4: summarize() returned a bare {} for mixtures and finite discrete families."""

    def test_gaussian_mixture_moments_are_the_closed_form(self):
        mix = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 4.0)], [0.25, 0.75])
        s = mixle.summarize(mix)
        self.assertAlmostEqual(s["mean"], 2.25)  # .25*0 + .75*3
        self.assertAlmostEqual(s["variance"], 4.9375)  # .25*(1+0) + .75*(4+9) - 2.25**2
        self.assertAlmostEqual(s["std"], np.sqrt(4.9375))

    def test_mixture_moments_match_monte_carlo(self):
        mix = MixtureDistribution([GaussianDistribution(-2.0, 1.0), GaussianDistribution(5.0, 9.0)], [0.4, 0.6])
        s = mixle.summarize(mix)
        draws = np.asarray(mix.sampler(seed=0).sample(200_000), dtype=float)
        self.assertAlmostEqual(s["mean"], float(draws.mean()), delta=0.05)
        self.assertAlmostEqual(s["variance"], float(draws.var()), delta=0.3)

    def test_integer_categorical_moments_are_exact(self):
        d = IntegerCategoricalDistribution(min_val=2, p_vec=[0.2, 0.5, 0.3])
        s = mixle.summarize(d)
        self.assertAlmostEqual(s["mean"], 3.1)
        self.assertAlmostEqual(s["variance"], 0.49)
        self.assertAlmostEqual(s["entropy"], -sum(p * np.log(p) for p in (0.2, 0.5, 0.3)))
        self.assertEqual(s["median"], 3.0)
        self.assertEqual(s["mode"], 3)

    def test_beta_binomial_moments_are_the_closed_form(self):
        n, a, b = 10, 2.0, 3.0
        s = mixle.summarize(BetaBinomialDistribution(n, a, b))
        self.assertAlmostEqual(s["mean"], n * a / (a + b))  # 4.0
        expected_var = n * a * b * (a + b + n) / ((a + b) ** 2 * (a + b + 1))  # 6.0
        self.assertAlmostEqual(s["variance"], expected_var)

    def test_label_categorical_reports_why_moments_are_missing(self):
        s = mixle.summarize(CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2}))
        self.assertNotIn("mean", s)
        self.assertIn("not numeric", s["_status"]["mean"]["reason"])
        self.assertAlmostEqual(s["entropy"], -sum(p * np.log(p) for p in (0.5, 0.3, 0.2)))
        self.assertEqual(s["mode"], "a")  # highest probability label

    def test_unavailable_statistics_are_receipted_not_blank(self):
        s = mixle.summarize(OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.1))
        status = s["_status"]
        self.assertTrue(status)  # never again a bare empty receipt
        for key in ("mean", "variance", "std", "entropy", "median"):
            self.assertEqual(status[key]["status"], "unavailable", key)
            self.assertTrue(status[key]["reason"], key)

    def test_mixture_of_momentless_components_reports_why(self):
        mvn = MultivariateGaussianDistribution(np.zeros(2), np.eye(2))
        s = mixle.summarize(MixtureDistribution([mvn, mvn], [0.5, 0.5]))
        self.assertNotIn("mean", s)
        self.assertIn("MultivariateGaussianDistribution", s["_status"]["mean"]["reason"])

    def test_oversized_finite_support_is_receipted_with_the_cap(self):
        big = IntegerCategoricalDistribution(min_val=0, p_vec=np.full(100_001, 1.0 / 100_001))
        s = mixle.summarize(big)
        self.assertNotIn("mean", s)
        self.assertIn("support_size=100001", s["_status"]["mean"]["reason"])

    def test_plain_gaussian_summary_is_unchanged(self):
        s = mixle.summarize(GaussianDistribution(1.0, 4.0))
        self.assertAlmostEqual(s["mean"], 1.0)
        self.assertAlmostEqual(s["variance"], 4.0)
        self.assertAlmostEqual(s["std"], 2.0)
        self.assertAlmostEqual(s["median"], 1.0)


class CompareRowsAreDistinguishableTest(unittest.TestCase):
    """t5: compare() labelled every regression row 'RegressionResult'."""

    @classmethod
    def setUpClass(cls):
        from mixle.ppl import Field, Normal, free

        rng = np.random.RandomState(0)
        n = 200
        flip, dep = rng.normal(0, 1, n), rng.normal(0, 1, n)
        cls.y = 1.5 * flip - 0.7 * dep + rng.normal(0, 1, n)
        given = {"flip": flip, "dep": dep}
        cls.m1 = Normal(free * Field("flip") + free, free).fit(cls.y, given=given)
        cls.m2 = Normal(free * Field("flip") + free * Field("dep") + free, free).fit(cls.y, given=given)
        cls.m2_again = Normal(free * Field("flip") + free * Field("dep") + free, free).fit(cls.y, given=given)

    def test_regression_rows_carry_formula_identity(self):
        from mixle.ppl import compare

        labels = {row["model"] for row in compare([self.m1, self.m2], self.y)}
        self.assertEqual(len(labels), 2)
        self.assertIn("RegressionResult(flip + intercept)", labels)
        self.assertIn("RegressionResult(flip + dep + intercept)", labels)

    def test_identical_formulas_still_map_back_to_input_positions(self):
        from mixle.ppl import compare

        labels = [row["model"] for row in compare([self.m1, self.m2, self.m2_again], self.y)]
        self.assertEqual(len(set(labels)), 3)  # every row identifiable
        self.assertIn("RegressionResult(flip + dep + intercept) (model 2)", labels)
        self.assertIn("RegressionResult(flip + dep + intercept) (model 3)", labels)
        self.assertIn("RegressionResult(flip + intercept)", labels)  # unique label needs no suffix

    def test_explicit_model_names_win(self):
        from mixle.ppl import Normal, compare, free

        rng = np.random.RandomState(1)
        data = rng.normal(0.0, 1.0, 80)
        named = Normal(free, free, name="baseline").fit(data)
        rows = compare([named], data)
        self.assertEqual(rows[0]["model"], "baseline")


class FitNamesMissingDataTest(unittest.TestCase):
    """t5/t4: NaN data was rejected as 'requires support x in (-inf,inf)' with no remedy named."""

    @staticmethod
    def _nan_data():
        rng = np.random.RandomState(0)
        y = rng.normal(4200.0, 45.0, 120)
        y[3] = np.nan
        y[77] = np.nan
        return y

    def test_nan_error_names_missing_data_and_the_remedy(self):
        from mixle.ppl import Normal, free

        with self.assertRaises(ValueError) as ctx:
            Normal(free, free).fit(self._nan_data())
        message = str(ctx.exception)
        self.assertIn("2 NaN", message)
        self.assertIn("missing", message)
        self.assertIn("missing='marginalize'", message)
        self.assertIsNotNone(ctx.exception.__cause__)  # the original rejection stays chained

    def test_marginalize_still_fits_the_present_rows(self):
        from mixle.ppl import Normal, free

        y = self._nan_data()
        fitted = Normal(free, free).fit(y, missing="marginalize")
        self.assertAlmostEqual(fitted.params["mean"], float(np.nanmean(y)), places=6)

    def test_infinite_data_keeps_the_genuine_support_error(self):
        from mixle.ppl import Normal, free

        y = np.random.RandomState(0).normal(0.0, 1.0, 60)
        y[5] = np.inf
        with self.assertRaises(ValueError) as ctx:
            Normal(free, free).fit(y)
        message = str(ctx.exception)
        self.assertNotIn("NaN", message)  # inf is out-of-support, not missing
        self.assertIn("support", message)


class AutoregressiveUnrankContractTest(unittest.TestCase):
    """t5: unrank's docstring promised exact descending order while the index order is quantized."""

    @staticmethod
    def _model(**kw):
        from mixle.enumeration.autoregressive import AutoregressiveEnumerable

        tables = {}

        def next_logprobs(prefix):
            key = prefix[-1] if prefix else -1
            if key not in tables:
                r = np.random.RandomState(1000 + key)
                tables[key] = np.log(r.dirichlet(np.ones(6) * 2.0))
            return list(enumerate(tables[key]))

        return AutoregressiveEnumerable(next_logprobs, eos=0, max_depth=6, **kw)

    def test_unrank_log_probabilities_are_always_exact(self):
        model = self._model()
        for i in range(10):
            seq, lp = model.unrank(i)
            self.assertAlmostEqual(lp, model.log_density(seq), places=12)

    def test_top_k_streams_strictly_descending(self):
        head = self._model().top_k(10)
        lps = [lp for _, lp in head]
        self.assertEqual(lps, sorted(lps, reverse=True))

    def test_fine_resolution_restores_exact_inversion(self):
        model = self._model(bin_width_bits=0.01)
        for i in range(8):
            self.assertEqual(model.rank(model.unrank(i)[0]).rank, i)
        top = [seq for seq, _ in model.top_k(8)]
        unranked = [model.unrank(i)[0] for i in range(8)]
        self.assertEqual(top, unranked)

    def test_unrank_contract_documents_the_quantized_order(self):
        from mixle.enumeration.autoregressive import AutoregressiveEnumerable

        doc = AutoregressiveEnumerable.unrank.__doc__
        self.assertIn("QUANTIZED", doc)
        self.assertIn("oversample", doc)
        self.assertIn("exact", doc.lower())
        self.assertIn("bucket", AutoregressiveEnumerable.threshold.__doc__)

    def test_default_resolution_misordering_stays_within_one_fine_bucket(self):
        # The documented contract: order is exact BETWEEN fine buckets, so any inversion among
        # neighbouring unrank results must involve log-probs closer than one bucket's width
        # (bin_width_bits / oversample bits, in nats) times the depth-roundoff bound.
        model = self._model()
        items = [model.unrank(i) for i in range(12)]
        bucket_nats = (model.bin_width_bits / model.oversample) * np.log(2.0)
        depth_bound = model._depth  # per-step floor rounding accumulates at most one bucket per step
        for (_, lp_a), (_, lp_b) in zip(items, items[1:]):
            if lp_b > lp_a:  # an inversion: must be a near-tie within the quantization tolerance
                self.assertLess(lp_b - lp_a, bucket_nats * depth_bound)


class SingleChainEssMachineryTest(unittest.TestCase):
    """Guards the helpers the ppl single-chain diagnostics diffs rely on (t5 wave-3, A2/A3)."""

    def test_public_bulk_and_tail_ess_accept_a_single_chain(self):
        from mixle.ppl import bulk_ess, tail_ess

        rng = np.random.RandomState(0)
        chain = rng.standard_normal((1, 1000))
        self.assertTrue(np.isfinite(bulk_ess(chain)))
        self.assertTrue(np.isfinite(tail_ess(chain)))


class SingleChainReportedDiagnosticsTest(unittest.TestCase):
    """t5 A2/A3: apply together with the mixle/ppl/inference.py and mixle/ppl/diagnostics.py patches.

    Single-chain fits must report the finite single-chain ess_bulk/ess_tail (split-chain estimators)
    with split_r_hat NaN as the mixing caveat, and convergence_diagnostics() must return its
    availability receipt for one chain instead of raising."""

    def test_convergence_diagnostics_receipts_a_single_chain(self):
        from mixle.ppl import convergence_diagnostics

        draws = np.random.RandomState(0).standard_normal((1, 1000))
        receipt = convergence_diagnostics(draws)  # must not raise
        self.assertTrue(np.isnan(receipt["split_rhat"]))
        self.assertTrue(np.isfinite(receipt["bulk_ess"]))
        self.assertTrue(np.isfinite(receipt["tail_ess"]))
        self.assertEqual(receipt["unavailable"], ["split_rhat"])
        self.assertIn("chains", receipt["unavailable_because"]["split_rhat"])

    def test_constant_chain_receipt_still_reports_everything_unavailable(self):
        from mixle.ppl import convergence_diagnostics

        receipt = convergence_diagnostics(np.zeros((2, 500)))
        self.assertEqual(set(receipt["unavailable"]), {"split_rhat", "bulk_ess", "tail_ess"})
        self.assertEqual(receipt["status"], "unavailable")

    def test_single_chain_nuts_summary_reports_finite_ess(self):
        import warnings

        from mixle.ppl import Normal, bulk_ess, free

        rng = np.random.RandomState(0)
        y = rng.normal(4200.0, 45.0, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = Normal(free, free).fit(y, how="nuts", draws=500, burn=300, rng=np.random.RandomState(1))
        row = fitted.summary()["arg0"]
        self.assertTrue(np.isfinite(row["ess_bulk"]))
        self.assertTrue(np.isfinite(row["ess_tail"]))
        self.assertTrue(np.isnan(row["split_r_hat"]))  # mixing stays unassessable on one chain
        draws = np.asarray(fitted.result.samples())
        col = draws[:, 0] if draws.ndim == 2 else draws
        self.assertAlmostEqual(row["ess_bulk"], float(bulk_ess(col.reshape(1, -1))), places=6)

    def test_single_chain_laplace_summary_reports_finite_ess(self):
        import warnings

        from mixle.ppl import Normal, free

        y = np.random.RandomState(0).normal(10.0, 2.0, 150)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = Normal(free, free).fit(y, how="laplace")
        row = fitted.summary()["arg0"]
        self.assertTrue(np.isfinite(row["ess_bulk"]))
        self.assertTrue(np.isfinite(row["ess_tail"]))


if __name__ == "__main__":
    unittest.main()
