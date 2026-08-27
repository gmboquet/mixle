"""ppl/dist campaign-2 regressions: T4-5 (MAP boundary refusal), T2-06 (naming), T2-05 (parameterization).

T4-5: ``Normal(c, free).fit(data)`` on data constant at ``c`` raised ``FloatingPointError`` from the
autograd MAP target -- the free sd's MLE is the ``sigma -> 0`` boundary, the log-likelihood diverges,
and the L-BFGS walk underflowed. The all-free sibling ``Normal(free, free).fit`` accepts the same
data and answers with the closed-form estimator's scale-relative variance floor plus a
``numerical_repairs()`` receipt. The MAP path now retries inside a floored box and disclosed-repairs
to the same value.

While reproducing T4-5's LogNormal variant two more defects surfaced in the same autograd module and
are pinned here: (a) the LogNormal scorer fed RAW data to a backend documented as taking log-encoded
data, so every autograd route (``how='map'``/samplers/VI) silently fitted the Gaussian of the raw
data -- and mis-scored LogNormal priors identically; (b) the ``missing='marginalize'`` NaN sentinel
was 0.0, which the logarithmic preps (Gamma/Weibull/Beta/LogNormal) turned into ``-inf`` data terms,
so the documented marginalization raised instead of integrating the row out.

T2-06: the log-normal family was ``LogGaussian`` in the core catalog and ``LogNormal`` in the ppl
dialect with no bridge; a search under either name missed the other surface. Both directions now
alias: ``mixle.dist.LogNormalDistribution``/``LogNormalEstimator`` and ``mixle.ppl.LogGaussian``.

T2-05: five shared families use silently different parameterizations across the dialect and the
catalog (Normal sd vs variance, LogNormal/EMG sigma vs sigma2, Gamma/Exponential rate vs scale) and
Binomial swaps positional order. The divergence set was measured by density probing, is now
documented on both surfaces, and is pinned here so the documentation cannot rot.
"""

import unittest

import numpy as np

import mixle.dist
import mixle.ppl
import mixle.stats
from mixle.ppl import EMG, Exponential, Gamma, LogNormal, Normal, free
from mixle.ppl.core import _FAMILIES, lower


class FixedMeanScaleFitTest(unittest.TestCase):
    """T4-5: the MAP path accepts the degenerate scale fit the closed-form sibling accepts."""

    def test_fixed_mean_constant_data_fits_with_disclosed_floor(self):
        # Before the fix: FloatingPointError("cannot differentiate a non-finite autograd log target.")
        fitted = Normal(3.0, free).fit([3.0] * 5)
        sibling = Normal(free, free).fit([3.0] * 5)
        self.assertAlmostEqual(fitted._dist.mu, 3.0)
        # Same scale-relative variance floor the all-free sibling reports (1e-8 * scale^2).
        np.testing.assert_allclose(fitted._dist.sigma2, sibling._dist.sigma2, rtol=1e-6)
        self.assertTrue(fitted._dist.numerical_repairs(), "a binding floor must be disclosed")
        self.assertTrue(sibling._dist.numerical_repairs())

    def test_fixed_mean_constant_data_explicit_map_route(self):
        fitted = Normal(3.0, free).fit([3.0] * 5, how="map")
        np.testing.assert_allclose(fitted._dist.sigma2, 9.0e-8, rtol=1e-6)
        self.assertTrue(fitted._dist.numerical_repairs())

    def test_ordinary_fixed_mean_fit_is_untouched(self):
        # The repair path must not perturb an interior optimum: exact fixed-mean MLE, no receipt.
        rng = np.random.RandomState(0)
        data = (2.0 + 1.5 * rng.randn(4000)).tolist()
        fitted = Normal(2.0, free).fit(data)
        mle_sigma2 = float(np.mean((np.asarray(data) - 2.0) ** 2))
        np.testing.assert_allclose(fitted._dist.sigma2, mle_sigma2, rtol=1e-5)
        self.assertEqual(fitted._dist.numerical_repairs(), ())

    def test_lognormal_fixed_mu_constant_data_gets_the_same_repair(self):
        # data == 1.0 puts log-data exactly at the fixed mu=0: the same boundary collapse.
        fitted = LogNormal(0.0, free).fit([1.0] * 6)
        self.assertAlmostEqual(fitted._dist.mu, 0.0)
        self.assertLessEqual(fitted._dist.sigma2, 1.0e-7)  # floored near zero, not a silent 1.0
        self.assertTrue(fitted._dist.numerical_repairs())


class LogNormalAutogradTargetTest(unittest.TestCase):
    """The autograd LogNormal scorer must score log-encoded data, matching the closed form."""

    def test_map_agrees_with_the_closed_form_fit(self):
        rng = np.random.RandomState(1)
        data = np.exp(0.5 + 0.8 * rng.randn(3000)).tolist()
        em = LogNormal(free, free).fit(data)
        mp = LogNormal(free, free).fit(data, how="map")
        # Flat-prior MAP == MLE. Before the fix MAP returned the raw-data Gaussian
        # (mu ~ 2.29, sigma2 ~ 4.99 here) instead of the log-data fit (~0.51, ~0.63).
        np.testing.assert_allclose(mp._dist.mu, em._dist.mu, rtol=1e-3)
        np.testing.assert_allclose(mp._dist.sigma2, em._dist.sigma2, rtol=1e-3)


class MarginalizeSentinelTest(unittest.TestCase):
    """missing='marginalize' must integrate a NaN row out for log-prep families too."""

    def test_gamma_map_marginalize_equals_dropping_the_row(self):
        rng = np.random.RandomState(2)
        data = rng.gamma(2.0, 1.5, 500).tolist()
        data[10] = float("nan")
        # Before the fix: FloatingPointError (the 0.0 sentinel made the Gamma log(x) data term -inf).
        marginalized = Gamma(2.0, free).fit(data, how="map", missing="marginalize")
        dropped = Gamma(2.0, free).fit([v for i, v in enumerate(data) if i != 10], how="map")
        np.testing.assert_allclose(1.0 / marginalized._dist.theta, 1.0 / dropped._dist.theta, rtol=1e-9)


class LogNormalNamingBridgeTest(unittest.TestCase):
    """T2-06: the family is findable under both names on both surfaces."""

    def test_catalog_aliases_are_the_same_objects(self):
        self.assertIs(mixle.dist.LogNormalDistribution, mixle.dist.LogGaussianDistribution)
        self.assertIs(mixle.dist.LogNormalEstimator, mixle.dist.LogGaussianEstimator)
        self.assertIn("LogNormalDistribution", mixle.dist.__all__)
        self.assertIn("LogNormalEstimator", mixle.dist.__all__)

    def test_catalog_aliases_do_not_mutate_the_stats_surface(self):
        # dist.__all__ used to BE stats.__all__ (one shared list); the aliases must not leak back.
        self.assertNotIn("LogNormalDistribution", mixle.stats.__all__)
        self.assertNotIn("LogNormalEstimator", mixle.stats.__all__)

    def test_ppl_alias_is_the_same_constructor(self):
        self.assertIs(mixle.ppl.LogGaussian, mixle.ppl.LogNormal)
        self.assertIn("LogGaussian", mixle.ppl.__all__)
        lowered = lower(mixle.ppl.LogGaussian(0.4, 1.3), target="dist")
        self.assertIsInstance(lowered, mixle.dist.LogNormalDistribution)

    def test_both_docstrings_cross_reference_the_other_name(self):
        self.assertIn("LogGaussian", LogNormal.__doc__)
        self.assertIn("LogNormalDistribution", mixle.dist.__doc__)
        self.assertIn("LogNormal", mixle.dist.__doc__)


class ParameterizationMapTest(unittest.TestCase):
    """T2-05: the measured dialect-vs-catalog divergence set, and the documented conversions."""

    # In-support probe arguments valid on BOTH surfaces, per shared scalar family.
    PROBE_ARGS = {
        "Normal": (0.7, 1.6),
        "Poisson": (2.3,),
        "Gamma": (2.0, 1.5),
        "Exponential": (2.0,),
        "Bernoulli": (0.3,),
        "Geometric": (0.4,),
        "Beta": (2.0, 3.0),
        "StudentT": (5.0, 0.5, 1.5),
        "LogNormal": (0.4, 1.3),
        "EMG": (0.5, 1.2, 2.0),
        "NegativeBinomial": (3.0, 0.4),
        "HalfNormal": (1.7,),
        "InverseGamma": (3.0, 2.0),
        "InverseGaussian": (1.5, 2.0),
        "Gumbel": (0.5, 1.5),
        "SkewNormal": (0.5, 1.5, 2.0),
        "Skellam": (2.0, 1.0),
        "LogSeries": (0.4,),
        "VonMises": (0.5, 2.0),
        "GEV": (0.5, 1.5, 0.2),
        "Tweedie": (2.0, 1.5),
        "GeneralizedGaussian": (0.5, 1.5, 1.7),
        "GeneralizedPareto": (1.5, 0.3, 0.0),
        "Nakagami": (1.5, 2.0),
        "Rician": (1.0, 1.5),
        "Weibull": (1.5, 2.0),
        "Laplace": (0.5, 1.5),
        "Logistic": (0.5, 1.5),
        "Uniform": (0.0, 2.0),
        "Rayleigh": (1.5,),
        "Pareto": (1.0, 2.5),
    }
    # The families whose parameterization silently diverges (measured, then documented).
    DOCUMENTED_DIVERGENT = {"Normal", "Gamma", "Exponential", "LogNormal", "EMG"}

    def _probe_points(self, name):
        if name in {"Poisson", "Bernoulli", "Geometric", "NegativeBinomial", "LogSeries"}:
            return np.arange(0, 8)
        if name == "Skellam":
            return np.arange(-4, 5)
        if name == "VonMises":
            return np.linspace(-3.0, 3.0, 9)
        return np.linspace(0.05, 4.0, 9)

    def test_divergence_set_is_exactly_the_documented_one(self):
        divergent = set()
        for name, args in self.PROBE_ARGS.items():
            lowered = lower(getattr(mixle.ppl, name)(*args), target="dist")
            positional = _FAMILIES[name].dist_cls(*args)
            points = self._probe_points(name)
            got = np.array([lowered.log_density(x) for x in points])
            same_args = np.array([positional.log_density(x) for x in points])
            if not np.allclose(got, same_args, rtol=1e-9, atol=1e-12):
                divergent.add(name)
        self.assertEqual(divergent, self.DOCUMENTED_DIVERGENT)

    def test_documented_conversions_hold(self):
        self.assertAlmostEqual(lower(Normal(0.7, 1.6), target="dist").sigma2, 1.6**2)
        self.assertAlmostEqual(lower(Gamma(2.0, 1.5), target="dist").theta, 1.0 / 1.5)
        self.assertAlmostEqual(lower(Exponential(2.0), target="dist").beta, 1.0 / 2.0)
        self.assertAlmostEqual(lower(LogNormal(0.4, 1.3), target="dist").sigma2, 1.3**2)
        self.assertAlmostEqual(lower(EMG(0.5, 1.2, 2.0), target="dist").sigma2, 1.2**2)

    def test_binomial_positional_order_is_swapped_but_meanings_agree(self):
        lowered = lower(mixle.ppl.Binomial(7, 0.3), target="dist")
        positional = _FAMILIES["Binomial"].dist_cls(0.3, 7)  # catalog order: (p, n)
        points = np.arange(0, 8)
        np.testing.assert_allclose(
            [lowered.log_density(x) for x in points],
            [positional.log_density(x) for x in points],
            rtol=1e-12,
        )

    def test_mapping_is_documented_on_both_surfaces(self):
        for doc in (mixle.ppl.distributions.__doc__, mixle.dist.__doc__):
            for family in sorted(self.DOCUMENTED_DIVERGENT) + ["Binomial"]:
                self.assertIn(family, doc)
            self.assertIn("sigma2 = sd**2", doc)
            self.assertIn("theta = 1 / rate", doc)
            self.assertIn("beta = 1 / rate", doc)


if __name__ == "__main__":
    unittest.main()
