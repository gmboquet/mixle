"""CopulaDistribution (mixle.stats.combinator.copula): glue arbitrary marginals to a copula core via
Sklar's theorem, fit by IFM. Recovers a known correlation + heterogeneous marginals, beats independence,
samples with the right marginals and rank dependence, and composes as a mixle five-piece distribution."""

import unittest

import numpy as np
from scipy.stats import gamma as spgamma
from scipy.stats import norm, spearmanr

import mixle.stats as st
from mixle.inference import optimize
from mixle.stats.combinator.copula import (
    CopulaDistribution,
    CopulaEstimator,
    CopulaIFMStatistics,
    CopulaQuantileError,
)
from mixle.stats.multivariate.frank_copula import FrankCopulaDistribution
from mixle.stats.multivariate.gaussian_copula import GaussianCopulaDistribution


def _correlated_heterogeneous(seed, n=800, r=0.7):
    # latent Gaussian dependence pushed through a Gamma(2,2) marginal and a Gaussian(5,2) marginal
    rng = np.random.RandomState(seed)
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, r], [r, 1.0]], size=n)
    u = norm.cdf(z)
    x0 = spgamma.ppf(u[:, 0], a=2.0, scale=2.0)
    x1 = norm.ppf(u[:, 1], loc=5.0, scale=2.0)
    return list(zip(x0.tolist(), x1.tolist()))


def _proto():
    return CopulaDistribution(
        [st.GammaDistribution(1.0, 1.0), st.GaussianDistribution(0.0, 1.0)],
        GaussianCopulaDistribution(np.eye(2)),
    )


class _BadCdfMarginal:
    """A stand-in marginal whose cdf() always returns a fixed, possibly-invalid value -- simulates a
    broken marginal CDF implementation (or a caller feeding a value outside the marginal's support)."""

    def __init__(self, bad_value):
        self.bad_value = bad_value

    def cdf(self, x):
        return self.bad_value

    def quantile(self, q):
        return 0.0

    def log_density(self, x):
        return 0.0


class _MissingQuantileMarginal:
    def cdf(self, x):
        return norm.cdf(x)

    def log_density(self, x):
        return norm.logpdf(x)


class _BrokenQuantileMarginal(_BadCdfMarginal):
    def __init__(self):
        super().__init__(0.5)


class _RecordingEstimator:
    def __init__(self, delegate):
        self.delegate = delegate
        self.counts = []

    def accumulator_factory(self):
        return self.delegate.accumulator_factory()

    def estimate(self, nobs, suff_stat):
        self.counts.append(nobs)
        return self.delegate.estimate(nobs, suff_stat)


class CopulaDistributionTest(unittest.TestCase):
    def test_ifm_recovers_the_correlation_and_the_marginals(self):
        data = _correlated_heterogeneous(0, r=0.7)
        fit = optimize(data, _proto().estimator(), prev_estimate=_proto(), max_its=5, out=None)
        self.assertAlmostEqual(float(fit.copula.corr[0, 1]), 0.7, delta=0.08)
        # Gamma(shape k=2, scale theta=2); GaussianDistribution stores (mean mu, variance sigma2) = (5, 4)
        self.assertAlmostEqual(float(fit.marginals[0].k), 2.0, delta=0.4)
        self.assertAlmostEqual(float(fit.marginals[1].mu), 5.0, delta=0.3)

    def test_beats_the_independence_copula_on_dependent_data(self):
        data = _correlated_heterogeneous(1, r=0.7)
        fit = optimize(data, _proto().estimator(), prev_estimate=_proto(), max_its=5, out=None)
        indep = CopulaDistribution(fit.marginals, GaussianCopulaDistribution(np.eye(2)))
        ll_fit = float(np.sum(fit.seq_log_density(fit.dist_to_encoder().seq_encode(data))))
        ll_indep = float(np.sum(indep.seq_log_density(indep.dist_to_encoder().seq_encode(data))))
        self.assertGreater(ll_fit, ll_indep + 50.0)  # dependence is real and worth many nats

    def test_scalar_log_density_matches_the_sklar_decomposition(self):
        cop = CopulaDistribution(
            [st.GammaDistribution(2.0, 2.0), st.GaussianDistribution(5.0, 4.0)],
            GaussianCopulaDistribution(np.array([[1.0, 0.5], [0.5, 1.0]])),
        )
        x = (3.0, 4.5)
        u = np.clip([cop.marginals[0].cdf(x[0]), cop.marginals[1].cdf(x[1])], 1e-12, 1 - 1e-12)
        expected = cop.marginals[0].log_density(x[0]) + cop.marginals[1].log_density(x[1]) + cop.copula.log_density(u)
        self.assertAlmostEqual(cop.log_density(x), expected, places=10)

    def test_seq_and_scalar_log_density_agree(self):
        data = _correlated_heterogeneous(2, n=50)
        cop = CopulaDistribution(
            [st.GammaDistribution(2.0, 2.0), st.GaussianDistribution(5.0, 4.0)],
            GaussianCopulaDistribution(np.array([[1.0, 0.6], [0.6, 1.0]])),
        )
        seq = cop.seq_log_density(cop.dist_to_encoder().seq_encode(data))
        scalar = np.array([cop.log_density(x) for x in data])
        np.testing.assert_allclose(seq, scalar, atol=1e-9)

    def test_sampling_has_the_right_marginals_and_dependence(self):
        cop = CopulaDistribution(
            [st.GammaDistribution(2.0, 2.0), st.GaussianDistribution(5.0, 4.0)],
            GaussianCopulaDistribution(np.array([[1.0, 0.8], [0.8, 1.0]])),
        )
        s = np.array(cop.sampler(0).sample(3000))
        self.assertAlmostEqual(s[:, 0].mean(), 4.0, delta=0.4)  # Gamma mean = shape*scale = 4
        self.assertAlmostEqual(s[:, 1].mean(), 5.0, delta=0.3)  # Gaussian mean = 5
        rho, _ = spearmanr(s[:, 0], s[:, 1])
        self.assertGreater(rho, 0.6)  # strong positive rank dependence, as the copula prescribes

    def test_requires_at_least_two_marginals(self):
        with self.assertRaises(ValueError):
            CopulaDistribution([st.GaussianDistribution(0.0, 1.0)], GaussianCopulaDistribution(np.eye(1)))

    def test_rejects_a_discrete_marginal(self):
        # CopulaDistribution's log_density/seq_log_density apply the CONTINUOUS Sklar decomposition
        # f(x) = c(F(x)) * prod_i f_i(x_i), valid only when every f_i is a genuine density. For a discrete
        # marginal, log_density is a probability MASS, not a density -- plugging a mass into the continuous
        # copula-density formula does not produce a model that sums to 1 (see the reviewer's concrete
        # example below), so this must be rejected at construction rather than silently scored.
        with self.assertRaises(ValueError):
            CopulaDistribution(
                [st.BernoulliDistribution(0.5), st.GaussianDistribution(0.0, 1.0)],
                GaussianCopulaDistribution(np.array([[1.0, 0.7], [0.7, 1.0]])),
            )

    def test_rejects_a_mixed_discrete_continuous_pair_even_though_one_marginal_is_continuous(self):
        # a single discrete marginal is enough to break the continuous-Sklar formula for the whole joint --
        # "one of the marginals happens to be continuous" does not rescue it.
        with self.assertRaises(ValueError):
            CopulaDistribution(
                [st.GaussianDistribution(0.0, 1.0), st.BernoulliDistribution(0.3)],
                GaussianCopulaDistribution(np.array([[1.0, 0.7], [0.7, 1.0]])),
            )

    def test_two_discrete_bernoulli_marginals_would_not_sum_to_one_if_construction_were_allowed(self):
        # Regression for the concrete failure mode the discrete-marginal rejection above is guarding
        # against: bypass the constructor's guard (build via __new__ to set up the same state the checked
        # __init__ would) and confirm the old unconditional Sklar formula really does report a total
        # probability far from 1 across the four Bernoulli x Bernoulli outcomes -- i.e. the rejection in
        # test_rejects_a_discrete_marginal is not a false positive on an actually-fine computation.
        cop = object.__new__(CopulaDistribution)
        cop.marginals = [st.BernoulliDistribution(0.5), st.BernoulliDistribution(0.5)]
        cop.dim = 2
        cop.copula = GaussianCopulaDistribution(np.array([[1.0, 0.7], [0.7, 1.0]]))
        cop.name = None
        cop.keys = None
        total = sum(cop.density((a, b)) for a in (0, 1) for b in (0, 1))
        self.assertGreater(total, 1000.0)  # nowhere near a valid probability model (should be exactly 1)

    def test_rejects_a_copula_dimension_mismatched_with_the_marginal_count(self):
        # three marginals but a two-dimensional Frank copula used to succeed at construction (and even at
        # scoring, since seq_log_density only ever touches as many columns as the copula core expects) and
        # only blow up later, at sampling time, with an opaque reshape error.
        with self.assertRaises(ValueError):
            CopulaDistribution(
                [
                    st.GaussianDistribution(0.0, 1.0),
                    st.GaussianDistribution(0.0, 1.0),
                    st.GaussianDistribution(0.0, 1.0),
                ],
                FrankCopulaDistribution(2, 3.0),
            )

    def test_matching_copula_dimension_and_marginal_count_still_constructs(self):
        # negative control for the dimension check: equal counts must still work.
        cop = CopulaDistribution(
            [st.GaussianDistribution(0.0, 1.0), st.GaussianDistribution(0.0, 1.0), st.GaussianDistribution(0.0, 1.0)],
            GaussianCopulaDistribution(np.eye(3)),
        )
        self.assertEqual(cop.dim, 3)
        self.assertEqual(cop.copula.dim, 3)

    def test_requires_deterministic_quantiles(self):
        with self.assertRaises(TypeError):
            CopulaDistribution(
                [_MissingQuantileMarginal(), st.GaussianDistribution(0.0, 1.0)],
                GaussianCopulaDistribution(np.eye(2)),
            )

    def test_sampling_rejects_a_broken_quantile_cdf_round_trip(self):
        cop = CopulaDistribution(
            [_BrokenQuantileMarginal(), st.GaussianDistribution(0.0, 1.0)],
            GaussianCopulaDistribution(np.eye(2)),
        )
        with self.assertRaises(CopulaQuantileError):
            cop.sampler(4).sample()

    def test_scalar_and_batch_paths_require_exact_finite_geometry(self):
        cop = _proto()
        for row in ((1.0,), (1.0, 2.0, 3.0), (1.0, float("nan"))):
            with self.subTest(row=row), self.assertRaises(ValueError):
                cop.log_density(row)
            with self.subTest(encoded_row=row), self.assertRaises(ValueError):
                cop.dist_to_encoder().seq_encode([row])

    def test_empty_batch_has_canonical_shape_and_empty_scores(self):
        cop = _proto()
        enc = cop.dist_to_encoder().seq_encode([])
        self.assertEqual(enc[1].shape, (0, 2))
        self.assertEqual(cop.seq_log_density(enc).shape, (0,))

    def test_unsupported_copula_pseudo_count_fails_after_explicit_forwarding(self):
        marginals = [st.GaussianDistribution(0.0, 1.0), st.GaussianDistribution(0.0, 1.0)]
        core = GaussianCopulaDistribution(np.eye(2))
        calls = []
        for index, marginal in enumerate(marginals):
            original = marginal.estimator
            marginal.estimator = (
                lambda pseudo_count=None, original=original, index=index: (
                    calls.append(("marginal", index, pseudo_count)),
                    original(pseudo_count=pseudo_count),
                )[1]
            )
        original_core = core.estimator
        core.estimator = lambda pseudo_count=None: (
            calls.append(("copula", pseudo_count)),
            original_core(pseudo_count=pseudo_count),
        )[1]
        with self.assertRaises(ValueError):
            CopulaDistribution(marginals, core).estimator(pseudo_count=3.5)
        self.assertEqual(
            calls,
            [("marginal", 0, 3.5), ("marginal", 1, 3.5), ("copula", 3.5)],
        )

    def test_ifm_statistics_validate_geometry_weights_and_counts(self):
        cop = _proto()
        accumulator = cop.estimator().accumulator_factory().make()
        encoded = cop.dist_to_encoder().seq_encode([(1.0, 2.0), (2.0, 3.0)])
        accumulator.seq_update(encoded, np.array([0.25, 1.75]), cop)
        valid = accumulator.value()
        malformed = (
            valid._replace(marginal_statistics=valid.marginal_statistics[:1]),
            valid._replace(columns=np.zeros((2, 3))),
            valid._replace(weights=np.ones(1)),
            valid._replace(weights=np.array([1.0, -1.0])),
            valid._replace(marginal_effective_count=99.0),
            tuple(valid),
        )
        for value in malformed:
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                accumulator.from_value(value)
            with self.subTest(estimate=type(value).__name__), self.assertRaises(ValueError):
                cop.estimator().estimate(None, value)

    def test_ifm_forwards_the_buffered_effective_count_not_outer_nobs(self):
        cop = _proto()
        marginal_estimators = [_RecordingEstimator(m.estimator()) for m in cop.marginals]
        core_estimator = _RecordingEstimator(cop.copula.estimator())
        estimator = CopulaEstimator(
            marginal_estimators,
            core_estimator,
            dim=2,
            copula_prototype=cop.copula,
        )
        accumulator = estimator.accumulator_factory().make()
        encoded = cop.dist_to_encoder().seq_encode([(1.0, 2.0), (2.0, 3.0), (4.0, 5.0)])
        accumulator.seq_update(encoded, np.array([0.5, 1.0, 2.0]), cop)
        estimator.estimate(1000.0, accumulator.value())
        self.assertEqual([est.counts for est in marginal_estimators], [[3.5], [3.5]])
        self.assertEqual(core_estimator.counts, [3.5])

    def test_ifm_statistics_are_versioned(self):
        stats = CopulaIFMStatistics(2, (None, None), np.zeros((0, 2)), np.zeros(0), 0.0, 0.0)
        with self.assertRaises(ValueError):
            _proto().estimator().estimate(None, stats)

    def test_scalar_pit_rejects_a_marginal_cdf_outside_the_unit_interval(self):
        # a broken marginal.cdf() (bug, or a value outside the marginal's support) used to be silently
        # np.clip-ed onto the (0,1) boundary and scored as though it were a legitimate PIT observation,
        # instead of raising -- the same bug class already fixed in the copula cores themselves.
        cop = CopulaDistribution(
            [_BadCdfMarginal(1.5), st.GaussianDistribution(0.0, 1.0)], GaussianCopulaDistribution(np.eye(2))
        )
        with self.assertRaises(ValueError):
            cop.log_density((0.0, 0.0))

    def test_scalar_pit_rejects_a_nan_marginal_cdf(self):
        cop = CopulaDistribution(
            [_BadCdfMarginal(float("nan")), st.GaussianDistribution(0.0, 1.0)], GaussianCopulaDistribution(np.eye(2))
        )
        with self.assertRaises(ValueError):
            cop.log_density((0.0, 0.0))

    def test_batched_pit_rejects_a_marginal_cdf_outside_the_unit_interval(self):
        # _pit_columns (used by seq_log_density) has its own clip call, independent of the scalar
        # path's _pit_row -- both needed the same guard, so both get their own regression test.
        cop = CopulaDistribution(
            [_BadCdfMarginal(-0.2), st.GaussianDistribution(0.0, 1.0)], GaussianCopulaDistribution(np.eye(2))
        )
        with self.assertRaises(ValueError):
            cop._pit_columns(np.zeros((3, 2)))

    def test_batched_pit_rejects_a_nan_marginal_cdf(self):
        cop = CopulaDistribution(
            [_BadCdfMarginal(float("nan")), st.GaussianDistribution(0.0, 1.0)], GaussianCopulaDistribution(np.eye(2))
        )
        with self.assertRaises(ValueError):
            cop._pit_columns(np.zeros((3, 2)))

    def test_composes_inside_a_mixture(self):
        # two dependence regimes: a positively- and a negatively-correlated cluster, same marginals
        rng = np.random.RandomState(3)
        a = _correlated_heterogeneous(10, n=300, r=0.8)
        b = _correlated_heterogeneous(11, n=300, r=-0.8)
        data = a + b
        rng.shuffle(data)
        comp = [
            CopulaDistribution(
                [st.GammaDistribution(2.0, 2.0), st.GaussianDistribution(5.0, 4.0)],
                GaussianCopulaDistribution(np.array([[1.0, s], [s, 1.0]])),
            )
            for s in (0.5, -0.5)
        ]
        mix = st.MixtureDistribution(comp, [0.5, 0.5])
        fit = optimize(data, mix.estimator(), prev_estimate=mix, max_its=8, out=None)
        corrs = sorted(float(c.copula.corr[0, 1]) for c in fit.components)
        self.assertLess(corrs[0], -0.3)  # one regime recovered as negatively dependent
        self.assertGreater(corrs[1], 0.3)  # the other as positively dependent


if __name__ == "__main__":
    unittest.main()
