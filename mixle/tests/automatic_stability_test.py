"""Stability fixes for mixle.utils.automatic: no crashes on degenerate input, and detector-backed
families no longer silently drop pseudo_count/the family's own already-computed fit.

Before this file's fixes:
  - get_poisson_estimator/get_gaussian_estimator/get_lognormal_estimator raised ZeroDivisionError on
    an empty (or entirely non-finite/non-positive) vdict -- reachable via get_length_estimator on an
    empty length distribution, among other paths.
  - every detector-registered family's _factory() (weibull, beta, gumbel, ...) discarded the
    pseudo_count argument and returned a fixed-parameter estimator unrelated to the fit the SAME
    detector already computed in _score()/_fit() moments earlier -- pseudo_count=1.0, the default in
    get_estimator/get_prototype, silently applied NO regularization for any detector-selected family.
"""

import unittest

import numpy as np

from mixle.utils.automatic.factories import (
    get_gaussian_estimator,
    get_length_estimator,
    get_lognormal_estimator,
    get_poisson_estimator,
)


class DegenerateInputDoesNotCrashTest(unittest.TestCase):
    def test_poisson_empty_vdict(self):
        get_poisson_estimator({})  # must not raise ZeroDivisionError

    def test_gaussian_empty_vdict(self):
        get_gaussian_estimator({})

    def test_lognormal_empty_vdict(self):
        get_lognormal_estimator({})

    def test_poisson_all_nan_keys(self):
        get_poisson_estimator({float("nan"): 5.0})

    def test_lognormal_all_non_positive_keys(self):
        # every key filtered out (log-normal needs k > 0), same empty-accumulator shape as an empty dict
        get_lognormal_estimator({-1.0: 3.0, -2.0: 5.0})

    def test_length_estimator_on_an_empty_length_distribution(self):
        # get_length_estimator falls through to get_poisson_estimator on a dense/empty support;
        # an empty len_dict must not crash the fallback.
        get_length_estimator({})

    def test_gaussian_and_lognormal_still_use_real_data_when_present(self):
        # the guard must not swallow the normal, well-populated path -- the estimator should still
        # carry a suff_stat computed from the data, not silently fall back to None/defaults.
        vdict = {1.0: 3.0, 2.0: 5.0, 3.0: 2.0}
        est = get_gaussian_estimator(vdict, pseudo_count=None)
        self.assertIsNotNone(est.suff_stat[0])


class DetectorFactoryUsesTheAlreadyComputedFitTest(unittest.TestCase):
    """The core bug: every detector _factory() ignored its own _fit()/_params() and pseudo_count."""

    def test_weibull_factory_seeds_from_the_actual_fit_not_a_hardcoded_default(self):
        from mixle.utils.automatic.detectors.weibull import _factory, _fit

        rng = np.random.RandomState(0)
        from scipy import stats

        data = stats.weibull_min.rvs(5.0, scale=200.0, size=500, random_state=rng)
        vdict: dict[float, float] = {}
        for x in data:
            vdict[float(x)] = vdict.get(float(x), 0.0) + 1.0

        shape, scale = _fit(data)
        est = _factory(vdict, pseudo_count=1.0, emp_suff_stat=True, use_bstats=False)
        # suff_stat is the (mean, second-moment) of a Weibull(shape, scale) -- NOT of Weibull(1, 1),
        # which would give (1.0, 2.0) regardless of the data.
        mean, second = est.suff_stat
        self.assertGreater(mean, 50.0)  # Weibull(1,1) would give mean=1.0; the real fit's mean is ~180
        self.assertNotAlmostEqual(mean, 1.0, places=3)
        self.assertNotAlmostEqual(second, 2.0, places=3)

    def test_weibull_factory_threads_pseudo_count_through(self):
        from mixle.utils.automatic.detectors.weibull import _factory

        vdict = {1.0: 1.0, 2.0: 1.0, 3.0: 1.0}
        est_with_pc = _factory(vdict, pseudo_count=5.0, emp_suff_stat=True, use_bstats=False)
        est_without_pc = _factory(vdict, pseudo_count=None, emp_suff_stat=True, use_bstats=False)
        self.assertEqual(est_with_pc.pseudo_count, 5.0)
        self.assertIsNone(est_without_pc.pseudo_count)
        self.assertIsNotNone(est_with_pc.suff_stat)
        self.assertIsNone(est_without_pc.suff_stat)  # no pseudo_count -> estimator() builds no prior

    def test_negative_binomial_factory_seeds_from_the_actual_fit(self):
        from mixle.utils.automatic.detectors.negative_binomial import _factory, _params

        rng = np.random.RandomState(1)
        from scipy import stats

        data = stats.nbinom.rvs(8.0, 0.3, size=500, random_state=rng).astype(float)
        vdict: dict[float, float] = {}
        for x in data:
            vdict[float(x)] = vdict.get(float(x), 0.0) + 1.0

        r_fit, p_fit = _params(data)
        est = _factory(vdict, pseudo_count=None, emp_suff_stat=True, use_bstats=False)
        self.assertIsNone(est.suff_stat)  # no pseudo_count given -> no prior suff_stat
        # with a pseudo_count, NegativeBinomialDistribution(r, p).estimator() builds suff_stat=p and
        # holds r fixed at the CONSTRUCTED r -- confirm both are the actual fit, not the hardcoded
        # (1.0, 0.5) default (p=0.5, r=1.0 would fail these given p_fit/r_fit are far from that here).
        est_with_pc = _factory(vdict, pseudo_count=2.0, emp_suff_stat=True, use_bstats=False)
        self.assertAlmostEqual(est_with_pc.suff_stat, p_fit, places=6)
        self.assertAlmostEqual(est_with_pc.r, r_fit, places=6)


if __name__ == "__main__":
    unittest.main()
