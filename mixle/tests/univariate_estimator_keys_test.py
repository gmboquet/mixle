"""Regression test: 9 univariate families' estimator()/estimate() round-trip previously dropped the
keys= constructor parameter entirely -- their own __init__ methods didn't even accept keys=, unlike
28 sibling families (e.g. LaplaceDistribution) that already thread it through construction, .estimator(),
and .estimate() consistently. keys= ties sufficient statistics together across a shared merge key (e.g.
tying two mixture components' parameters); silently losing it on a fit round-trip breaks that tying with
no error, only a model that quietly stops sharing statistics the caller asked it to share.
"""

import unittest

from mixle.stats.univariate.continuous.exponential import ExponentialDistribution
from mixle.stats.univariate.continuous.gamma import GammaDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.continuous.log_gaussian import LogGaussianDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution
from mixle.stats.univariate.discrete.geometric import GeometricDistribution
from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalDistribution
from mixle.stats.univariate.discrete.integer_uniform_spike import IntegerUniformSpikeDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution

KEY = "shared-key"


def _fit_round_trip(dist, data):
    """estimator() -> accumulate data -> estimate(): the path the audit found dropping keys=."""
    est = dist.estimator()
    acc = est.accumulator_factory().make()
    for x in data:
        acc.update(x, 1.0, None)
    return est, est.estimate(len(data), acc.value())


class UnivariateEstimatorKeysTestCase(unittest.TestCase):
    def test_gaussian(self):
        d = GaussianDistribution(0.0, 1.0, keys=KEY)
        est, fit = _fit_round_trip(d, [0.1, -0.2, 0.3, 0.05])
        self.assertEqual(est.keys, KEY)
        self.assertEqual(fit.keys, KEY)

    def test_log_gaussian(self):
        d = LogGaussianDistribution(0.0, 1.0, keys=KEY)
        est, fit = _fit_round_trip(d, [1.1, 0.9, 1.3, 0.8])
        self.assertEqual(est.keys, KEY)
        self.assertEqual(fit.keys, KEY)

    def test_exponential(self):
        d = ExponentialDistribution(2.0, keys=KEY)
        est, fit = _fit_round_trip(d, [1.0, 2.0, 0.5, 3.0])
        self.assertEqual(est.keys, KEY)
        self.assertEqual(fit.keys, KEY)

    def test_gamma(self):
        d = GammaDistribution(2.0, 1.5, keys=KEY)
        est, fit = _fit_round_trip(d, [1.0, 2.0, 0.5, 3.0])
        self.assertEqual(est.keys, KEY)
        self.assertEqual(fit.keys, KEY)

    def test_geometric(self):
        d = GeometricDistribution(0.4, keys=KEY)
        est, fit = _fit_round_trip(d, [1, 2, 1, 3])
        self.assertEqual(est.keys, KEY)
        self.assertEqual(fit.keys, KEY)

    def test_poisson(self):
        d = PoissonDistribution(3.0, keys=KEY)
        est, fit = _fit_round_trip(d, [2, 3, 4, 1])
        self.assertEqual(est.keys, KEY)
        self.assertEqual(fit.keys, KEY)

    def test_categorical(self):
        d = CategoricalDistribution(pmap={"a": 0.5, "b": 0.5}, keys=KEY)
        est, fit = _fit_round_trip(d, ["a", "b", "a", "a"])
        self.assertEqual(est.keys, KEY)
        self.assertEqual(fit.keys, KEY)

    def test_integer_categorical(self):
        d = IntegerCategoricalDistribution(0, [0.3, 0.3, 0.4], keys=KEY)
        est, fit = _fit_round_trip(d, [0, 1, 2, 0])
        self.assertEqual(est.keys, KEY)
        self.assertEqual(fit.keys, KEY)

    def test_integer_uniform_spike(self):
        d = IntegerUniformSpikeDistribution(k=2, num_vals=5, p=0.6, keys=KEY)
        est, fit = _fit_round_trip(d, [2, 2, 0, 3])
        self.assertEqual(est.keys, KEY)
        self.assertEqual(fit.keys, KEY)

    def test_estimator_preserves_keys_without_pseudo_count_too(self):
        # estimator()'s two branches (with/without pseudo_count) both dropped keys -- exercise the
        # pseudo_count=None path (the default, most common call) explicitly for a couple of families.
        self.assertEqual(GaussianDistribution(0.0, 1.0, keys=KEY).estimator().keys, KEY)
        self.assertEqual(GammaDistribution(2.0, 1.5, keys=KEY).estimator().keys, KEY)
        self.assertEqual(PoissonDistribution(3.0, keys=KEY).estimator().keys, KEY)

    def test_estimator_preserves_keys_with_pseudo_count(self):
        # and the pseudo_count-is-set branch, which builds a different Estimator call entirely.
        self.assertEqual(GaussianDistribution(0.0, 1.0, keys=KEY).estimator(pseudo_count=1.0).keys, KEY)
        self.assertEqual(GammaDistribution(2.0, 1.5, keys=KEY).estimator(pseudo_count=1.0).keys, KEY)
        self.assertEqual(PoissonDistribution(3.0, keys=KEY).estimator(pseudo_count=1.0).keys, KEY)

    def test_str_reprs_include_keys(self):
        # __str__ was updated alongside the constructor for every family -- spot-check a few.
        self.assertIn("keys='%s'" % KEY, str(GaussianDistribution(0.0, 1.0, keys=KEY)))
        self.assertIn("keys='%s'" % KEY, str(PoissonDistribution(3.0, keys=KEY)))
        self.assertIn("keys='%s'" % KEY, str(IntegerUniformSpikeDistribution(k=0, num_vals=3, p=0.5, keys=KEY)))

    def test_default_keys_is_none_for_every_family(self):
        # backward compatibility: omitting keys= must behave exactly as before (None).
        self.assertIsNone(GaussianDistribution(0.0, 1.0).keys)
        self.assertIsNone(LogGaussianDistribution(0.0, 1.0).keys)
        self.assertIsNone(ExponentialDistribution(2.0).keys)
        self.assertIsNone(GammaDistribution(2.0, 1.5).keys)
        self.assertIsNone(GeometricDistribution(0.4).keys)
        self.assertIsNone(PoissonDistribution(3.0).keys)
        self.assertIsNone(CategoricalDistribution(pmap={"a": 1.0}).keys)
        self.assertIsNone(IntegerCategoricalDistribution(0, [1.0]).keys)
        self.assertIsNone(IntegerUniformSpikeDistribution(k=0, num_vals=3, p=0.5).keys)


if __name__ == "__main__":
    unittest.main()
