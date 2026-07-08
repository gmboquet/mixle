"""Automatic-inference detectors for 7 previously-unreachable univariate families (mixle.utils.automatic):
half_normal, rayleigh, inverse_gamma (continuous); geometric, binomial, beta_binomial (discrete).
Each: BIC-selected as the winner on data drawn from it, without disturbing the Gaussian/Poisson defaults."""

import unittest

import numpy as np
from scipy import stats

import mixle.utils.automatic.profiling as P


def _winner(values):
    """The auto-inference family recommendation for a single column of ``values``."""
    node = P.DatumNode(data=[(v,) for v in values])
    child = node.children[0]
    if child.float_count > 0:
        arr = P._value_array_from_vdict(child.vdict)
        return P._numeric_model_recommendation(P._numeric_candidate_bics(arr, arr.size))
    name, _ = P._recommended_integer_model(child.vdict)
    return name


class ContinuousDetectorTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)

    def test_half_normal_is_detected(self):
        vals = [abs(float(x)) for x in self.rng.randn(3000) * 2.0]  # mode at 0
        self.assertEqual(_winner(vals), "half_normal")

    def test_rayleigh_is_detected(self):
        vals = list(stats.rayleigh.rvs(scale=1.5, size=3000, random_state=self.rng))
        self.assertEqual(_winner(vals), "rayleigh")

    def test_inverse_gamma_is_detected(self):
        vals = list(stats.invgamma.rvs(3.0, scale=2.0, size=3000, random_state=self.rng))
        self.assertEqual(_winner(vals), "inverse_gamma")


class DiscreteDetectorTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)

    def test_geometric_is_detected(self):
        vals = list(stats.geom.rvs(0.3, size=3000, random_state=self.rng))  # support {1,2,...}
        self.assertEqual(_winner(vals), "geometric")

    def test_binomial_is_detected_when_n_is_observable(self):
        # binomial is identifiable only when p is high enough that the sample max reaches n
        vals = list(stats.binom.rvs(10, 0.6, size=4000, random_state=self.rng))
        self.assertEqual(_winner(vals), "binomial")

    def test_beta_binomial_is_detected_on_overdispersed_bounded_counts(self):
        vals = list(stats.betabinom.rvs(20, 2.0, 5.0, size=4000, random_state=self.rng))
        self.assertEqual(_winner(vals), "beta_binomial")

    def test_plain_binomial_data_is_not_stolen_by_beta_binomial(self):
        # beta-binomial nests binomial as overdispersion -> 0; genuine binomial data must stay binomial,
        # not flip to the extra-parameter family on sampling noise.
        vals = list(stats.binom.rvs(8, 0.5, size=4000, random_state=self.rng))
        self.assertEqual(_winner(vals), "binomial")


class NoRegressionTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)

    def test_gaussian_still_wins_for_gaussian_data(self):
        self.assertEqual(_winner(list(self.rng.randn(3000) * 2.0 + 5.0)), "gaussian")

    def test_poisson_still_wins_for_poisson_data(self):
        vals = list(stats.poisson.rvs(3.0, size=3000, random_state=self.rng))
        self.assertEqual(_winner(vals), "poisson")

    def test_gamma_still_wins_for_gamma_data(self):
        vals = list(stats.gamma.rvs(2.0, scale=2.0, size=3000, random_state=self.rng))
        self.assertEqual(_winner(vals), "gamma")


class DetectorFactoryTest(unittest.TestCase):
    def test_each_new_detector_builds_a_working_estimator(self):
        # the winning family's factory must produce a fittable estimator that yields a finite log-density
        from mixle.inference import optimize
        from mixle.utils.automatic import get_estimator

        rng = np.random.RandomState(1)
        cases = {
            "half_normal": [(abs(float(x)),) for x in rng.randn(400) * 2.0],
            "rayleigh": [(float(v),) for v in stats.rayleigh.rvs(scale=1.5, size=400, random_state=rng)],
            "geometric": [(int(v),) for v in stats.geom.rvs(0.3, size=400, random_state=rng)],
        }
        for _name, data in cases.items():
            est = get_estimator(data)
            model = optimize(data, est, max_its=5, out=None)
            ld = model.seq_log_density(model.dist_to_encoder().seq_encode(data))
            self.assertTrue(np.all(np.isfinite(ld)))


if __name__ == "__main__":
    unittest.main()
