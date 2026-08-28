"""T3-03: ProbabilityDistribution.to_json() must not silently produce a write-only artifact.

Before this fix, VonMisesDistribution.to_json() succeeded with no exception or warning, and the
resulting text then failed in from_json() -- on a different object, possibly in a different
process. mixle.stats.dump_models(verify=True) already refuses this case immediately by reading the
JSON back before returning it; to_json() had none of that protection despite being documented
(docs/api-overview.rst, docs/models.rst) as a safe artifact route.
"""

import unittest

import numpy as np

from mixle.utils.serialization import SerializationError


def _fit_von_mises():
    import mixle
    from mixle.stats import VonMisesEstimator

    angles = [float(a) for a in np.random.default_rng(3).vonmises(1.1, 4.0, 200) + np.pi]
    return mixle.Model(VonMisesEstimator()).fit(angles)


class DistributionToJsonWriteOnlyTest(unittest.TestCase):
    def test_to_json_refuses_a_fitted_von_mises_it_cannot_read_back(self):
        fitted = _fit_von_mises().fitted
        with self.assertRaises(SerializationError) as ctx:
            fitted.to_json()
        message = str(ctx.exception)
        self.assertIn("cannot read back", message)
        self.assertIn("to_dict", message)  # names a working escape hatch, not just "no"

    def test_gaussian_to_json_still_round_trips(self):
        # Ordinary, non-degenerate case exercising the same to_json() code path: the added
        # read-back check must not change behavior for a family that already round-trips cleanly.
        import mixle
        from mixle.stats import GaussianDistribution, GaussianEstimator

        rng = np.random.default_rng(4)
        fitted = mixle.Model(GaussianEstimator()).fit([float(v) for v in rng.normal(size=200)]).fitted
        text = fitted.to_json()
        loaded = GaussianDistribution.from_json(text)
        self.assertAlmostEqual(loaded.log_density(0.3), fitted.log_density(0.3), places=12)


if __name__ == "__main__":
    unittest.main()
