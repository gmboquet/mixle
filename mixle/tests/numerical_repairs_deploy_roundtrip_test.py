"""numerical_repairs() must survive a Model.deploy()/Model.load() round trip.

Pins the defect where a fitted distribution's numerical_repairs() silently returned () after
deploy()/load(), even though the repaired VALUE (a clamped p, a floored variance) came back
correctly and fit_provenance().repairs still carried the same names. serialization.py gave
_fit_provenance an explicit envelope slot on encode and a restore step on decode, but
_numerical_repairs -- also listed in _NON_STATE_ATTRIBUTES -- had neither, so the documented
disclosure surface (CHANGELOG's headline claim for 0.8.0) went quiet on exactly the artifacts it
exists to describe.
"""

import tempfile
import unittest

import mixle
from mixle.stats.univariate.continuous.gaussian import GaussianEstimator
from mixle.stats.univariate.discrete.bernoulli import BernoulliEstimator


class NumericalRepairsSurviveDeployTest(unittest.TestCase):
    def test_bernoulli_boundary_clamp_disclosure_survives_round_trip(self):
        m = mixle.Model(BernoulliEstimator())
        m.fit([True] * 50)
        before = m.fitted.numerical_repairs()
        self.assertEqual(before, ("bernoulli-p-clamped(1 -> 1 - 1e-12)",))

        with tempfile.TemporaryDirectory() as d:
            path = m.deploy(d + "/bernoulli")
            back = mixle.Model.load(path)

        self.assertEqual(back.fitted.p, m.fitted.p)
        self.assertEqual(back.fitted.numerical_repairs(), before)

    def test_gaussian_variance_floor_disclosure_survives_round_trip(self):
        m = mixle.Model(GaussianEstimator())
        m.fit([5.0] * 30)
        before = m.fitted.numerical_repairs()
        self.assertNotEqual(before, ())

        with tempfile.TemporaryDirectory() as d:
            path = m.deploy(d + "/gaussian")
            back = mixle.Model.load(path)

        self.assertEqual(back.fitted.sigma2, m.fitted.sigma2)
        self.assertEqual(back.fitted.numerical_repairs(), before)

    def test_ordinary_fit_with_no_repair_stays_empty_after_round_trip(self):
        # Guards against a fix that widens the envelope treatment into fabricating a repair, or
        # that breaks the ordinary no-repair path while chasing the degenerate one.
        m = mixle.Model(GaussianEstimator())
        m.fit([1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 1.5, 4.5, 3.0] * 5)
        self.assertEqual(m.fitted.numerical_repairs(), ())

        with tempfile.TemporaryDirectory() as d:
            path = m.deploy(d + "/gaussian_ok")
            back = mixle.Model.load(path)

        self.assertEqual(back.fitted.numerical_repairs(), ())
        self.assertEqual(back.fitted.mu, m.fitted.mu)
        self.assertEqual(back.fitted.sigma2, m.fitted.sigma2)

    def test_to_dict_from_dict_round_trip_also_carries_the_disclosure(self):
        m = mixle.Model(BernoulliEstimator())
        m.fit([True] * 50)
        payload = m.fitted.to_dict()
        self.assertIn("numerical_repairs", payload)

        rebuilt = type(m.fitted).from_dict(payload)
        self.assertEqual(rebuilt.numerical_repairs(), m.fitted.numerical_repairs())

    def test_legacy_artifact_without_the_new_envelope_field_still_decodes(self):
        m = mixle.Model(BernoulliEstimator())
        m.fit([True] * 50)
        payload = m.fitted.to_dict()
        del payload["numerical_repairs"]  # simulate an artifact written before this fix

        rebuilt = type(m.fitted).from_dict(payload)
        self.assertEqual(rebuilt.numerical_repairs(), ())


if __name__ == "__main__":
    unittest.main()
