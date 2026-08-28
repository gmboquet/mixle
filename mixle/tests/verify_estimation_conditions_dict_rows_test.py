"""Found while fixing campaign-six T3-01: verify_estimation_conditions (mixle/inference/planning.py)
handled a RecordDistribution (dict-keyed fields, e.g. RecordEstimator's fitted output) through the
same branch as CompositeDistribution -- both carry `.dists` -- but indexed each row with a bare
integer `row[index]`. Against a dict row this raises KeyError, which the branch's own
`except (IndexError, TypeError)` does not catch, so certify() on ANY dict-shaped fit raised
unhandled -- caught only incidentally by Model.fit()'s own broad try/except, which silently records
evidence["certificate"] = {"status": "failed", "error_type": "KeyError", ...} for every dict-row
propose(fit=True) call regardless of whether anything was actually wrong with the fit.
"""

from __future__ import annotations

import unittest

import numpy as np

from mixle.inference import Guarantee, certify, optimize
from mixle.stats.combinator.record import RecordEstimator
from mixle.stats.univariate.continuous.gaussian import GaussianEstimator
from mixle.stats.univariate.discrete.poisson import PoissonEstimator


class VerifyEstimationConditionsDictRowsTest(unittest.TestCase):
    def test_certify_succeeds_on_dict_shaped_rows(self):
        rows = [
            {"x": float(np.random.RandomState(i).randn()), "y": int(np.random.RandomState(i).poisson(3))}
            for i in range(300)
        ]
        model = optimize(rows, RecordEstimator({"x": GaussianEstimator(), "y": PoissonEstimator()}), out=None)
        cert = certify(model, data=rows)
        self.assertEqual(cert.guarantee, Guarantee.GLOBAL_UNIQUE)
        self.assertEqual(len(cert.blocks), 2)

    def test_certify_still_succeeds_on_the_equivalent_tuple_shaped_rows(self):
        # Regression guard: the fix must not change the pre-existing, already-working
        # CompositeDistribution (positional/tuple-row) path.
        rows = [(float(np.random.RandomState(i).randn()), int(np.random.RandomState(i).poisson(3))) for i in range(300)]
        import mixle.stats as st

        model = optimize(rows, st.CompositeEstimator((GaussianEstimator(), PoissonEstimator())), out=None)
        cert = certify(model, data=rows)
        self.assertEqual(cert.guarantee, Guarantee.GLOBAL_UNIQUE)
        self.assertEqual(len(cert.blocks), 2)

    def test_a_dict_row_missing_a_field_is_skipped_not_raised(self):
        # A malformed/partial dict row must fall through the same KeyError-tolerant path a
        # tuple row's IndexError already did, not crash certify() outright.
        rows = [
            {"x": float(np.random.RandomState(i).randn()), "y": int(np.random.RandomState(i).poisson(3))}
            for i in range(50)
        ]
        model = optimize(rows, RecordEstimator({"x": GaussianEstimator(), "y": PoissonEstimator()}), out=None)
        malformed = [dict(r) for r in rows]
        del malformed[0]["y"]
        cert = certify(model, data=malformed)  # must not raise
        self.assertIsNotNone(cert)


if __name__ == "__main__":
    unittest.main()
