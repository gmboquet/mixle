"""JSON codec round trips for the campaign-c9eb9d2e families (finding T3-10).

Three families silently deployed as pickle because their safe-JSON codecs were broken:
BernoulliSetDistribution encoded a ``mappingproxy`` (no JSON form), ThurstoneDistribution shipped
its derived, unregistered approximation-diagnostics record, and SpearmanRankingDistribution stored
its location under ``_sigma`` while the constructor takes ``sigma`` -- so it encoded cleanly and
refused every read back. Each family now serializes constructor-owned state through a
``__pysp_getstate__``/``__pysp_setstate__`` pair and its fit-diagnostics value class is registered.
These tests pin the full contract: dump -> load -> bit-identical scores on the same encoded data,
provenance preserved, and the lifecycle deploy path writing ``format: json`` with no pickle
fallback.
"""

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

import mixle.stats as stats
from mixle.lifecycle import Model
from mixle.stats.rankings.spearman_rho import SpearmanRankingDistribution, SpearmanRankingEstimator
from mixle.stats.rankings.thurstone import ThurstoneEstimator
from mixle.stats.sets.bernoulli_set import BernoulliSetEstimator
from mixle.stats.univariate.continuous.beta import BetaDistribution

BERNOULLI_SET_DATA = [["a", "b"], ["a"], ["b", "c"], ["a", "c"], ["a"]]
THURSTONE_DATA = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [0, 1, 2], [2, 0, 1]]
SPEARMAN_DATA = [[0, 1, 2, 3], [0, 2, 1, 3], [1, 0, 2, 3], [0, 1, 3, 2], [0, 1, 2, 3]]


def _fit(estimator, data):
    accumulator = estimator.accumulator_factory().make()
    for row in data:
        accumulator.update(row, 1.0, None)
    return estimator.estimate(None, accumulator.value())


class CampaignCodecRoundTripTestCase(unittest.TestCase):
    def assert_scores_bit_identical(self, dist, data):
        """dump -> load must reproduce every score to the exact bit, not merely to a tolerance."""
        encoded = dist.dist_to_encoder().seq_encode(data)
        loaded = stats.load_models(stats.dump_models([dist]))[0]
        self.assertIsInstance(loaded, type(dist))
        original = np.asarray(dist.seq_log_density(encoded))
        restored = np.asarray(loaded.seq_log_density(encoded))
        self.assertEqual(original.tobytes(), restored.tobytes())
        return loaded

    def test_bernoulli_set_round_trip_scores_bit_identical(self):
        dist = _fit(BernoulliSetEstimator(), BERNOULLI_SET_DATA)
        loaded = self.assert_scores_bit_identical(dist, BERNOULLI_SET_DATA)
        self.assertEqual(dict(loaded.pmap), dict(dist.pmap))
        self.assertEqual(loaded.min_prob, dist.min_prob)

    def test_bernoulli_set_conjugate_round_trip_preserves_prior_and_posteriors(self):
        dist = _fit(BernoulliSetEstimator(prior=BetaDistribution(2.0, 3.0)), BERNOULLI_SET_DATA)
        loaded = self.assert_scores_bit_identical(dist, BERNOULLI_SET_DATA)
        self.assertEqual(loaded.get_posteriors(), dist.get_posteriors())
        self.assertEqual(loaded.get_prior().get_parameters(), dist.get_prior().get_parameters())
        for row in BERNOULLI_SET_DATA:
            self.assertEqual(loaded.expected_log_density(row), dist.expected_log_density(row))

    def test_thurstone_round_trip_scores_bit_identical(self):
        dist = _fit(ThurstoneEstimator(dim=3, n_mc=200), THURSTONE_DATA)
        loaded = self.assert_scores_bit_identical(dist, THURSTONE_DATA)
        # Constructor-carried estimation provenance survives; the approximation record is NOT
        # shipped -- it is re-derived in __init__ and must land on the identical value.
        self.assertEqual(loaded.fit_diagnostics, dist.fit_diagnostics)
        self.assertEqual(loaded.approximation_diagnostics, dist.approximation_diagnostics)
        self.assertEqual(loaded.mu.tobytes(), dist.mu.tobytes())

    def test_spearman_round_trip_scores_bit_identical(self):
        dist = _fit(SpearmanRankingEstimator(dim=4), SPEARMAN_DATA)
        loaded = self.assert_scores_bit_identical(dist, SPEARMAN_DATA)
        self.assertEqual(loaded.fit_diagnostics, dist.fit_diagnostics)
        self.assertEqual(loaded.sigma.tobytes(), dist.sigma.tobytes())
        self.assertEqual(loaded.rho, dist.rho)

    def test_spearman_direct_construction_round_trip(self):
        # The pre-fix defect was specific to read back (state named `_sigma`, constructor `sigma`),
        # so also pin the un-fitted shape that used to encode into a write-only artifact.
        dist = SpearmanRankingDistribution([0.25, 1.5, 2.0, 2.25], rho=0.5, name="direct")
        self.assert_scores_bit_identical(dist, SPEARMAN_DATA)

    def test_lifecycle_deploy_writes_json_not_pickle(self):
        cases = [
            ("bernoulli_set", BernoulliSetEstimator(), BERNOULLI_SET_DATA),
            ("thurstone", ThurstoneEstimator(dim=3, n_mc=200), THURSTONE_DATA),
            ("spearman", SpearmanRankingEstimator(dim=4), SPEARMAN_DATA),
        ]
        for label, estimator, data in cases:
            with self.subTest(family=label):
                model = Model(estimator).fit(data)
                with tempfile.TemporaryDirectory() as tmp:
                    out = Path(tmp) / label
                    with warnings.catch_warnings():
                        # A pickle fallback is disclosed with a warning; make it a failure here.
                        warnings.simplefilter("error")
                        artifact = model.deploy(str(out))
                    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
                    self.assertEqual(artifact.format, "json")
                    self.assertEqual(manifest["format"], "json")
                    self.assertIsNone(manifest["format_fallback"])
                    self.assertEqual(manifest["model_file"], "model.json")
                    loaded = Model.load(str(out))
                    encoded = model.fitted.dist_to_encoder().seq_encode(data)
                    self.assertEqual(
                        np.asarray(model.fitted.seq_log_density(encoded)).tobytes(),
                        np.asarray(loaded.fitted.seq_log_density(encoded)).tobytes(),
                    )


if __name__ == "__main__":
    unittest.main()
