"""Reproducible artifacts for the PPL surface: provenance headers for fitted RandomVariables."""

import unittest

import numpy as np

from mixle.data.hashing import dataset_hash
from mixle.inference.production.provenance import Header
from mixle.ppl import Normal, PPLFitResult, fit_with_provenance, free


class PPLProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.data = np.random.RandomState(0).normal(3.0, 2.0, 400).tolist()

    def test_em_path_full_header(self):
        fitted, header = fit_with_provenance(Normal(free, free), self.data, how="em", max_its=30)
        self.assertEqual(header.training["method"], "em")
        self.assertEqual(header.training["surface"], "ppl")
        self.assertEqual(header.schema, [("value", "Real")])  # built from the lowered distribution
        self.assertIsNotNone(header.final_loglik)
        self.assertIn("duration_s", header.timing)
        # the fitted RV recovers the truth, so the header reflects a real fit
        self.assertAlmostEqual(fitted.dist.mu, 3.0, delta=0.4)

    def test_map_path_full_header(self):
        _, header = fit_with_provenance(Normal(free, free), self.data, how="map", max_its=50)
        self.assertEqual(header.training["method"], "map")
        self.assertIsNotNone(header.final_loglik)
        self.assertEqual(len(header.dataset_hash), 64)

    def test_header_round_trips(self):
        _, header = fit_with_provenance(Normal(free, free), self.data, how="em", max_its=20)
        back = Header.from_dict(header.to_dict())
        self.assertEqual(back.dataset_hash, header.dataset_hash)
        self.assertEqual(back.training["surface"], "ppl")

    def test_seed_is_the_exact_rng_passed_to_fit(self):
        class Spy:
            def __init__(self):
                self.draw = None

            def fit(self, data, **kw):
                self.draw = int(kw["rng"].randint(0, 2**31))
                return self

        spy = Spy()
        result = fit_with_provenance(spy, [1.0, 2.0], seed=17)
        expected = int(np.random.RandomState(17).randint(0, 2**31))
        self.assertIsInstance(result, PPLFitResult)
        self.assertEqual(spy.draw, expected)
        self.assertEqual(result.header.training["seed"], 17)
        self.assertEqual(result.header.training["random_state"]["effective_seed"], 17)
        self.assertEqual(len(result.header.training["random_state"]["initial_state_sha256"]), 64)

    def test_one_shot_data_fit_and_hash_use_the_same_materialization(self):
        class Spy:
            def __init__(self):
                self.seen = None

            def fit(self, data, **_kw):
                self.seen = list(data)
                return self

        records = [1.0, 2.0, 3.0]
        spy = Spy()
        result = fit_with_provenance(spy, (value for value in records), seed=3)
        self.assertEqual(spy.seen, records)
        self.assertEqual(result.header.dataset_hash, dataset_hash(records))
        self.assertEqual(result.header.n_records, len(records))

    def test_result_does_not_depend_on_attaching_to_fitted_object(self):
        class Slotted:
            __slots__ = ("seen",)

            def __init__(self):
                self.seen = []

            def fit(self, data, **_kw):
                self.seen = list(data)
                return self

        fitted, header = fit_with_provenance(Slotted(), [1.0], seed=0)
        self.assertEqual(fitted.seen, [1.0])
        self.assertEqual(header.training["surface"], "ppl")
        self.assertFalse(hasattr(fitted, "header"))

    def test_rng_override_and_invalid_seeds_are_rejected(self):
        rv = Normal(free, free)
        with self.assertRaisesRegex(ValueError, "pass seed"):
            fit_with_provenance(rv, self.data, rng=np.random.RandomState(0))
        with self.assertRaises(TypeError):
            fit_with_provenance(rv, self.data, seed=True)
        with self.assertRaises(ValueError):
            fit_with_provenance(rv, self.data, seed=-1)


if __name__ == "__main__":
    unittest.main()
