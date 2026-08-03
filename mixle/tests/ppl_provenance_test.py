"""Reproducible artifacts for the PPL surface: provenance headers for fitted RandomVariables."""

import unittest
from collections.abc import Iterator

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

    def test_one_shot_iterable_that_is_not_an_iterator_is_snapshotted_before_the_fit(self):
        """MXR-080-1897: only ``Iterator`` was snapshotted, so a one-shot *iterable* was hashed empty."""

        class OneShotIterable:
            """Iterable, but not an ``Iterator``: re-iterating a drained source yields nothing."""

            def __init__(self, records):
                self._records = list(records)
                self._spent = False

            def __iter__(self):
                if self._spent:
                    return iter(())
                self._spent = True
                return iter(self._records)

        class Spy:
            def fit(self, data, **_kw):
                self.seen = list(data)
                return self

        records = [1.0, 2.0, 3.0, 4.0]
        source = OneShotIterable(records)
        self.assertNotIsInstance(source, Iterator)  # why the old rule let it through

        fitted, header = fit_with_provenance(Spy(), source, seed=0)
        self.assertEqual(fitted.seen, records)
        # used to be dataset_hash([]) with n_records=None -- a header claiming an empty dataset
        self.assertNotEqual(header.dataset_hash, dataset_hash([]))
        self.assertEqual(header.dataset_hash, dataset_hash(records))
        self.assertEqual(header.n_records, len(records))
        self.assertTrue(header.training["data_materialized"])

    def test_data_the_fit_mutates_is_refused_rather_than_hashed_after_the_fact(self):
        """MXR-080-1897: the hash was taken after the fit, so in-place edits rewrote what it described."""

        class MutatingSpy:
            def fit(self, data, **_kw):
                self.seen = list(data)
                for index in range(len(data)):
                    data[index] = 0.0
                return self

        payload = [1.0, 2.0, 3.0]
        with self.assertRaisesRegex(RuntimeError, "changed while rv.fit was running"):
            fit_with_provenance(MutatingSpy(), payload, seed=0)
        # the caller's own list is a snapshot source, not the fit's scratch space
        self.assertEqual(payload, [1.0, 2.0, 3.0])

    def test_arrays_stay_arrays_and_are_detached_from_the_caller(self):
        """The snapshot must not silently retype an ndarray fit input (MXR-080-1897)."""

        class Spy:
            def fit(self, data, **_kw):
                self.seen = data
                return self

        payload = np.array([1.0, 2.0, 3.0])
        fitted, header = fit_with_provenance(Spy(), payload, seed=0)
        self.assertIsInstance(fitted.seen, np.ndarray)
        self.assertIsNot(fitted.seen, payload)
        self.assertEqual(header.dataset_hash, dataset_hash(payload))
        self.assertEqual(header.n_records, 3)

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
