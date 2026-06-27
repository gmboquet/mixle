"""Reproducible model artifacts: dataset hashing, model headers/provenance, dataset checking, and
serialization of encoded data."""

import os
import tempfile
import unittest

import numpy as np

from pysp.data import check_dataset, dataset_hash, load_encoded, save_encoded
from pysp.inference import ModelHeader, fit_with_provenance
from pysp.stats import CategoricalDistribution, CompositeDistribution, GaussianDistribution


class DatasetHashTest(unittest.TestCase):
    def test_stable_and_order_sensitivity(self):
        d = [1.0, 2.0, 3.0]
        self.assertEqual(dataset_hash(d), dataset_hash([1.0, 2.0, 3.0]))
        self.assertNotEqual(dataset_hash(d), dataset_hash([3.0, 2.0, 1.0]))  # order-sensitive by default
        self.assertEqual(dataset_hash(d, sort=True), dataset_hash([3.0, 2.0, 1.0], sort=True))  # commutative

    def test_distinguishes_content_and_count(self):
        self.assertNotEqual(dataset_hash([1.0, 2.0]), dataset_hash([1.0, 2.0, 2.0]))
        self.assertNotEqual(dataset_hash([1.0, 2.0]), dataset_hash([1.0, 9.0]))

    def test_tuple_records(self):
        a = [(1.0, "x"), (2.0, "y")]
        self.assertEqual(dataset_hash(a), dataset_hash([(1.0, "x"), (2.0, "y")]))
        self.assertNotEqual(dataset_hash(a), dataset_hash([(1.0, "x"), (2.0, "z")]))


class ProvenanceHeaderTest(unittest.TestCase):
    def test_fit_with_provenance_populates_header(self):
        data = np.random.RandomState(0).normal(3.0, 2.0, 400).tolist()
        model, header = fit_with_provenance(
            data, GaussianDistribution(0.0, 1.0).estimator(), max_its=20, delta=1e-7, out=None
        )
        self.assertIs(model.header, header)
        self.assertEqual(header.model_type, "GaussianDistribution")
        self.assertEqual(header.n_records, 400)
        self.assertEqual(header.dataset_hash, dataset_hash(data))
        self.assertEqual(header.schema, [("value", "Real")])
        self.assertIsNotNone(header.final_loglik)
        self.assertEqual(header.training["method"], "em")
        self.assertIn("duration_s", header.timing)
        self.assertIsNotNone(header.environment["python"])

    def test_header_round_trips_through_dict(self):
        data = [1.0, 2.0, 3.0, 4.0]
        _, header = fit_with_provenance(data, GaussianDistribution(0.0, 1.0).estimator(), max_its=5, out=None)
        back = ModelHeader.from_dict(header.to_dict())
        self.assertEqual(back.dataset_hash, header.dataset_hash)
        self.assertEqual(back.schema, header.schema)
        self.assertEqual(back.model_type, header.model_type)

    def test_composite_schema(self):
        data = [(1.0, "x"), (2.0, "y"), (1.5, "x")]
        model = CompositeDistribution((GaussianDistribution(0, 1), CategoricalDistribution({"x": 0.5, "y": 0.5})))
        _, header = fit_with_provenance(data, model.estimator(), max_its=5, out=None)
        self.assertEqual(len(header.schema), 2)


class CheckDatasetTest(unittest.TestCase):
    def test_flags_nonconforming_and_out_of_support(self):
        rep = check_dataset(GaussianDistribution(0.0, 1.0), [1.0, 2.0, "oops", 4.0])
        self.assertFalse(rep.ok)
        self.assertTrue(any("conform" in i for i in rep.issues))

    def test_clean_passes(self):
        rep = check_dataset(GaussianDistribution(0.0, 1.0), [1.0, 2.0, 3.0])
        self.assertTrue(rep.ok)
        self.assertEqual(rep.n_checked, 3)

    def test_raise_on_error(self):
        with self.assertRaises(ValueError):
            check_dataset(GaussianDistribution(0.0, 1.0), ["nope"], raise_on_error=True)


class EncodedIoTest(unittest.TestCase):
    def test_round_trip_with_integrity(self):
        model = GaussianDistribution(0.0, 1.0)
        enc = model.dist_to_encoder().seq_encode([1.0, 2.0, 3.0])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "enc.pspenc")
            digest = save_encoded(enc, path, encoder=model.dist_to_encoder())
            loaded = load_encoded(path, encoder=model.dist_to_encoder())
            # the reloaded encoding scores identically
            np.testing.assert_allclose(model.seq_log_density(loaded), model.seq_log_density(enc))
            self.assertEqual(len(digest), 64)

    def test_corruption_detected(self):
        enc = GaussianDistribution(0, 1).dist_to_encoder().seq_encode([1.0, 2.0])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "enc.pspenc")
            save_encoded(enc, path)
            with open(path, "ab") as f:
                f.write(b"corrupt")
            with self.assertRaises(ValueError):
                load_encoded(path)


if __name__ == "__main__":
    unittest.main()
