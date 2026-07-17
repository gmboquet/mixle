"""Reproducible model artifacts: dataset hashing, model headers/provenance, dataset checking, and
serialization of encoded data."""

import os
import tempfile
import unittest

import numpy as np

from mixle.data import check_dataset, dataset_hash, load_encoded, save_encoded
from mixle.inference.production import Header, fit_with_provenance
from mixle.stats import CategoricalDistribution, CompositeDistribution, GaussianDistribution


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
        back = Header.from_dict(header.to_dict())
        self.assertEqual(back.dataset_hash, header.dataset_hash)
        self.assertEqual(back.schema, header.schema)
        self.assertEqual(back.model_type, header.model_type)
        self.assertEqual(back.resources, header.resources)

    def test_resources_captured(self):
        data = np.random.RandomState(7).normal(0.0, 1.0, 2000).tolist()
        _, header = fit_with_provenance(data, GaussianDistribution(0.0, 1.0).estimator(), max_its=20, out=None)
        # resource module exists on this platform; if so, peak RSS and CPU time are recorded
        if header.resources:
            self.assertIn("peak_rss_mb", header.resources)
            self.assertIn("cpu_time_s", header.resources)
            self.assertGreaterEqual(header.resources["cpu_time_s"], 0.0)

    def test_provenance_from_datasource(self):
        import os
        import tempfile

        from mixle.data import Field, Real, Schema, open_source

        data = np.random.RandomState(8).normal(2.0, 1.0, 500).tolist()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.csv")
            with open(path, "w") as f:
                f.write("x\n" + "\n".join(map(str, data)))
            src = open_source("csv", path, columns=["x"], schema=Schema((Field("x", Real()),)))
            _, header = fit_with_provenance(src, GaussianDistribution(0.0, 1.0).estimator(), max_its=20, out=None)
        self.assertEqual(header.n_records, 500)  # DataSource length flows through
        self.assertIsNotNone(header.final_loglik)  # and it can still be scored for the header

    def test_convergence_trace_captured(self):
        data = np.random.RandomState(1).normal(0.0, 1.0, 300).tolist()
        _, header = fit_with_provenance(data, GaussianDistribution(5.0, 1.0).estimator(), max_its=30, delta=1e-7)
        conv = header.training["convergence"]
        self.assertGreater(len(conv), 0)
        self.assertEqual(header.training["iterations"], conv[-1]["iter"])
        self.assertTrue(header.training["converged"])
        lls = [r["loglik"] for r in conv]
        self.assertTrue(all(lls[i] <= lls[i + 1] + 1e-6 for i in range(len(lls) - 1)))  # EM monotone
        self.assertIsNone(conv[0]["delta"])  # first delta nulled (no prior) -> JSON-clean

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

    def test_header_is_json_not_pickle(self):
        # The header carrying the digest must be plain JSON: parsing it must never itself be able to
        # execute code, unlike the digest-verified pickle body that follows it.
        import json

        enc = GaussianDistribution(0, 1).dist_to_encoder().seq_encode([1.0, 2.0])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "enc.pspenc")
            save_encoded(enc, path, encoder=GaussianDistribution(0, 1).dist_to_encoder())
            with open(path, "rb") as f:
                magic = f.read(8)
                header_line = f.readline()
            self.assertEqual(magic, b"PSPENC1\n")
            meta = json.loads(header_line)  # raises if this were pickle bytes, not JSON
            self.assertEqual(len(meta["digest"]), 64)
            self.assertIn("Gaussian", meta["encoder"])

    def test_header_digest_mismatch_rejected(self):
        # A header whose digest does not match the body must be rejected, including when the body
        # itself is well-formed pickle -- proving the check gates on the digest, not on a parse error.
        enc = GaussianDistribution(0, 1).dist_to_encoder().seq_encode([1.0, 2.0])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "enc.pspenc")
            save_encoded(enc, path)
            with open(path, "rb") as f:
                raw = f.read()
            nl = raw.index(b"\n", 8)
            tampered = raw[: nl + 1].replace(b'"digest": "', b'"digest": "0000000000000000') + raw[nl + 1 :]
            with open(path, "wb") as f:
                f.write(tampered)
            with self.assertRaises(ValueError):
                load_encoded(path)


if __name__ == "__main__":
    unittest.main()
