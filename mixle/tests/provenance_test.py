"""Reproducible model artifacts: dataset hashing, model headers/provenance, dataset checking, and
serialization of encoded data."""

import dataclasses
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mixle.data import check_dataset, dataset_hash, load_encoded, save_encoded
from mixle.data.encoded_io import _envelope_digest
from mixle.data.hashing import _canonical
from mixle.inference.production import Header, fit_with_provenance
from mixle.stats import CategoricalDistribution, CompositeDistribution, GaussianDistribution
from mixle.stats.multivariate.diagonal_gaussian import DiagonalGaussianDistribution


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

    def test_distinct_records_do_not_collide_across_separator_bytes(self):
        # Audit finding: _canonical used to join list/dict elements with a bare "," / ":" and encode
        # strings/bytes as a tag plus raw, un-length-prefixed content. A string, key, or record
        # containing those separator bytes could then make one structure's join land on exactly the
        # same bytes as a different structure's join -- so genuinely distinct data (different values
        # and different shapes, not just different metadata) could get identical provenance hashes.
        # Three independent shapes of the same bug, all now must produce distinct hashes:
        self.assertNotEqual(
            dataset_hash(["x", "|sy"]),  # two records: "x" and "|sy"
            dataset_hash(["x|s", "y"]),  # two records: "x|s" and "y" -- used to both hash to 1d0c57ad...
        )
        self.assertNotEqual(
            dataset_hash([["X", "Y"]]),  # one record: a 2-element list
            dataset_hash([["X,sY"]]),  # one record: a 1-element list -- used to both hash to 1be84588...
        )
        self.assertNotEqual(
            dataset_hash([{"a": 1, "b": 2}]),  # one record: two named columns
            dataset_hash([{"a:i1,sb": 2}]),  # one record: one named column -- used to both hash to 497ea36c...
        )

    def test_identical_datasets_still_hash_identically(self):
        # The fix for the collisions above must not turn genuinely equal data into a false mismatch.
        self.assertEqual(dataset_hash(["x", "|sy"]), dataset_hash(["x", "|sy"]))
        self.assertEqual(dataset_hash([["X", "Y"]]), dataset_hash([["X", "Y"]]))
        self.assertEqual(dataset_hash([{"a": 1, "b": 2}]), dataset_hash([{"b": 2, "a": 1}]))  # key order-free
        self.assertEqual(dataset_hash([np.array([1.0, 2.0])]), dataset_hash([np.array([1.0, 2.0])]))

    def test_arrays_are_distinguished_by_value_not_just_shape_or_dtype(self):
        self.assertNotEqual(dataset_hash([np.array([1.0, 2.0])]), dataset_hash([np.array([1.0, 3.0])]))
        self.assertNotEqual(
            dataset_hash([np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)]),
            dataset_hash([np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1)]),
        )

    def test_nan_normalizes_but_is_never_confused_with_a_real_float(self):
        self.assertEqual(dataset_hash([float("nan")]), dataset_hash([float("nan")]))  # normalizes consistently
        self.assertNotEqual(dataset_hash([float("nan")]), dataset_hash([0.0]))

    def test_object_dtype_array_hashes_identically_across_processes(self):
        # Reviewer finding: dtype=object arrays store PyObject* pointers, not the elements' own bytes.
        # A naive arr.tobytes() bakes those process-specific addresses into the hash, so the identical
        # logical array hashes differently from one process to the next. A single-process comparison
        # can't catch this (nothing forces an object's address to move within one run), so reproduce it
        # for real across two separate fresh interpreters.
        script = (
            "import numpy as np\n"
            "from mixle.data import dataset_hash\n"
            "print(dataset_hash([np.array(['a', 'b', 'c'], dtype=object)]))\n"
        )
        env = dict(os.environ)
        hashes = {
            subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env
            ).stdout.strip()
            for _ in range(2)
        }
        self.assertEqual(len(hashes), 1, f"same logical object array hashed differently across processes: {hashes}")

    def test_object_dtype_array_content_equality_is_independent_of_object_identity(self):
        # Same check as above without spawning a subprocess: build the "same" logical array through two
        # different code paths so the underlying PyObject identities (and thus, pre-fix, the hashed
        # pointer bytes) genuinely differ within a single process, and confirm content -- not identity --
        # drives the hash.
        a = np.array(["alpha", "beta", "gamma"], dtype=object)
        b = np.array(["".join(["al", "pha"]), "".join(["be", "ta"]), "".join(["gam", "ma"])], dtype=object)
        for x, y in zip(a, b):
            self.assertIsNot(x, y)  # sanity: genuinely distinct objects, not accidentally interned
        self.assertEqual(dataset_hash([a]), dataset_hash([b]))
        c = np.array(["alpha", "beta", "different"], dtype=object)
        self.assertNotEqual(dataset_hash([a]), dataset_hash([c]))

    def test_object_dtype_array_shape_and_nested_content_still_distinguished(self):
        # The dtype=object fix must not regress what already worked: shape still matters, and elements
        # that are themselves containers still canonicalize (recursively) by value.
        flat = np.array(["a", "b", "c", "d", "e", "f"], dtype=object).reshape(2, 3)
        other_shape = np.array(["a", "b", "c", "d", "e", "f"], dtype=object).reshape(3, 2)
        self.assertNotEqual(dataset_hash([flat]), dataset_hash([other_shape]))

        nested_a = np.array([[1, 2], {"x": 1}, None], dtype=object)
        nested_b = np.array([[1, 2], {"x": 1}, None], dtype=object)
        nested_c = np.array([[1, 2], {"x": 2}, None], dtype=object)
        self.assertEqual(dataset_hash([nested_a]), dataset_hash([nested_b]))
        self.assertNotEqual(dataset_hash([nested_a]), dataset_hash([nested_c]))

    def test_truncated_prefix_never_collides_with_a_genuinely_complete_dataset(self):
        # Reviewer finding: max_records only mixed in *how many* records were hashed, not *whether*
        # hashing stopped early -- so a hash truncated to N records was byte-identical to the hash of a
        # genuinely complete N-record dataset, contradicting the documented contract ("a truncated hash
        # never collides with a full one").
        self.assertNotEqual(dataset_hash([1, 2, 3], max_records=2), dataset_hash([1, 2]))
        self.assertNotEqual(dataset_hash([1, 2, 3], max_records=2, sort=True), dataset_hash([1, 2], sort=True))

    def test_max_records_past_the_end_is_not_truncation(self):
        # max_records at or beyond the actual record count never cuts anything off, so it must still
        # match the plain untruncated hash -- pins that the fix above doesn't over-mark truncation.
        self.assertEqual(dataset_hash([1, 2], max_records=5), dataset_hash([1, 2]))
        self.assertEqual(dataset_hash([1, 2], max_records=2), dataset_hash([1, 2]))
        self.assertNotEqual(dataset_hash([1, 2, 3], max_records=0), dataset_hash([]))  # truncated-to-nothing


class ClosedCanonicalSchemaTest(unittest.TestCase):
    """MXR-080-1601: the canonical encoder claimed structurally different inputs cannot collide and
    that its digests are stable, while three cases broke both halves of that claim."""

    def test_lists_and_tuples_no_longer_share_one_tag(self):
        # audit repro: both were tagged b"t", so these hashed identically
        self.assertNotEqual(dataset_hash([[1, 2]]), dataset_hash([(1, 2)]))
        self.assertNotEqual(_canonical([1, 2]), _canonical((1, 2)))
        # nested, and as dict values, too
        self.assertNotEqual(dataset_hash([{"a": [1, 2]}]), dataset_hash([{"a": (1, 2)}]))
        self.assertNotEqual(dataset_hash([[[1], [2]]]), dataset_hash([[(1,), (2,)]]))
        # each kind still hashes equal to itself
        self.assertEqual(dataset_hash([[1, 2]]), dataset_hash([[1, 2]]))
        self.assertEqual(dataset_hash([(1, 2)]), dataset_hash([(1, 2)]))

    def test_sets_hash_by_content_not_by_iteration_order(self):
        self.assertEqual(dataset_hash([{1, 2, 3}]), dataset_hash([{3, 1, 2}]))
        self.assertEqual(dataset_hash([{"a", "b"}]), dataset_hash([frozenset({"b", "a"})]))
        self.assertNotEqual(dataset_hash([{1, 2, 3}]), dataset_hash([{1, 2, 4}]))
        # a set is not the list or tuple of the same elements
        self.assertNotEqual(dataset_hash([{1, 2}]), dataset_hash([[1, 2]]))
        self.assertNotEqual(dataset_hash([{1, 2}]), dataset_hash([(1, 2)]))

    def test_set_hash_is_independent_of_the_process_hash_seed(self):
        # The old repr() fallback ordered a set by the per-process string hash seed, so the SAME set
        # record produced different provenance digests run to run. Only a real cross-process check can
        # catch this: within one interpreter the seed never changes.
        script = "from mixle.data import dataset_hash\nprint(dataset_hash([{'a', 'b', 'c', 'd', 'e', 'f'}]))\n"
        hashes = set()
        for seed in ("1", "2", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            hashes.add(
                subprocess.run(
                    [sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env
                ).stdout.strip()
            )
        self.assertEqual(len(hashes), 1, f"set record hashed differently across hash seeds: {hashes}")

    def test_dataclass_records_hash_by_field_value_not_by_instance_address(self):
        @dataclasses.dataclass
        class Row:
            label: str
            weight: float

        # two distinct instances holding equal values: repr() embedded the address, so these differed
        self.assertEqual(dataset_hash([Row("a", 1.0)]), dataset_hash([Row("a", 1.0)]))
        self.assertNotEqual(dataset_hash([Row("a", 1.0)]), dataset_hash([Row("a", 2.0)]))
        self.assertNotEqual(dataset_hash([Row("a", 1.0)]), dataset_hash([Row("b", 1.0)]))

        @dataclasses.dataclass
        class OtherRow:
            label: str
            weight: float

        # same field names and values, different declared type -- must not collide
        self.assertNotEqual(dataset_hash([Row("a", 1.0)]), dataset_hash([OtherRow("a", 1.0)]))

    def test_unsupported_objects_are_rejected_rather_than_hashed_by_repr(self):
        class Plain:
            def __init__(self, value):
                self.value = value

        with self.assertRaises(TypeError):
            dataset_hash([Plain(1)])
        with self.assertRaises(TypeError):
            _canonical(Plain(1))
        with self.assertRaises(TypeError):
            dataset_hash([{"nested": Plain(1)}])

    def test_supported_types_negative_control(self):
        """Closing the schema must not start rejecting the ordinary record shapes it exists to hash."""
        record = {
            "id": 7,
            "name": "row",
            "flag": True,
            "missing": None,
            "weights": np.array([1.0, 2.0]),
            "tags": {"a", "b"},
            "pair": (1, "x"),
            "items": [1, 2, 3],
            "blob": b"\x00\x01",
        }
        digest = dataset_hash([record])
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, dataset_hash([dict(record)]))


class CanonicalEncodingTest(unittest.TestCase):
    """Unit-level pin on ``_canonical`` itself (imported directly by ``mixle.analysis.emissions`` too),
    not just observed indirectly through ``dataset_hash``."""

    def test_list_join_no_longer_ambiguous(self):
        self.assertNotEqual(_canonical(["X", "Y"]), _canonical(["X,sY"]))
        self.assertNotEqual(_canonical(["x", "|sy"]), _canonical(["x|s", "y"]))

    def test_dict_join_no_longer_ambiguous(self):
        self.assertNotEqual(_canonical({"a": 1, "b": 2}), _canonical({"a:i1,sb": 2}))

    def test_string_encoding_carries_an_explicit_length_prefix(self):
        # The fix: a tag byte, then an 8-byte big-endian length, then exactly that many content bytes --
        # the length is stated, not inferred by scanning content for an unescaped separator.
        encoded = _canonical("ab")
        self.assertEqual(encoded[:1], b"s")
        self.assertEqual(int.from_bytes(encoded[1:9], "big"), 2)
        self.assertEqual(encoded[9:], b"ab")

    def test_object_dtype_array_payload_is_recursively_canonicalized_elements(self):
        # Reviewer finding: object arrays store PyObject* pointers, so the array-encoding's payload must
        # be built from each element's own _canonical(...) bytes (self-delimiting, so this composes with
        # the same framing as every other container), not from a raw, address-dependent arr.tobytes().
        encoded = _canonical(np.array(["ab", "cd"], dtype=object))
        self.assertEqual(encoded[:1], b"a")
        pos = 1
        dtype_len = int.from_bytes(encoded[pos : pos + 8], "big")
        pos += 8 + dtype_len
        self.assertEqual(encoded[pos - dtype_len : pos], b"object")
        shape_len = int.from_bytes(encoded[pos : pos + 8], "big")
        pos += 8 + shape_len
        payload_len = int.from_bytes(encoded[pos : pos + 8], "big")
        pos += 8
        payload = encoded[pos : pos + payload_len]
        self.assertEqual(pos + payload_len, len(encoded))  # payload length prefix accounts for the whole rest
        self.assertEqual(payload, _canonical("ab") + _canonical("cd"))

    def test_numeric_dtype_array_payload_is_still_the_raw_tobytes_fast_path(self):
        # The dtype=object fix must not move fixed-width numeric arrays onto the (slower, recursive)
        # element-by-element path -- they have no pointers to leak, so the raw-bytes fast path stays.
        arr = np.array([1.0, 2.0, 3.0])
        encoded = _canonical(arr)
        pos = 1
        dtype_len = int.from_bytes(encoded[pos : pos + 8], "big")
        pos += 8 + dtype_len
        shape_len = int.from_bytes(encoded[pos : pos + 8], "big")
        pos += 8 + shape_len
        payload_len = int.from_bytes(encoded[pos : pos + 8], "big")
        pos += 8
        payload = encoded[pos : pos + payload_len]
        self.assertEqual(payload, arr.tobytes())


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

    def test_default_max_its_and_delta_are_recorded_not_none(self):
        # training["max_its"]/["delta"] used to read straight from the caller-supplied kwargs dict,
        # so a caller relying on optimize()'s own defaults (not passing max_its=/delta= explicitly,
        # exactly like this call) got them recorded as bare None -- and training["converged"] was
        # never even set, since that check was gated on delta not being None -- silently breaking
        # the audit trail this function exists to build.
        # note: no out= kwarg here (unlike the other tests in this file) -- passing out= at all,
        # even out=None, disables convergence capture entirely per this function's own docstring,
        # which would trivially make "converged" absent for an unrelated, documented reason.
        data = np.random.RandomState(2).normal(1.0, 1.0, 300).tolist()
        _, header = fit_with_provenance(data, GaussianDistribution(0.0, 1.0).estimator())
        self.assertEqual(header.training["max_its"], 10)  # optimize()'s own default
        self.assertEqual(header.training["delta"], 1.0e-9)  # optimize()'s own default
        self.assertIn("converged", header.training)

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

    def test_one_shot_data_is_materialized_once_for_fit_score_and_hash(self):
        records = [1.0, 2.0, 3.0, 4.0]
        consumed = []

        def one_shot():
            for value in records:
                consumed.append(value)
                yield value

        _, header = fit_with_provenance(one_shot(), GaussianDistribution(0.0, 1.0).estimator(), max_its=3)

        self.assertEqual(consumed, records)
        self.assertEqual(header.n_records, len(records))
        self.assertEqual(header.dataset_hash, dataset_hash(records))
        self.assertIsNotNone(header.final_loglik)
        self.assertTrue(header.training["data_materialized"])
        self.assertEqual(header.training["fit_request"]["data_hash"], dataset_hash(records))

    def test_recorded_seed_is_the_seed_passed_to_optimization(self):
        captured = {}

        def fake_optimize(data, estimator, max_its=10, delta=1.0e-9, **kwargs):
            captured["data"] = list(data)
            captured["seed"] = kwargs.get("seed")
            return GaussianDistribution(0.0, 1.0)

        with patch("mixle.inference.estimation.optimize", fake_optimize):
            _, header = fit_with_provenance(
                (value for value in [1.0, 2.0, 3.0]),
                GaussianDistribution(0.0, 1.0).estimator(),
                seed=17,
            )

        self.assertEqual(captured["data"], [1.0, 2.0, 3.0])
        self.assertEqual(captured["seed"], 17)
        self.assertEqual(header.training["seed"], 17)

    def test_environment_records_dirty_worktree_state(self):
        _, header = fit_with_provenance([1.0, 2.0, 3.0], GaussianDistribution(0.0, 1.0).estimator(), max_its=2)
        self.assertIn("git_dirty", header.environment)
        self.assertIn("git_worktree_digest", header.environment)
        if header.environment["git_dirty"]:
            self.assertIsInstance(header.environment["git_worktree_digest"], str)
            self.assertEqual(len(header.environment["git_worktree_digest"]), 64)


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

    def test_zero_and_negative_sample_rejected(self):
        # MXR-080-0069: sample=0 used to check nothing and still return
        # DataReport(ok=True, n_checked=0) -- a false certification for data nobody looked at -- and a
        # negative (or fractional) sample used to fail deep inside itertools.islice with an error that
        # never mentions `sample` at all (e.g. "Stop argument for islice() must be ..."). All three must
        # now be rejected clearly and immediately, before any record is read: assert on the message, not
        # just the exception type, since islice's own opaque error is also (incidentally) a ValueError
        # and a message-blind assertRaises(ValueError) would not distinguish the fix from the old bug.
        for sample in (0, -5, 2.5):
            with self.subTest(sample=repr(sample)):
                with self.assertRaisesRegex(ValueError, "sample"):
                    check_dataset(GaussianDistribution(0.0, 1.0), [1.0, 2.0, 3.0], sample=sample)

    def test_positive_sample_negative_control(self):
        # Negative control: a normal positive sample on real data still produces a genuine certifying
        # report -- both the passing and the legitimately-failing case -- and still only inspects
        # exactly `sample` records, not the whole dataset.
        rep_ok = check_dataset(GaussianDistribution(0.0, 1.0), [1.0, 2.0, 3.0], sample=2)
        self.assertTrue(rep_ok.ok)
        self.assertEqual(rep_ok.n_checked, 2)  # only the first `sample` records were inspected

        rep_bad = check_dataset(GaussianDistribution(0.0, 1.0), [1.0, 2.0, "oops", 4.0], sample=10)
        self.assertFalse(rep_bad.ok)
        self.assertEqual(rep_bad.n_checked, 4)  # sample beyond dataset size just caps at what exists

    def test_multivariate_list_data_passes_the_same_as_ndarray_data(self):
        # Bug-2 regression: a dataset of plain Python lists (the natural shape for a multivariate
        # observation) must be accepted exactly like the same data expressed as a list of np.ndarray --
        # previously the list form false-rejected every record (misread as "one record, too many
        # top-level values" against the derived single-Vector-field schema), while the identical data
        # as ndarrays passed fine.
        dist = DiagonalGaussianDistribution([0.0, 0.0], [1.0, 1.0])
        raw = [[1.0, 2.0], [0.5, -0.5], [3.0, 1.0]]

        rep_list = check_dataset(dist, raw)
        rep_array = check_dataset(dist, [np.array(r) for r in raw])

        self.assertTrue(rep_list.ok, rep_list)
        self.assertTrue(rep_array.ok, rep_array)
        self.assertEqual(rep_list.n_checked, 3)


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
        encoder = GaussianDistribution(0, 1).dist_to_encoder()
        enc = encoder.seq_encode([1.0, 2.0])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "enc.pspenc")
            save_encoded(enc, path, encoder=encoder)
            with open(path, "ab") as f:
                f.write(b"corrupt")
            with self.assertRaises(ValueError):
                load_encoded(path, encoder=encoder)

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
            self.assertEqual(magic, b"PSPENC3\n")
            meta = json.loads(header_line)  # raises if this were pickle bytes, not JSON
            self.assertEqual(len(meta["digest"]), 64)
            self.assertIn("Gaussian", meta["encoder"]["signature"])

    def test_header_digest_mismatch_rejected(self):
        # A header whose digest does not match the body must be rejected, including when the body
        # itself is well-formed pickle -- proving the check gates on the digest, not on a parse error.
        import json

        encoder = GaussianDistribution(0, 1).dist_to_encoder()
        enc = encoder.seq_encode([1.0, 2.0])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "enc.pspenc")
            save_encoded(enc, path, encoder=encoder)
            with open(path, "rb") as f:
                raw = f.read()
            nl = raw.index(b"\n", 8)
            meta = json.loads(raw[8:nl])
            meta["digest"] = "0" * 64
            tampered = raw[:8] + json.dumps(meta, separators=(",", ":")).encode() + b"\n" + raw[nl + 1 :]
            with open(path, "wb") as f:
                f.write(tampered)
            with self.assertRaises(ValueError):
                load_encoded(path, encoder=encoder)

    def test_header_tampering_detected(self):
        # MXR-080-0052: the digest previously covered only the pickle body, so the header's
        # "encoder" field could be edited on disk -- leaving the body and the stored digest
        # completely untouched -- and load_encoded would still accept the payload under the
        # forged identity. This is the audit's exact repro: replace the recorded encoder identity
        # without changing the body or digest, then load under the now-forged identity.
        import json

        enc_2d = DiagonalGaussianDistribution([0.0, 0.0], [1.0, 1.0]).dist_to_encoder()
        enc_3d = DiagonalGaussianDistribution([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]).dist_to_encoder()
        self.assertNotEqual(str(enc_2d), str(enc_3d))

        data = enc_2d.seq_encode([[1.0, 2.0], [3.0, 4.0]])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "enc.pspenc")
            save_encoded(data, path, encoder=enc_2d)
            with open(path, "rb") as f:
                raw = f.read()
            magic, rest = raw[:8], raw[8:]
            nl = rest.index(b"\n")
            header_line, body = rest[:nl], rest[nl + 1 :]
            meta = json.loads(header_line)

            # Forge only the "encoder" field; the stored digest and the body bytes are untouched --
            # exactly the audit's repro (swap "Encoder-A" for "Encoder-B" without touching the rest).
            forged = dict(meta)
            forged["encoder"] = str(enc_3d)
            self.assertEqual(forged["digest"], meta["digest"])  # digest itself was NOT re-derived
            tampered = magic + json.dumps(forged).encode("utf-8") + b"\n" + body
            tampered_path = os.path.join(d, "tampered.pspenc")
            with open(tampered_path, "wb") as f:
                f.write(tampered)

            # A load requesting the forged identity must be rejected -- previously this succeeded
            # and silently returned 2D data mislabeled as 3D.
            with self.assertRaises(ValueError):
                load_encoded(tampered_path, encoder=enc_3d)

    def test_envelope_digest_covers_header_fields_not_just_body(self):
        # Direct confirmation of the mechanism behind the regression test above: two different
        # header-field dicts over the *same* body must produce different digests. Before
        # MXR-080-0052's fix, the digest was a pure function of the body, so it was invariant to
        # the header entirely -- this would fail against the pre-fix computation.
        body = b"identical-body-bytes"
        digest_a = _envelope_digest({"encoder": "Encoder-A"}, body)
        digest_b = _envelope_digest({"encoder": "Encoder-B"}, body)
        self.assertNotEqual(digest_a, digest_b)

    def test_envelope_digest_is_canonical_regardless_of_header_key_order(self):
        # The digest must depend on header *content*, not on incidental serialization details like
        # dict/JSON key insertion order -- otherwise a legitimately-written file could fail its own
        # integrity check purely from key ordering, a false "corrupt" signal.
        body = b"identical-body-bytes"
        fields_one_order = {"encoder": "Encoder-A", "extra": "value"}
        fields_other_order = {"extra": "value", "encoder": "Encoder-A"}
        self.assertNotEqual(list(fields_one_order.items()), list(fields_other_order.items()))
        self.assertEqual(
            _envelope_digest(fields_one_order, body),
            _envelope_digest(fields_other_order, body),
        )

    def test_composite_field_count_mismatch_rejected(self):
        # Regression: a dataset encoded with a one-field CompositeDataEncoder must not be accepted
        # as compatible with a two-field CompositeDataEncoder request. Both encoders share the same
        # class name ("CompositeDataEncoder"), so a compatibility check that only compared
        # type(encoder).__name__ silently missed this -- the shape mismatch would then surface as a
        # crash or silently wrong results wherever the mis-shapen loaded data was actually used.
        one_field_encoder = CompositeDistribution([GaussianDistribution(0.0, 1.0)]).dist_to_encoder()
        two_field_encoder = CompositeDistribution(
            [GaussianDistribution(0.0, 1.0), GaussianDistribution(0.0, 1.0)]
        ).dist_to_encoder()
        # Same class name, but structurally different -- this is exactly the gap a class-name-only
        # check misses.
        self.assertEqual(type(one_field_encoder).__name__, type(two_field_encoder).__name__)
        self.assertNotEqual(one_field_encoder, two_field_encoder)

        enc = one_field_encoder.seq_encode([(1.0,), (2.0,), (3.0,)])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "enc.pspenc")
            save_encoded(enc, path, encoder=one_field_encoder)
            with self.assertRaises(ValueError):
                load_encoded(path, encoder=two_field_encoder)

    def test_composite_matching_shape_round_trips(self):
        # Negative control for the regression above: two separately-constructed but structurally
        # equal composite encoders (same field count, same component encoder types) must still be
        # accepted -- the stricter shape check must not false-reject genuinely compatible encoders.
        encoder_a = CompositeDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(0.0, 1.0)])
        encoder_b = CompositeDistribution([GaussianDistribution(1.0, 2.0), GaussianDistribution(3.0, 4.0)])
        encoder_a = encoder_a.dist_to_encoder()
        encoder_b = encoder_b.dist_to_encoder()
        self.assertEqual(encoder_a, encoder_b)

        enc = encoder_a.seq_encode([(1.0, 2.0), (3.0, 4.0)])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "enc.pspenc")
            save_encoded(enc, path, encoder=encoder_a)
            loaded = load_encoded(path, encoder=encoder_b)
            np.testing.assert_allclose(loaded[0], enc[0])
            np.testing.assert_allclose(loaded[1], enc[1])

    def test_same_class_different_dim_leaf_encoder_rejected(self):
        # Broader coverage: the fix is a general structural check (str(encoder)), not special-cased
        # to CompositeDataEncoder -- a same-class leaf encoder with a different structural shape
        # (dimension) must also be rejected.
        enc_2d = DiagonalGaussianDistribution([0.0, 0.0], [1.0, 1.0]).dist_to_encoder()
        enc_3d = DiagonalGaussianDistribution([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]).dist_to_encoder()
        self.assertEqual(type(enc_2d).__name__, type(enc_3d).__name__)
        self.assertNotEqual(enc_2d, enc_3d)

        enc = enc_2d.seq_encode([[1.0, 2.0], [3.0, 4.0]])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "enc.pspenc")
            save_encoded(enc, path, encoder=enc_2d)
            with self.assertRaises(ValueError):
                load_encoded(path, encoder=enc_3d)


if __name__ == "__main__":
    unittest.main()
