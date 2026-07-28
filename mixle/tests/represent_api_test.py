"""fit_embedder / Embedder: one-call embeddings + retrieval over raw heterogeneous data."""

import json
import pathlib
import tempfile
import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _records(n, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        z = i % 2
        out.append(
            {
                "kind": ["refund", "question"][z],
                "amount": float(rng.gamma(2.0, 50.0 if z == 0 else 500.0)),
                "region": ["us", "eu"][rng.randint(0, 2)],
            }
        )
    return out


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class EmbedderTest(unittest.TestCase):
    def test_records_embed_and_retrieve_their_near_duplicates(self):
        from mixle.represent import Embedder, fit_embedder

        data = _records(120)
        emb = fit_embedder(data, dim=16, epochs=150, seed=0)
        self.assertEqual(emb.transform(data[:7]).shape, (7, 16))

        # a light perturbation of record 3 must retrieve record 3 first (or with near-max similarity)
        q = dict(data[3])
        q["amount"] = q["amount"] * 1.01
        hits = emb.retrieve(q, k=3)
        top_idx, top_sim = hits[0]
        self.assertGreater(top_sim, 0.95)
        self.assertEqual(top_idx % 2, 3 % 2)  # at minimum, the right latent cluster

        with tempfile.TemporaryDirectory() as d:
            path = emb.save(d + "/emb")
            back = Embedder.load(path, trust_code=True)
            np.testing.assert_allclose(back.transform(q), emb.transform(q), atol=1e-6)
            self.assertEqual(back.retrieve(q, k=1)[0][0], hits[0][0])

    def test_text_kind_sniffing(self):
        from mixle.represent import fit_embedder

        texts = [f"refund request number {i}" for i in range(20)] + [f"weather question {i}" for i in range(20)]
        emb = fit_embedder(texts, dim=8, epochs=100, seed=0)
        self.assertEqual(emb.kind, "text")
        hits = emb.retrieve("refund request number 3", k=2)
        self.assertLess(hits[0][0], 20)  # retrieves from the refund half


class EmbedderKindSchemaTest(unittest.TestCase):
    """MXR-080-1784: one validated input schema, resolved before any featurizing or fitting.

    No torch needed -- every case here must fail before ``fit_autoencoder`` is ever reached, which
    is precisely the point: an unrecognized kind used to route to the record featurizer and only
    surface after a complete fit, and an un-declared mixed corpus used to be typed off item 0 alone.
    """

    def test_unknown_kind_is_rejected_by_the_featurizer_router(self):
        from mixle.represent.api import _featurizer

        # The router itself must not treat "anything that isn't 'text'" as a record request.
        with self.assertRaisesRegex(ValueError, "kind must be one of"):
            _featurizer("txet", 16, 0)

    def test_unknown_kind_is_rejected_before_fitting(self):
        from mixle.represent import fit_embedder

        with self.assertRaisesRegex(ValueError, "kind must be one of"):
            fit_embedder(["a", "b", "c", "d"], dim=4, kind="txet", feature_dim=16, epochs=2)

    def test_inferred_kind_rejects_a_mixed_corpus(self):
        from mixle.represent import fit_embedder

        # Typed off item 0 alone this fit as kind="text" and str()-coerced the dict record.
        with self.assertRaisesRegex(ValueError, "mixing"):
            fit_embedder(["a", {"x": 1}, "c", "d"], dim=4, feature_dim=16, epochs=2)

    def test_inferred_kind_accepts_mixed_record_containers(self):
        from mixle.represent.api import _kind_of

        # dict/tuple/list are all one kind ("record"), so a corpus mixing them stays inferable.
        self.assertEqual({_kind_of(x) for x in ({"a": 1}, (1, 2), [3, 4])}, {"record"})


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class EmbedderRetrieveKValidationTest(unittest.TestCase):
    """retrieve(k=...) must reject a k that can't mean "top k": negative or fractional."""

    def setUp(self):
        from mixle.represent import fit_embedder

        self.data = _records(12)
        self.emb = fit_embedder(self.data, dim=4, epochs=5, seed=0)

    def test_negative_k_raises(self):
        # Python's [:k] slicing treats a negative k as "all but the last |k| items", not
        # empty/error -- a negative retrieval count has no sensible meaning and must be
        # rejected rather than silently returning the wrong-sized result.
        with self.assertRaises(ValueError):
            self.emb.retrieve(self.data[0], k=-1)

    def test_non_integer_k_raises(self):
        # A fractional k (e.g. 2.7) must not be silently truncated by an int() cast.
        with self.assertRaises(ValueError):
            self.emb.retrieve(self.data[0], k=2.7)

    def test_zero_k_returns_empty(self):
        self.assertEqual(self.emb.retrieve(self.data[0], k=0), [])

    def test_k_larger_than_corpus_returns_everything(self):
        hits = self.emb.retrieve(self.data[0], k=10_000)
        self.assertEqual(len(hits), len(self.data))

    def test_exact_integer_float_k_still_works(self):
        hits = self.emb.retrieve(self.data[0], k=3.0)
        self.assertEqual(len(hits), 3)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class EmbedderRecordVsBatchContractTest(unittest.TestCase):
    """MXR-080-1783: a list-valued record must not be mistaken for a batch at serving time.

    ``_kind_of`` accepts a list as ONE record when fitting, so training and serving have to agree
    about what ``[1, 2]`` is. ``transform_one``/``transform_batch`` state it explicitly; the
    convenience ``transform`` refuses the shape it cannot tell apart instead of guessing wrong.
    """

    def setUp(self):
        from mixle.represent import fit_embedder

        self.rows = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [2.0, 9.0], [4.0, 1.0]]
        self.emb = fit_embedder(self.rows, dim=4, feature_dim=16, epochs=5, seed=0)

    def test_list_record_embeds_to_one_vector_like_its_tuple_and_dict_equivalents(self):
        from mixle.represent.api import _kind_of

        self.assertEqual(_kind_of([1.0, 2.0]), "record")  # a list IS one record at fit time
        self.assertEqual(self.emb.transform_one([1.0, 2.0]).shape, (self.emb.dim,))
        self.assertEqual(self.emb.transform((1.0, 2.0)).shape, (self.emb.dim,))
        # the same declared record through either container is the SAME single embedding
        np.testing.assert_allclose(
            self.emb.transform_one([1.0, 2.0]), self.emb.transform((1.0, 2.0)), rtol=1e-6, atol=1e-6
        )

    def test_transform_batch_always_returns_a_block(self):
        self.assertEqual(self.emb.transform_batch(self.rows[:3]).shape, (3, self.emb.dim))
        self.assertEqual(self.emb.transform_batch([[1.0, 2.0]]).shape, (1, self.emb.dim))

    def test_transform_refuses_the_ambiguous_scalar_element_list(self):
        # [1.0, 2.0] used to silently produce TWO embedding rows for one declared record.
        with self.assertRaises(ValueError):
            self.emb.transform([1.0, 2.0])

    def test_transform_still_batches_an_unambiguous_sequence_of_records(self):
        self.assertEqual(self.emb.transform(self.rows[:3]).shape, (3, self.emb.dim))

    def test_retrieve_treats_a_list_query_as_one_record(self):
        hits = self.emb.retrieve([1.0, 2.0], k=2)
        self.assertEqual(len(hits), 2)
        self.assertTrue(all(np.isfinite(s) for _, s in hits))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class EmbedderCorpusOwnershipTest(unittest.TestCase):
    """MXR-080-1785: retrieval evidence must not be mutable into arbitrary results.

    The corpus vectors ARE the retrieval evidence. They used to be the caller's own writable array,
    published as a plain attribute with no shape/finiteness check, so setting one entry to NaN made
    retrieve() return an ordinary-looking ranked list whose similarities were all NaN.
    """

    def setUp(self):
        from mixle.represent import fit_embedder

        self.data = _records(12)
        self.emb = fit_embedder(self.data, dim=4, epochs=5, seed=0)

    def test_corpus_vectors_are_read_only(self):
        with self.assertRaises(ValueError):
            self.emb.corpus_vectors[0, 0] = np.nan

    def test_corpus_vectors_attribute_cannot_be_rebound(self):
        with self.assertRaises(AttributeError):
            self.emb.corpus_vectors = np.zeros((3, 4), dtype=np.float32)

    def test_constructor_does_not_alias_the_callers_array(self):
        from mixle.represent import Embedder

        supplied = np.array(self.emb.corpus_vectors, dtype=np.float32)
        clone = Embedder(self.emb.featurizer, self.emb.result, self.emb.kind, supplied)
        supplied[:] = np.nan  # the caller still holds their array and corrupts it
        hits = clone.retrieve(self.data[0], k=3)
        self.assertTrue(all(np.isfinite(s) for _, s in hits))

    def test_constructor_rejects_non_finite_corpus(self):
        from mixle.represent import Embedder

        bad = np.array(self.emb.corpus_vectors, dtype=np.float32)
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            Embedder(self.emb.featurizer, self.emb.result, self.emb.kind, bad)

    def test_constructor_rejects_wrong_rank_and_empty_corpus(self):
        from mixle.represent import Embedder

        for bad in (np.zeros((3, 2, 2), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)):
            with self.assertRaises(ValueError):
                Embedder(self.emb.featurizer, self.emb.result, self.emb.kind, bad)

    def test_constructor_rejects_a_width_the_fitted_encoder_cannot_produce(self):
        from mixle.represent import Embedder

        with self.assertRaises(ValueError):
            Embedder(self.emb.featurizer, self.emb.result, self.emb.kind, np.zeros((5, 9), dtype=np.float32))

    def test_corpus_vectors_stay_unit_normalized(self):
        norms = np.linalg.norm(self.emb.corpus_vectors, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class EmbedderArtifactManifestTest(unittest.TestCase):
    """MXR-080-1786: the manifest must describe AND protect the artifact it ships with.

    Save used to write a pickle and a separate manifest non-atomically with no digest, and load
    ignored the manifest completely, so stale, swapped, partially written or tampered state loaded
    happily under an unrelated description.
    """

    def _fit(self, dim=4, seed=0, n=12):
        from mixle.represent import fit_embedder

        return fit_embedder(_records(n, seed=seed), dim=dim, epochs=5, seed=seed)

    def _envelope(self, path):
        from mixle.represent.api import _ARTIFACT_NAME

        return pathlib.Path(path) / _ARTIFACT_NAME

    def test_round_trip_preserves_the_fitted_state(self):
        from mixle.represent import Embedder

        with tempfile.TemporaryDirectory() as d:
            emb = self._fit()
            back = Embedder.load(emb.save(d + "/emb"), trust_code=True)
            self.assertEqual(back.kind, emb.kind)
            np.testing.assert_allclose(back.corpus_vectors, emb.corpus_vectors, atol=1e-6)

    def test_a_swapped_body_cannot_load_under_another_manifest(self):
        from mixle.represent import Embedder

        with tempfile.TemporaryDirectory() as d:
            four = self._fit(dim=4, seed=0).save(d + "/a")
            six = self._fit(dim=6, seed=1).save(d + "/b")
            # graft b's whole envelope body under a's manifest line
            a_bytes = self._envelope(four).read_bytes()
            b_bytes = self._envelope(six).read_bytes()
            head = a_bytes[: a_bytes.index(b"\n", len(b"MIXLEEMB2\n")) + 1]
            tail = b_bytes[b_bytes.index(b"\n", len(b"MIXLEEMB2\n")) + 1 :]
            self._envelope(four).write_bytes(head + tail)
            with self.assertRaises(ValueError):
                Embedder.load(four, trust_code=True)

    def test_a_tampered_manifest_field_is_rejected(self):
        from mixle.represent import Embedder

        with tempfile.TemporaryDirectory() as d:
            path = self._fit(dim=4).save(d + "/emb")
            raw = self._envelope(path).read_bytes()
            self._envelope(path).write_bytes(raw.replace(b'"dim":4', b'"dim":8', 1))
            with self.assertRaises(ValueError):
                Embedder.load(path, trust_code=True)

    def test_a_truncated_artifact_is_rejected_before_unpickling(self):
        from mixle.represent import Embedder

        with tempfile.TemporaryDirectory() as d:
            path = self._fit().save(d + "/emb")
            env = self._envelope(path)
            env.write_bytes(env.read_bytes()[:-32])
            with self.assertRaises(ValueError):
                Embedder.load(path, trust_code=True)

    def test_a_foreign_payload_under_a_valid_digest_is_rejected(self):
        import pickle

        from mixle.represent import Embedder
        from mixle.represent.api import _ARTIFACT_ID, _ARTIFACT_MAGIC, _envelope_digest

        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d) / "emb"
            root.mkdir()
            body = pickle.dumps({"featurizer": None, "result": None, "kind": "nope", "corpus_vectors": "not an array"})
            manifest = {"mixle_artifact": _ARTIFACT_ID, "kind": "record", "dim": 4, "n_corpus": 12, "created_at": 0.0}
            meta = {"digest": _envelope_digest(manifest, body), **manifest}
            self._envelope(root).write_bytes(
                _ARTIFACT_MAGIC + json.dumps(meta, sort_keys=True, separators=(",", ":")).encode() + b"\n" + body
            )
            with self.assertRaises(ValueError):
                Embedder.load(root, trust_code=True)

    def test_a_legacy_unbound_artifact_is_named_not_silently_loaded(self):
        from mixle.represent import Embedder

        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d) / "emb"
            root.mkdir()
            (root / "embedder.pkl").write_bytes(b"anything")
            (root / "manifest.json").write_text('{"mixle_artifact": "represent.Embedder/v1"}')
            with self.assertRaisesRegex(ValueError, "legacy"):
                Embedder.load(root, trust_code=True)

    def test_save_leaves_no_partial_artifact_behind(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(self._fit().save(d + "/emb"))
            self.assertEqual([p.name for p in path.iterdir()], ["embedder.mixle"])


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class EmbedderLoadTrustGateTest(unittest.TestCase):
    """Embedder.load unpickles a live torch module -- it must refuse without explicit trust."""

    def _saved_path(self, d):
        from mixle.represent import fit_embedder

        emb = fit_embedder(_records(20), dim=4, epochs=5, seed=0)
        return emb.save(d + "/emb")

    def test_load_refuses_without_trust_code(self):
        from mixle.represent import Embedder
        from mixle.utils.serialization import SerializationError

        with tempfile.TemporaryDirectory() as d:
            path = self._saved_path(d)
            with self.assertRaises(SerializationError):
                Embedder.load(path)  # gate closed by default: no trust given

    def test_load_succeeds_with_trust_code_true(self):
        from mixle.represent import Embedder

        with tempfile.TemporaryDirectory() as d:
            path = self._saved_path(d)
            back = Embedder.load(path, trust_code=True)
            self.assertIsInstance(back, Embedder)

    def test_load_succeeds_inside_trusted_deserialization(self):
        from mixle.represent import Embedder
        from mixle.utils.serialization import trusted_deserialization

        with tempfile.TemporaryDirectory() as d:
            path = self._saved_path(d)
            with trusted_deserialization():
                back = Embedder.load(path)
            self.assertIsInstance(back, Embedder)


if __name__ == "__main__":
    unittest.main()
