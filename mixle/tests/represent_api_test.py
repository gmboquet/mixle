"""fit_embedder / Embedder: one-call embeddings + retrieval over raw heterogeneous data."""

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
