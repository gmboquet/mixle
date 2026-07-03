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
            back = Embedder.load(path)
            np.testing.assert_allclose(back.transform(q), emb.transform(q), atol=1e-6)
            self.assertEqual(back.retrieve(q, k=1)[0][0], hits[0][0])

    def test_text_kind_sniffing(self):
        from mixle.represent import fit_embedder

        texts = [f"refund request number {i}" for i in range(20)] + [f"weather question {i}" for i in range(20)]
        emb = fit_embedder(texts, dim=8, epochs=100, seed=0)
        self.assertEqual(emb.kind, "text")
        hits = emb.retrieve("refund request number 3", k=2)
        self.assertLess(hits[0][0], 20)  # retrieves from the refund half


if __name__ == "__main__":
    unittest.main()
