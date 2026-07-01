"""Heterogeneous representation layer (mixle.represent): every modality into one space, discretize only if wanted.

The design must (a) embed text/image/signal/structure into ONE shared dim, (b) train end to end to a generative
or downstream objective, and (c) discretize the shared space into a LEARNED cross-modal vocabulary on demand --
without any modality committing to a vocabulary upstream.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.represent import (  # noqa: E402
    ByteSegmenter,
    CategoricalEmbedding,
    FeatureEmbedding,
    HeterogeneousEncoder,
    PatchSegmenter,
    SetSegmenter,
    VectorQuantizer,
    WindowSegmenter,
)

DIM = 16


def _hetero_encoder():
    enc = HeterogeneousEncoder(dim=DIM)
    enc.register("text", ByteSegmenter(), CategoricalEmbedding(256, DIM))  # discrete bytes
    enc.register("image", PatchSegmenter(patch=4), FeatureEmbedding(3 * 4 * 4, DIM))  # continuous patches
    enc.register("seismic", WindowSegmenter(window=8, hop=8), FeatureEmbedding(8, DIM))  # continuous windows
    enc.register("molecule", SetSegmenter(), FeatureEmbedding(5, DIM))  # a structure = a set of atom features
    return enc


def _record(seed=0):
    rng = np.random.RandomState(seed)
    return {
        "text": "hello",  # 5 byte units
        "image": rng.rand(3, 8, 8).astype(np.float32),  # (8/4)^2 = 4 patch units
        "seismic": rng.randn(24).astype(np.float32),  # 3 window units
        "molecule": rng.rand(6, 5).astype(np.float32),  # 6 atom-feature units
    }


class ShapeTest(unittest.TestCase):
    def test_all_modalities_land_in_one_shared_space(self):
        enc = _hetero_encoder()
        stream, tags = enc.encode_numpy(_record())
        # 5 bytes + 4 patches + 3 windows + 6 atoms = 18 units, each a DIM vector
        self.assertEqual(stream.shape, (18, DIM))
        self.assertEqual(tags.shape, (18,))
        self.assertEqual(len(set(tags.tolist())), 4)  # four distinct modality tags

    def test_unknown_modality_raises(self):
        enc = _hetero_encoder()
        with self.assertRaises(KeyError):
            enc.encode({"proteins": "MKV"})

    def test_dim_mismatch_rejected(self):
        enc = HeterogeneousEncoder(dim=DIM)
        with self.assertRaises(ValueError):
            enc.register("text", ByteSegmenter(), CategoricalEmbedding(256, DIM + 1))


class TrainabilityTest(unittest.TestCase):
    def test_encoders_train_end_to_end(self):
        # a downstream objective: pool the stream -> a linear head -> binary label; gradients must reach the encoders
        enc = _hetero_encoder()
        head = torch.nn.Linear(DIM, 2)
        params = enc.parameters() + list(head.parameters())
        opt = torch.optim.Adam(params, lr=1e-2)
        records = [_record(i) for i in range(12)]
        labels = torch.tensor([i % 2 for i in range(12)])

        before = enc.encoders["image"].embedding.module()[0].weight.detach().clone()
        loss0 = None
        for step in range(15):
            opt.zero_grad()
            logits = torch.stack([enc.encode(r)[0].mean(dim=0) for r in records])  # mean-pool each record
            loss = torch.nn.functional.cross_entropy(head(logits), labels)
            if step == 0:
                loss0 = float(loss.detach())
            loss.backward()
            opt.step()
        self.assertLess(float(loss), loss0)  # the objective drove the encoders
        after = enc.encoders["image"].embedding.module()[0].weight.detach()
        self.assertFalse(torch.allclose(before, after))  # a continuous encoder actually trained

    def test_shared_embedding_ties_two_modalities(self):
        # the same FeatureEmbedding instance used by two modalities -> one shared tensor (as before, but continuous)
        enc = HeterogeneousEncoder(dim=DIM)
        shared = FeatureEmbedding(8, DIM, name="shared")
        enc.register("seismic", WindowSegmenter(window=8, hop=8), shared)
        enc.register("audio", WindowSegmenter(window=8, hop=8), shared)
        w1 = enc.encoders["seismic"].embedding.module()[0].weight
        w2 = enc.encoders["audio"].embedding.module()[0].weight
        self.assertIs(w1, w2)


class QuantizeTest(unittest.TestCase):
    def test_learned_codebook_quantizes_and_reconstructs(self):
        rng = np.random.RandomState(0)
        # three well-separated clusters in the shared space
        vecs = np.vstack([rng.randn(60, DIM) + c for c in ([0] * DIM, [6] * DIM, [-6] * DIM)])
        vq = VectorQuantizer(num_codes=3, dim=DIM, seed=0).fit(vecs)
        ids = vq.quantize(vecs)
        self.assertEqual(len(set(ids.tolist())), 3)  # recovered the three codes
        self.assertLess(vq.reconstruction_error(vecs), 2.0 * DIM)  # near the cluster centers

    def test_cross_modal_vocabulary(self):
        # one codebook fit on the whole heterogeneous stream -> a shared vocabulary across modalities
        enc = _hetero_encoder()
        stream, _ = enc.encode_numpy(_record(3))
        vq = VectorQuantizer(num_codes=8, dim=DIM, seed=0).fit(stream)
        ids = vq.quantize(stream)
        self.assertEqual(ids.shape, (stream.shape[0],))
        self.assertTrue(set(ids.tolist()).issubset(set(range(8))))

    def test_straight_through_passes_gradient(self):
        vecs = torch.randn(10, DIM, requires_grad=True)
        vq = VectorQuantizer(num_codes=4, dim=DIM, seed=0).fit(vecs.detach().numpy())
        q = vq.straight_through(vecs)
        q.sum().backward()
        self.assertIsNotNone(vecs.grad)  # gradient flows through the discrete bottleneck
        self.assertTrue(torch.allclose(vecs.grad, torch.ones_like(vecs)))  # identity backward


if __name__ == "__main__":
    unittest.main()
