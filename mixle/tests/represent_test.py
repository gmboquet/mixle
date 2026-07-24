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

    def test_encode_order_follows_registration_not_record_key_order(self):
        # regression: encode() used to concatenate in the INPUT record dict's own key order (record.items())
        # instead of registration order, despite the docstring's promise of registration order -- so two
        # dicts holding the same modality values but built with different key-insertion order produced
        # different (even fully reversed) token streams for semantically-identical records.
        enc = _hetero_encoder()  # registers text(id0), image(id1), seismic(id2), molecule(id3), in that order
        record = _record()
        reversed_record = dict(reversed(list(record.items())))  # same k/v pairs, opposite insertion order
        self.assertNotEqual(list(record), list(reversed_record))  # sanity: the two dicts really do differ in order

        stream, tags = enc.encode_numpy(record)
        rev_stream, rev_tags = enc.encode_numpy(reversed_record)

        # 5 bytes (text) + 4 patches (image) + 3 windows (seismic) + 6 atoms (molecule), in registration order
        expected_tags = [0] * 5 + [1] * 4 + [2] * 3 + [3] * 6
        self.assertEqual(tags.tolist(), expected_tags)
        self.assertEqual(rev_tags.tolist(), expected_tags)  # record's own key order must not matter
        np.testing.assert_array_equal(stream, rev_stream)

    def test_record_missing_registered_modality_raises(self):
        enc = _hetero_encoder()
        record = _record()
        del record["seismic"]  # drop one of the four registered modalities
        with self.assertRaises(KeyError):
            enc.encode(record)

    def test_patch_segmenter_rejects_image_smaller_than_patch(self):
        # regression: an image smaller than the patch size in either dimension used to silently
        # produce a (0, features) array (h // p == 0) instead of erroring -- no patches, no warning.
        seg = PatchSegmenter(patch=8)
        with self.assertRaises(ValueError):
            seg.segment(np.random.rand(5, 5))  # smaller than patch in both dims
        with self.assertRaises(ValueError):
            seg.segment(np.random.rand(3, 5, 20))  # (C, H, W) form, smaller than patch in H only
        # an image exactly the patch size is the boundary case and must still yield one patch.
        out = seg.segment(np.random.rand(8, 8))
        self.assertEqual(out.shape, (1, 8 * 8))

    def test_window_segmenter_pads_short_signal_without_discarding_it(self):
        # regression: a signal shorter than `window` used to be silently replaced by one
        # fabricated all-zero window instead of keeping the real (if short) signal -- [1, 2] and
        # [9, 8] under window=4 both produced the identical all-zero output, so genuinely different
        # short signals became indistinguishable. WindowSegmenter's contract is to always return at
        # least one window (unlike PatchSegmenter, which has no such contract and instead rejects a
        # too-small image outright); the fix must honor that contract honestly, padding only the
        # slots beyond the real samples rather than overwriting the real samples too.
        seg = WindowSegmenter(window=4)
        out_12 = seg.segment([1, 2])
        out_98 = seg.segment([9, 8])
        self.assertEqual(out_12.shape, (1, 4))
        np.testing.assert_array_equal(out_12, [[1.0, 2.0, 0.0, 0.0]])  # real samples first, zero-padded after
        np.testing.assert_array_equal(out_98, [[9.0, 8.0, 0.0, 0.0]])
        self.assertFalse(np.array_equal(out_12, out_98))  # different inputs must not collapse to the same output
        # the empty signal is the degenerate case: nothing real to place, so all-zero is honest here
        # (nothing is being discarded, unlike the [1, 2] / [9, 8] cases above).
        np.testing.assert_array_equal(seg.segment([]), [[0.0, 0.0, 0.0, 0.0]])
        # a signal exactly `window` long is the boundary case and must still yield one real
        # window -- containing the actual observed samples unchanged, no padding involved.
        out_full = seg.segment([1, 2, 3, 4])
        self.assertEqual(out_full.shape, (1, 4))
        np.testing.assert_array_equal(out_full, [[1.0, 2.0, 3.0, 4.0]])

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


class ModalityEmbeddingRegistrationTest(unittest.TestCase):
    def test_reregistering_existing_modality_preserves_learned_tag_embedding(self):
        # regression: register_encoder() used to unconditionally reset _modality_embedding to None on every
        # call, even when re-registering an ALREADY-registered modality (the modality set/count unchanged) --
        # so any already-learned tag-embedding weights were silently discarded for what is otherwise a no-op.
        enc = HeterogeneousEncoder(dim=DIM)
        enc.register("text", ByteSegmenter(), CategoricalEmbedding(256, DIM))
        enc.register("image", PatchSegmenter(patch=4), FeatureEmbedding(3 * 4 * 4, DIM))

        tag_module = enc._modality_embed().module()  # force-build, standing in for "has been used"
        with torch.no_grad():
            tag_module.weight.fill_(1.0)  # obviously non-random stand-in for "learned" state
        embedding_obj = enc._modality_embedding
        learned_weight = tag_module.weight.clone()

        enc.register("text", ByteSegmenter(), CategoricalEmbedding(256, DIM))  # re-register, same modality set

        self.assertIs(enc._modality_embedding, embedding_obj)  # not rebuilt
        self.assertTrue(torch.allclose(enc._modality_embedding.module().weight, learned_weight))  # not reset

    def test_registering_new_modality_grows_tag_embedding_preserving_old_rows(self):
        # negative control / contrast: a genuine modality-SET change (a real new modality) must still be
        # reflected in the tag embedding -- it just has to preserve, not discard, the already-learned rows.
        enc = HeterogeneousEncoder(dim=DIM)
        enc.register("text", ByteSegmenter(), CategoricalEmbedding(256, DIM))
        enc.register("image", PatchSegmenter(patch=4), FeatureEmbedding(3 * 4 * 4, DIM))

        tag_module = enc._modality_embed().module()
        with torch.no_grad():
            tag_module.weight.fill_(1.0)
        learned_weight = tag_module.weight.clone()
        old_num_categories = enc._modality_embedding.num_categories

        enc.register("seismic", WindowSegmenter(window=8, hop=8), FeatureEmbedding(8, DIM))  # brand new modality

        self.assertEqual(enc._modality_embedding.num_categories, old_num_categories + 1)
        grown_weight = enc._modality_embedding.module().weight
        self.assertEqual(tuple(grown_weight.shape), (old_num_categories + 1, DIM))
        self.assertTrue(torch.allclose(grown_weight[:old_num_categories], learned_weight))  # old rows preserved

    def test_registering_before_any_use_stays_lazily_unbuilt(self):
        # sanity: registration alone (no training/no encode call) must not force-build the tag embedding --
        # the lazy-build contract for a never-yet-used encoder is untouched by the fix.
        enc = HeterogeneousEncoder(dim=DIM)
        enc.register("text", ByteSegmenter(), CategoricalEmbedding(256, DIM))
        self.assertIsNone(enc._modality_embedding)
        enc.register("image", PatchSegmenter(patch=4), FeatureEmbedding(3 * 4 * 4, DIM))
        self.assertIsNone(enc._modality_embedding)


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

    def test_fit_with_fewer_samples_than_codes_shrinks_num_codes(self):
        # 2 vectors can't support 5 centers -> fit can only produce one center per sample, and
        # num_codes must report that ACTUAL count instead of silently overstating capacity as 5
        vq = VectorQuantizer(num_codes=5, dim=2, seed=0)
        vq.fit(np.array([[0.0, 0.0], [10.0, 10.0]]))
        self.assertEqual(vq.num_codes, 2)  # downgraded from the declared 5, not left stale
        self.assertEqual(vq.codebook.shape, (2, 2))

    def test_dequantize_rejects_id_beyond_actual_codebook_after_downgrade(self):
        # id 4 is "nominally valid" against the originally-requested 5 codes, but the codebook
        # actually only has 2 rows after the downgrade -- must be rejected with a clear error
        vq = VectorQuantizer(num_codes=5, dim=2, seed=0).fit(np.array([[0.0, 0.0], [10.0, 10.0]]))
        with self.assertRaises(IndexError) as ctx:
            vq.dequantize(4)
        message = str(ctx.exception)
        # our own validation message, not numpy's incidental "index 4 is out of bounds for axis 0
        # with size 2" -- which would (confusingly) also contain "4" and "2" as raw digits
        self.assertIn("codebook", message)
        self.assertIn("4", message)
        self.assertIn("2", message)  # names the actual codebook size, not just "out of bounds"
        self.assertEqual(vq.dequantize(1).shape, (2,))  # a genuinely in-range id still works

    def test_fit_with_enough_samples_keeps_declared_num_codes(self):
        # guard against over-correcting: the common case (enough data for the full request) must
        # keep reporting the declared count unchanged
        rng = np.random.RandomState(0)
        vecs = np.vstack([rng.randn(60, DIM) + c for c in ([0] * DIM, [6] * DIM, [-6] * DIM)])
        vq = VectorQuantizer(num_codes=3, dim=DIM, seed=0).fit(vecs)
        self.assertEqual(vq.num_codes, 3)
        self.assertEqual(vq.codebook.shape, (3, DIM))


if __name__ == "__main__":
    unittest.main()
