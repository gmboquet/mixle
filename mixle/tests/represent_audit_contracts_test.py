"""Focused regressions for the 0.8.0 representation audit."""

from __future__ import annotations

import unittest

import numpy as np

from mixle.represent.graph import GraphEmbedding, GraphEncoder
from mixle.represent.heterogeneous import HeterogeneousEncoder
from mixle.represent.modality import image_features, signal_features, vectorize
from mixle.represent.quantize import VectorQuantizer
from mixle.represent.segment import (
    ByteSegmenter,
    ElementSegmenter,
    PatchSegmenter,
    SetSegmenter,
    WholeSegmenter,
    WindowSegmenter,
)


class SegmentContractTest(unittest.TestCase):
    def test_patch_segmentation_rejects_implicit_border_loss(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            PatchSegmenter(8).segment(np.zeros((10, 10)))
        for patch in (True, 0, -1, 1.5):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                PatchSegmenter(patch)

    def test_window_covers_and_pads_the_final_tail(self):
        segmenter = WindowSegmenter(window=4, hop=4)
        np.testing.assert_array_equal(
            segmenter.segment([1, 2, 3, 4, 5]),
            np.asarray([[1, 2, 3, 4], [5, 0, 0, 0]], dtype=np.float32),
        )
        for bad in (0, -1, 1.5, True):
            with self.subTest(window=bad), self.assertRaises(ValueError):
                WindowSegmenter(window=bad)

    def test_integer_is_not_interpreted_as_a_byte_count(self):
        for bad in (3, True, np.int64(2)):
            with self.subTest(raw=bad), self.assertRaises(TypeError):
                ByteSegmenter().segment(bad)

    def test_generic_feature_segmenters_reject_featureless_values(self):
        for segmenter in (WholeSegmenter(), SetSegmenter()):
            with self.subTest(segmenter=type(segmenter).__name__), self.assertRaises(ValueError):
                segmenter.segment([])
        with self.assertRaises(ValueError):
            SetSegmenter().segment(3.0)
        with self.assertRaises(ValueError):
            SetSegmenter().segment([[1.0], [1.0, 2.0]])

    def test_unknown_symbols_do_not_impersonate_the_first_alphabet_entry(self):
        # MXR-080-1671: unknowns mapped to id 0, which is already "A" -- an out-of-vocabulary residue
        # became positive evidence for a genuinely observed one.
        segmenter = ElementSegmenter(["A", "C"])
        ids = segmenter.segment(["A", "C", "X"])
        self.assertEqual(ids.tolist(), [0, 1, segmenter.unknown_id])
        self.assertNotIn(segmenter.unknown_id, (0, 1))
        self.assertEqual(segmenter.num_categories, 3)  # alphabet + the reserved unknown state
        self.assertLess(int(ids.max()), segmenter.num_categories)

    def test_duplicate_alphabet_entries_are_rejected(self):
        # a repeat overwrote its own index while still inflating num_categories, leaving an id
        # segment() could never emit.
        with self.assertRaisesRegex(ValueError, "unique alphabet"):
            ElementSegmenter(["A", "A", "C"])


class ModalityContractTest(unittest.TestCase):
    def test_absent_and_nonfinite_signals_are_not_ordinary_evidence(self):
        # MXR-080-1670: signal_features([], dim=4) returned [0, 0, 0, 0] -- byte-identical to a real,
        # measured all-zero trace -- and [0, NaN] returned NaN features padded out to dim.
        with self.assertRaisesRegex(ValueError, "non-empty"):
            signal_features([], dim=4)
        with self.assertRaisesRegex(ValueError, "finite"):
            signal_features([0.0, np.nan], dim=4)
        with self.assertRaisesRegex(ValueError, "finite"):
            image_features([[0.0, np.inf]], dim=4)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            image_features(np.empty((0, 3)), dim=4)
        # a genuinely measured zero trace is still a perfectly ordinary descriptor
        np.testing.assert_array_equal(signal_features([0.0, 0.0], dim=4), [0.0, 0.0, 0.0, 0.0])
        for bad in (0, -1, True, 1.5):
            with self.subTest(dim=bad), self.assertRaises(ValueError):
                signal_features([1.0, 2.0], dim=bad)
            with self.subTest(windows=bad), self.assertRaises(ValueError):
                signal_features([1.0, 2.0], dim=4, windows=bad)

    def test_text_vectorization_requires_one_identified_space(self):
        # MXR-080-1669: the default fitted a fresh autoencoder on four copies of the single item, so
        # separately vectorized items came back in different learned bases while being presented as a
        # common fixed-length vector.
        for kind in ("text", "record"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "requires a fitted embedder"):
                vectorize("alpha alpha", kind)

        class SharedSpace:
            def transform(self, item):
                return np.asarray([float(len(item)), 1.0])

        np.testing.assert_array_equal(vectorize("abc", "text", embedder=SharedSpace()), [3.0, 1.0])


class QuantizerContractTest(unittest.TestCase):
    def test_fit_holds_the_declared_geometry_and_finiteness(self):
        # MXR-080-1673: VectorQuantizer(2, 2) accepted a (2, 3) fit and kept dim == 2 beside a (2, 3)
        # codebook; a NaN sample produced a NaN codebook; iters=0 published the init centers as fitted.
        with self.assertRaisesRegex(ValueError, "declared width"):
            VectorQuantizer(2, 2).fit(np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        with self.assertRaisesRegex(ValueError, "finite"):
            VectorQuantizer(2, 2).fit(np.asarray([[1.0, 2.0], [np.nan, 5.0]]))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            VectorQuantizer(2, 2).fit(np.empty((0, 2)))
        for bad in (0, -1, True, 2.5):
            with self.subTest(iters=bad), self.assertRaises(ValueError):
                VectorQuantizer(2, 2).fit(np.asarray([[1.0, 2.0], [4.0, 5.0]]), iters=bad)
        for bad in (0, -1, True):
            with self.subTest(num_codes=bad), self.assertRaises(ValueError):
                VectorQuantizer(bad, 2)
            with self.subTest(dim=bad), self.assertRaises(ValueError):
                VectorQuantizer(2, bad)

    def test_fractional_ids_are_not_silently_truncated_into_tokens(self):
        quantizer = VectorQuantizer(2, 2).fit(np.asarray([[0.0, 0.0], [10.0, 10.0]]))
        with self.assertRaisesRegex(ValueError, "exact integers"):
            quantizer.dequantize(np.asarray([0.9, 1.9]))
        np.testing.assert_array_equal(quantizer.dequantize(np.asarray([0.0, 1.0])).shape, (2, 2))
        with self.assertRaisesRegex(ValueError, "not Booleans"):
            quantizer.dequantize(np.asarray([True, False]))
        # serving vectors are held to the same declared geometry as training vectors
        with self.assertRaisesRegex(ValueError, "declared width"):
            quantizer.quantize(np.asarray([[1.0, 2.0, 3.0]]))


class GraphContractTest(unittest.TestCase):
    def test_graph_encoder_rejects_malformed_or_nonfinite_graphs(self):
        encoder = GraphEncoder(GraphEmbedding(in_features=2, dim=3))
        invalid = [
            (np.empty((0, 2)), np.empty((0, 0))),
            (np.ones((2, 3)), np.eye(2)),
            (np.ones((2, 2)), np.ones((2, 3))),
            (np.ones((2, 2)), np.asarray([[0.0, np.nan], [1.0, 0.0]])),
            (np.ones((2, 2)), np.asarray([[0.0, -1.0], [1.0, 0.0]])),
        ]
        for graph in invalid:
            with self.subTest(shapes=(graph[0].shape, graph[1].shape)), self.assertRaises(ValueError):
                encoder.encode(graph)

    def test_heterogeneous_encoder_rejects_broadcastable_wrong_width(self):
        import torch

        class WrongWidth:
            dim = 2
            embedding = None

            def encode(self, raw):
                return torch.ones((2, 1))

        encoder = HeterogeneousEncoder(2).register_encoder("bad", WrongWidth())
        with self.assertRaisesRegex(ValueError, "n_units"):
            encoder.encode({"bad": object()})

    def test_growth_after_export_cannot_silently_detach_from_the_optimizer(self):
        # MXR-080-1672: registering a new modality replaced the tag embedding with a larger module, so
        # the optimizer still owned the old parameter object -- the grown weight had a nonzero gradient,
        # was absent from every param group, and stayed bit-identical after step().
        import torch

        from mixle.models.embedding import CategoricalEmbedding

        encoder = HeterogeneousEncoder(4)
        encoder.register("text", ByteSegmenter(), CategoricalEmbedding(256, 4, name="text"))
        optimizer = torch.optim.SGD(encoder.parameters(), lr=1.0)
        with self.assertRaisesRegex(RuntimeError, "rebind=True"):
            encoder.register("other", ByteSegmenter(), CategoricalEmbedding(256, 4, name="other"))
        self.assertNotIn("other", encoder.encoders)  # the refused registration did not half-apply

        encoder.register("other", ByteSegmenter(), CategoricalEmbedding(256, 4, name="other"), rebind=True)
        optimizer = torch.optim.SGD(encoder.parameters(), lr=1.0)
        tag_weight = encoder._modality_embed().module().weight
        owned = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertTrue(any(p is tag_weight for p in owned))

        stream, _ = encoder.encode({"text": "ab", "other": "cd"})
        stream.sum().backward()
        before = tag_weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, tag_weight.detach()))  # the grown tag actually trains

    def test_registering_before_export_is_unaffected(self):
        # negative control: the ordinary register-everything-then-train flow must not be disturbed.
        from mixle.models.embedding import CategoricalEmbedding

        encoder = HeterogeneousEncoder(4)
        encoder.register("text", ByteSegmenter(), CategoricalEmbedding(256, 4, name="text"))
        encoder.encode({"text": "ab"})  # builds the tag embedding, but exports nothing
        encoder.register("other", ByteSegmenter(), CategoricalEmbedding(256, 4, name="other"))
        self.assertEqual(sorted(encoder.encoders), ["other", "text"])


if __name__ == "__main__":
    unittest.main()
