"""Focused regressions for the 0.8.0 representation audit."""

from __future__ import annotations

import unittest

import numpy as np

from mixle.represent.graph import GraphEmbedding, GraphEncoder
from mixle.represent.heterogeneous import HeterogeneousEncoder
from mixle.represent.segment import ByteSegmenter, SetSegmenter, WholeSegmenter, WindowSegmenter


class SegmentContractTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
