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
            with self.subTest(patch=repr(patch)), self.assertRaises(ValueError):
                PatchSegmenter(patch)

    def test_window_covers_and_pads_the_final_tail(self):
        segmenter = WindowSegmenter(window=4, hop=4)
        np.testing.assert_array_equal(
            segmenter.segment([1, 2, 3, 4, 5]),
            np.asarray([[1, 2, 3, 4], [5, 0, 0, 0]], dtype=np.float32),
        )
        for bad in (0, -1, 1.5, True):
            with self.subTest(window=repr(bad)), self.assertRaises(ValueError):
                WindowSegmenter(window=bad)

    def test_integer_is_not_interpreted_as_a_byte_count(self):
        for bad in (3, True, np.int64(2)):
            with self.subTest(raw=repr(bad)), self.assertRaises(TypeError):
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
            with self.subTest(dim=repr(bad)), self.assertRaises(ValueError):
                signal_features([1.0, 2.0], dim=bad)
            with self.subTest(windows=repr(bad)), self.assertRaises(ValueError):
                signal_features([1.0, 2.0], dim=4, windows=bad)

    def test_text_vectorization_requires_one_identified_space(self):
        # MXR-080-1669: the default fitted a fresh autoencoder on four copies of the single item, so
        # separately vectorized items came back in different learned bases while being presented as a
        # common fixed-length vector.
        for kind in ("text", "record"):
            with self.subTest(kind=repr(kind)), self.assertRaisesRegex(ValueError, "requires a fitted embedder"):
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
            with self.subTest(iters=repr(bad)), self.assertRaises(ValueError):
                VectorQuantizer(2, 2).fit(np.asarray([[1.0, 2.0], [4.0, 5.0]]), iters=bad)
        for bad in (0, -1, True):
            with self.subTest(num_codes=repr(bad)), self.assertRaises(ValueError):
                VectorQuantizer(bad, 2)
            with self.subTest(dim=repr(bad)), self.assertRaises(ValueError):
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
            with self.subTest(shapes=repr((graph[0].shape, graph[1].shape))), self.assertRaises(ValueError):
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


class QuantizerOwnershipTest(unittest.TestCase):
    """MXR-080-1906: the codebook is the learned vocabulary, and the seed identifies a draw."""

    def test_fitted_codebook_cannot_be_edited_or_rebound(self):
        # REPRODUCED before the fix: `codebook` was a plain public attribute. After fit,
        # `vq.codebook[0, 0] = nan` collapsed quantize() from [0, 2, 0, 0, 2] to [0, 0, 0, 0, 0] --
        # every vector assigned to code 0, since a NaN distance never wins an argmin -- and
        # reconstruction_error() returned nan instead of raising. `vq.codebook = np.zeros((99, 7))`
        # was also accepted, beside an unchanged dim=2 and num_codes=3.
        vectors = np.random.RandomState(0).randn(40, 2)
        quantizer = VectorQuantizer(3, 2, seed=0).fit(vectors)
        before = quantizer.quantize(vectors[:5]).tolist()
        self.assertFalse(quantizer.codebook.flags.writeable)
        with self.assertRaises(ValueError):
            quantizer.codebook[0, 0] = np.nan
        with self.assertRaises(AttributeError):
            quantizer.codebook = np.zeros((99, 7))
        self.assertEqual(quantizer.quantize(vectors[:5]).tolist(), before)
        # decoded vectors are still ordinary writable arrays (fancy indexing copies)
        self.assertTrue(quantizer.dequantize(np.asarray([0, 1])).flags.writeable)

    def test_seed_is_an_identifier_not_a_magnitude(self):
        # REPRODUCED before the fix: `self.seed = int(seed)` truncated, so seed=2.9 and seed=2 named
        # the same stream and produced a bit-identical codebook, and seed=True became 1. seed=-1 was
        # accepted at construction and only failed later inside RandomState at fit time.
        for bad in (2.9, True, np.float64(3.7), -1, 2**32):
            with self.subTest(seed=repr(bad)), self.assertRaises(ValueError):
                VectorQuantizer(4, 2, seed=bad)
        self.assertEqual(VectorQuantizer(4, 2, seed=np.int64(7)).seed, 7)  # exact integers still fine

    def test_straight_through_still_passes_gradients_over_a_frozen_codebook(self):
        # The freeze must not break the VQ-VAE path: torch.as_tensor would try to SHARE the
        # non-writable buffer and warn, so straight_through copies instead.
        import warnings

        import pytest

        pytest.importorskip("torch")
        import torch

        vectors = np.random.RandomState(0).randn(40, 2)
        quantizer = VectorQuantizer(3, 2, seed=0).fit(vectors)
        x = torch.as_tensor(vectors[:5], dtype=torch.float32).requires_grad_(True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            quantizer.straight_through(x).sum().backward()
        self.assertTrue(bool((x.grad != 0).any()))


class AutoencoderResultBindingTest(unittest.TestCase):
    """MXR-080-1906: a result must be evidence of a fit, not a shape that merely looks like one."""

    def test_untrained_state_cannot_be_dressed_as_a_result(self):
        # REPRODUCED before the fix: with no __post_init__, this was accepted and indistinguishable
        # from a trained result -- the exact state AutoencoderFitError exists to keep out of the
        # shared representation space, and which the class docstring already claimed was impossible.
        import pytest

        pytest.importorskip("torch")
        from mixle.represent.embed import FeatureEmbedding
        from mixle.represent.generative import AutoencoderResult

        with self.assertRaisesRegex(ValueError, "non-empty loss curve"):
            AutoencoderResult(encoder=FeatureEmbedding(6, 3), decoder=object(), quantizer=None, losses=[])
        with self.assertRaisesRegex(ValueError, "encoder and a fitted decoder"):
            AutoencoderResult(encoder=FeatureEmbedding(6, 3), decoder=None, quantizer=None, losses=[1.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            AutoencoderResult(encoder=FeatureEmbedding(6, 3), decoder=object(), quantizer=None, losses=[np.nan])
        with self.assertRaisesRegex(ValueError, "unfitted quantizer"):
            AutoencoderResult(
                encoder=FeatureEmbedding(6, 3), decoder=object(), quantizer=VectorQuantizer(2, 3), losses=[1.0]
            )

    def test_loss_curve_is_not_the_live_training_list(self):
        # REPRODUCED before the fix: fit_autoencoder handed over the SAME list the training loop was
        # appending to, so `res.losses.append(-999.0)` rewrote recorded evidence.
        import pytest

        pytest.importorskip("torch")
        from mixle.represent.generative import fit_autoencoder

        units = np.random.RandomState(1).randn(20, 6).astype(np.float32)
        result = fit_autoencoder(units, 3, epochs=5, seed=0)
        self.assertIsInstance(result.losses, tuple)
        self.assertEqual(len(result.losses), 5)
        with self.assertRaises(AttributeError):
            result.losses.append(-999.0)


class RetrievalIdentityTest(unittest.TestCase):
    """MXR-080-1906: `k` must be a count, and an index must name the corpus it indexes."""

    def test_k_refuses_a_boolean_but_still_accepts_an_integral_float(self):
        # REPRODUCED before the fix: `float(k)` then `kf != round(kf)` admits True, because
        # float(True) == 1.0 -- so k=True silently meant "retrieve exactly one".
        from mixle.represent.identity import exact_count

        for bad in (True, np.True_, -1, 2.5, np.nan, np.inf):
            with self.subTest(k=repr(bad)), self.assertRaises(ValueError):
                exact_count(bad, "k")
        # integral floats stay accepted on purpose: the previous contract allowed them, computed
        # counts arrive as floats, and nothing is truncated.
        self.assertEqual(exact_count(5.0, "k"), 5)
        self.assertEqual(exact_count(np.int64(3), "k"), 3)
        self.assertEqual(exact_count(0, "k"), 0)

    def test_posterior_retriever_owns_its_corpus_and_weights(self):
        # REPRODUCED before the fix: `self.corpus = list(corpus)` was public and mutable, so
        # `r.corpus.append(rec)` silently changed what every previously returned index meant, and
        # `self.field_weights = field_weights` aliased the caller's array, letting them reweight the
        # similarity function of an already-built retriever.
        from mixle.represent.posterior import PosteriorRetriever

        class FakeMixture:
            components = ()
            log_w = np.zeros(1)

        rows = [(1.0,), (2.0,), (3.0,)]
        weights = np.asarray([1.0, 2.0])
        retriever = PosteriorRetriever(FakeMixture(), rows, field_weights=weights)
        self.assertIsInstance(retriever.corpus, tuple)
        with self.assertRaises(AttributeError):
            retriever.corpus = ()
        rows.append((4.0,))
        self.assertEqual(len(retriever.corpus), 3)  # the caller's later append did not move indices
        self.assertFalse(retriever.field_weights.flags.writeable)
        weights[0] = 99.0
        self.assertEqual(retriever.field_weights[0], 1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            PosteriorRetriever(FakeMixture(), rows, field_weights=[1.0, np.nan])

    def test_retrievers_record_which_corpus_their_indices_refer_to(self):
        # REPRODUCED before the fix: neither retriever recorded model or corpus identity at all, so
        # results over different corpora were indistinguishable.
        from mixle.represent.posterior import PosteriorRetriever

        class FakeMixture:
            components = ()
            log_w = np.zeros(1)

        a = PosteriorRetriever(FakeMixture(), [(1.0,), (2.0,), (3.0,)])
        same = PosteriorRetriever(FakeMixture(), [(1.0,), (2.0,), (3.0,)])
        different = PosteriorRetriever(FakeMixture(), [(1.0,), (2.0,), (9.0,)])
        self.assertEqual(a.identity.corpus_size, 3)
        self.assertTrue(a.identity.matches(same.identity))
        self.assertFalse(a.identity.matches(different.identity))

    def test_corpus_digest_is_absent_rather_than_faked_for_unencodable_records(self):
        # A retrieval corpus may hold arbitrary payloads, and the canonical encoder is closed over
        # the types it supports. Reporting `None` keeps the retriever working without inventing a
        # digest that would not be reproducible run to run.
        from mixle.represent.identity import RetrievalIdentity, records_digest

        class Opaque:
            pass

        self.assertIsNone(records_digest([Opaque()]))
        unknown = RetrievalIdentity(model="m", corpus_size=1, corpus_digest=None)
        self.assertFalse(unknown.matches(unknown))  # unknown is never evidence of a match


if __name__ == "__main__":
    unittest.main()
