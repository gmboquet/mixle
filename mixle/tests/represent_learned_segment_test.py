"""Learned segmentation (mixle.represent.learned_segment): infer boundaries by HMM, not fixed cuts.

On a signal made of clear regime runs, the learned segmenter should collapse the long atomic stream into a few
variable-length tokens at the regime changes -- boundaries chosen by likelihood -- and plug into the pipeline.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")  # only for the compose test's embedding; the segmenter itself is torch-free

from mixle.represent import (  # noqa: E402
    ByteSegmenter,
    FeatureEmbedding,
    HeterogeneousEncoder,
    LearnedSegmenter,
    WindowSegmenter,
)
from mixle.represent.segment import Segmenter  # noqa: E402


def _regime_signal(seed, runs=3, run_len=40):
    # a 1-D signal that switches between low/high regimes in long runs
    rng = np.random.RandomState(seed)
    levels = [-3.0, 3.0]
    return np.concatenate([rng.randn(run_len) + levels[i % 2] for i in range(runs)]).astype(np.float32)


class LearnedSegmentTest(unittest.TestCase):
    def test_collapses_atoms_into_few_learned_tokens(self):
        atomic = WindowSegmenter(window=4, hop=4)  # fine atomic units
        seg = LearnedSegmenter(atomic, n_states=2, seed=0).fit([_regime_signal(i) for i in range(6)])
        sig = _regime_signal(99, runs=3, run_len=40)  # ~30 atomic frames
        n_atoms = len(atomic.segment(sig))
        tokens = seg.segment(sig)
        self.assertEqual(tokens.shape[1], 4)  # pooled to the atomic feature width (window)
        self.assertLess(len(tokens), n_atoms)  # genuinely coarser than the atomic stream
        self.assertLessEqual(len(tokens), 8)  # ~3 regimes -> a handful of tokens, not one per frame

    def test_requires_fit(self):
        with self.assertRaises(RuntimeError):
            LearnedSegmenter(WindowSegmenter(window=4)).segment(_regime_signal(0))

    def test_fit_on_all_empty_sequences_does_not_raise_when_atomic_has_num_categories(self):
        # regression: getattr(self.atomic, "num_categories", 1 + max(...)) evaluates the max(...)
        # fallback EAGERLY regardless of whether num_categories exists, so an all-empty-sequences
        # batch raised ValueError (max() of an empty generator) even though ByteSegmenter declares
        # num_categories=256 and the fallback should never have been needed at all.
        seg = LearnedSegmenter(ByteSegmenter(), n_states=2, max_its=3, seed=0)
        seg.fit(["", ""])  # every raw segments to a zero-length byte sequence
        self.assertEqual(seg.feat, 256)

    def test_plugs_into_heterogeneous_encoder(self):
        atomic = WindowSegmenter(window=4, hop=4)
        seg = LearnedSegmenter(atomic, n_states=2, seed=0).fit([_regime_signal(i) for i in range(6)])
        enc = HeterogeneousEncoder(dim=8)
        enc.register("signal", seg, FeatureEmbedding(4, 8))  # learned segmenter feeds a continuous embedding
        stream, tags = enc.encode_numpy({"signal": _regime_signal(7)})
        self.assertEqual(stream.shape[1], 8)  # lands in the shared space
        self.assertGreater(stream.shape[0], 0)


class _ExactFrameSegmenter(Segmenter):
    """Cut a 1-D signal into exact non-overlapping frames -- and into ZERO frames when it is empty.

    ``WindowSegmenter`` pads a short signal up to one full window, which is the right behaviour for
    it but hides the empty case this test is about.
    """

    discrete = False

    def __init__(self, width=4):
        self.width = int(width)

    def segment(self, raw):
        signal = np.asarray(raw, dtype=np.float32).ravel()
        n_frames = len(signal) // self.width
        return signal[: n_frames * self.width].reshape(n_frames, self.width)


class EmptyEvidenceContractTest(unittest.TestCase):
    """MXR-080-1659: no fabricated tokens from an empty input or from an evidence-free fit."""

    def test_a_zero_atom_fit_is_an_explicit_no_evidence_state(self):
        seg = LearnedSegmenter(ByteSegmenter(), n_states=2, max_its=3, seed=0)
        seg.fit(["", ""])
        self.assertEqual(seg.n_fit_atoms, 0)  # the HMM saw no emissions at all
        with self.assertRaisesRegex(RuntimeError, "zero atomic units"):
            seg.segment("")
        with self.assertRaisesRegex(RuntimeError, "zero atomic units"):
            seg.segment("abc")

    def test_empty_input_yields_zero_units_not_one_invented_token(self):
        atomic = _ExactFrameSegmenter(width=4)
        seg = LearnedSegmenter(atomic, n_states=2, seed=0).fit([_regime_signal(i) for i in range(4)])
        self.assertGreater(seg.n_fit_atoms, 0)
        empty = np.zeros(0, dtype=np.float32)
        self.assertEqual(len(atomic.segment(empty)), 0)  # the atomic stream really is empty
        tokens = seg.segment(empty)
        self.assertEqual(tokens.shape, (0, seg.feat))  # zero units preserved, nothing invented
        self.assertGreater(len(seg.segment(_regime_signal(50))), 0)  # a real input still segments

    def test_plugs_into_heterogeneous_encoder(self):
        atomic = WindowSegmenter(window=4, hop=4)
        seg = LearnedSegmenter(atomic, n_states=2, seed=0).fit([_regime_signal(i) for i in range(6)])
        enc = HeterogeneousEncoder(dim=8)
        enc.register("signal", seg, FeatureEmbedding(4, 8))  # learned segmenter feeds a continuous embedding
        stream, tags = enc.encode_numpy({"signal": _regime_signal(7)})
        self.assertEqual(stream.shape[1], 8)  # lands in the shared space
        self.assertGreater(stream.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
