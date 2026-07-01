"""Learned segmentation (mixle.represent.learned_segment): infer boundaries by HMM, not fixed cuts.

On a signal made of clear regime runs, the learned segmenter should collapse the long atomic stream into a few
variable-length tokens at the regime changes -- boundaries chosen by likelihood -- and plug into the pipeline.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")  # only for the compose test's embedding; the segmenter itself is torch-free

from mixle.represent import FeatureEmbedding, HeterogeneousEncoder, LearnedSegmenter, WindowSegmenter  # noqa: E402


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
