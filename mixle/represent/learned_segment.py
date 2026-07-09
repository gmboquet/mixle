"""Learned segmentation -- infer WHERE the tokens are, instead of cutting at fixed positions.

A fixed segmenter cuts every N bytes / every patch; but the *right* boundaries depend on the data. This fits an
HMM over a finer atomic stream and cuts where the latent state changes: the boundaries are chosen by
maximum likelihood, so the segmentation is *inferred* -- the objective-coupled tokenizer. A run of same-state
atoms becomes one variable-length token, pooled into a fixed feature vector for the embedding (a bag-of-symbols
histogram for a discrete atomic stream, the mean vector for a continuous one).

``LearnedSegmenter`` wraps any atomic :class:`~mixle.represent.segment.Segmenter`, is ``fit`` on example raws, and
then plugs into the representation pipeline exactly like a fixed segmenter -- so tokenization is a *model* you
train, reusing mixle's HMM inference, not a hand-set rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.represent.segment import Segmenter


class LearnedSegmenter(Segmenter):
    """Cut a raw object into variable-length tokens at HMM state changes over its atomic units (pooled to features)."""

    discrete = False  # segments are pooled feature vectors, whatever the atomic stream was

    def __init__(self, atomic: Segmenter, n_states: int = 4, *, max_its: int = 30, seed: int = 0) -> None:
        self.atomic = atomic
        self.n_states = int(n_states)
        self.max_its = int(max_its)
        self.seed = int(seed)
        self.hmm: Any = None
        self.feat: int | None = None  # dimension of a pooled segment feature

    def fit(self, raws: Sequence[Any]) -> LearnedSegmenter:
        """Fit the boundary HMM on example raws -- the segmentation is learned to maximize their likelihood."""
        import mixle.stats as st
        from mixle.inference import optimize

        seqs = [self.atomic.segment(r) for r in raws]
        if self.atomic.discrete:
            self.feat = int(getattr(self.atomic, "num_categories", 1 + max(int(s.max()) for s in seqs if len(s))))
            est = st.HiddenMarkovEstimator([st.CategoricalEstimator() for _ in range(self.n_states)])
            data = [[int(v) for v in s] for s in seqs]
        else:
            self.feat = int(seqs[0].shape[1])
            est = st.HiddenMarkovEstimator([st.DiagonalGaussianEstimator(dim=self.feat) for _ in range(self.n_states)])
            data = [[np.asarray(v, dtype=np.float64) for v in s] for s in seqs]
        self.hmm = optimize(data, est, max_its=self.max_its, rng=np.random.RandomState(self.seed), out=None)
        return self

    def segment(self, raw: Any) -> np.ndarray:
        """Segment raw input using fitted HMM state assignments."""
        if self.hmm is None:
            raise RuntimeError("call fit(...) before segment(...)")
        atoms = self.atomic.segment(raw)
        if len(atoms) == 0:
            return np.zeros((1, self.feat or 1), dtype=np.float32)
        if self.atomic.discrete:
            obs = [int(v) for v in atoms]
        else:
            obs = [np.asarray(v, dtype=np.float64) for v in atoms]
        path = np.asarray(self.hmm.viterbi(obs))
        cuts = np.flatnonzero(np.diff(path)) + 1  # boundaries where the latent state changes
        runs = np.split(np.arange(len(path)), cuts)
        return np.stack([self._pool(atoms, run) for run in runs]).astype(np.float32)

    def _pool(self, atoms: np.ndarray, run: np.ndarray) -> np.ndarray:
        if self.atomic.discrete:
            return np.bincount(atoms[run].astype(int), minlength=self.feat or 1).astype(np.float64) / len(run)
        return atoms[run].mean(axis=0)
