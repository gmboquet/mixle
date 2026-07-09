"""Speculative enumeration: build the index with a low-cost DRAFT model, score results with the TARGET.

Speculative decoding's economics applied to enumeration. Building any autoregressive index costs one
forward per live prefix -- prohibitive when the model is a large transformer. But the *ordering* work
(which sequences are near a rank/threshold) tolerates approximation, while the *scores* must be the real
model's. So: let a low-cost draft (an n-gram, a distilled student, a quantized twin) pay for the tree or
envelope build, and touch the target only for the sequences a query actually returns -- one batched
teacher-forcing forward for all of them (:meth:`AutoregressiveEnumerable.score_sequences`).

Contract: every returned ``log_prob`` is the **target's exact** score. The *order* is
draft-approximate, repaired locally by window reranking: ``top_k(k)`` / ``slice`` pull
``k + rerank_window`` draft-ordered candidates, rescore them all with the target in one batch, and sort by
target score. That is exact whenever no unpulled sequence out-scores the returned ones -- guaranteed if the
draft-to-target log-prob gap is globally bounded by ``assumed_gap`` and the window edge clears it (the
``certified`` flag); without an assumed bound the observed ``gap`` diagnostic is reported and the
certificate is reported as ``None``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["RescoredIndex"]


class RescoredIndex:
    """Draft-ordered, target-scored enumeration with window reranking.

    Args:
        draft_index: any index with ``unrank(i) -> (sequence, draft_log_prob)`` -- a
            :class:`~mixle.enumeration.seek_index.SeekIndex` over a low-cost
            :class:`~mixle.enumeration.autoregressive.AutoregressiveEnumerable`, an
            :class:`~mixle.enumeration.envelope.AREnvelopeIndex`, or anything equivalent.
        target: the expensive model -- an :class:`AutoregressiveEnumerable` (its
            :meth:`score_sequences` batch scorer is used) or a bare callable ``[seqs] -> log_probs``.
        rerank_window: extra draft candidates pulled around a query and reranked by target score.
            Larger = more robust to draft/target disagreement, one batched forward either way.
        assumed_gap: optional global bound on ``|target_lp - draft_lp|`` (nats). When supplied, results
            carry a sound ``certified`` verdict; otherwise ``certified`` is ``None`` and the observed
            ``gap`` is reported as a diagnostic.
    """

    def __init__(
        self,
        draft_index: Any,
        target: Any,
        *,
        rerank_window: int = 64,
        assumed_gap: float | None = None,
    ) -> None:
        self.draft_index = draft_index
        self._score = target.score_sequences if hasattr(target, "score_sequences") else target
        self.rerank_window = int(rerank_window)
        self.assumed_gap = None if assumed_gap is None else float(assumed_gap)
        self.observed_gap: float = 0.0  # running max |target - draft| over everything rescored
        self.target_forig_calls: int = 0  # batched target scoring calls (the cost being economized)

    # -- internals -----------------------------------------------------------------------------------------

    def _pull(self, n: int) -> tuple[list[tuple], np.ndarray, np.ndarray]:
        """First ``n`` draft-ordered sequences with draft and (batch-rescored) target scores."""
        seqs: list[tuple] = []
        draft_lps: list[float] = []
        for i in range(n):
            try:
                seq, dlp = self.draft_index.unrank(i)
            except IndexError:
                break  # draft support exhausted: everything is pulled
            seqs.append(tuple(seq))
            draft_lps.append(float(dlp))
        if not seqs:
            return [], np.zeros(0), np.zeros(0)
        target_lps = np.asarray(self._score(seqs), dtype=float).reshape(len(seqs))
        self.target_forig_calls += 1
        draft_arr = np.asarray(draft_lps, dtype=float)
        finite = np.isfinite(target_lps) & np.isfinite(draft_arr)
        if finite.any():
            self.observed_gap = max(self.observed_gap, float(np.max(np.abs(target_lps[finite] - draft_arr[finite]))))
        return seqs, draft_arr, target_lps

    def _certify(self, kth_target_lp: float, edge_draft_lp: float | None) -> bool | None:
        """Sound only under ``assumed_gap``: every unpulled draft item scores below the window edge, so its
        target score is below ``edge + gap``; the k-th returned item clearing that bound proves the top-k."""
        if self.assumed_gap is None:
            return None
        if edge_draft_lp is None:  # the draft support was exhausted: nothing unpulled exists
            return True
        return bool(kth_target_lp >= edge_draft_lp + self.assumed_gap)

    # -- queries -------------------------------------------------------------------------------------------

    def top_k(self, k: int) -> dict[str, Any]:
        """The ``k`` best sequences by TARGET score among the ``k + rerank_window`` draft head.

        Returns ``{"items": [(seq, target_lp), ...], "certified": bool | None, "gap": float}`` --
        target-exact scores, draft+window-approximate completeness (see the class docstring).
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        n = k + self.rerank_window
        seqs, draft_lps, target_lps = self._pull(n)
        if not seqs:
            return {"items": [], "certified": True, "gap": self.observed_gap}
        order = np.argsort(-target_lps, kind="stable")[: min(k, len(seqs))]
        items = [(seqs[i], float(target_lps[i])) for i in order.tolist()]
        exhausted = len(seqs) < n
        edge = None if exhausted else float(draft_lps[-1])
        certified = self._certify(items[-1][1], edge)
        return {"items": items, "certified": certified, "gap": self.observed_gap}

    def slice(self, start: int, k: int) -> dict[str, Any]:
        """Target-reranked ``[start, start + k)`` slice of the pulled ``start + k + rerank_window`` head.

        Same semantics as :meth:`top_k`: order within the pulled set is target-exact; the certificate
        covers whether an unpulled sequence could belong in (or before) the slice.
        """
        if start < 0 or k < 1:
            raise ValueError("start must be >= 0 and k >= 1")
        n = start + k + self.rerank_window
        seqs, draft_lps, target_lps = self._pull(n)
        if not seqs:
            return {"items": [], "certified": True, "gap": self.observed_gap}
        order = np.argsort(-target_lps, kind="stable")
        window = order[start : start + k]
        items = [(seqs[i], float(target_lps[i])) for i in window.tolist()]
        exhausted = len(seqs) < n
        edge = None if exhausted else float(draft_lps[-1])
        boundary = items[-1][1] if items else float("inf")
        certified = self._certify(boundary, edge)
        return {"items": items, "certified": certified, "gap": self.observed_gap}

    def unrank(self, i: int) -> tuple[tuple, float]:
        """The draft's rank-``i`` sequence with the TARGET's exact log-probability.

        The rank coordinate is the draft's (no reranking): the low-cost random-access primitive. Use
        :meth:`top_k` / :meth:`slice` when local target-order matters.
        """
        seq, _draft_lp = self.draft_index.unrank(i)
        lp = float(np.asarray(self._score([tuple(seq)]), dtype=float).reshape(-1)[0])
        self.target_forig_calls += 1
        return tuple(seq), lp
