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
target score. That is exact whenever no unpulled sequence out-scores the returned ones.

Soundness (the ``certified`` verdict on :class:`RescoreResult`): the draft index's rank order is only
guaranteed *up to its own quantized fine-bucket width* (:class:`~mixle.enumeration.seek_index.SeekIndex`
and :class:`~mixle.enumeration.envelope.AREnvelopeIndex` are both built over a
:class:`~mixle.enumeration.quantization.core.Quantizer`) -- so the last PULLED candidate's own exact
draft score does NOT bound every unpulled one: an unpulled candidate can share the edge item's quantized
bucket, or even score higher, while still being ranked later (MXR-080-0233). The sound bound instead
comes from the draft index's OWN quantizer: every candidate ranked at or after the pulled window has a
fine bucket at least as deep as the edge item's, so that bucket's own best-case (least negative) log-prob
edge -- strictly looser than the edge item's exact score, never tighter -- is a certified upper bound on
any unpulled candidate's draft score. ``certified`` is ``True`` only when the worst target score in the
returned set provably clears ``bound + assumed_gap``; ``False`` when certification was requested but
could not be established (no sound bound available -- e.g. the draft index exposes no ``quantizer`` -- or
the bound did not clear the gap); ``None`` when no ``assumed_gap`` was supplied at all, so certification
was never attempted. The observed ``gap`` is always reported as a diagnostic, certified or not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["RescoredIndex", "RescoreResult"]

_LOG2 = math.log(2.0)


@dataclass
class RescoreResult:
    """Outcome of a :class:`RescoredIndex` query: target-exact items plus an explicit certificate.

    ``top_k``/``slice`` are speculative: the returned items are always target-exact and
    target-ordered, but completeness (whether some unpulled candidate could outscore them) is only
    provable under an ``assumed_gap`` AND a draft index that can supply a sound bound -- a caller
    that needs the proof, not just a plausible answer, must check ``certified is True`` rather than
    trusting ``items`` alone (MXR-080-0233: a bare boolean bolted onto an unqualified value cannot
    make that distinction).

    Attributes:
        items: ``(sequence, target_log_prob)`` pairs, target-exact and sorted by target score.
        certified: ``None`` when no ``assumed_gap`` was supplied to the index -- certification was
            never attempted and ``items`` is a plausible, uncertified answer. ``True`` only when
            ``bound`` was established and the returned set's worst target score provably clears
            every unpulled candidate (``bound + assumed_gap``). ``False`` when certification was
            requested but could not be proven -- either the draft index offered no sound bound, or
            the bound did not clear the gap; ``items`` is then still the best-effort draft/window
            answer, never a proof.
        bound: the certified upper bound on any unpulled candidate's draft log-probability that the
            certificate was checked against, or ``None`` when no bound applies -- nothing was left
            unpulled (the draft support was exhausted), no ``assumed_gap`` was supplied, or the
            draft index could not supply a sound bound.
        gap: running max observed ``|target_lp - draft_lp|`` over everything rescored on this index
            so far -- a diagnostic of what the gap has actually been, never a certificate.
    """

    items: list[tuple[tuple, float]]
    certified: bool | None
    bound: float | None
    gap: float


class RescoredIndex:
    """Draft-ordered, target-scored enumeration with window reranking.

    Args:
        draft_index: any index with ``unrank(i) -> (sequence, draft_log_prob)`` -- a
            :class:`~mixle.enumeration.seek_index.SeekIndex` over a low-cost
            :class:`~mixle.enumeration.autoregressive.AutoregressiveEnumerable`, an
            :class:`~mixle.enumeration.envelope.AREnvelopeIndex`, or anything equivalent. Sound
            certification additionally requires a ``.quantizer`` attribute exposing the
            :class:`~mixle.enumeration.quantization.core.Quantizer` the index was built with --
            both accepted index types have one; without it, queries still work but never certify.
        target: the expensive model -- an :class:`AutoregressiveEnumerable` (its
            :meth:`score_sequences` batch scorer is used) or a bare callable ``[seqs] -> log_probs``.
        rerank_window: extra draft candidates pulled around a query and reranked by target score.
            Larger = more robust to draft/target disagreement, one batched forward either way. Must
            be a non-negative integer -- a negative window used to silently shrink or empty the
            pulled set while still reporting a falsely certified result (MXR-080-0233).
        assumed_gap: optional global bound on ``|target_lp - draft_lp|`` (nats). Must be finite and
            non-negative. When supplied, results carry a genuine ``certified`` verdict (see
            :class:`RescoreResult`); otherwise ``certified`` is ``None`` and the observed ``gap`` is
            reported as a diagnostic only.
    """

    def __init__(
        self,
        draft_index: Any,
        target: Any,
        *,
        rerank_window: int = 64,
        assumed_gap: float | None = None,
    ) -> None:
        rerank_window = int(rerank_window)
        if rerank_window < 0:
            # MXR-080-0233: a negative window used to make n = k + rerank_window collapse to <= 0,
            # so `range(n)` pulled NOTHING and top_k/slice still returned certified=True on an empty
            # result -- reject it outright rather than silently degrading the pulled set.
            raise ValueError(f"rerank_window must be non-negative, got {rerank_window!r}")
        if assumed_gap is not None:
            assumed_gap = float(assumed_gap)
            if not math.isfinite(assumed_gap) or assumed_gap < 0:
                # MXR-080-0233: NaN and negative gaps used to pass straight through -- a NaN gap
                # poisons every certify comparison into a silent (always-False) no-op, and a
                # negative gap shrinks the required margin below what an absolute-value bound on
                # |target_lp - draft_lp| could ever honestly mean, letting weaker evidence certify.
                raise ValueError(f"assumed_gap must be finite and non-negative, got {assumed_gap!r}")
        self.draft_index = draft_index
        self._score = target.score_sequences if hasattr(target, "score_sequences") else target
        self.rerank_window = rerank_window
        self.assumed_gap = assumed_gap
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

    def _sound_draft_bound(self, edge_draft_lp: float) -> float | None:
        """A certified upper bound on the draft log-prob of any candidate ranked at/after the edge.

        The draft index's rank order only guarantees non-decreasing *fine-bucket* index (the
        :class:`~mixle.enumeration.quantization.core.Quantizer` both accepted index types are built
        over), never non-increasing exact log-prob -- so the edge item's own exact draft score is
        NOT itself a bound on what comes after it (MXR-080-0233): a later-ranked candidate can share
        the edge item's bucket, or even score higher, while its bucket index is still >= the edge's.
        Every candidate ranked at/after the edge therefore has a fine bucket >= the edge item's own,
        hence bits >= that bucket's floor, hence log-prob <= that bucket's own best-case (least
        negative) edge -- a bound that is provably looser than the edge item's exact score, which is
        exactly the honest slack the old "use the edge score itself" premise was missing.

        Requires ``draft_index.quantizer`` -- the ``Quantizer`` the index itself was built with, as
        exposed by both :class:`~mixle.enumeration.seek_index.SeekIndex` and
        :class:`~mixle.enumeration.envelope.AREnvelopeIndex`/``LatticeEnvelopeIndex``. Returns
        ``None`` when unavailable, or when ``edge_draft_lp`` is non-finite (no bucket to bound from)
        -- never a guessed number standing in for an unproven one.
        """
        if not math.isfinite(edge_draft_lp):
            return None
        quantizer = getattr(self.draft_index, "quantizer", None)
        bin_width_bits = getattr(quantizer, "bin_width_bits", None)
        oversample = getattr(quantizer, "oversample", None)
        fine_bucket = getattr(quantizer, "fine_bucket", None)
        if quantizer is None or bin_width_bits is None or oversample is None or not callable(fine_bucket):
            return None
        fb = fine_bucket(edge_draft_lp)
        bucket_width_bits = float(bin_width_bits) / float(oversample)
        return -float(fb) * bucket_width_bits * _LOG2

    def _certify(self, kth_target_lp: float, edge_draft_lp: float | None) -> tuple[bool | None, float | None]:
        """``(certified, bound)`` -- sound only under ``assumed_gap`` AND a real draft-index bound.

        ``bound`` (see :meth:`_sound_draft_bound`) is a certified upper bound on every unpulled
        candidate's draft score; under the global ``assumed_gap`` its target score is then at most
        ``bound + assumed_gap``, so the k-th returned item clearing that sum proves the top-k.
        """
        if self.assumed_gap is None:
            return None, None
        if edge_draft_lp is None:  # the draft support was exhausted: nothing unpulled exists
            return True, None
        bound = self._sound_draft_bound(edge_draft_lp)
        if bound is None:  # no sound bound available: never claim a proof this index cannot back
            return False, None
        return bool(kth_target_lp >= bound + self.assumed_gap), bound

    def _empty_result(self) -> RescoreResult:
        """The (rare) result when the draft index has no items at all: vacuously complete."""
        certified, bound = (None, None) if self.assumed_gap is None else (True, None)
        return RescoreResult(items=[], certified=certified, bound=bound, gap=self.observed_gap)

    # -- queries -------------------------------------------------------------------------------------------

    def top_k(self, k: int) -> RescoreResult:
        """The ``k`` best sequences by TARGET score among the ``k + rerank_window`` draft head.

        Returns a :class:`RescoreResult` -- target-exact scores, draft+window-approximate
        completeness (see the class/module docstrings for exactly what ``certified`` proves).
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        n = k + self.rerank_window
        seqs, draft_lps, target_lps = self._pull(n)
        if not seqs:
            return self._empty_result()
        order = np.argsort(-target_lps, kind="stable")[: min(k, len(seqs))]
        items = [(seqs[i], float(target_lps[i])) for i in order.tolist()]
        exhausted = len(seqs) < n
        edge = None if exhausted else float(draft_lps[-1])
        certified, bound = self._certify(items[-1][1], edge)
        return RescoreResult(items=items, certified=certified, bound=bound, gap=self.observed_gap)

    def slice(self, start: int, k: int) -> RescoreResult:
        """Target-reranked ``[start, start + k)`` slice of the pulled ``start + k + rerank_window`` head.

        Same semantics as :meth:`top_k`: order within the pulled set is target-exact; the certificate
        covers whether an unpulled sequence could belong in (or before) the slice.
        """
        if start < 0 or k < 1:
            raise ValueError("start must be >= 0 and k >= 1")
        n = start + k + self.rerank_window
        seqs, draft_lps, target_lps = self._pull(n)
        if not seqs:
            return self._empty_result()
        order = np.argsort(-target_lps, kind="stable")
        window = order[start : start + k]
        items = [(seqs[i], float(target_lps[i])) for i in window.tolist()]
        exhausted = len(seqs) < n
        edge = None if exhausted else float(draft_lps[-1])
        boundary = items[-1][1] if items else float("inf")
        certified, bound = self._certify(boundary, edge)
        return RescoreResult(items=items, certified=certified, bound=bound, gap=self.observed_gap)

    def unrank(self, i: int) -> tuple[tuple, float]:
        """The draft's rank-``i`` sequence with the TARGET's exact log-probability.

        The rank coordinate is the draft's (no reranking): the low-cost random-access primitive. Use
        :meth:`top_k` / :meth:`slice` when local target-order matters.
        """
        seq, _draft_lp = self.draft_index.unrank(i)
        lp = float(np.asarray(self._score([tuple(seq)]), dtype=float).reshape(-1)[0])
        self.target_forig_calls += 1
        return tuple(seq), lp
