"""Online / streaming estimation over batched sufficient statistics.

``streaming_accumulate`` folds one batch's sufficient statistics; ``StreamingEstimator`` and
``IncrementalEstimator`` drive incremental fitting across a stream of batches (with optional
forgetting/step schedules). Distinct from the Bayesian ``BayesianStreamingEstimator`` in
``estimation.py``.
"""

import copy
from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.random import RandomState

from mixle.stats.compute.pdist import (
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    validate_estimator_keys,
)

# Import seq_initialize from its canonical home rather than the mixle.inference package re-export:
# streaming.py is itself imported by mixle.inference.__init__, so depending on the package surface here
# would be a load-order-fragile cycle.
from mixle.stats.compute.sequence import seq_encode, seq_initialize

T = TypeVar("T")
E0 = TypeVar("E0")


from mixle.inference.estimation import _local_encoded_chunks, harmonic


def streaming_accumulate(
    enc_data: Any, estimator: ParameterEstimator, model: SequenceEncodableProbabilityDistribution
) -> tuple[float, Any]:
    """Return one batch's globally tied sufficient-stat accumulator.

    Encoded-data handles can implement ``pysp_stream_accumulate`` to do the
    local/distributed fold themselves.  Plain encoded chunks use the legacy
    in-process ``seq_update`` loop.
    """
    validate_estimator_keys(estimator)
    if hasattr(enc_data, "pysp_stream_accumulate"):
        nobs, value = enc_data.pysp_stream_accumulate(estimator, model)
        return nobs, estimator.accumulator_factory().make().from_value(value)

    chunks = _local_encoded_chunks(enc_data)
    acc = estimator.accumulator_factory().make()
    nobs = 0.0
    for sz, enc in chunks:
        nobs += sz
        acc.seq_update(enc, np.ones(sz), model)
    stats_dict = dict()
    acc.key_merge(stats_dict)
    acc.key_replace(stats_dict)
    return nobs, acc


class _StreamingBase:
    """Shared plumbing for the online estimators: construction, batch encoding, lazy model init.

    Subclasses differ only in how :meth:`update` folds a batch into the running statistics
    (decayed step schedule vs. Neal-Hinton chunk replacement). Everything else -- the estimator/
    model/encoder state, ``value``, ``reset`` -- is common and lives here.
    """

    def __init__(
        self,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution | None = None,
        init_estimator: ParameterEstimator | None = None,
        init_p: float = 0.1,
        rng: RandomState | None = None,
        encoder=None,
        num_chunks: int = 1,
    ) -> None:
        validate_estimator_keys(estimator)
        self.estimator = estimator
        self.init_estimator = estimator if init_estimator is None else init_estimator
        self.model = model
        self.init_p = init_p
        self.rng = RandomState(0) if rng is None else rng  # fixed default: an un-seeded fit is deterministic
        self.encoder = encoder if encoder is not None else (model.dist_to_encoder() if model is not None else None)
        self.num_chunks = num_chunks
        self.running_accumulator = None
        self.nobs = 0.0
        self.step = 0

    def _encode_batch(self, data, enc_data):
        if enc_data is not None:
            if hasattr(enc_data, "as_seq_chunk"):
                return [enc_data.as_seq_chunk()]
            return enc_data
        if data is None:
            raise ValueError("%s.update requires data or enc_data." % type(self).__name__)
        if self.encoder is None:
            self.encoder = (
                self.model.dist_to_encoder()
                if self.model is not None
                else self.init_estimator.accumulator_factory().make().acc_to_encoder()
            )
        return seq_encode(data, encoder=self.encoder, num_chunks=self.num_chunks)

    def _ensure_model(self, enc_data):
        if self.model is None:
            p = min(max(self.init_p, 0.0), 1.0) if self.init_p > 0.0 else 0.1
            self.model = seq_initialize(enc_data, self.init_estimator, self.rng, p)
            self.encoder = self.model.dist_to_encoder()

    def value(self):
        """Return the running sufficient-statistic payload."""
        return None if self.running_accumulator is None else self.running_accumulator.value()

    def reset(self) -> None:
        """Drop running statistics and fitted model state."""
        self.running_accumulator = None
        self.model = None
        self.nobs = 0.0
        self.step = 0


class StreamingEstimator(_StreamingBase):
    """Decay-mode online estimator built from accumulator scaling and M-steps."""

    def __init__(
        self,
        estimator: ParameterEstimator,
        schedule=None,
        model: SequenceEncodableProbabilityDistribution | None = None,
        init_estimator: ParameterEstimator | None = None,
        init_p: float = 0.1,
        rng: RandomState | None = None,
        encoder=None,
        num_chunks: int = 1,
    ) -> None:
        super().__init__(
            estimator,
            model=model,
            init_estimator=init_estimator,
            init_p=init_p,
            rng=rng,
            encoder=encoder,
            num_chunks=num_chunks,
        )
        self.schedule = harmonic(0.7) if schedule is None else schedule

    def update(
        self, data: Sequence[T] | None = None, *, enc_data: list[tuple[int, E0]] | None = None
    ) -> SequenceEncodableProbabilityDistribution:
        """Consume one batch and return the updated model."""
        enc_batch = self._encode_batch(data, enc_data)
        self._ensure_model(enc_batch)
        batch_nobs, batch_acc = streaming_accumulate(enc_batch, self.estimator, self.model)

        if self.running_accumulator is None:
            self.running_accumulator = batch_acc
            self.nobs = batch_nobs
        else:
            rho = float(self.schedule(self.step + 1))
            if rho <= 0.0 or rho > 1.0:
                raise ValueError("streaming schedule returned %r; expected 0 < rho <= 1." % rho)
            self.running_accumulator.scale(1.0 - rho)
            batch_acc.scale(rho)
            self.running_accumulator.combine(batch_acc.value())
            self.nobs = (1.0 - rho) * self.nobs + rho * batch_nobs

        self.model = self.estimator.estimate(self.nobs, self.running_accumulator.value())
        self.step += 1
        return self.model


def _max_cancelled_bits(pre: Any, post: Any) -> float:
    """Worst per-element magnitude collapse, in bits, between two payload snapshots.

    ``log2(|pre| / |post|)`` measures how many leading bits a running statistic lost
    when a payload was subtracted out of it -- the cheap form of the cancellation
    escalation signal described in :mod:`mixle.engines.affine` (a subtraction at a
    cancellation point blows up the relative-error bound).  Non-numeric leaves and
    structure mismatches contribute 0: the guard may under-fire but never breaks
    the update path.
    """
    if isinstance(pre, dict) and isinstance(post, dict):
        return max((_max_cancelled_bits(pre[k], post[k]) for k in pre if k in post), default=0.0)
    if isinstance(pre, (list, tuple)) and isinstance(post, (list, tuple)):
        return max((_max_cancelled_bits(a, b) for a, b in zip(pre, post)), default=0.0)
    if pre is None or post is None:
        return 0.0
    try:
        a = np.abs(np.asarray(pre, dtype=np.float64))
        b = np.abs(np.asarray(post, dtype=np.float64))
    except (TypeError, ValueError):
        return 0.0
    if a.shape != b.shape or a.size == 0:
        return 0.0
    mask = a > 0.0
    if not bool(np.any(mask)):
        return 0.0
    with np.errstate(divide="ignore"):
        bits = np.log2(a[mask] / b[mask])
    return float(np.max(bits))


class IncrementalEstimator(_StreamingBase):
    """Neal-Hinton style incremental EM over replaceable data chunks.

    Each chunk contributes a sufficient-statistic payload computed under the
    current model.  Revisiting a chunk subtracts that chunk's previous payload,
    adds the new payload, and runs the ordinary estimator M-step on the pooled
    statistics.  No distribution-specific estimation code lives here; the class
    only uses ``scale(-1)``, ``combine()``, and ``estimate()``.

    The subtract step is O(1) per revisit, but floating-point subtraction does not
    invert addition: every revisit leaves roundoff residue in the running
    statistics, and removing a payload whose magnitude dominated the running sums
    can cancel catastrophically (e.g. the pooled second moment driven negative
    after smaller chunks were absorbed at add time).  Because every chunk payload
    is retained in ``chunk_values``, the exact state is always recoverable:
    :meth:`rebase` rebuilds the running statistics by combining the stored
    payloads in canonical order.  ``rebase_every=n`` does so automatically every
    ``n`` updates; ``cancellation_bits=b`` watches each subtract and auto-rebases
    when any statistic loses more than ``b`` bits of magnitude
    (``cancellation_rebases`` counts the triggers).  Both default to off, keeping
    the historical O(1) fast path unchanged.
    """

    def __init__(
        self,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution | None = None,
        init_estimator: ParameterEstimator | None = None,
        init_p: float = 0.1,
        rng: RandomState | None = None,
        encoder=None,
        num_chunks: int = 1,
        rebase_every: int | None = None,
        cancellation_bits: float | None = None,
    ) -> None:
        if rebase_every is not None and rebase_every < 1:
            raise ValueError("rebase_every must be a positive integer or None; got %r." % (rebase_every,))
        if cancellation_bits is not None and cancellation_bits <= 0.0:
            raise ValueError("cancellation_bits must be positive or None; got %r." % (cancellation_bits,))
        super().__init__(
            estimator,
            model=model,
            init_estimator=init_estimator,
            init_p=init_p,
            rng=rng,
            encoder=encoder,
            num_chunks=num_chunks,
        )
        self.rebase_every = rebase_every
        self.cancellation_bits = cancellation_bits
        self.cancellation_rebases = 0
        self.chunk_values = dict()
        self.nobs_by_chunk = dict()

    def update(
        self,
        data: Sequence[T] | None = None,
        *,
        enc_data: list[tuple[int, E0]] | None = None,
        chunk_id: Any = None,
    ) -> SequenceEncodableProbabilityDistribution:
        """Replace one chunk contribution and return the updated model.

        ``chunk_id`` is keyword-only so this matches :meth:`StreamingEstimator.update`'s
        ``(data, *, enc_data)`` shape across the streaming surface; it is required (a ``None``
        ``chunk_id`` raises) because the Neal-Hinton update keys each batch's contribution by it.
        """
        if chunk_id is None:
            raise ValueError("IncrementalEstimator.update requires a non-None chunk_id (pass chunk_id=...).")
        enc_batch = self._encode_batch(data, enc_data)
        self._ensure_model(enc_batch)
        batch_nobs, batch_acc = streaming_accumulate(enc_batch, self.estimator, self.model)

        if self.running_accumulator is None:
            self.running_accumulator = self.estimator.accumulator_factory().make()

        needs_rebase = False
        if chunk_id in self.chunk_values:
            pre_value = copy.deepcopy(self.running_accumulator.value()) if self.cancellation_bits is not None else None
            old_acc = self.estimator.accumulator_factory().make()
            old_acc.from_value(copy.deepcopy(self.chunk_values[chunk_id]))
            old_acc.scale(-1.0)
            self.running_accumulator.combine(old_acc.value())
            self.nobs -= self.nobs_by_chunk[chunk_id]
            if pre_value is not None and (
                _max_cancelled_bits(pre_value, self.running_accumulator.value()) > self.cancellation_bits
            ):
                needs_rebase = True

        self.running_accumulator.combine(batch_acc.value())
        self.nobs += batch_nobs
        self.chunk_values[chunk_id] = copy.deepcopy(batch_acc.value())
        self.nobs_by_chunk[chunk_id] = batch_nobs
        if needs_rebase:
            self.cancellation_rebases += 1
            self._rebase_stats()
        elif self.rebase_every is not None and (self.step + 1) % self.rebase_every == 0:
            self._rebase_stats()
        self.model = self.estimator.estimate(self.nobs, self.running_accumulator.value())
        self.step += 1
        return self.model

    def _canonical_chunk_ids(self) -> list:
        """Chunk ids in canonical (sorted) order; insertion order if ids are unorderable."""
        try:
            return sorted(self.chunk_values)
        except TypeError:
            return list(self.chunk_values)

    def _rebase_stats(self) -> None:
        """Rebuild running statistics exactly from the stored chunk payloads."""
        ids = self._canonical_chunk_ids()
        acc = self.estimator.accumulator_factory().make()
        acc.from_value(copy.deepcopy(self.chunk_values[ids[0]]))
        for cid in ids[1:]:
            acc.combine(copy.deepcopy(self.chunk_values[cid]))
        self.running_accumulator = acc
        nobs = 0.0
        for cid in ids:
            nobs += self.nobs_by_chunk[cid]
        self.nobs = nobs

    def rebase(self) -> SequenceEncodableProbabilityDistribution | None:
        """Recompute the running statistics from the stored chunk payloads and re-estimate.

        The subtract-based fast path in :meth:`update` accumulates roundoff residue
        (and can cancel catastrophically); this rebuilds the exact canonical-order
        reduction of ``chunk_values`` -- identical to a from-scratch reduce over the
        current chunk set -- and refreshes the model from it.  No data re-encoding
        happens; cost is one ``combine`` per stored chunk plus one M-step.  A no-op
        when no chunks have been folded yet.
        """
        if not self.chunk_values:
            return self.model
        self._rebase_stats()
        self.model = self.estimator.estimate(self.nobs, self.running_accumulator.value())
        return self.model

    def chunk_value(self, chunk_id: Any):
        """Return a copy of one stored chunk contribution."""
        if chunk_id not in self.chunk_values:
            raise KeyError(chunk_id)
        return copy.deepcopy(self.chunk_values[chunk_id])

    def reset(self) -> None:
        """Drop all chunk contributions and fitted model state."""
        super().reset()
        self.chunk_values = dict()
        self.nobs_by_chunk = dict()
        self.cancellation_rebases = 0
