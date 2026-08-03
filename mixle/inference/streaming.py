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


def _finite_count(value: Any, name: str) -> float:
    """Return ``value`` as a finite non-negative float count.

    ``streaming_accumulate`` may source ``nobs`` from a remote/distributed handle, so the count is
    checked before it can scale an accumulator or the running ``nobs``.
    """
    try:
        count = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("%s must be a real number, got %r" % (name, value)) from exc
    if not np.isfinite(count) or count < 0.0:
        raise ValueError("%s must be finite and non-negative, got %r" % (name, count))
    return count


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
        dataset_size: float | None = None,
    ) -> None:
        validate_estimator_keys(estimator)
        self.estimator = estimator
        self.init_estimator = estimator if init_estimator is None else init_estimator
        self.model = model
        self.init_p = init_p
        self.rng = RandomState(0) if rng is None else rng  # fixed default: an un-seeded fit is deterministic
        self.encoder = encoder if encoder is not None else (model.dist_to_encoder() if model is not None else None)
        self.num_chunks = num_chunks
        self.dataset_size = None if dataset_size is None else float(dataset_size)
        if self.dataset_size is not None and not (np.isfinite(self.dataset_size) and self.dataset_size > 0.0):
            raise ValueError("dataset_size must be positive and finite when supplied.")
        self.last_batch_scale = 1.0
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
        self.last_batch_scale = 1.0


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
        dataset_size: float | None = None,
    ) -> None:
        super().__init__(
            estimator,
            model=model,
            init_estimator=init_estimator,
            init_p=init_p,
            rng=rng,
            encoder=encoder,
            num_chunks=num_chunks,
            dataset_size=dataset_size,
        )
        self.schedule = harmonic(0.7) if schedule is None else schedule

    def update(
        self, data: Sequence[T] | None = None, *, enc_data: list[tuple[int, E0]] | None = None
    ) -> SequenceEncodableProbabilityDistribution:
        """Consume one batch and return the updated model.

        Every scale and count is validated finite *before* it touches the running accumulator, and the
        running state is committed only once the M-step has succeeded -- so a failed update leaves the
        estimator exactly where it was and the caller can retry.
        """
        enc_batch = self._encode_batch(data, enc_data)
        self._ensure_model(enc_batch)
        batch_nobs, batch_acc = streaming_accumulate(enc_batch, self.estimator, self.model)
        batch_nobs = _finite_count(batch_nobs, "streaming batch nobs")
        effective_nobs = batch_nobs
        batch_scale = 1.0
        if self.dataset_size is not None:
            if batch_nobs <= 0.0:
                raise ValueError("cannot scale an empty minibatch to dataset_size.")
            batch_scale = self.dataset_size / batch_nobs
            batch_acc.scale(batch_scale)
            effective_nobs = self.dataset_size

        if self.running_accumulator is None:
            # nothing to roll back yet: the running state stays empty until the M-step succeeds
            model = self.estimator.estimate(effective_nobs, batch_acc.value())
            self.running_accumulator = batch_acc
            self.nobs = effective_nobs
        else:
            rho = float(self.schedule(self.step + 1))
            if not np.isfinite(rho) or rho <= 0.0 or rho > 1.0:
                raise ValueError("streaming schedule returned %r; expected a finite 0 < rho <= 1." % rho)
            previous_value = copy.deepcopy(self.running_accumulator.value())
            updated_nobs = (1.0 - rho) * self.nobs + rho * effective_nobs
            try:
                self.running_accumulator.scale(1.0 - rho)
                batch_acc.scale(rho)
                self.running_accumulator.combine(batch_acc.value())
                model = self.estimator.estimate(updated_nobs, self.running_accumulator.value())
            except Exception:
                self.running_accumulator.from_value(previous_value)
                raise
            self.nobs = updated_nobs

        self.last_batch_scale = batch_scale
        self.model = model
        self.step += 1
        return self.model


class IncrementalEstimator(_StreamingBase):
    """Neal-Hinton style incremental EM over replaceable data chunks.

    Each chunk contributes a sufficient-statistic payload computed under the
    current model. Revisiting a chunk replaces that payload and rebuilds the
    pooled accumulator from the stored chunk summaries before running the
    ordinary M-step. Rebuilding is essential because mergeable statistics form
    a monoid, not necessarily a group: sums can be subtracted, but support
    minima/maxima, weighted medians, sketches, and similar summaries cannot.
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
        super().__init__(
            estimator,
            model=model,
            init_estimator=init_estimator,
            init_p=init_p,
            rng=rng,
            encoder=encoder,
            num_chunks=num_chunks,
        )
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

        The chunk store, running statistics and model are committed only once the M-step has
        succeeded, so a failed update leaves the estimator exactly where it was -- in particular the
        previous payload for a revisited ``chunk_id`` survives -- and the caller can retry.
        """
        if chunk_id is None:
            raise ValueError("IncrementalEstimator.update requires a non-None chunk_id (pass chunk_id=...).")
        enc_batch = self._encode_batch(data, enc_data)
        self._ensure_model(enc_batch)
        batch_nobs, batch_acc = streaming_accumulate(enc_batch, self.estimator, self.model)
        batch_nobs = _finite_count(batch_nobs, "incremental batch nobs")

        # Stage, then commit (MXR-080-1899). The chunk store used to be overwritten *before* the
        # pooled rebuild and the M-step, so an estimator that refused the new state was left holding
        # a chunk contribution that was never accepted: revisiting chunk "a" with a bad batch
        # destroyed "a"'s previous good payload, advanced `nobs` and the running accumulator to
        # include the rejected batch, and left `self.model` behind at the value from before -- an
        # internally inconsistent estimator whose very next successful update silently baked the
        # rejected batch in. StreamingEstimator.update already promised exactly this ("a failed
        # update leaves the estimator exactly where it was and the caller can retry"); this is the
        # sibling path that did not.
        replacing = chunk_id in self.chunk_values
        staged_value = copy.deepcopy(batch_acc.value())
        # The append path folds the batch into the LIVE accumulator, so its pre-update value is
        # captured for rollback; the replace path pools into a fresh accumulator and needs no copy.
        previous_running_value = (
            None if replacing or self.running_accumulator is None else copy.deepcopy(self.running_accumulator.value())
        )
        try:
            if replacing:
                # Rebuild into a FRESH accumulator (mergeable statistics are a monoid, not a group --
                # see the class docstring), substituting the staged payload for the revisited chunk so
                # the stored one survives untouched until the M-step has succeeded.
                pooled = self.estimator.accumulator_factory().make()
                pooled_nobs = 0.0
                for stored_id, value in self.chunk_values.items():
                    contribution = staged_value if stored_id == chunk_id else value
                    pooled.combine(copy.deepcopy(contribution))
                    pooled_nobs += batch_nobs if stored_id == chunk_id else self.nobs_by_chunk[stored_id]
            else:
                pooled = (
                    self.estimator.accumulator_factory().make()
                    if self.running_accumulator is None
                    else self.running_accumulator
                )
                pooled.combine(batch_acc.value())
                pooled_nobs = self.nobs + batch_nobs
            model = self.estimator.estimate(pooled_nobs, pooled.value())
        except Exception:
            # Only the append path mutates the live accumulator in place (the replace path builds a
            # fresh one), so that is the only state needing restoration -- and it is restored the
            # same way StreamingEstimator.update restores its own: from the value captured before
            # the combine. Everything else (chunk store, nobs, model, step) is still untouched
            # because it is written only in the commit block below.
            if not replacing and previous_running_value is not None:
                self.running_accumulator.from_value(previous_running_value)
            raise

        self.chunk_values[chunk_id] = staged_value
        self.nobs_by_chunk[chunk_id] = batch_nobs
        self.running_accumulator = pooled
        self.nobs = pooled_nobs
        self.model = model
        self.step += 1
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
