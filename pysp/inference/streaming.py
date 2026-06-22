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

from pysp.inference import seq_initialize
from pysp.stats import (
    seq_encode,
    validate_estimator_keys,
)
from pysp.stats.compute.pdist import ParameterEstimator, SequenceEncodableProbabilityDistribution

T = TypeVar("T")
E0 = TypeVar("E0")


from pysp.inference.estimation import _local_encoded_chunks, harmonic


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


class StreamingEstimator:
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
        validate_estimator_keys(estimator)
        self.estimator = estimator
        self.init_estimator = estimator if init_estimator is None else init_estimator
        self.schedule = harmonic(0.7) if schedule is None else schedule
        self.model = model
        self.init_p = init_p
        self.rng = RandomState() if rng is None else rng
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
            raise ValueError("StreamingEstimator.update requires data or enc_data.")
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

    def update(
        self, data: Sequence[T] | None = None, enc_data: list[tuple[int, E0]] | None = None
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

    def value(self):
        """Return the running sufficient-statistic payload."""
        return None if self.running_accumulator is None else self.running_accumulator.value()

    def reset(self) -> None:
        """Drop running statistics and fitted model state."""
        self.running_accumulator = None
        self.model = None
        self.nobs = 0.0
        self.step = 0


class IncrementalEstimator:
    """Neal-Hinton style incremental EM over replaceable data chunks.

    Each chunk contributes a sufficient-statistic payload computed under the
    current model.  Revisiting a chunk subtracts that chunk's previous payload,
    adds the new payload, and runs the ordinary estimator M-step on the pooled
    statistics.  No distribution-specific estimation code lives here; the class
    only uses ``scale(-1)``, ``combine()``, and ``estimate()``.
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
        self.rng = RandomState() if rng is None else rng
        self.encoder = encoder if encoder is not None else (model.dist_to_encoder() if model is not None else None)
        self.num_chunks = num_chunks
        self.running_accumulator = None
        self.chunk_values = dict()
        self.nobs_by_chunk = dict()
        self.nobs = 0.0
        self.step = 0

    def _encode_batch(self, data, enc_data):
        if enc_data is not None:
            if hasattr(enc_data, "as_seq_chunk"):
                return [enc_data.as_seq_chunk()]
            return enc_data
        if data is None:
            raise ValueError("IncrementalEstimator.update requires data or enc_data.")
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

    def update(
        self, chunk_id: Any, data: Sequence[T] | None = None, enc_data: list[tuple[int, E0]] | None = None
    ) -> SequenceEncodableProbabilityDistribution:
        """Replace one chunk contribution and return the updated model."""
        if chunk_id is None:
            raise ValueError("IncrementalEstimator.update requires a non-None chunk_id.")
        enc_batch = self._encode_batch(data, enc_data)
        self._ensure_model(enc_batch)
        batch_nobs, batch_acc = streaming_accumulate(enc_batch, self.estimator, self.model)

        if self.running_accumulator is None:
            self.running_accumulator = self.estimator.accumulator_factory().make()

        if chunk_id in self.chunk_values:
            old_acc = self.estimator.accumulator_factory().make()
            old_acc.from_value(copy.deepcopy(self.chunk_values[chunk_id]))
            old_acc.scale(-1.0)
            self.running_accumulator.combine(old_acc.value())
            self.nobs -= self.nobs_by_chunk[chunk_id]

        self.running_accumulator.combine(batch_acc.value())
        self.nobs += batch_nobs
        self.chunk_values[chunk_id] = copy.deepcopy(batch_acc.value())
        self.nobs_by_chunk[chunk_id] = batch_nobs
        self.model = self.estimator.estimate(self.nobs, self.running_accumulator.value())
        self.step += 1
        return self.model

    def value(self):
        """Return the current pooled sufficient-statistic payload."""
        return None if self.running_accumulator is None else self.running_accumulator.value()

    def chunk_value(self, chunk_id: Any):
        """Return a copy of one stored chunk contribution."""
        if chunk_id not in self.chunk_values:
            raise KeyError(chunk_id)
        return copy.deepcopy(self.chunk_values[chunk_id])

    def reset(self) -> None:
        """Drop all chunk contributions and fitted model state."""
        self.running_accumulator = None
        self.chunk_values = dict()
        self.nobs_by_chunk = dict()
        self.model = None
        self.nobs = 0.0
        self.step = 0
