"""Named-record distributions for dict/DataFrame observations.

``RecordDistribution`` is a product distribution like ``CompositeDistribution``,
but its children are addressed by field name instead of tuple position.  It is
small on purpose: named observations reuse the same distribution, estimator,
accumulator, encoder, kernel, and engine protocols as the rest of
``mixle.stats``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mixle.enumeration.algorithms import BufferedStream, ProductEnumerator
from mixle.stats.compute.pdist import (
    DataSequenceEncoder,
    DistributionEnumerator,
    DistributionSampler,
    EnumerationError,
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    SequenceEncodableStatisticAccumulator,
    StatisticAccumulatorFactory,
    child_enumerator,
)

FieldSpec = Any


def field(name: Any, source: Any = None) -> tuple[Any, Any]:
    """Declare a named model field and the input key/column it reads.

    ``field('x')`` reads input key ``'x'`` and names the model field ``'x'``.
    ``field('x_copy', source='x')`` names a second model variable that reads
    the same input key, useful for dependence features and repeated views.
    """
    return name, name if source is None else source


def _field_name(spec: FieldSpec) -> Any:
    if isinstance(spec, tuple) and len(spec) == 2:
        return spec[0]
    return spec


def _field_source(spec: FieldSpec) -> Any:
    if isinstance(spec, tuple) and len(spec) == 2:
        return spec[1]
    return spec


def _normalize_fields(fields: Sequence[FieldSpec]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    specs = tuple(fields)
    return tuple(_field_name(spec) for spec in specs), tuple(_field_source(spec) for spec in specs)


def _split_fields(
    fields: Any, values: Sequence[Any] | None = None
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    if isinstance(fields, Mapping):
        if values is not None:
            raise TypeError("values must be omitted when fields is a mapping.")
        names, sources = _normalize_fields(tuple(fields.keys()))
        return names, sources, tuple(fields.values())
    if values is None:
        raise TypeError("values must be supplied when fields is not a mapping.")
    keys, sources = _normalize_fields(tuple(fields))
    vals = tuple(values)
    if len(keys) != len(vals):
        raise ValueError("field/value length mismatch: %d fields, %d values." % (len(keys), len(vals)))
    return keys, sources, vals


def _record_get(row: Any, source: Any) -> Any:
    if isinstance(row, Mapping):
        return row.get(source, None)
    return None


class RecordDistribution(SequenceEncodableProbabilityDistribution):
    """Product distribution over mapping records with a fixed field set."""

    def __init__(self, fields: Any, dists: Sequence[SequenceEncodableProbabilityDistribution] | None = None) -> None:
        self.fields, self.sources, self.dists = _split_fields(fields, dists)
        self.keys = self.fields
        self.count = len(self.fields)

    def compute_capabilities(self):
        """Return engine support inherited from all child distributions."""
        from mixle.stats.compute.capabilities import DistributionCapabilities, intersect_engine_ready

        return DistributionCapabilities(engine_ready=intersect_engine_ready(tuple(self.dists)), kernel_status="generic")

    def compute_declaration(self):
        """Return a child-role declaration for generated metadata consumers."""
        from mixle.stats.compute.declarations import DistributionDeclaration, StatisticSpec, declaration_for

        children = tuple(declaration_for(d) for d in self.dists)
        children = tuple(d for d in children if d is not None)
        return DistributionDeclaration(
            name="record",
            distribution_type=type(self),
            parameters=(),
            statistics=(StatisticSpec("fields", kind="mapping"),),
            support="mapping_record",
            children=children,
            child_roles=tuple(str(field) for field in self.fields[: len(children)]),
            differentiable=all(child.differentiable for child in children),
        )

    def __str__(self) -> str:
        pairs = ["%s: %s" % (repr(k), str(d)) for k, d in zip(self.fields, self.dists)]
        return "RecordDistribution({%s})" % ", ".join(pairs)

    def density(self, x: Mapping[Any, Any]) -> float:
        """Return probability density/mass for one mapping record."""
        return math.exp(self.log_density(x))

    def log_density(self, x: Mapping[Any, Any]) -> float:
        """Return summed child log densities for one mapping record."""
        if not isinstance(x, Mapping):
            return -np.inf
        rv = 0.0
        for source, dist in zip(self.sources, self.dists):
            rv += dist.log_density(_record_get(x, source))
        return rv

    def seq_log_density(self, x: tuple[Any, ...]) -> np.ndarray:
        """Return per-row log densities for encoded record fields."""
        if self.count == 0:
            if isinstance(x, tuple) and len(x) == 1 and isinstance(x[0], (int, np.integer)):
                return np.zeros(int(x[0]), dtype=float)
            return np.zeros(0, dtype=float)
        rv = self.dists[0].seq_log_density(x[0])
        for i in range(1, self.count):
            rv += self.dists[i].seq_log_density(x[i])
        return rv

    def backend_seq_log_density(self, x: tuple[Any, ...], engine: Any) -> Any:
        """Return per-row log densities using backend-aware child scorers."""
        if self.count == 0:
            n = int(x[0]) if isinstance(x, tuple) and len(x) == 1 else 0
            return engine.zeros(n)
        from mixle.stats.compute.backend import backend_seq_log_density

        rv = backend_seq_log_density(self.dists[0], x[0], engine)
        for i in range(1, self.count):
            rv = rv + backend_seq_log_density(self.dists[i], x[i], engine)
        return rv

    @classmethod
    def backend_stacked_params(cls, dists: Sequence[RecordDistribution], engine: Any) -> dict[str, Any]:
        """Return stacked child parameters for homogeneous named-record mixtures."""
        from mixle.stats.compute.stacked import stacked_component_params

        fields = dists[0].fields
        sources = dists[0].sources
        count = dists[0].count
        if any(d.fields != fields or d.sources != sources for d in dists):
            raise ValueError("Stacked RecordDistribution components require matching field/source layout.")
        children = []
        for i in range(count):
            child_dists = [d.dists[i] for d in dists]
            try:
                children.append(stacked_component_params(child_dists, engine))
            except ValueError as exc:
                raise ValueError(
                    "Record field %s child %s is not stackable: %s"
                    % (repr(fields[i]), type(child_dists[0]).__name__, exc)
                )
        return {"children": tuple(children), "fields": fields, "sources": sources, "num_components": len(dists)}

    @classmethod
    def backend_stacked_log_density(cls, x: tuple[Any, ...], params: dict[str, Any], engine: Any) -> Any:
        """Return an ``(n, k)`` matrix of record log densities."""
        from mixle.stats.compute.stacked import stacked_component_log_density

        children = params["children"]
        if not children:
            n = int(x[0]) if isinstance(x, tuple) and len(x) == 1 else 0
            return engine.zeros((n, int(params["num_components"])))
        rv = stacked_component_log_density(x[0], children[0], engine)
        for i in range(1, len(children)):
            rv = rv + stacked_component_log_density(x[i], children[i], engine)
        return rv

    @classmethod
    def backend_stacked_sufficient_statistics_with_estimator(
        cls, x: tuple[Any, ...], weights: Any, params: dict[str, Any], engine: Any, estimator: Any
    ) -> tuple[Any, ...]:
        """Return per-component legacy named-record sufficient statistics."""
        from mixle.stats.compute.stacked import (
            StackedEstimatorView,
            stacked_component_sufficient_statistics,
            unstack_component_stats,
        )

        ww = engine.asarray(weights)
        num_components = int(tuple(getattr(ww, "shape", (0, 0)))[1])
        outer_estimators = tuple(getattr(estimator, "estimators", ()))
        child_payloads = []
        for i, route in enumerate(params["children"]):
            component_estimators = tuple(
                getattr(component_est, "estimators", ())[i]
                for component_est in outer_estimators
                if len(getattr(component_est, "estimators", ())) > i
            )
            child_estimator = (
                StackedEstimatorView(component_estimators) if len(component_estimators) == num_components else None
            )
            child_stats = stacked_component_sufficient_statistics(x[i], ww, route, engine, child_estimator)
            child_payloads.append(unstack_component_stats(child_stats, num_components))
        return tuple(tuple(child[i] for child in child_payloads) for i in range(num_components))

    def gradient_fit_state(self, engine: Any, torch: Any, leaves: list[Any], recurse: Any, tensor_param: Any) -> Any:
        """Return distribution-owned state for autograd fitting."""
        from mixle.stats.compute.gradient import RecordGradientFitState

        return RecordGradientFitState(self, [recurse(dist, engine, torch, leaves) for dist in self.dists])

    def seq_ld_lambda(self) -> list[Any]:
        """Return legacy sequence log-density callables for this distribution."""
        return [self.seq_log_density]

    def support_size(self) -> int | None:
        """Product of field support sizes (``None`` if any field is infinite)."""
        total = 1
        for d in self.dists:
            s = d.support_size()
            if s is None:
                return None
            total *= s
        return total

    def sampler(self, seed: int | None = None) -> RecordSampler:
        """Return a sampler that draws mapping records field-by-field."""
        return RecordSampler(self, seed)

    def estimator(self, pseudo_count: float | None = None) -> RecordEstimator:
        """Return a record estimator with child estimators from each field."""
        return RecordEstimator(
            tuple(zip(self.fields, self.sources)), [d.estimator(pseudo_count=pseudo_count) for d in self.dists]
        )

    def decomposition(self):
        """Record fields are independent: split along the field (factor) axis, sufficient stats SUM-reduce."""
        from mixle.stats.compute.decomposition import DecompAxis, Decomposition, ReductionOp

        return Decomposition(
            axis=DecompAxis.FACTOR,
            num_units=self.count,
            reduction=ReductionOp.SUM,
            exact=True,
            child_roles=tuple(self.fields),
        )

    def dist_to_encoder(self) -> RecordDataEncoder:
        """Return a data encoder for this record field/source layout."""
        return RecordDataEncoder(tuple(zip(self.fields, self.sources)), [d.dist_to_encoder() for d in self.dists])

    def _row_from_values(self, values: Sequence[Any]) -> dict[Any, Any]:
        """Assemble a mapping record from per-field values (keyed by source, like the sampler)."""
        return {source: value for source, value in zip(self.sources, values)}

    def enumerator(self) -> RecordEnumerator:
        """Creates RecordEnumerator iterating mapping records in descending joint probability order."""
        return RecordEnumerator(self)

    def conditional_enumerator(self, given: Mapping[Any, Any]) -> RecordConditionalEnumerator:
        """Enumerate complete records consistent with the fixed fields in ``given``.

        ``given`` is a mapping ``{source: value}`` pinning a subset of fields (the canonical
        most-probable-completion / imputation query: complete missing fields, best first). Because the
        fields are independent, ``P(record | given)`` is proportional to the joint ``P(record)``, so
        descending order over the *free* fields is also descending conditional order. Each yielded
        value is a complete record with the fixed fields merged back in, and its ``log_prob`` is the
        full joint ``log_density`` (the enumerator contract: ``lp == dist.log_density(value)``) -- the
        fixed fields enter as a constant offset, which only shifts every score by the same amount.

        Raises ValueError if ``given`` names a field this record does not have.
        """
        if not isinstance(given, Mapping):
            raise TypeError("given must be a mapping of {source: value}.")
        valid_sources = set(self.sources)
        unknown = [k for k in given if k not in valid_sources]
        if unknown:
            raise ValueError("given names fields not in this record: %r" % unknown)
        return RecordConditionalEnumerator(self, dict(given))

    def structural_fine_bucket(self, value, quantizer) -> int:
        """Sum of child structural buckets -- mirrors the count index's child convolution.

        Like :meth:`CompositeDistribution.structural_fine_bucket`, but fields are addressed by
        source name rather than tuple position.
        """
        return sum(
            self.dists[i].structural_fine_bucket(_record_get(value, self.sources[i]), quantizer)
            for i in range(self.count)
        )

    def quantized_count_index(self, quantizer, max_fine_bucket: int):
        """Structural count index: the additive law -- the carrier's n-ary product over fields.

        Identical reduction to :meth:`CompositeDistribution.quantized_count_index` (the joint
        histogram is the convolution of the per-field histograms in the witness-retaining count
        semiring); the only difference is that the unranked structural tuple is relabelled into a
        mapping record keyed by source, so witnesses match what the model actually scores.
        """
        from mixle.enumeration.quantization.semiring import CountSemiring

        semiring = CountSemiring()
        children = []
        truncated = False
        for i, dist in enumerate(self.dists):
            try:
                child_index, child_truncated = dist.quantized_count_index(quantizer, max_fine_bucket)
            except EnumerationError as e:
                path = "RecordDistribution.dists[%d]" % i
                new_path = path if not e.path else "%s -> %s" % (path, e.path)
                raise EnumerationError(e.leaf, path=new_path, reason=e.reason) from None
            children.append(child_index)
            truncated = truncated or child_truncated

        joint = semiring.product(children, quantizer, max_fine_bucket)
        sources = self.sources
        relabelled = semiring.map_values(
            joint, lambda values, sources=sources: {source: value for source, value in zip(sources, values)}
        )
        return relabelled, truncated


class RecordEnumerator(DistributionEnumerator):
    """Enumerate named records over field supports in descending joint probability order."""

    def __init__(self, dist: RecordDistribution) -> None:
        """Enumerates mapping records over the field supports in descending joint probability order.

        Joint log-density is the sum of per-field log-densities, so this is a best-first search over
        the product of the (sorted) field enumerations -- the named-field analogue of
        :class:`CompositeEnumerator`. All fields must support enumeration; the combined value is a
        mapping keyed by source (matching :class:`RecordSampler`).

        Args:
            dist (RecordDistribution): Distribution whose support is enumerated.
        """
        super().__init__(dist)
        streams = [
            BufferedStream(child_enumerator(d, "RecordDistribution.dists[%d]" % i)) for i, d in enumerate(dist.dists)
        ]
        self._product = ProductEnumerator(streams, combine=dist._row_from_values)

    def __next__(self) -> tuple[dict[Any, Any], float]:
        return next(self._product)


class RecordConditionalEnumerator(DistributionEnumerator):
    """Enumerate complete records that agree with fixed conditioning fields."""

    def __init__(self, dist: RecordDistribution, given: dict[Any, Any]) -> None:
        """Enumerate complete records consistent with the fixed fields ``given``, best-first.

        Best-first over the product of the *free* fields' enumerations, offset by the fixed fields'
        summed log-density so each emitted score is the full joint ``log_density``. If a fixed value
        is impossible (its field assigns it ``-inf``) the support is empty.

        Args:
            dist (RecordDistribution): Distribution whose conditional support is enumerated.
            given (dict): Fixed ``{source: value}`` assignments (already validated by the caller).
        """
        super().__init__(dist)
        free_idx = [i for i in range(dist.count) if dist.sources[i] not in given]
        with np.errstate(divide="ignore"):
            fixed_lp = sum(
                dist.dists[i].log_density(given[dist.sources[i]]) for i in range(dist.count) if dist.sources[i] in given
            )
        if fixed_lp == -np.inf:
            self._product: Any = iter(())
            return
        free_sources = [dist.sources[i] for i in free_idx]

        def combine(free_values: Sequence[Any], _given=given, _free_sources=free_sources) -> dict[Any, Any]:
            row = dict(_given)
            row.update(zip(_free_sources, free_values))
            return row

        streams = [
            BufferedStream(child_enumerator(dist.dists[i], "RecordDistribution.dists[%d]" % i)) for i in free_idx
        ]
        self._product = ProductEnumerator(streams, combine=combine, offset=float(fixed_lp))

    def __next__(self) -> tuple[dict[Any, Any], float]:
        return next(self._product)


class RecordSampler(DistributionSampler):
    """Draw mapping records from a ``RecordDistribution``."""

    def __init__(self, dist: RecordDistribution, seed: int | None = None) -> None:
        super().__init__(dist, seed)
        self.dist = dist
        self.samplers = [d.sampler(seed=self.new_seed()) for d in dist.dists]

    def sample(self, size: int | None = None, *, batched: bool = True):
        """Draw one record or a list of records from the child samplers."""
        if size is None:
            return {source: sampler.sample() for source, sampler in zip(self.dist.sources, self.samplers)}
        rows = [dict() for _ in range(size)]
        for source, sampler in zip(self.dist.sources, self.samplers):
            values = sampler.sample(size=size)
            for i, value in enumerate(values):
                rows[i][source] = value
        return rows


class RecordAccumulator(SequenceEncodableStatisticAccumulator):
    """Accumulate field-wise sufficient statistics for record data."""

    def __init__(self, fields: Sequence[Any], accumulators: Sequence[SequenceEncodableStatisticAccumulator]) -> None:
        self.fields, self.sources = _normalize_fields(fields)
        self.keys = self.fields
        self.accumulators = list(accumulators)
        self.count = len(self.fields)
        self._init_rng = False
        self._acc_rng: list[np.random.RandomState] | None = None

    def update(self, x: Mapping[Any, Any], weight: float, estimate: RecordDistribution | None) -> None:
        """Accumulate one weighted mapping record."""
        row = x if isinstance(x, Mapping) else {}
        for i, source in enumerate(self.sources):
            child_estimate = None if estimate is None else estimate.dists[i]
            self.accumulators[i].update(_record_get(row, source), weight, child_estimate)

    def _rng_initialize(self, rng: np.random.RandomState) -> None:
        seeds = rng.randint(2**31, size=self.count)
        self._acc_rng = [np.random.RandomState(seed=int(seed)) for seed in seeds]
        self._init_rng = True

    def initialize(self, x: Mapping[Any, Any], weight: float, rng: np.random.RandomState) -> None:
        """Randomly initialize child accumulators from one mapping record."""
        if not self._init_rng:
            self._rng_initialize(rng)
        row = x if isinstance(x, Mapping) else {}
        for i, source in enumerate(self.sources):
            self.accumulators[i].initialize(_record_get(row, source), weight, self._acc_rng[i])

    def seq_update(self, x: tuple[Any, ...], weights: np.ndarray, estimate: RecordDistribution | None) -> None:
        """Accumulate encoded records with per-row weights."""
        for i in range(self.count):
            child_estimate = None if estimate is None else estimate.dists[i]
            self.accumulators[i].seq_update(x[i], weights, child_estimate)

    def seq_initialize(self, x: tuple[Any, ...], weights: np.ndarray, rng: np.random.RandomState) -> None:
        """Randomly initialize child accumulators from encoded records."""
        if not self._init_rng:
            self._rng_initialize(rng)
        for i in range(self.count):
            self.accumulators[i].seq_initialize(x[i], weights, self._acc_rng[i])

    def get_seq_lambda(self) -> list[Any]:
        """Return child sequence update callables for legacy orchestration."""
        rv = []
        for acc in self.accumulators:
            rv.extend(acc.get_seq_lambda())
        return rv

    def combine(self, suff_stat: tuple[Any, ...]) -> RecordAccumulator:
        """Merge field-wise sufficient statistics into this accumulator."""
        for i in range(self.count):
            self.accumulators[i].combine(suff_stat[i])
        return self

    def value(self) -> tuple[Any, ...]:
        """Return field-wise sufficient-statistic payloads in estimator order."""
        return tuple(acc.value() for acc in self.accumulators)

    def from_value(self, x: tuple[Any, ...]) -> RecordAccumulator:
        """Restore child accumulators from a field-wise payload."""
        self.accumulators = [self.accumulators[i].from_value(x[i]) for i in range(len(x))]
        self.count = len(x)
        return self

    def scale(self, c: float) -> RecordAccumulator:
        """Scale each field accumulator using its family-specific protocol."""
        for acc in self.accumulators:
            acc.scale(c)
        return self

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge keyed child statistics into the shared stats dictionary."""
        for acc in self.accumulators:
            acc.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace keyed child statistics from the shared stats dictionary."""
        for acc in self.accumulators:
            acc.key_replace(stats_dict)

    def acc_to_encoder(self) -> RecordDataEncoder:
        """Return a record encoder composed from child accumulator encoders."""
        return RecordDataEncoder(
            tuple(zip(self.fields, self.sources)), [acc.acc_to_encoder() for acc in self.accumulators]
        )


class RecordAccumulatorFactory(StatisticAccumulatorFactory):
    """Factory that creates record accumulators with matching child factories."""

    def __init__(self, fields: Sequence[Any], factories: Sequence[StatisticAccumulatorFactory]) -> None:
        self.fields, self.sources = _normalize_fields(fields)
        self.factories = tuple(factories)

    def make(self) -> RecordAccumulator:
        """Create a fresh record accumulator."""
        return RecordAccumulator(tuple(zip(self.fields, self.sources)), [factory.make() for factory in self.factories])


class RecordEstimator(ParameterEstimator):
    """Estimator for independent named fields."""

    def __init__(self, fields: Any, estimators: Sequence[ParameterEstimator] | None = None) -> None:
        self.fields, self.sources, self.estimators = _split_fields(fields, estimators)
        self.keys = self.fields
        self.count = len(self.fields)

    def accumulator_factory(self) -> RecordAccumulatorFactory:
        """Return a factory for record sufficient-statistic accumulators."""
        return RecordAccumulatorFactory(
            tuple(zip(self.fields, self.sources)), [est.accumulator_factory() for est in self.estimators]
        )

    def estimate(self, nobs: float | None, suff_stat: tuple[Any, ...]) -> RecordDistribution:
        """Estimate each child field and return a fitted record distribution."""
        return RecordDistribution(
            tuple(zip(self.fields, self.sources)),
            [est.estimate(nobs, ss) for est, ss in zip(self.estimators, suff_stat)],
        )


class RecordDataEncoder(DataSequenceEncoder):
    """Encode a sequence of mapping records field-by-field."""

    def __init__(self, fields: Sequence[Any], encoders: Sequence[DataSequenceEncoder]) -> None:
        self.fields, self.sources = _normalize_fields(fields)
        self.keys = self.fields
        self.encoders = tuple(encoders)

    def __str__(self) -> str:
        parts = ["%s: %s" % (repr(k), str(e)) for k, e in zip(self.fields, self.encoders)]
        return "RecordDataEncoder({%s})" % ", ".join(parts)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RecordDataEncoder) and self.fields == other.fields and self.encoders == other.encoders

    def seq_encode(self, x: Sequence[Mapping[Any, Any]]) -> tuple[Any, ...]:
        """Encode a sequence of mapping records field-by-field."""
        if len(self.fields) == 0:
            return (len(x),)
        encoded = []
        for source, encoder in zip(self.sources, self.encoders):
            encoded.append(encoder.seq_encode([_record_get(row, source) for row in x]))
        return tuple(encoded)


def record(fields: Mapping[Any, SequenceEncodableProbabilityDistribution]) -> RecordDistribution:
    """Create a ``RecordDistribution`` from a field-to-distribution mapping."""
    return RecordDistribution(fields)


def record_estimator(fields: Mapping[Any, ParameterEstimator]) -> RecordEstimator:
    """Create a ``RecordEstimator`` from a field-to-estimator mapping."""
    return RecordEstimator(fields)


DictRecordDistribution = RecordDistribution
DictRecordSampler = RecordSampler
DictRecordAccumulator = RecordAccumulator
DictRecordAccumulatorFactory = RecordAccumulatorFactory
DictRecordEstimator = RecordEstimator
DictRecordDataEncoder = RecordDataEncoder


__all__ = [
    "DictRecordAccumulator",
    "DictRecordAccumulatorFactory",
    "DictRecordDataEncoder",
    "DictRecordDistribution",
    "DictRecordEstimator",
    "DictRecordSampler",
    "RecordAccumulator",
    "RecordAccumulatorFactory",
    "RecordDataEncoder",
    "RecordDistribution",
    "RecordEstimator",
    "RecordSampler",
    "field",
    "record",
    "record_estimator",
]
