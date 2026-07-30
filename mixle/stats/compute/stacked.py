"""Stacked mixture kernels built from distribution-owned backend math.

The module prepares homogeneous component parameters, resident sufficient
statistics, and generated or explicit stacked scoring routes for efficient
mixture E-steps across supported compute engines.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from mixle.capability import SupportsStackedBackend, supports
from mixle.engines import ComputeEngine
from mixle.stats.compute.capability_decline import KernelCapabilityDeclinedError
from mixle.stats.compute.declarations import (
    declaration_for,
    generated_stacked_available,
    generated_stacked_log_density,
    generated_stacked_params,
    generated_stacked_preferred,
    generated_stacked_strategy,
    generated_stacked_sufficient_statistics,
    generated_stacked_sufficient_statistics_available,
)
from mixle.stats.compute.kernel import Kernel, KernelFactory
from mixle.stats.compute.mixture_evidence import normalize_engine_mixture_log_scores
from mixle.stats.compute.pdist import ParameterEstimator, SequenceEncodableProbabilityDistribution

_COMPONENT_AXIS_KEY = "__pysp_component_axis__"


@dataclass(frozen=True)
class StackedComponentParams:
    """A generic route for homogeneous component scoring."""

    component_type: type[Any]
    strategy: str
    params: Any


@dataclass(frozen=True)
class StackedEstimatorView:
    """Minimal estimator view for component-stacked resident reductions."""

    estimators: Sequence[Any]


@dataclass(frozen=True)
class ComponentShardLayout:
    """Explicit global component indices represented by one local payload."""

    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(isinstance(index, bool) or int(index) != index or int(index) < 0 for index in self.indices):
            raise ValueError("component shard indices must be non-negative integers.")
        normalized = tuple(int(index) for index in self.indices)
        if len(set(normalized)) != len(normalized):
            raise ValueError("component shard indices must be unique.")
        object.__setattr__(self, "indices", normalized)

    @property
    def contiguous(self) -> bool:
        """Return whether the indices form one ascending contiguous interval."""
        return not self.indices or self.indices == tuple(range(self.indices[0], self.indices[0] + len(self.indices)))

    @property
    def ranges(self) -> tuple[tuple[int, int], ...]:
        """Return minimal half-open ranges preserving local component order."""
        if not self.indices:
            return ()
        ranges = []
        start = previous = self.indices[0]
        for index in self.indices[1:]:
            if index != previous + 1:
                ranges.append((start, previous + 1))
                start = index
            previous = index
        ranges.append((start, previous + 1))
        return tuple(ranges)

    def legacy_selection(self) -> int | tuple[int, ...]:
        """Return the old start integer when contiguous, explicit indices otherwise."""
        return (self.indices[0] if self.indices else 0) if self.contiguous else self.indices


@dataclass(frozen=True)
class StackedMixtureResidentStats:
    """Engine-resident sufficient statistics for a homogeneous mixture."""

    component_counts: Any
    component_stats: Any
    engine: ComputeEngine
    component_type: type[Any]

    def value(self) -> Any:
        """Return sufficient statistics in the legacy ``MixtureEstimator`` format."""
        counts = np.asarray(self.engine.to_numpy(self.component_counts), dtype=np.float64)
        stats = _engine_to_numpy(self.component_stats, self.engine)
        return counts, _unstack_component_stats(stats, len(counts))

    def local_value(self) -> tuple[int | tuple[int, ...], Any]:
        """Return this rank's explicitly indexed component-stat payload.

        DTensor-backed model-parallel runs should run component M-steps on the
        local component shard and all-reduce only scalar mixture-weight totals.
        This method therefore prefers DTensor ``to_local()`` chunks over
        ``full_tensor()``. A contiguous shard retains the legacy integer start;
        a non-contiguous shard returns its global-index tuple. Every resident
        statistic tensor must expose the same component layout as the counts.
        """
        counts = np.asarray(_engine_local_to_numpy(self.component_counts, self.engine), dtype=np.float64)
        layout = _local_component_layout(self.component_counts, len(counts), "component_counts")
        stats = _engine_local_to_numpy(self.component_stats, self.engine)
        _validate_local_stat_layout(self.component_stats, stats, layout, "component_stats")
        return layout.legacy_selection(), (counts, _unstack_component_stats(stats, len(counts)))

    def estimate(self, estimator: ParameterEstimator) -> Any:
        """Estimate a distribution after converting to the legacy estimator protocol."""
        return estimator.estimate(None, self.value())

    def estimate_component_shard(
        self, estimator: ParameterEstimator, total_count: float | None = None
    ) -> StackedMixtureShardEstimate:
        """Run the component-local part of a mixture M-step for this shard.

        Component distributions are independent given posterior sufficient
        statistics.  Mixture weights only need the scalar global total count
        (or fixed weights), so a model-parallel orchestrator can call this on
        each shard after an all-reduce of that scalar instead of gathering all
        component statistics to the driver.
        """
        start, local = self.local_value()
        return estimate_component_shard_value(estimator, start, local, total_count=total_count)


@dataclass(frozen=True)
class StackedMixtureShardEstimate:
    """Component-local M-step result for a homogeneous mixture shard."""

    component_indices: tuple[int, ...]
    components: tuple[Any, ...]
    weights: np.ndarray
    total_count: float

    @property
    def component_start(self) -> int:
        """Lowest selected index (legacy contiguous-shard compatibility)."""
        return min(self.component_indices, default=0)

    @property
    def component_stop(self) -> int:
        """Exclusive upper bound (legacy contiguous-shard compatibility)."""
        return max(self.component_indices, default=-1) + 1

    @property
    def component_ranges(self) -> tuple[tuple[int, int], ...]:
        """Minimal half-open global ranges owned by this result."""
        return ComponentShardLayout(self.component_indices).ranges


def _component_selection_layout(
    selection: int | Sequence[int] | ComponentShardLayout,
    local_count: int,
) -> ComponentShardLayout:
    if isinstance(selection, ComponentShardLayout):
        layout = selection
    elif isinstance(selection, (int, np.integer)) and not isinstance(selection, (bool, np.bool_)):
        start = int(selection)
        layout = ComponentShardLayout(tuple(range(start, start + int(local_count))))
    else:
        try:
            layout = ComponentShardLayout(tuple(selection))
        except TypeError as exc:
            raise TypeError("component selection must be an integer start or a sequence of global indices.") from exc
    if len(layout.indices) != int(local_count):
        raise ValueError(
            "component selection has %d indices but the local payload has %d components."
            % (len(layout.indices), local_count)
        )
    return layout


def estimate_component_shard_value(
    estimator: ParameterEstimator,
    component_selection: int | Sequence[int] | ComponentShardLayout,
    value: tuple[Any, tuple[Any, ...]],
    total_count: float | None = None,
) -> StackedMixtureShardEstimate:
    """Estimate the component-local part of a mixture M-step from an explicit shard value."""
    counts, comp_suff_stats = value
    counts = np.asarray(counts, dtype=np.float64)
    num_components = int(getattr(estimator, "num_components", len(counts)))
    layout = _component_selection_layout(component_selection, len(counts))
    if layout.indices and max(layout.indices) >= num_components:
        raise ValueError(
            "component shard indices %s exceed estimator with %d components." % (layout.indices, num_components)
        )
    if len(comp_suff_stats) != len(counts):
        raise ValueError(
            "component shard has %d count entries but %d component statistics." % (len(counts), len(comp_suff_stats))
        )

    estimators = getattr(estimator, "estimators", None)
    if estimators is None:
        raise ValueError("estimate_component_shard_value requires a mixture-style estimator with component estimators.")
    components = tuple(
        estimators[index].estimate(counts[offset], comp_suff_stats[offset])
        for offset, index in enumerate(layout.indices)
    )

    fixed_weights = getattr(estimator, "fixed_weights", None)
    if fixed_weights is not None:
        weights = np.asarray(fixed_weights, dtype=np.float64)[list(layout.indices)]
        global_total = float(np.asarray(total_count if total_count is not None else counts.sum()))
    else:
        if total_count is None:
            if layout.indices != tuple(range(num_components)):
                raise ValueError("total_count is required when estimating weights for a component shard.")
            total_count = float(counts.sum())
        global_total = float(total_count)
        pseudo_count = getattr(estimator, "pseudo_count", None)
        suff_stat = getattr(estimator, "suff_stat", None)
        if pseudo_count is not None and suff_stat is None:
            p = float(pseudo_count) / float(num_components)
            weights = (counts + p) / (global_total + float(pseudo_count))
        elif pseudo_count is not None and suff_stat is not None:
            prior = np.asarray(suff_stat, dtype=np.float64)[list(layout.indices)]
            weights = (counts + prior * float(pseudo_count)) / (global_total + float(pseudo_count))
        elif global_total == 0.0:
            weights = np.ones(len(counts), dtype=np.float64) / float(num_components)
        else:
            weights = counts / global_total

    return StackedMixtureShardEstimate(
        component_indices=layout.indices,
        components=components,
        weights=np.asarray(weights, dtype=np.float64),
        total_count=global_total,
    )


def tie_component_shard_values(
    estimator: ParameterEstimator,
    shard_values: Sequence[tuple[int | Sequence[int] | ComponentShardLayout, tuple[Any, tuple[Any, ...]]]],
) -> tuple[tuple[int | tuple[int, ...], tuple[np.ndarray, tuple[Any, ...]]], ...]:
    """Apply mixle key tying to component-sharded mixture statistics.

    ``MixtureAccumulator.key_merge`` / ``key_replace`` assume a full component
    vector is materialized on one worker.  Component-sharded model-parallel
    runs instead hold ranges ``[component_start, component_stop)``.  This helper
    applies the same key protocol to those local ranges by materializing only
    the component accumulators owned by each shard.
    """
    if not shard_values:
        return ()
    estimators = getattr(estimator, "estimators", None)
    if estimators is None:
        raise ValueError("tie_component_shard_values requires a mixture-style estimator.")
    num_components = int(getattr(estimator, "num_components", len(estimators)))
    keyed_accs = []
    keyed_counts = []
    for selection, value in shard_values:
        counts, comp_stats = value
        counts = np.asarray(counts, dtype=np.float64).copy()
        layout = _component_selection_layout(selection, len(counts))
        if layout.indices and max(layout.indices) >= num_components:
            raise ValueError(
                "component shard indices %s exceed estimator with %d components." % (layout.indices, num_components)
            )
        if len(comp_stats) != len(counts):
            raise ValueError(
                "component shard has %d count entries but %d component statistics." % (len(counts), len(comp_stats))
            )
        accs = []
        for index, suff_stat in zip(layout.indices, comp_stats):
            acc = estimators[index].accumulator_factory().make()
            acc.from_value(suff_stat)
            accs.append(acc)
        keyed_accs.append([layout, tuple(accs)])
        keyed_counts.append([layout, counts])

    keys = getattr(estimator, "keys", (None, None))
    weight_key = keys[0] if isinstance(keys, tuple) and len(keys) > 0 else None
    comp_key = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else None

    if weight_key is not None:
        pooled_counts = {}
        for layout, counts in keyed_counts:
            for index, count in zip(layout.indices, counts):
                pooled_counts[index] = pooled_counts.get(index, 0.0) + float(count)
        for layout, counts in keyed_counts:
            for offset, index in enumerate(layout.indices):
                counts[offset] = pooled_counts[index]

    stats_dict = {}
    pooled_components = {}
    if comp_key is not None:
        for layout, accs in keyed_accs:
            for index, acc in zip(layout.indices, accs):
                if index in pooled_components:
                    pooled_components[index].combine(acc.value())
                else:
                    pooled = estimators[index].accumulator_factory().make()
                    pooled.combine(acc.value())
                    pooled_components[index] = pooled

    for _, accs in keyed_accs:
        for acc in accs:
            acc.key_merge(stats_dict)

    if comp_key is not None:
        keyed_accs = [
            [layout, tuple(pooled_components[index] for index in layout.indices)] for layout, accs in keyed_accs
        ]

    for _, accs in keyed_accs:
        for acc in accs:
            acc.key_replace(stats_dict)

    return tuple(
        (layout.legacy_selection(), (counts, tuple(acc.value() for acc in accs)))
        for (layout, counts), (_, accs) in zip(keyed_counts, keyed_accs)
    )


def stacked_component_params(dists: Sequence[Any], engine: ComputeEngine) -> StackedComponentParams:
    """Return a generic stacked-scoring route for homogeneous components.

    Generated declaration-backed routes are attempted first.  Families with
    object lookups, table layouts, derived parameters, or wrapper structure can
    keep explicit ``backend_stacked_*`` hooks and still compose through this
    single dispatcher.
    """
    if not dists:
        raise ValueError("stacked_component_params requires at least one component.")
    component_type: type[Any] = type(dists[0])
    if any(type(component) is not component_type for component in dists):
        raise ValueError("stacked component scoring requires homogeneous component types.")

    params_fn = getattr(component_type, "backend_stacked_params", None)
    score_fn = getattr(component_type, "backend_stacked_log_density", None)
    has_explicit = callable(params_fn) and callable(score_fn)
    # Only take the generated route when generated stacked *scoring* is actually available:
    # generated_stacked_params can build a parameter bundle for a family whose generated scorer
    # then can't run (e.g. a backend hook keyed on a derived parameter, or an exp-family spec with
    # runtime_scoring=False). In that case prefer the family's explicit backend_stacked_* hooks.
    if generated_stacked_available(component_type) or not has_explicit:
        try:
            return StackedComponentParams(
                component_type=component_type,
                strategy="generated",
                params=_place_stacked_params(generated_stacked_params(dists, engine), engine, default_axis=0),
            )
        except KernelCapabilityDeclinedError:
            if not has_explicit:
                raise
    return StackedComponentParams(
        component_type=component_type,
        strategy="explicit",
        params=_place_stacked_params(params_fn(dists, engine), engine, default_axis=None),
    )


def stacked_component_log_density(enc: Any, route: StackedComponentParams, engine: ComputeEngine) -> Any:
    """Evaluate a route returned by ``stacked_component_params``."""
    if route.strategy == "generated":
        return generated_stacked_log_density(enc, route.params, engine)
    if route.strategy == "explicit":
        return route.component_type.backend_stacked_log_density(enc, route.params, engine)
    raise ValueError("Unknown stacked component strategy %s." % route.strategy)


def _estimator_resident_supported(estimator: Any) -> bool:
    """Return whether ``estimator`` (or every component of a mixture estimator) accepts resident stats."""
    component_estimators = getattr(estimator, "estimators", None)
    if component_estimators is not None:
        return all(_estimator_resident_supported(e) for e in component_estimators)
    supported = getattr(estimator, "resident_accumulation_supported", None)
    return bool(supported()) if callable(supported) else True


def stacked_component_sufficient_statistics(
    enc: Any,
    weights: Any,
    route: StackedComponentParams,
    engine: ComputeEngine,
    estimator: ParameterEstimator | None = None,
) -> Any:
    """Return component-stacked legacy sufficient statistics for a stacked route."""
    stats_fn = getattr(route.component_type, "backend_stacked_sufficient_statistics", None)
    estimator_stats_fn = getattr(route.component_type, "backend_stacked_sufficient_statistics_with_estimator", None)
    if callable(stats_fn):
        return stats_fn(enc, weights, route.params, engine)
    if estimator is not None and callable(estimator_stats_fn):
        return estimator_stats_fn(enc, weights, route.params, engine, estimator)
    if route.strategy == "generated":
        return generated_stacked_sufficient_statistics(enc, weights, route.params, engine)
    raise NotImplementedError(
        "%s does not provide resident stacked sufficient statistics." % route.component_type.__name__
    )


def unstack_component_stats(value: Any, num_components: int) -> tuple[Any, ...]:
    """Return per-component legacy statistics from a component-stacked payload."""
    return _unstack_component_stats(value, num_components)


class StackedMixtureKernel(Kernel):
    """Homogeneous mixture kernel with stacked component parameters.

    The mixture mechanics live here: component matrix scoring, row-wise
    logsumexp, posterior weights, and legacy sufficient-stat dispatch. The leaf
    family still owns the actual component log-density math through
    ``backend_stacked_params`` and ``backend_stacked_log_density``.
    """

    def __init__(self, dist: Any, engine: ComputeEngine, estimator: ParameterEstimator | None = None) -> None:
        self.dist = dist
        self.engine = engine
        self.estimator = estimator
        self.route = stacked_component_params(dist.components, engine)
        self.component_type: type[Any] = self.route.component_type
        self._generated = self.route.strategy == "generated"
        self.params = self.route.params
        self.log_w = _place_component_array(engine.asarray(dist.log_w), engine, axis=0)

    def encode(self, data: Any) -> Any:
        """Encode raw observations with the mixture's ordinary encoder."""
        return self.dist.dist_to_encoder().seq_encode(data)

    def component_scores(self, enc: Any) -> Any:
        """Return unweighted component log densities with shape ``(n, k)``."""
        enc = getattr(enc, "engine_payload", enc)
        ll = stacked_component_log_density(enc, self.route, self.engine)
        zw = getattr(self.dist, "zw", None)
        if zw is not None and np.any(zw):
            mask = self.engine.asarray(np.asarray(zw))
            impossible = self.engine.zeros(ll.shape) + self.engine.asarray(-np.inf)
            ll = self.engine.where(mask[None, :], impossible, ll)
        return ll

    def score(self, enc: Any) -> Any:
        """Return row log densities after adding mixture log weights."""
        weighted = self.component_scores(enc) + self.log_w
        return normalize_engine_mixture_log_scores(weighted, self.engine).log_evidence

    def posteriors(self, enc: Any) -> Any:
        """Return posterior component weights for each encoded row."""
        weighted = self.component_scores(enc) + self.log_w
        return normalize_engine_mixture_log_scores(weighted, self.engine).responsibilities

    @property
    def has_resident_accumulate(self) -> bool:
        """Return true when the leaf family can accumulate sufficient stats on the engine."""
        if self.estimator is not None and not _estimator_resident_supported(self.estimator):
            return False
        if callable(getattr(self.component_type, "backend_stacked_sufficient_statistics", None)):
            return True
        if self.estimator is not None and callable(
            getattr(self.component_type, "backend_stacked_sufficient_statistics_with_estimator", None)
        ):
            return True
        return self.route.strategy == "generated" and generated_stacked_sufficient_statistics_available(
            self.route.params
        )

    def resident_accumulate(self, enc: Any, weights: Any) -> StackedMixtureResidentStats:
        """Return engine-resident mixture sufficient statistics.

        The mixture owns only the posterior mechanics. Leaf families own the
        sufficient-statistic algebra through ``backend_stacked_sufficient_statistics``.
        """
        gamma = self.posteriors(enc)
        weights = self.engine.asarray(weights)
        gamma = gamma * weights[:, None]
        counts = self.engine.sum(gamma, axis=0)
        engine_enc = getattr(enc, "engine_payload", enc)
        stats = stacked_component_sufficient_statistics(engine_enc, gamma, self.route, self.engine, self.estimator)
        return StackedMixtureResidentStats(
            component_counts=counts,
            component_stats=stats,
            engine=self.engine,
            component_type=self.component_type,
        )

    def accumulate(self, enc: Any, weights: Any) -> Any:
        """Return mixture sufficient statistics in the legacy estimator format."""
        if self.estimator is None:
            raise ValueError("StackedMixtureKernel.accumulate requires an estimator.")
        if self.has_resident_accumulate:
            resident = self.resident_accumulate(enc, weights).value()
            self._assert_resident_statistics_complete(resident)
            return resident
        host_enc = getattr(enc, "host_payload", enc)
        gamma = self.posteriors(enc)
        weights = self.engine.asarray(weights)
        gamma = gamma * weights[:, None]
        gamma_np = np.asarray(self.engine.to_numpy(gamma), dtype=np.float64)
        comp_counts = gamma_np.sum(axis=0)
        comp_stats = []
        for i, acc in enumerate(self.estimator.accumulator_factory().make().accumulators):
            acc.seq_update(host_enc, gamma_np[:, i], self.dist.components[i])
            comp_stats.append(acc.value())
        return comp_counts, tuple(comp_stats)

    def _assert_resident_statistics_complete(self, resident: Any) -> None:
        """Refuse a resident statistic tuple that is shorter than the family declares.

        has_resident_accumulate selects a backend because the method *exists*, never because it is
        current. GEV added observed min/max to its declaration and left its stacked backend returning
        the four moments; the stale method was chosen over the correct host accumulator and the
        support bounds vanished, so the engine-resident path scored -inf on rows the host path fits
        (MXR-080-1846). Presence is not currency, so check the arity at the point of use, where it
        cannot go stale.
        """
        declaration = getattr(self.component_type, "compute_declaration", None)
        if not callable(declaration):
            return
        try:
            declared = len(declaration().statistics)
        except Exception:  # noqa: BLE001 - a family that cannot describe itself is another test's problem
            return
        per_component = resident[1] if isinstance(resident, tuple) and len(resident) == 2 else resident
        first = None
        if isinstance(per_component, (tuple, list)) and per_component:
            first = per_component[0]
        if not isinstance(first, (tuple, list)) or declared == 0:
            return
        if len(first) < declared:
            raise ValueError(
                f"{self.component_type.__name__}.backend_stacked_sufficient_statistics returned "
                f"{len(first)} statistics but the family declares {declared}. The engine-resident "
                "path would drop the rest; update the backend or remove it so the host accumulator runs."
            )

    def refresh(self, dist: SequenceEncodableProbabilityDistribution) -> None:
        """Refresh stacked parameters after an M-step without changing structure."""
        self.dist = dist
        self.route = stacked_component_params(dist.components, self.engine)
        self.component_type = self.route.component_type
        self._generated = self.route.strategy == "generated"
        self.params = self.route.params
        self.log_w = _place_component_array(self.engine.asarray(dist.log_w), self.engine, axis=0)


def _stackable_mixture(dist: Any) -> bool:
    components = getattr(dist, "components", None)
    if not components:
        return False
    component_type = type(components[0])
    if not all(type(component) is component_type for component in components):
        return False
    return supports(component_type, SupportsStackedBackend) or generated_stacked_available(component_type)


def _has_generated_backend_hook(dist_type: type[Any]) -> bool:
    if generated_stacked_strategy(dist_type) != "backend_log_density_from_params":
        return False
    declaration = declaration_for(dist_type)
    if declaration is None:
        return False
    generated_constraints = {
        "real",
        "real_vector",
        "positive",
        "positive_vector",
        "unit_interval",
        "integer",
        "positive_integer",
        "non_negative_integer",
        "optional_integer",
        "fixed",
    }
    for spec in declaration.parameters:
        if str(spec.constraint).startswith("greater_than:"):
            continue
        if spec.constraint not in generated_constraints:
            return False
    return True


def stacked_component_strategy(dist_type: type[Any]) -> str:
    """Describe how homogeneous mixture component scoring will be dispatched."""
    if generated_stacked_preferred(dist_type):
        return "generated_exp_family"
    if _has_generated_backend_hook(dist_type):
        return "generated_backend_hook"
    if supports(dist_type, SupportsStackedBackend):
        return "explicit_stacked"
    return "generic"


class StackedMixtureKernelFactory(KernelFactory):
    """Factory for homogeneous mixtures with distribution-owned stacked math."""

    def __init__(self, fallback: KernelFactory | None = None) -> None:
        # numpy mixtures fall through to generated numba (legacy-enc compatible);
        # GeneratedNumbaKernelFactory itself defers to the generic kernel when no
        # generated scorer exists, so this remains a guaranteed fallback chain
        if fallback is None:
            from mixle.stats.compute.kernel import GeneratedNumbaKernelFactory

            fallback = GeneratedNumbaKernelFactory()
        self.fallback = fallback

    def build(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        engine: ComputeEngine,
        estimator: ParameterEstimator | None = None,
    ) -> Kernel:
        """Build a stacked mixture kernel when safe, otherwise use fallback."""
        if getattr(engine, "name", None) == "torch" and _stackable_mixture(dist):
            if not dist.supports_engine(engine):
                return self.fallback.build(dist, engine, estimator=estimator)
            try:
                return StackedMixtureKernel(dist, engine=engine, estimator=estimator)
            except KernelCapabilityDeclinedError:
                return self.fallback.build(dist, engine, estimator=estimator)
        return self.fallback.build(dist, engine, estimator=estimator)


def _place_stacked_params(params: Any, engine: ComputeEngine, default_axis: int | None) -> Any:
    """Apply engine-owned component placement to stacked parameter payloads."""
    if not callable(getattr(engine, "place_component_axis", None)):
        return params
    return _place_value(params, engine, default_axis)


def _place_value(value: Any, engine: ComputeEngine, axis: int | None) -> Any:
    if isinstance(value, StackedComponentParams):
        return replace(value, params=_place_value(value.params, engine, axis))
    if isinstance(value, dict):
        axis_spec = value.get(_COMPONENT_AXIS_KEY, axis)
        rv = {}
        for key, child in value.items():
            if key == _COMPONENT_AXIS_KEY:
                rv[key] = child
                continue
            if str(key).startswith("__pysp_"):
                rv[key] = child
                continue
            child_axis = axis_spec.get(key) if isinstance(axis_spec, dict) else axis_spec
            rv[key] = _place_value(child, engine, child_axis)
        return rv
    if isinstance(value, tuple):
        return tuple(_place_value(child, engine, axis) for child in value)
    if isinstance(value, list):
        return [_place_value(child, engine, axis) for child in value]
    if axis is None:
        return value
    return _place_component_array(value, engine, axis)


def _place_component_array(value: Any, engine: ComputeEngine, axis: int) -> Any:
    if isinstance(value, (str, bytes, bool, int, float, np.number, type)):
        return value
    try:
        arr = np.asarray(value)
    except Exception:  # noqa: BLE001
        arr = None
    if arr is not None:
        if arr.dtype.kind in ("O", "U", "S"):
            return value
        if arr.ndim == 0:
            return value
    return engine.place_component_axis(value, axis=axis)


def _scalar_like(value: Any) -> Any:
    """Unwrap a rank-0 array to the Python scalar a legacy accumulator would have produced.

    Legacy accumulators hand estimators plain Python scalars; routing the same statistic through an
    engine tensor and ``np.asarray`` yields a rank-0 ndarray instead. Estimators can tell the
    difference: IntegerCategoricalDistribution.estimate runs the ``min_val`` half of its
    ``(min_val, counts)`` statistic through ``exact_integer_observations``, which rejects a rank-0
    array outright -- and would equally reject the ``0.0`` a blanket float() cast produces, so
    ``item()`` is what keeps an integer support integral (and a state label a str).
    """
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _engine_to_numpy(value: Any, engine: ComputeEngine) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _engine_to_numpy(child, engine) for key, child in value.items()}
    if isinstance(value, tuple):
        converted = tuple(_engine_to_numpy(child, engine) for child in value)
        return type(value)(*converted) if hasattr(value, "_fields") else converted
    if isinstance(value, list):
        return [_engine_to_numpy(child, engine) for child in value]
    try:
        return _scalar_like(np.asarray(engine.to_numpy(value)))
    except Exception:  # noqa: BLE001
        return value


def _engine_local_to_numpy(value: Any, engine: ComputeEngine) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _engine_local_to_numpy(child, engine) for key, child in value.items()}
    if isinstance(value, tuple):
        converted = tuple(_engine_local_to_numpy(child, engine) for child in value)
        return type(value)(*converted) if hasattr(value, "_fields") else converted
    if isinstance(value, list):
        return [_engine_local_to_numpy(child, engine) for child in value]
    local_fn = getattr(value, "to_local", None)
    if callable(local_fn):
        try:
            return _tensor_like_to_numpy(local_fn())
        except Exception:  # noqa: BLE001
            pass
    return _engine_to_numpy(value, engine)


def _tensor_like_to_numpy(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    try:
        return _scalar_like(np.asarray(value))
    except Exception:  # noqa: BLE001
        return value


def _local_component_layout(value: Any, local_count: int, path: str) -> ComponentShardLayout:
    """Read every local DTensor chunk and return its global component indices."""
    chunk_fn = getattr(value, "__create_chunk_list__", None)
    if callable(chunk_fn):
        chunks = tuple(chunk_fn())
        if chunks:
            indices = []
            for chunk_index, chunk in enumerate(chunks):
                offsets = tuple(int(x) for x in chunk.offsets)
                sizes = tuple(int(x) for x in chunk.sizes)
                if not offsets or len(offsets) != len(sizes):
                    raise ValueError("%s chunk %d has invalid offset/size metadata." % (path, chunk_index))
                if offsets[0] < 0 or sizes[0] < 0:
                    raise ValueError("%s chunk %d has a negative component offset/size." % (path, chunk_index))
                indices.extend(range(offsets[0], offsets[0] + sizes[0]))
            layout = ComponentShardLayout(tuple(indices))
            if len(layout.indices) != int(local_count):
                raise ValueError(
                    "%s chunk metadata describes %d components but its local tensor contains %d."
                    % (path, len(layout.indices), local_count)
                )
            return layout
    return ComponentShardLayout(tuple(range(int(local_count))))


def _validate_local_stat_layout(
    original: Any,
    local: Any,
    expected: ComponentShardLayout,
    path: str,
) -> None:
    """Validate every component-stacked local statistic tensor against the count-tensor layout.

    Resident statistics come in the two layouts :func:`_unstack_component_stats` accepts: tensors
    whose axis 0 *is* the component axis, and a per-component sequence whose element ``i`` holds
    component ``i``'s own statistics with whatever shape that child distribution produces. Only the
    first has a component axis to check here. Descending into the second and demanding that every
    leaf lead with the component count rejects payloads that are perfectly well formed -- a
    two-component Markov chain mixture stores a ``(4,)`` transition vector per component, and a
    multinomial one a ``(3,)`` category vector, neither of which is two wide nor meant to be.
    """
    if original is None:
        return
    num_components = len(expected.indices)
    if isinstance(original, dict):
        if not isinstance(local, dict) or set(local) != set(original):
            raise ValueError("%s local statistic mapping does not match its resident structure." % path)
        for key, child in original.items():
            _validate_local_stat_layout(child, local[key], expected, "%s.%s" % (path, key))
        return
    if isinstance(original, (tuple, list)):
        if not isinstance(local, (tuple, list)) or len(local) != len(original):
            raise ValueError("%s local statistic sequence does not match its resident structure." % path)
        # Same precedence as _unstack_component_stats: component-stacked first, per-component second.
        if not _all_component_stacked(tuple(original), num_components) and len(original) == num_components:
            return
        for index, (child, local_child) in enumerate(zip(original, local)):
            _validate_local_stat_layout(child, local_child, expected, "%s[%d]" % (path, index))
        return

    shape = getattr(local, "shape", None)
    if shape is None:
        try:
            shape = np.asarray(local).shape
        except (TypeError, ValueError):
            return
    if not shape:
        return
    local_count = int(shape[0])
    if local_count != len(expected.indices):
        raise ValueError(
            "%s has local component axis %d but component_counts has %d." % (path, local_count, len(expected.indices))
        )
    if callable(getattr(original, "__create_chunk_list__", None)):
        actual = _local_component_layout(original, local_count, path)
        if actual.indices != expected.indices:
            raise ValueError(
                "%s maps global components %s but component_counts maps %s." % (path, actual.indices, expected.indices)
            )


def _unstack_component_stats(value: Any, num_components: int) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        if _all_component_stacked(value, num_components):
            return tuple(tuple(_take_component(child, i) for child in value) for i in range(num_components))
        if len(value) == num_components:
            return value
    if isinstance(value, list):
        if _all_component_stacked(tuple(value), num_components):
            return tuple(tuple(_take_component(child, i) for child in value) for i in range(num_components))
        if len(value) == num_components:
            return tuple(value)
    if _is_component_stacked(value, num_components):
        return tuple(_take_component(value, i) for i in range(num_components))
    raise ValueError("Resident component statistics do not expose a component axis of length %d." % num_components)


def _all_component_stacked(values: tuple[Any, ...], num_components: int) -> bool:
    return bool(values) and all(_is_component_stacked(value, num_components) for value in values)


def _is_component_stacked(value: Any, num_components: int) -> bool:
    shape = getattr(value, "shape", None)
    return shape is not None and len(shape) > 0 and int(shape[0]) == num_components


def _take_component(value: Any, index: int) -> Any:
    rv = _scalar_like(value[index])
    return rv
