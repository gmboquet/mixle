"""Backend-neutral evaluation kernel contracts.

The generic kernel is a thin adapter over the existing seq_* protocol.  It is
the guaranteed fallback for engine-aware orchestration; specialized factories
can override code shape for performance without changing estimators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from mixle.capability import EngineResidentEStep, SupportsBackendComponentScoring, supports
from mixle.engines import NUMPY_ENGINE, ComputeEngine
from mixle.stats.compute.pdist import ParameterEstimator, SequenceEncodableProbabilityDistribution


class EngineNotSupportedError(ValueError):
    """Raised when no kernel can safely evaluate a distribution on an engine."""

    pass


def _estimator_resident_supported(estimator: Any) -> bool:
    """Return whether ``estimator`` (or every component of a mixture estimator) accepts resident stats.

    Estimators whose M-step needs more than fixed-width resident sufficient statistics
    (e.g. the negative-binomial dispersion solve, which needs the full count histogram)
    override ``resident_accumulation_supported`` to ``False`` so generated kernels fall
    back to the host accumulator and stay identical to the numpy/seq path.
    """
    component_estimators = getattr(estimator, "estimators", None)
    if component_estimators is not None:
        return all(_estimator_resident_supported(e) for e in component_estimators)
    supported = getattr(estimator, "resident_accumulation_supported", None)
    return bool(supported()) if callable(supported) else True


class Kernel(ABC):
    """Evaluation kernel for a fitted distribution."""

    @abstractmethod
    def score(self, enc: Any) -> Any:
        """Return per-row log densities for an encoded observation batch."""
        ...

    def component_scores(self, enc: Any) -> Any:
        """Return per-row, per-component log densities where meaningful."""
        raise NotImplementedError("%s has no component_scores implementation." % type(self).__name__)

    @abstractmethod
    def accumulate(self, enc: Any, weights: Any) -> Any:
        """Return sufficient statistics in the legacy estimator format."""
        ...

    @abstractmethod
    def refresh(self, dist: SequenceEncodableProbabilityDistribution) -> None:
        """Refresh kernel parameters after an EM M-step without rebuilding structure."""
        ...


class KernelFactory(ABC):
    """Factory that builds a Kernel for a distribution and engine."""

    @abstractmethod
    def build(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        engine: ComputeEngine,
        estimator: ParameterEstimator | None = None,
    ) -> Kernel:
        """Return a kernel for ``dist`` on ``engine``.

        ``estimator`` is optional for pure scoring and required for kernels
        that need to emit sufficient statistics for an M-step.
        """
        ...


class GenericKernel(Kernel):
    """Fallback kernel over distribution-owned backend hooks or existing seq_* methods."""

    def __init__(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        engine: ComputeEngine = NUMPY_ENGINE,
        estimator: ParameterEstimator | None = None,
    ) -> None:
        self.dist = dist
        self.engine = engine
        self.estimator = estimator

    def score(self, enc: Any) -> Any:
        """Return per-row log densities using backend hooks when available."""
        enc = getattr(enc, "engine_payload", enc)
        from mixle.stats.compute.backend import BackendScoringError, backend_seq_log_density

        try:
            return backend_seq_log_density(self.dist, enc, self.engine)
        except BackendScoringError:
            # The numpy seq_log_density fallback returns host numpy arrays, so it is only valid on a
            # numpy-host engine; other engines must surface the failure.
            if not getattr(self.engine, "supports_numba", False):
                raise
        return self.dist.seq_log_density(enc)

    def component_scores(self, enc: Any) -> Any:
        """Return per-row component log densities for mixture-like models."""
        enc = getattr(enc, "engine_payload", enc)
        if supports(self.dist, SupportsBackendComponentScoring):
            from mixle.stats.compute.backend import backend_seq_component_log_density

            return backend_seq_component_log_density(self.dist, enc, self.engine)
        if hasattr(self.dist, "seq_component_log_density"):
            return self.dist.seq_component_log_density(enc)
        return super().component_scores(enc)

    def accumulate(self, enc: Any, weights: Any) -> Any:
        """Accumulate weighted sufficient statistics in estimator-owned format."""
        if self.estimator is None:
            raise ValueError("GenericKernel.accumulate requires an estimator.")
        from mixle.stats.compute.declarations import (
            generated_sufficient_statistics,
            generated_sufficient_statistics_available,
        )

        if generated_sufficient_statistics_available(self.dist):
            return generated_sufficient_statistics(self.dist, getattr(enc, "engine_payload", enc), weights, self.engine)
        accumulator = self.estimator.accumulator_factory().make()
        enc = getattr(enc, "host_payload", enc)
        # Engine-resident E-step: when the accumulator provides a backend update and the engine
        # prefers staying resident, run the sufficient-statistic accumulation on the active engine
        # instead of falling back to the host seq_update path.
        if getattr(self.engine, "resident_estep", True) and supports(accumulator, EngineResidentEStep):
            accumulator.seq_update_engine(enc, weights, self.dist, self.engine)
            return accumulator.value()
        weights = np.asarray(self.engine.to_numpy(weights), dtype=np.float64)
        accumulator.seq_update(enc, weights, self.dist)
        return accumulator.value()

    def refresh(self, dist: SequenceEncodableProbabilityDistribution) -> None:
        """Replace the fitted distribution while preserving kernel structure."""
        self.dist = dist


class GenericKernelFactory(KernelFactory):
    """Guaranteed fallback factory for distributions that support the engine."""

    def build(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        engine: ComputeEngine,
        estimator: ParameterEstimator | None = None,
    ) -> GenericKernel:
        """Build a generic kernel or fail fast when the engine is unsupported."""
        if not dist.supports_engine(engine):
            raise EngineNotSupportedError(
                "%s does not declare support for the %s engine. Register a specialized "
                "KernelFactory or keep this model on a supported engine: %s."
                % (type(dist).__name__, engine.name, ", ".join(dist.supported_engines()))
            )
        return GenericKernel(dist, engine=engine, estimator=estimator)


class NumbaKernel(Kernel):
    """Kernel adapter over the existing fused-numba ``CompiledMixture`` path.

    This kernel intentionally uses the columnar encoding returned by
    ``encode(data)`` rather than the legacy ``dist_to_encoder().seq_encode``
    payload.  That keeps the high-performance path explicit while giving it the
    same score/accumulate/refresh surface as other engines.
    """

    def __init__(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        engine: ComputeEngine = NUMPY_ENGINE,
        estimator: ParameterEstimator | None = None,
    ) -> None:
        if not getattr(engine, "supports_numba", False):
            raise ValueError("NumbaKernel requires a numba-capable (host numpy) engine.")
        from mixle.stats.compute.fused_kernels import CompiledMixture

        self.dist = dist
        self.engine = engine
        self.estimator = estimator
        self.compiled = CompiledMixture(dist)

    def encode(self, data: Any) -> Any:
        """Encode raw observations into the fused columnar kernel format."""
        return self.compiled.encode(data)

    def score(self, enc: Any) -> np.ndarray:
        """Return per-row log densities from the fused numba mixture kernel."""
        return self.compiled.seq_log_density(enc, model=self.dist)

    def component_scores(self, enc: Any) -> np.ndarray:
        """Return per-row, per-component log densities from the fused kernel."""
        return self.compiled.seq_component_log_density(enc, model=self.dist)

    def accumulate(self, enc: Any, weights: Any) -> Any:
        """Use fused posteriors plus row weights to produce legacy statistics."""
        if self.estimator is None:
            raise ValueError("NumbaKernel.accumulate requires an estimator.")
        row_weights = np.asarray(self.engine.to_numpy(weights), dtype=np.float64)
        if row_weights.ndim != 1:
            raise ValueError("NumbaKernel.accumulate expects per-row weights with shape (n,).")
        gamma = self.compiled.posteriors(enc, model=self.dist)
        gamma *= row_weights.reshape(-1, 1)
        return self.compiled.weighted_suff_stats(enc, gamma, model=self.dist)

    def refresh(self, dist: SequenceEncodableProbabilityDistribution) -> None:
        """Refresh parameters after an M-step without rebuilding the compiled object."""
        self.dist = dist
        self.compiled.model = dist


class GeneratedNumbaKernel(Kernel):
    """Generated numba kernel from declaration exponential-family metadata."""

    def __init__(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        engine: ComputeEngine = NUMPY_ENGINE,
        estimator: ParameterEstimator | None = None,
    ) -> None:
        if not getattr(engine, "supports_numba", False):
            raise ValueError("GeneratedNumbaKernel requires a numba-capable (host numpy) engine.")
        if not _generated_numba_kernel_available(dist):
            raise ValueError("%s has no declaration-generated numba scorer." % type(dist).__name__)
        self.dist = dist
        self.engine = engine
        self.estimator = estimator
        self.components = _generated_numba_components(dist)

    def encode(self, data: Any) -> Any:
        """Encode raw observations with the distribution's ordinary encoder."""
        return self.dist.dist_to_encoder().seq_encode(data)

    def score(self, enc: Any) -> np.ndarray:
        """Return per-row log densities from declaration-generated numba code."""
        from mixle.stats.compute.declarations import generated_numba_log_density

        enc = getattr(enc, "engine_payload", enc)  # unwrap resident payloads
        if self.components is not None:
            ll = self.component_scores(enc) + np.asarray(self.dist.log_w, dtype=np.float64).reshape(1, -1)
            mx = ll.max(axis=1, keepdims=True)
            good = np.isfinite(mx[:, 0])
            rv = np.full(ll.shape[0], -np.inf)
            rv[good] = np.log(np.exp(ll[good] - mx[good]).sum(axis=1)) + mx[good, 0]
            return rv
        return generated_numba_log_density(self.dist, enc)

    def component_scores(self, enc: Any) -> np.ndarray:
        """Return generated component scores for homogeneous generated mixtures."""
        enc = getattr(enc, "engine_payload", enc)  # unwrap resident payloads
        if self.components is None:
            return super().component_scores(enc)
        return _generated_numba_component_scores(enc, self.components, self.engine)

    def accumulate(self, enc: Any, weights: Any) -> Any:
        """Accumulate generated sufficient statistics for leaves or mixtures."""
        if self.estimator is None:
            raise ValueError("GeneratedNumbaKernel.accumulate requires an estimator.")
        from mixle.stats.compute.declarations import (
            generated_sufficient_statistics,
            generated_sufficient_statistics_available,
        )

        enc = getattr(enc, "engine_payload", enc)  # unwrap resident payloads
        resident_ok = _estimator_resident_supported(self.estimator)
        if self.components is not None:
            row_weights = np.asarray(self.engine.to_numpy(weights), dtype=np.float64)
            try:
                if not resident_ok:
                    raise ValueError("estimator requires host-side sufficient statistics")
                gamma = self.posteriors(enc)
                gamma *= row_weights.reshape(-1, 1)
                component_stats = _generated_numba_component_stats(enc, gamma, self.components, self.engine)
                component_counts = gamma.sum(axis=0)
                return component_counts, _unstack_numba_component_stats(component_stats, len(component_counts))
            except ValueError:
                # A component family has no generated stacked scorer / sufficient-statistic hook (or
                # a width mismatch), or its M-step needs more than resident statistics: fall back to the
                # host mixture accumulator, which handles any component family.
                accumulator = self.estimator.accumulator_factory().make()
                accumulator.seq_update(getattr(enc, "host_payload", enc), row_weights, self.dist)
                return accumulator.value()
        if resident_ok and generated_sufficient_statistics_available(self.dist):
            return generated_sufficient_statistics(self.dist, enc, weights, self.engine)
        # Scorer-only leaf (numba scorer but no generated suff-stat hook): accumulate via the host
        # accumulator so the M-step still receives its expected statistics.
        accumulator = self.estimator.accumulator_factory().make()
        host_enc = getattr(enc, "host_payload", enc)
        row_weights = np.asarray(self.engine.to_numpy(weights), dtype=np.float64)
        accumulator.seq_update(host_enc, row_weights, self.dist)
        return accumulator.value()

    def posteriors(self, enc: Any) -> np.ndarray:
        """Return normalized mixture posterior weights for generated mixtures."""
        if self.components is None:
            ll = self.score(enc).reshape(-1, 1)
            logw = np.zeros((1, 1))
        else:
            logw = np.asarray(self.dist.log_w, dtype=np.float64).reshape(1, -1)
            ll = self.component_scores(enc) + logw
        mx = ll.max(axis=1, keepdims=True)
        # a row with no supporting component has max=-inf, so -inf-(-inf)=nan; fall back to the prior
        # weights for those rows (matches StackedMixtureKernel.posteriors) instead of emitting NaN
        bad = ~np.isfinite(mx[:, 0])
        if bad.any():
            ll[bad] = logw
            mx[bad, 0] = ll[bad].max(axis=1)
        ll = ll - mx
        np.exp(ll, out=ll)
        ll /= ll.sum(axis=1, keepdims=True)
        return ll

    def refresh(self, dist: SequenceEncodableProbabilityDistribution) -> None:
        """Refresh the distribution and regenerated component metadata."""
        if not _generated_numba_kernel_available(dist):
            raise ValueError("%s has no declaration-generated numba scorer." % type(dist).__name__)
        self.dist = dist
        self.components = _generated_numba_components(dist)


class NumbaKernelFactory(KernelFactory):
    """Factory for generated declaration numba kernels with legacy fused fallback."""

    def build(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        engine: ComputeEngine,
        estimator: ParameterEstimator | None = None,
    ) -> Kernel:
        """Prefer generated numba kernels, then fused kernels, then stacked fallback."""
        if _generated_numba_kernel_available(dist):
            return GeneratedNumbaKernel(dist, engine=engine, estimator=estimator)
        try:
            return NumbaKernel(dist, engine=engine, estimator=estimator)
        except ValueError:
            stacked = _stacked_kernel_after_numba_decline(dist, engine, estimator)
            if stacked is not None:
                return stacked
            raise


class GeneratedNumbaKernelFactory(KernelFactory):
    """Default-safe factory that prefers declaration-generated numba kernels.

    Unlike :class:`NumbaKernelFactory`, this never selects the fused
    ``CompiledMixture`` adapter (whose columnar encoding is incompatible with
    the legacy ``seq_encode`` payloads that the engine estimation path feeds
    kernels) and never raises: when a generated numba scorer is unavailable, or
    the engine is not numpy, it defers to a guaranteed fallback (the generic
    kernel). That makes it safe to register as a default on the kernel
    dispatch path while still accelerating mixtures of declared
    exponential-family leaves on the numpy engine.
    """

    def __init__(self, fallback: KernelFactory | None = None) -> None:
        self.fallback = GenericKernelFactory() if fallback is None else fallback

    def build(
        self,
        dist: SequenceEncodableProbabilityDistribution,
        engine: ComputeEngine,
        estimator: ParameterEstimator | None = None,
    ) -> Kernel:
        """Build a generated numba kernel on numpy when available, else fall back."""
        if getattr(engine, "supports_numba", False) and _generated_numba_kernel_available(dist):
            try:
                return GeneratedNumbaKernel(dist, engine=engine, estimator=estimator)
            except ValueError:
                pass
        return self.fallback.build(dist, engine, estimator=estimator)


def _stacked_kernel_after_numba_decline(
    dist: SequenceEncodableProbabilityDistribution, engine: ComputeEngine, estimator: ParameterEstimator | None
) -> Kernel | None:
    if not engine.supports_numba:
        return None
    components = getattr(dist, "components", None)
    log_w = getattr(dist, "log_w", None)
    if components is None or log_w is None:
        return None
    try:
        from mixle.stats.compute.stacked import StackedMixtureKernel

        return StackedMixtureKernel(dist, engine=engine, estimator=estimator)
    except ValueError:
        return None


def _generated_numba_kernel_available(dist: SequenceEncodableProbabilityDistribution) -> bool:
    from mixle.stats.compute.declarations import generated_numba_log_density_available

    if generated_numba_log_density_available(dist):
        return True
    components = _generated_numba_components(dist)
    if components is None:
        return False
    return _generated_numba_components_available(components)


def _generated_numba_components(dist: SequenceEncodableProbabilityDistribution) -> tuple | None:
    components = getattr(dist, "components", None)
    log_w = getattr(dist, "log_w", None)
    if components is None or log_w is None:
        return None
    components = tuple(components)
    return components if components else None


def _generated_numba_components_available(components: tuple) -> bool:
    if not components:
        return False
    component_type = type(components[0])
    if not all(type(component) is component_type for component in components):
        return False
    from mixle.stats.compute.declarations import generated_numba_stacked_available, generated_stacked_params

    if generated_numba_stacked_available(components[0]):
        try:
            generated_stacked_params(components, NUMPY_ENGINE)
        except ValueError:
            return False
        return True
    sequence_child_sets = _generated_numba_sequence_child_sets(components)
    if sequence_child_sets is not None:
        element_set, length_set = sequence_child_sets
        return _generated_numba_components_available(element_set) and (
            length_set is None or _generated_numba_components_available(length_set)
        )
    optional_child_set = _generated_numba_optional_child_set(components)
    if optional_child_set is not None:
        return _generated_numba_components_available(optional_child_set)
    child_sets = _generated_numba_child_component_sets(components)
    return bool(child_sets) and all(_generated_numba_components_available(child_set) for child_set in child_sets)


def _generated_numba_child_component_sets(components: tuple) -> tuple | None:
    child_count = getattr(components[0], "count", None)
    child_dists = getattr(components[0], "dists", None)
    if child_count is None or child_dists is None:
        return None
    child_count = int(child_count)
    if any(getattr(component, "count", None) != child_count for component in components):
        return None
    return tuple(tuple(component.dists[i] for component in components) for i in range(child_count))


def _generated_numba_sequence_child_sets(components: tuple) -> tuple | None:
    required = ("dist", "len_dist", "len_normalized", "null_len_dist")
    if any(not all(hasattr(component, name) for name in required) for component in components):
        return None
    len_normalized = bool(components[0].len_normalized)
    null_len_dist = bool(components[0].null_len_dist)
    if any(
        bool(component.len_normalized) != len_normalized or bool(component.null_len_dist) != null_len_dist
        for component in components
    ):
        return None
    element_set = tuple(component.dist for component in components)
    length_set = None if null_len_dist else tuple(component.len_dist for component in components)
    return element_set, length_set


def _generated_numba_optional_child_set(components: tuple) -> tuple | None:
    required = ("dist", "has_p", "log_p", "log_pn", "missing_value", "missing_value_is_nan")
    if any(not all(hasattr(component, name) for name in required) for component in components):
        return None
    first = components[0]
    if first.missing_value_is_nan:
        if any(not component.missing_value_is_nan for component in components):
            return None
    elif any(
        component.missing_value_is_nan or component.missing_value != first.missing_value for component in components
    ):
        return None
    return tuple(component.dist for component in components)


def _generated_numba_component_scores(enc: Any, components: tuple, engine: ComputeEngine) -> np.ndarray:
    from mixle.stats.compute.declarations import (
        generated_numba_stacked_available,
        generated_numba_stacked_log_density,
        generated_stacked_params,
    )

    if generated_numba_stacked_available(components[0]):
        params = generated_stacked_params(components, engine)
        return generated_numba_stacked_log_density(enc, params)
    sequence_child_sets = _generated_numba_sequence_child_sets(components)
    if sequence_child_sets is not None:
        return _generated_numba_sequence_component_scores(enc, components, sequence_child_sets, engine)
    optional_child_set = _generated_numba_optional_child_set(components)
    if optional_child_set is not None:
        return _generated_numba_optional_component_scores(enc, components, optional_child_set, engine)
    child_sets = _generated_numba_child_component_sets(components)
    if child_sets is None:
        raise ValueError("%s has no declaration-generated numba component scorer." % type(components[0]).__name__)
    scores = _generated_numba_component_scores(enc[0], child_sets[0], engine)
    for idx in range(1, len(child_sets)):
        scores = scores + _generated_numba_component_scores(enc[idx], child_sets[idx], engine)
    return scores


def _generated_numba_component_stats(enc: Any, weights: Any, components: tuple, engine: ComputeEngine) -> Any:
    from mixle.stats.compute.declarations import (
        generated_numba_stacked_available,
        generated_stacked_params,
        generated_stacked_sufficient_statistics,
    )

    if generated_numba_stacked_available(components[0]):
        params = generated_stacked_params(components, engine)
        return generated_stacked_sufficient_statistics(enc, weights, params, engine)
    sequence_child_sets = _generated_numba_sequence_child_sets(components)
    if sequence_child_sets is not None:
        return _generated_numba_sequence_component_stats(enc, weights, components, sequence_child_sets, engine)
    optional_child_set = _generated_numba_optional_child_set(components)
    if optional_child_set is not None:
        return _generated_numba_optional_component_stats(enc, weights, components, optional_child_set, engine)
    child_sets = _generated_numba_child_component_sets(components)
    if child_sets is None:
        raise ValueError("%s has no declaration-generated numba component-stat route." % type(components[0]).__name__)
    return tuple(
        _generated_numba_component_stats(enc[idx], weights, child_set, engine)
        for idx, child_set in enumerate(child_sets)
    )


def _generated_numba_sequence_component_scores(
    enc: Any, components: tuple, child_sets: tuple, engine: ComputeEngine
) -> np.ndarray:
    idx, icnt, _inz, enc_seq, enc_nseq = enc
    element_components, length_components = child_sets
    num_components = len(components)
    nseq = len(icnt)
    rv = np.zeros((nseq, num_components), dtype=np.float64)
    if len(idx):
        idx_arr = np.asarray(idx, dtype=np.int64)
        element_scores = _generated_numba_component_scores(enc_seq, element_components, engine)
        if bool(components[0].len_normalized):
            element_scores = element_scores * np.asarray(icnt, dtype=np.float64)[idx_arr, None]
        np.add.at(rv, idx_arr, element_scores)
    if length_components is not None and enc_nseq is not None:
        rv = rv + _generated_numba_component_scores(enc_nseq, length_components, engine)
    return rv


def _generated_numba_sequence_component_stats(
    enc: Any, weights: Any, components: tuple, child_sets: tuple, engine: ComputeEngine
) -> Any:
    idx, icnt, _inz, enc_seq, enc_nseq = enc
    element_components, length_components = child_sets
    ww = np.asarray(engine.to_numpy(weights), dtype=np.float64)
    num_components = len(components)
    if len(idx):
        idx_arr = np.asarray(idx, dtype=np.int64)
        element_weights = ww[idx_arr, :]
        if bool(components[0].len_normalized):
            element_weights = element_weights * np.asarray(icnt, dtype=np.float64)[idx_arr, None]
    else:
        element_weights = np.zeros((0, num_components), dtype=np.float64)
    element_stats = _generated_numba_component_stats(enc_seq, element_weights, element_components, engine)

    if length_components is None or enc_nseq is None:
        length_stats = None
    else:
        length_stats = _generated_numba_component_stats(enc_nseq, ww, length_components, engine)
    return element_stats, length_stats


def _generated_numba_optional_component_scores(
    enc: Any, components: tuple, child_components: tuple, engine: ComputeEngine
) -> np.ndarray:
    sz, z_idx, nz_idx, enc_data = enc
    num_components = len(components)
    rv = np.zeros((int(sz), num_components), dtype=np.float64)
    has_p = np.asarray([component.has_p for component in components], dtype=bool)
    log_p = np.asarray([component.log_p for component in components], dtype=np.float64)
    log_pn = np.asarray([component.log_pn for component in components], dtype=np.float64)
    if len(z_idx):
        rv[np.asarray(z_idx, dtype=np.int64), :] = np.where(has_p, log_p, 0.0).reshape(1, -1)
    if len(nz_idx):
        child_scores = _generated_numba_component_scores(enc_data, child_components, engine)
        rv[np.asarray(nz_idx, dtype=np.int64), :] = np.where(
            has_p.reshape(1, -1), child_scores + log_pn.reshape(1, -1), child_scores
        )
    return rv


def _generated_numba_optional_component_stats(
    enc: Any, weights: Any, components: tuple, child_components: tuple, engine: ComputeEngine
) -> Any:
    _, z_idx, nz_idx, enc_data = enc
    ww = np.asarray(engine.to_numpy(weights), dtype=np.float64)
    num_components = len(components)
    if len(z_idx):
        missing_counts = ww[np.asarray(z_idx, dtype=np.int64), :].sum(axis=0)
    else:
        missing_counts = np.zeros(num_components, dtype=np.float64)
    if len(nz_idx):
        observed_weights = ww[np.asarray(nz_idx, dtype=np.int64), :]
        observed_counts = observed_weights.sum(axis=0)
    else:
        observed_weights = np.zeros((0, num_components), dtype=np.float64)
        observed_counts = np.zeros(num_components, dtype=np.float64)
    child_stats = _generated_numba_component_stats(enc_data, observed_weights, child_components, engine)
    wrapper_counts = np.column_stack((missing_counts, observed_counts))
    return wrapper_counts, child_stats


def _unstack_numba_component_stats(stats: Any, count: int) -> tuple:
    return tuple(tuple(_numba_component_stat_value(stat, idx) for stat in stats) for idx in range(count))


def _numba_component_stat_value(value: Any, idx: int) -> Any:
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_numba_component_stat_value(child, idx) for child in value)
    if isinstance(value, list):
        return [_numba_component_stat_value(child, idx) for child in value]
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    component = arr[idx]
    if np.asarray(component).ndim == 0:
        return float(component)
    return np.asarray(component).copy()


_GENERIC_FACTORY = GenericKernelFactory()
# Default kernel for unregistered distributions: declaration-generated numba on
# the numpy engine where available, generic everywhere else.
_DEFAULT_FACTORY = GeneratedNumbaKernelFactory()
_KERNEL_FACTORIES: dict[type[Any], KernelFactory] = {}


def register_kernel_factory(dist_type: type[Any], factory: KernelFactory) -> None:
    """Register a specialized kernel factory for a distribution class."""
    _KERNEL_FACTORIES[dist_type] = factory


def kernel_for(
    dist: SequenceEncodableProbabilityDistribution,
    engine: ComputeEngine | None = None,
    estimator: ParameterEstimator | None = None,
) -> Kernel:
    """Build the best registered kernel for ``dist`` and ``engine``."""
    engine = NUMPY_ENGINE if engine is None else engine
    # On a numba-capable engine, a composite/mixture of low-cost leaves runs its whole E-step in ONE fused
    # nopython pass (no per-leaf boundary crossings/allocations) -- ~1.5-2.7x over the per-leaf kernels.
    # Only engage where fusion actually helps (multi-factor or multi-component); single leaves and
    # BLAS-bound / untemplated leaves fall through to the registered factories unchanged.
    if getattr(engine, "prefer_fused", False):
        from mixle.stats.compute.fused_codegen import FusedKernel, analyze, fusible, fusible_estep

        plan = analyze(dist)
        flat_worth_it = plan is not None and (plan.num_components > 1 or len(plan.leaf_templates) > 1)
        # nested scalar trees (Mixture-of-Mixture, Composite-with-Mixture-factor, ...) the flat analyzer
        # declines but the recursive path fuses -- they are always worth fusing (multi-node by construction)
        nested_worth_it = plan is None and fusible(dist)
        if flat_worth_it or nested_worth_it:
            if estimator is None or fusible_estep(dist):
                return FusedKernel(dist, engine, estimator=estimator)
    for cls in type(dist).mro():
        factory = _KERNEL_FACTORIES.get(cls)
        if factory is not None:
            return factory.build(dist, engine, estimator=estimator)
    return _DEFAULT_FACTORY.build(dist, engine, estimator=estimator)
