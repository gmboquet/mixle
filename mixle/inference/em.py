"""Expectation-maximization strategy helpers.

The strategies in this module are deliberately orchestration-level objects:
they move encoded data through existing estimators/kernels and never contain
distribution-specific likelihood math.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mixle.inference.estimation import _engine_seq_estimate, _engine_seq_log_density_sum, _local_encoded_chunks
from mixle.inference.transaction import AlgorithmStateSnapshot, MutableStateSnapshot
from mixle.stats.compute.pdist import (
    ParameterEstimator,
    SequenceEncodableProbabilityDistribution,
    merge_accumulator_keys,
)
from mixle.stats.compute.sequence import seq_estimate, seq_log_density_sum
from mixle.stats.parameter_packing import squarem_packer
from mixle.utils.exact import require_exact_bool


def _annealing_temperature(value: Any, name: str) -> float:
    """Return an annealing temperature as a finite non-negative float.

    Two values used to get through ``if temperature < 0.0`` and then be coerced by ``float(...)``
    (MXR-080-1899). ``float("nan")`` compares false against every bound, and a NaN temperature makes
    ``_transform`` divide the log-responsibilities by NaN: every row comes out non-finite, the
    ``row_sum > 0`` guard then writes zeros, and the M-step silently runs on all-zero
    responsibilities instead of reporting that the schedule was nonsense. And a Boolean is not a
    temperature -- ``True`` would quietly mean ordinary soft EM (1.0) and ``False`` hard EM (0.0),
    which are the two *opposite* algorithms this control selects between.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number (not a Boolean), got {value!r}")
    temperature = float(value)
    if not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {temperature!r}")
    return temperature


@dataclass
class EMStepResult:
    """Result from one EM-family strategy step."""

    model: SequenceEncodableProbabilityDistribution
    objective: float | None = None
    accepted: bool = True
    metadata: dict | None = None


@dataclass(frozen=True)
class SampledSufficientStatistics:
    """Unambiguous Monte-Carlo EM payload for an optional observation count."""

    suff_stat: Any
    nobs: float | None = None

    def __post_init__(self) -> None:
        if self.nobs is not None and (
            isinstance(self.nobs, (bool, np.bool_))
            or not isinstance(self.nobs, (int, float, np.integer, np.floating))
            or not np.isfinite(self.nobs)
            or self.nobs < 0.0
        ):
            raise ValueError("sampled sufficient-statistic nobs must be a finite non-negative number.")


@runtime_checkable
class EMStrategy(Protocol):
    """Structural contract for an EM-family strategy consumed by :func:`run_em`.

    Every strategy object in this module (``StandardEM``, ``PosteriorTransformEM``,
    ``AnnealedEM``, ...) satisfies this Protocol structurally by exposing a
    ``step(...) -> EMStepResult`` method.  ``run_em`` and ``_em_step_fn`` dispatch
    on it polymorphically; membership is decided by :func:`isinstance`.
    """

    def step(
        self,
        enc_data: Any,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution,
        engine: Any | None = ...,
        objective: Callable[[Any], float] | None = ...,
    ) -> EMStepResult:
        """Run one EM-style update and return the resulting model and objective metadata."""
        ...


class StandardEM:
    """The ordinary Dempster-Laird-Rubin EM update with an exact M-step."""

    def step(
        self,
        enc_data: Any,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution,
        engine: Any | None = None,
        objective: Callable[[Any], float] | None = None,
    ) -> EMStepResult:
        """Run one exact EM update and return the new model."""
        if engine is None:
            new_model = seq_estimate(enc_data, estimator, model)
        else:
            new_model = _engine_seq_estimate(enc_data, estimator, model, engine)
        return EMStepResult(new_model)


class CompiledEM:
    """Local full-mixture EM using the profiled fused component kernels.

    This is the reusable form of the full-tree execution path used by block EM. It keeps the
    ordinary EM fixed point and update semantics, but can avoid generic accumulator traversal for
    fusible subtrees of a :class:`~mixle.stats.MixtureDistribution`. Unsupported models or remote
    execution engines fall back to :class:`StandardEM`.
    """

    def __init__(self, *, compute_dtype: Any = None) -> None:
        self.compute_dtype = compute_dtype
        self._cache: Any | None = None

    def step(
        self,
        enc_data: Any,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution,
        engine: Any | None = None,
        objective: Callable[[Any], float] | None = None,
    ) -> EMStepResult:
        """Run one compiled full-mixture E/M sweep."""
        from mixle.inference.freeze_rollup import (
            FreezeRollupCache,
            compiled_em_step,
            is_compiled_em_eligible,
        )

        if engine is not None or not is_compiled_em_eligible(model, estimator):
            result = StandardEM().step(enc_data, estimator, model, engine=engine, objective=objective)
            result.metadata = {"compiled": False, "fallback": "unsupported_execution_shape"}
            return result

        if self._cache is None:
            self._cache = FreezeRollupCache()
        candidate, metadata = compiled_em_step(
            enc_data,
            estimator,
            model,
            compute_dtype=self.compute_dtype,
            cache=self._cache,
        )
        return EMStepResult(candidate, metadata=metadata)


class PosteriorTransformEM:
    """EM update that transforms mixture posteriors before the M-step.

    ``temperature=1`` gives the usual soft EM responsibilities. ``hard=True``
    gives classification/hard EM. Intermediate temperatures implement a simple
    deterministic-annealing style generalized EM update.
    """

    def __init__(self, temperature: float = 1.0, hard: bool = False) -> None:
        self.temperature = _annealing_temperature(temperature, "temperature")
        # `bool(hard)` made `hard="false"` a HARD/classification EM run (MXR-080-1899): a non-empty
        # string is truthy, so a strategy configured from serialized text with the word that names
        # the opposite ran a different algorithm than the configuration says, and the fitted model
        # is the only place that difference shows up.
        self.hard = require_exact_bool(hard, "hard")

    def step(
        self,
        enc_data: Any,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution,
        engine: Any | None = None,
        objective: Callable[[Any], float] | None = None,
    ) -> EMStepResult:
        """Run one posterior-transformed E-step followed by the estimator M-step."""
        if not _is_mixture_like(model):
            raise TypeError("PosteriorTransformEM requires a mixture-like model with components and seq_posterior.")
        acc = estimator.accumulator_factory().make()
        nobs = 0.0
        for sz, enc in _local_encoded_chunks(enc_data):
            gamma = _posterior_matrix(model, enc, engine)
            gamma = self._transform(gamma)
            acc.combine(_mixture_stats_from_gamma(model, estimator, enc, gamma))
            nobs += sz
        # The same key_merge/key_replace pass seq_estimate runs after accumulation: without it,
        # keyed (tied) parameters are silently untied under HardEM/AnnealedEM/PosteriorTransformEM.
        merge_accumulator_keys(acc)
        return EMStepResult(estimator.estimate(nobs, acc.value()))

    def _transform(self, gamma: np.ndarray) -> np.ndarray:
        if self.hard or self.temperature == 0.0:
            idx = np.argmax(gamma, axis=1)
            rv = np.zeros_like(gamma)
            rv[np.arange(gamma.shape[0]), idx] = 1.0
            return rv
        if self.temperature == 1.0:
            return gamma
        with np.errstate(divide="ignore", invalid="ignore"):
            log_gamma = np.log(gamma)
            log_gamma /= self.temperature
            log_gamma -= np.max(log_gamma, axis=1, keepdims=True)
            rv = np.exp(log_gamma)
            row_sum = rv.sum(axis=1, keepdims=True)
            return np.divide(rv, row_sum, out=np.zeros_like(rv), where=row_sum > 0.0)


class HardEM(PosteriorTransformEM):
    """Classification EM using maximum-posterior component assignments."""

    def __init__(self) -> None:
        super().__init__(temperature=0.0, hard=True)


class AnnealedEM:
    """Deterministic-annealing EM over a temperature schedule.

    Temperatures greater than one flatten mixture responsibilities early in a
    run, then later entries in the schedule can cool toward ordinary EM at
    temperature one or hard/classification EM at temperature zero.  The object
    owns only the schedule; posterior math and M-steps remain delegated to
    ``PosteriorTransformEM`` and the estimator.
    """

    def __init__(self, temperatures: Sequence[float], hard_final: bool = False) -> None:
        if len(temperatures) == 0:
            raise ValueError("AnnealedEM requires at least one temperature.")
        self.temperatures = tuple(_annealing_temperature(t, f"temperatures[{i}]") for i, t in enumerate(temperatures))
        self.hard_final = require_exact_bool(hard_final, "hard_final")
        self.iteration = 0

    @property
    def current_temperature(self) -> float:
        """Return the schedule temperature for the next annealed step."""
        idx = min(self.iteration, len(self.temperatures) - 1)
        return self.temperatures[idx]

    def step(
        self,
        enc_data: Any,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution,
        engine: Any | None = None,
        objective: Callable[[Any], float] | None = None,
    ) -> EMStepResult:
        """Run one annealed posterior-transform EM step and advance the schedule."""
        temperature = self.current_temperature
        hard = self.hard_final and self.iteration >= len(self.temperatures) - 1 and temperature == 0.0
        result = PosteriorTransformEM(temperature=temperature, hard=hard).step(
            enc_data, estimator, model, engine=engine, objective=objective
        )
        self.iteration += 1
        return result

    def reset(self) -> None:
        """Restart the annealing schedule for a new EM run."""
        self.iteration = 0


class GeneralizedEM:
    """Generalized EM wrapper around a caller-supplied candidate step.

    The candidate function is called as
    ``candidate_fn(enc_data, estimator, model, engine)``.  When
    ``require_improvement`` is true, the candidate is accepted only if the
    supplied objective (or observed log likelihood by default) does not
    decrease.
    """

    def __init__(
        self,
        candidate_fn: Callable[[Any, ParameterEstimator, Any, Any | None], Any],
        require_improvement: bool = True,
    ) -> None:
        self.candidate_fn = candidate_fn
        self.require_improvement = require_exact_bool(require_improvement, "require_improvement")

    def step(
        self,
        enc_data: Any,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution,
        engine: Any | None = None,
        objective: Callable[[Any], float] | None = None,
    ) -> EMStepResult:
        """Evaluate and optionally objective-gate one caller-supplied GEM step."""
        objective = observed_log_likelihood(enc_data, engine=engine) if objective is None else objective
        candidate = self.candidate_fn(enc_data, estimator, model, engine)
        if not self.require_improvement:
            return EMStepResult(candidate, objective(candidate), True)
        old_value = objective(model)
        new_value = objective(candidate)
        if new_value + 1.0e-12 >= old_value:
            return EMStepResult(candidate, new_value, True)
        return EMStepResult(model, old_value, False)


class MonotonicEM:
    """Objective-gated wrapper that rejects log-likelihood-decreasing or non-finite steps.

    Wraps any base EM-family strategy (``StandardEM`` by default). After the base step it
    evaluates the objective on the candidate; if the candidate objective is non-finite, or
    (with ``require_improvement``) it decreases beyond ``tolerance``, the previous model is
    kept and the step is marked rejected. This is the robust-path guard against the
    singular-covariance / NaN cascade and against EM steps that overshoot.
    """

    def __init__(
        self,
        base_strategy: Any | None = None,
        require_improvement: bool = True,
        tolerance: float = 1.0e-9,
    ) -> None:
        self.base_strategy = StandardEM() if base_strategy is None else base_strategy
        self.require_improvement = require_exact_bool(require_improvement, "require_improvement")
        self.tolerance = float(tolerance)

    def step(
        self,
        enc_data: Any,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution,
        engine: Any | None = None,
        objective: Callable[[Any], float] | None = None,
    ) -> EMStepResult:
        """Run the base step, then reject it if the objective is non-finite or decreases."""
        objective = observed_log_likelihood(enc_data, engine=engine) if objective is None else objective
        old_value = objective(model)
        mutable_state = MutableStateSnapshot.capture(model, estimator, self.base_strategy)
        strategy_state = AlgorithmStateSnapshot.capture(
            self.base_strategy, enc_data, estimator, model, engine, objective
        )
        try:
            base_result = self.base_strategy.step(enc_data, estimator, model, engine=engine, objective=objective)
            candidate = base_result.model
            new_value = objective(candidate) if base_result.objective is None else base_result.objective
        except (np.linalg.LinAlgError, FloatingPointError, ValueError, RuntimeError):
            # M-step blew up (e.g. a singular covariance slipped through): keep the last good model.
            mutable_state.restore()
            strategy_state.restore()
            return EMStepResult(model, old_value, False, metadata={"rejected": "exception"})
        except Exception:
            mutable_state.restore()
            strategy_state.restore()
            raise

        if not base_result.accepted:
            mutable_state.restore()
            strategy_state.restore()
            return EMStepResult(model, old_value, False, metadata={"rejected": "base_strategy"})
        if not np.isfinite(new_value):
            mutable_state.restore()
            strategy_state.restore()
            return EMStepResult(model, old_value, False, metadata={"rejected": "nonfinite"})
        if self.require_improvement and new_value + self.tolerance < old_value:
            mutable_state.restore()
            strategy_state.restore()
            return EMStepResult(model, old_value, False, metadata={"rejected": "decrease"})
        return EMStepResult(candidate, new_value, True)


class ConditionalMaximizationEM:
    """Expectation/conditional-maximization over caller-supplied CM steps."""

    def __init__(
        self,
        conditional_steps: Sequence[Callable[[Any, ParameterEstimator, Any, Any | None], Any]],
        require_improvement: bool = True,
    ) -> None:
        if len(conditional_steps) == 0:
            raise ValueError("ConditionalMaximizationEM requires at least one conditional step.")
        self.conditional_steps = tuple(conditional_steps)
        self.require_improvement = require_exact_bool(require_improvement, "require_improvement")

    def step(
        self,
        enc_data: Any,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution,
        engine: Any | None = None,
        objective: Callable[[Any], float] | None = None,
    ) -> EMStepResult:
        """Run each conditional maximization step with optional objective gates."""
        objective = observed_log_likelihood(enc_data, engine=engine) if objective is None else objective
        current = model
        current_value = objective(current)
        accepted = True
        for step_fn in self.conditional_steps:
            candidate = step_fn(enc_data, estimator, current, engine)
            candidate_value = objective(candidate)
            if (not self.require_improvement) or candidate_value + 1.0e-12 >= current_value:
                current = candidate
                current_value = candidate_value
            else:
                accepted = False
        return EMStepResult(current, current_value, accepted)


class MonteCarloEM:
    """Monte-Carlo EM over sampled sufficient statistics.

    ``sample_suff_stat_fn`` is called as
    ``fn(enc_data, estimator, model, rng, num_samples, engine)``. Return a bare
    sufficient statistic, or :class:`SampledSufficientStatistics` when the
    estimator also needs an explicit observation count. A two-tuple is always
    treated as a statistic because tuple-valued statistics are common.
    """

    def __init__(
        self,
        sample_suff_stat_fn: Callable[[Any, ParameterEstimator, Any, np.random.RandomState, int, Any | None], Any],
        num_samples: int = 1,
        seed: int | None = None,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        self.sample_suff_stat_fn = sample_suff_stat_fn
        self.num_samples = int(num_samples)
        self.rng = np.random.RandomState(seed)

    def step(
        self,
        enc_data: Any,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution,
        engine: Any | None = None,
        objective: Callable[[Any], float] | None = None,
    ) -> EMStepResult:
        """Estimate sufficient statistics by sampling latent completions."""
        sampled = self.sample_suff_stat_fn(enc_data, estimator, model, self.rng, self.num_samples, engine)
        nobs, suff_stat = _split_suff_stat(sampled)
        candidate = estimator.estimate(nobs, suff_stat)
        value = None if objective is None else objective(candidate)
        return EMStepResult(candidate, value, True)


class VariationalEM:
    """Free-energy EM over an explicit variational state.

    ``variational_step_fn`` updates or creates the variational state.  The
    ``m_step_fn`` maps that state to a new model.  A supplied
    ``free_energy_fn`` can report the model/state objective without requiring
    the generic observed-likelihood objective to know about the variational
    state.
    """

    def __init__(
        self,
        variational_step_fn: Callable[[Any, ParameterEstimator, Any, Any, Any | None], Any],
        m_step_fn: Callable[[Any, ParameterEstimator, Any, Any, Any | None], Any],
        initial_state: Any = None,
        free_energy_fn: Callable[[Any, ParameterEstimator, Any, Any, Any | None], float] | None = None,
    ) -> None:
        self.variational_step_fn = variational_step_fn
        self.m_step_fn = m_step_fn
        self.state = initial_state
        self.free_energy_fn = free_energy_fn

    def step(
        self,
        enc_data: Any,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution,
        engine: Any | None = None,
        objective: Callable[[Any], float] | None = None,
    ) -> EMStepResult:
        """Update the variational state, then map it to a candidate model."""
        self.state = self.variational_step_fn(enc_data, estimator, model, self.state, engine)
        candidate = self.m_step_fn(enc_data, estimator, model, self.state, engine)
        if self.free_energy_fn is not None:
            value = self.free_energy_fn(enc_data, estimator, candidate, self.state, engine)
        elif objective is not None:
            value = objective(candidate)
        else:
            value = None
        return EMStepResult(candidate, value, True)


class OnlineEM:
    """Decay-mode stochastic/online EM over encoded mini-batches.

    This adapter exposes ``StreamingEstimator`` through the strategy interface
    used by ``run_em``: each step folds one batch into decayed sufficient
    statistics and then reuses the estimator's ordinary M-step.
    """

    def __init__(
        self,
        schedule: Callable[[int], float] | None = None,
        init_estimator: ParameterEstimator | None = None,
        init_p: float = 0.1,
        rng: np.random.RandomState | None = None,
        encoder: Any | None = None,
        num_chunks: int = 1,
        dataset_size: float | None = None,
    ) -> None:
        self.schedule = schedule
        self.init_estimator = init_estimator
        self.init_p = init_p
        self.rng = rng
        self.encoder = encoder
        self.num_chunks = num_chunks
        self.dataset_size = dataset_size
        self._stream = None

    def step(
        self,
        enc_data: Any,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution,
        engine: Any | None = None,
        objective: Callable[[Any], float] | None = None,
    ) -> EMStepResult:
        """Fold one mini-batch into decayed sufficient statistics."""
        stream = self._ensure_stream(estimator, model)
        stream.model = model
        candidate = stream.update(enc_data=enc_data)
        value = None if objective is None else objective(candidate)
        return EMStepResult(
            candidate,
            value,
            True,
            metadata={
                "online_step": stream.step,
                "nobs": stream.nobs,
                "dataset_size": stream.dataset_size,
                "batch_scale": stream.last_batch_scale,
            },
        )

    def reset(self) -> None:
        """Drop running statistics before a new online EM run."""
        if self._stream is not None:
            self._stream.reset()
        self._stream = None

    def _ensure_stream(self, estimator: ParameterEstimator, model: SequenceEncodableProbabilityDistribution) -> Any:
        if self._stream is None:
            from mixle.inference.streaming import StreamingEstimator

            self._stream = StreamingEstimator(
                estimator,
                schedule=self.schedule,
                model=model,
                init_estimator=self.init_estimator,
                init_p=self.init_p,
                rng=self.rng,
                encoder=self.encoder,
                num_chunks=self.num_chunks,
                dataset_size=self.dataset_size,
            )
        elif self._stream.estimator is not estimator:
            raise ValueError("OnlineEM cannot change estimator after the first step; call reset().")
        return self._stream


class IncrementalEM:
    """Neal-Hinton style incremental EM over replaceable encoded chunks.

    Revisited chunks replace their previous sufficient-statistic contribution,
    allowing repeated passes over partitioned data without re-accumulating the
    whole dataset each iteration.
    """

    def __init__(
        self,
        chunk_id_fn: Callable[[Any, ParameterEstimator, Any, Any | None], Any] | None = None,
        init_estimator: ParameterEstimator | None = None,
        init_p: float = 0.1,
        rng: np.random.RandomState | None = None,
        encoder: Any | None = None,
        num_chunks: int = 1,
    ) -> None:
        self.chunk_id_fn = chunk_id_fn
        self.init_estimator = init_estimator
        self.init_p = init_p
        self.rng = rng
        self.encoder = encoder
        self.num_chunks = num_chunks
        self._incremental = None

    def step(
        self,
        enc_data: Any,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution,
        engine: Any | None = None,
        objective: Callable[[Any], float] | None = None,
    ) -> EMStepResult:
        """Replace the chunk chosen by ``chunk_id_fn`` and update the model."""
        if self.chunk_id_fn is None:
            raise ValueError("IncrementalEM.step requires chunk_id_fn or use step_chunk(...).")
        chunk_id = self.chunk_id_fn(enc_data, estimator, model, engine)
        return self.step_chunk(chunk_id, enc_data, estimator, model, engine=engine, objective=objective)

    def step_chunk(
        self,
        chunk_id: Any,
        enc_data: Any,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution,
        engine: Any | None = None,
        objective: Callable[[Any], float] | None = None,
    ) -> EMStepResult:
        """Replace one named chunk's sufficient statistics and update the model."""
        incremental = self._ensure_incremental(estimator, model)
        incremental.model = model
        candidate = incremental.update(enc_data=enc_data, chunk_id=chunk_id)
        value = None if objective is None else objective(candidate)
        return EMStepResult(
            candidate,
            value,
            True,
            metadata={
                "chunk_id": chunk_id,
                "incremental_step": incremental.step,
                "nobs": incremental.nobs,
            },
        )

    def chunk_value(self, chunk_id: Any) -> Any:
        """Return a stored chunk sufficient-statistic payload."""
        if self._incremental is None:
            raise KeyError(chunk_id)
        return self._incremental.chunk_value(chunk_id)

    def reset(self) -> None:
        """Drop stored chunks and running statistics before a new incremental EM run."""
        self._incremental = None

    def _ensure_incremental(
        self, estimator: ParameterEstimator, model: SequenceEncodableProbabilityDistribution
    ) -> Any:
        if self._incremental is None:
            from mixle.inference.streaming import IncrementalEstimator

            self._incremental = IncrementalEstimator(
                estimator,
                model=model,
                init_estimator=self.init_estimator,
                init_p=self.init_p,
                rng=self.rng,
                encoder=self.encoder,
                num_chunks=self.num_chunks,
            )
        elif self._incremental.estimator is not estimator:
            raise ValueError("IncrementalEM cannot change estimator after the first step; call reset().")
        return self._incremental


class AcceleratedEM:
    """Objective-gated acceleration wrapper around an EM-family strategy.

    The wrapped ``base_strategy`` performs the ordinary EM/GEM step.  The
    caller-supplied ``proposal_fn`` may then propose extrapolated candidates
    from ``(old_model, base_model, step_factor, enc_data, estimator, engine)``.
    This class owns only the orchestration and objective gate; model-specific
    extrapolation stays with the caller/model layer.
    """

    def __init__(
        self,
        proposal_fn: Callable[[Any, Any, float, Any, ParameterEstimator, Any | None], Any],
        base_strategy: Any | None = None,
        step_factors: Sequence[float] = (1.0, 0.5, 0.25),
        require_improvement: bool = True,
        tolerance: float = 1.0e-12,
    ) -> None:
        if not callable(proposal_fn):
            raise TypeError("AcceleratedEM requires a callable proposal_fn.")
        if len(step_factors) == 0:
            raise ValueError("AcceleratedEM requires at least one step factor.")
        self.step_factors = tuple(float(v) for v in step_factors)
        if any((not np.isfinite(v)) or v <= 0.0 for v in self.step_factors):
            raise ValueError("step_factors must be positive finite values.")
        self.proposal_fn = proposal_fn
        self.base_strategy = StandardEM() if base_strategy is None else base_strategy
        self.require_improvement = require_exact_bool(require_improvement, "require_improvement")
        self.tolerance = float(tolerance)

    def step(
        self,
        enc_data: Any,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution,
        engine: Any | None = None,
        objective: Callable[[Any], float] | None = None,
    ) -> EMStepResult:
        """Run the base strategy, test extrapolated candidates, and keep the best."""
        objective = observed_log_likelihood(enc_data, engine=engine) if objective is None else objective
        old_value = objective(model)
        base_result = self.base_strategy.step(enc_data, estimator, model, engine=engine, objective=objective)
        base_value = objective(base_result.model) if base_result.objective is None else base_result.objective

        if self.require_improvement and base_value + self.tolerance < old_value:
            return EMStepResult(
                model,
                old_value,
                False,
                metadata={
                    "accelerated": False,
                    "base_accepted": False,
                    "base_objective": base_value,
                    "old_objective": old_value,
                    "step_factor": None,
                },
            )

        best_model = base_result.model
        best_value = base_value
        best_factor = None
        for factor in self.step_factors:
            candidate = self.proposal_fn(model, base_result.model, factor, enc_data, estimator, engine)
            candidate_value = objective(candidate)
            if candidate_value > best_value + self.tolerance and (
                (not self.require_improvement) or candidate_value + self.tolerance >= old_value
            ):
                best_model = candidate
                best_value = candidate_value
                best_factor = factor

        return EMStepResult(
            best_model,
            best_value,
            True,
            metadata={
                "accelerated": best_factor is not None,
                "base_accepted": True,
                "base_objective": base_value,
                "old_objective": old_value,
                "step_factor": best_factor,
            },
        )


class SquaremEM:
    """SQUAREM acceleration (Varadhan & Roland 2008, SqS3) with an objective gate.

    One ``step`` runs one SQUAREM cycle -- THREE base-strategy sweeps: two ordinary EM sweeps give
    the secant pair ``r = theta1 - theta0``, ``v = (theta2 - theta1) - r``; the squared-extrapolation
    proposal ``theta0 - 2*alpha*r + alpha^2*v`` with ``alpha = min(-1, -sqrt(r.r/v.v))`` is then
    stabilized by a third EM sweep. The gate keeps the stabilized proposal only if its objective is
    at least the plain two-sweep result's; otherwise the cycle falls back to that plain result, so
    the accepted sequence is monotone whenever the base strategy is (StandardEM's exact M-step is).
    An unpack that rejects the proposal (a constraint violation surfacing as ``ValueError`` /
    ``FloatingPointError`` / ``OverflowError``) counts as a rejected cycle, never a crash.

    Cost accounting is honest: ``max_its`` iterations of ``optimize(strategy=SquaremEM())`` spend
    up to ``3 * max_its`` E-step sweeps. The probe receipt for the win (overlapping 6-component
    GMM, n=100k): plain EM needed 200 sweeps to a target log-likelihood that SQUAREM reached in 33
    sweeps, every cycle accepted, monotone throughout.

    ``packer``: ``(pack, unpack)`` between the model and an unconstrained parameter vector; default
    :func:`squarem_packer` (supported families documented there).

    The strategy is STATEFUL across steps (the adaptive trust-region cap on ``alpha``): use a fresh
    instance per fit, exactly as ``optimize(strategy=SquaremEM())`` constructs one.
    """

    def __init__(
        self,
        base_strategy: Any | None = None,
        packer: tuple[Callable[[Any], np.ndarray], Callable[[np.ndarray], Any]] | None = None,
        tolerance: float = 1.0e-12,
        step_growth: float = 4.0,
    ) -> None:
        self.base_strategy = StandardEM() if base_strategy is None else base_strategy
        self.packer = packer
        self.tolerance = float(tolerance)
        if step_growth <= 1.0:
            raise ValueError("step_growth must be > 1 (it is the trust-region growth factor).")
        self.step_growth = float(step_growth)
        # Trust-region cap on |alpha| (Varadhan & Roland's adaptive maximum step). Raw SqS3 alpha
        # EXPLODES near the fixed point (the secant ratio -sqrt(r.r/v.v) reaches -100 on the probe
        # fixture) and every uncapped proposal overshoots into the objective gate -- each rejection
        # then wastes the cycle's third sweep and SQUAREM decays to slower-than-plain EM. Capping
        # |alpha| and growing the cap by `step_growth` only when a boundary step succeeds (shrinking
        # it back on rejection) keeps proposals inside the region the gate accepts. alpha = -1
        # reproduces the plain two-sweep result exactly, so the cap can never make a cycle worse.
        self._alpha_cap = self.step_growth

    def step(
        self,
        enc_data: Any,
        estimator: ParameterEstimator,
        model: SequenceEncodableProbabilityDistribution,
        engine: Any | None = None,
        objective: Callable[[Any], float] | None = None,
    ) -> EMStepResult:
        """One SQUAREM cycle: two base sweeps, one gated extrapolation sweep."""
        objective = observed_log_likelihood(enc_data, engine=engine) if objective is None else objective
        pack, unpack = self.packer if self.packer is not None else squarem_packer(model)

        theta0 = pack(model)
        m1 = self.base_strategy.step(enc_data, estimator, model, engine=engine, objective=objective).model
        m2 = self.base_strategy.step(enc_data, estimator, m1, engine=engine, objective=objective).model
        base_value = objective(m2)

        r = pack(m1) - theta0
        v = (pack(m2) - pack(m1)) - r
        vv = float(v @ v)
        meta = {"squarem_alpha": None, "accelerated": False, "sweeps": 2, "fallback": None}
        if vv <= 0.0 or not np.isfinite(vv):
            meta["fallback"] = "degenerate_secant"  # theta already at (or numerically at) the fixed point
            return EMStepResult(m2, base_value, True, metadata=meta)

        alpha_raw = -float(np.sqrt(float(r @ r) / vv))
        alpha = -min(max(1.0, -alpha_raw), self._alpha_cap)
        meta["squarem_alpha"] = alpha
        meta["squarem_alpha_raw"] = alpha_raw
        meta["squarem_alpha_cap"] = self._alpha_cap
        try:
            proposal = unpack(theta0 - 2.0 * alpha * r + alpha * alpha * v)
            stabilized = self.base_strategy.step(
                enc_data, estimator, proposal, engine=engine, objective=objective
            ).model
            meta["sweeps"] = 3
            stabilized_value = objective(stabilized)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            meta["fallback"] = "invalid_proposal:%s" % type(exc).__name__
            self._alpha_cap = max(1.0, self._alpha_cap / self.step_growth)
            return EMStepResult(m2, base_value, True, metadata=meta)

        if np.isfinite(stabilized_value) and stabilized_value + self.tolerance >= base_value:
            meta["accelerated"] = True
            if alpha <= -self._alpha_cap + 1e-12:  # accepted a boundary step: widen the trust region
                self._alpha_cap *= self.step_growth
            return EMStepResult(stabilized, float(stabilized_value), True, metadata=meta)
        meta["fallback"] = "objective_gate"
        self._alpha_cap = max(1.0, self._alpha_cap / self.step_growth)
        return EMStepResult(m2, base_value, True, metadata=meta)


class RestartEM:
    """Run an EM-family strategy from several initial models and keep the best."""

    def __init__(
        self,
        initial_models: Sequence[SequenceEncodableProbabilityDistribution],
        strategy: Any | None = None,
        max_its: int = 10,
        delta: float | None = 1.0e-9,
        max_iter: int | None = None,
    ) -> None:
        if len(initial_models) == 0:
            raise ValueError("RestartEM requires at least one initial model.")
        if max_iter is not None:
            max_its = max_iter
        self.initial_models = tuple(initial_models)
        self.strategy = StandardEM() if strategy is None else strategy
        self.max_its = int(max_its)
        self.delta = delta

    def run(
        self,
        enc_data: Any,
        estimator: ParameterEstimator,
        engine: Any | None = None,
        objective: Callable[[Any], float] | None = None,
    ) -> SequenceEncodableProbabilityDistribution:
        """Run each initial model through EM and return the best final model."""
        objective = observed_log_likelihood(enc_data, engine=engine) if objective is None else objective
        best_model = None
        best_value = -np.inf
        roots = (estimator, *self.initial_models)
        mutable_baseline = MutableStateSnapshot.capture(*roots)
        algorithm_baseline = AlgorithmStateSnapshot.capture(roots, enc_data, engine, objective)
        try:
            for initial in self.initial_models:
                mutable_baseline.restore()
                algorithm_baseline.restore()
                try:
                    strategy = copy.deepcopy(self.strategy)
                except (TypeError, ValueError, RuntimeError) as exc:
                    raise TypeError(
                        "RestartEM requires a strategy whose initial state can be copied independently "
                        "for each restart."
                    ) from exc
                candidate = run_em(
                    enc_data,
                    estimator,
                    initial,
                    strategy=strategy,
                    max_its=self.max_its,
                    delta=self.delta,
                    engine=engine,
                    objective=objective,
                )
                value = objective(candidate)
                if best_model is None or value > best_value:
                    best_model = copy.deepcopy(candidate)
                    best_value = value
        finally:
            mutable_baseline.restore()
            algorithm_baseline.restore()
        return best_model


def _resolve_run_em_objective(
    objective: str | Callable[[Any], float] | None,
    enc_data: Any,
    estimator: ParameterEstimator,
    initial_model: SequenceEncodableProbabilityDistribution,
    engine: Any | None,
) -> Callable[[Any], float]:
    """Resolve ``run_em``'s ``objective`` into a ``model -> float`` scorer.

    Accepts the same spellings the high-level verbs do, so ``objective='map'`` means the same thing in
    ``run_em`` as in :func:`~mixle.inference.estimation.optimize`:

      * ``None`` -- observed-data log-likelihood (MLE), the historical default;
      * a selection string ``'auto'`` / ``'mle'`` / ``'map'`` / ``'vb'`` -- resolved against the
        estimator's prior exactly like ``optimize`` / ``fit`` and bound over ``enc_data``;
      * a ready ``model -> float`` callable -- used as-is (the power-user escape hatch).
    """
    if objective is None:
        return observed_log_likelihood(enc_data, engine=engine)
    if callable(objective):
        return objective
    from mixle.inference.estimation import _objective_scorer, _resolve_objective

    resolved = _resolve_objective(objective, estimator, initial_model)
    scorer = _objective_scorer(resolved, estimator, engine)
    return lambda model: scorer(enc_data, model)[1]


def _attach_run_em_provenance(
    model: SequenceEncodableProbabilityDistribution,
    *,
    strategy: Any,
    estimator: ParameterEstimator,
    objective_label: str,
    iterations: int,
    max_its: int,
    converged: bool,
    delta: float | None,
    final_objective: float | None,
    objective_gain: float | None,
    enc_data: Any,
) -> SequenceEncodableProbabilityDistribution:
    """Attach the run's :class:`~mixle.stats.compute.pdist.FitProvenance` receipt; never raises.

    Reuses ``optimize``'s recorder so a model fitted through ``run_em`` answers ``fit_provenance()``
    the same way: without it, a run stopped by the ``max_its`` cap (10 by default -- routinely still
    on the initialization plateau) was indistinguishable from one that converged on ``delta``.
    """
    from mixle.inference.estimation import _FitTrace, _record_fit_provenance

    trace = _FitTrace()
    trace.iterations = int(iterations)
    trace.converged = bool(converged)
    trace.final_objective = final_objective
    trace.objective_gain = objective_gain
    return _record_fit_provenance(
        model,
        trace,
        algorithm=f"run_em[{type(strategy).__name__}]",
        estimator=estimator,
        objective=objective_label,
        max_its=max(1, int(max_its)),
        delta=delta,
        enc_data=enc_data,
        seed=None,
    )


def _rejected_step_reason(result: EMStepResult, value: float, gain: float) -> str:
    """Describe, for the stalled-run disclosure, why an EM step was not accepted."""
    if not np.isfinite(value):
        return "its first step scored a non-finite objective"
    if not bool(result.accepted):
        detail = (result.metadata or {}).get("rejected")
        return "the strategy rejected its own first step" + (" (%s)" % detail if detail else "")
    return "its first step decreased the objective by %.3g" % (-gain)


def run_em(
    enc_data: Any,
    estimator: ParameterEstimator,
    initial_model: SequenceEncodableProbabilityDistribution,
    strategy: EMStrategy | None = None,
    max_its: int = 10,
    delta: float | None = 1.0e-9,
    engine: Any | None = None,
    objective: str | Callable[[Any], float] | None = None,
    max_iter: int | None = None,
    monotone: bool = True,
    tolerance: float = 1.0e-12,
) -> SequenceEncodableProbabilityDistribution:
    """Run an EM-family strategy until convergence or ``max_its``.

    ``objective`` takes the same values as :func:`~mixle.inference.estimation.optimize`: ``None`` (MLE),
    a selection string (``'auto'`` / ``'mle'`` / ``'map'`` / ``'vb'``), or a ready ``model -> float``
    callable. ``max_its`` is the canonical iteration-cap spelling (matching ``optimize`` / ``fit`` /
    ``best_of``); ``max_iter`` is accepted as a back-compat alias and overrides ``max_its`` when given.
    ``monotone=True`` transactionally rejects an objective decrease independently of whether
    ``delta`` is disabled. Set it to ``False`` only for a strategy whose changing surrogate objective
    intentionally permits temporary decreases in the supplied reporting objective.

    The returned model carries a :class:`~mixle.stats.compute.pdist.FitProvenance` receipt
    (``model.fit_provenance()``): iterations actually run, whether the run converged on ``delta`` or
    stopped at the ``max_its`` cap (or on a rejected step), and the final objective. A run exiting on
    the cap reports ``converged=False``, so a truncated fit is distinguishable from a finished one.
    A run whose very first step is rejected accepts nothing and returns ``initial_model`` unchanged;
    that case additionally raises a ``UserWarning``, because the returned model is the
    initialization rather than a fit and the receipt alone does not say so.
    """
    if max_iter is not None:
        max_its = max_iter
    strategy = StandardEM() if strategy is None else strategy
    objective_label = "mle" if objective is None else (objective if isinstance(objective, str) else "custom")
    objective = _resolve_run_em_objective(objective, enc_data, estimator, initial_model, engine)
    model = initial_model
    last_good = model
    old_value = objective(model)
    iterations = 0
    converged = False
    accepted_steps = 0
    stall_reason: str | None = None
    final_objective: float | None = float(old_value) if np.isfinite(old_value) else None
    objective_gain: float | None = None
    for _ in range(max(1, int(max_its))):
        transaction = MutableStateSnapshot.capture(model, estimator, strategy)
        strategy_transaction = AlgorithmStateSnapshot.capture(strategy, enc_data, estimator, model, engine, objective)
        iterations += 1
        try:
            result = strategy.step(enc_data, estimator, model, engine=engine, objective=objective)
            candidate = result.model
            value = objective(candidate) if result.objective is None else result.objective
            gain = value - old_value
            accepted = bool(result.accepted) and np.isfinite(value) and ((not monotone) or gain >= -float(tolerance))
        except Exception:
            transaction.restore()
            strategy_transaction.restore()
            raise
        if not accepted:
            transaction.restore()
            strategy_transaction.restore()
            if accepted_steps == 0:
                stall_reason = _rejected_step_reason(result, value, gain)
            break
        accepted_steps += 1
        model = candidate
        last_good = model
        final_objective = float(value)
        objective_gain = float(gain)
        if delta is not None and 0.0 <= gain < delta:
            converged = True
            break
        old_value = value
    if stall_reason is not None:
        # A run that accepts nothing returns `initial_model` byte-for-byte, and the receipt still
        # reports the full observation count -- so a stalled run is otherwise indistinguishable from
        # a fit, and the initialization (typically a random init_p subsample) reads as the answer.
        # Warn rather than raise: the initialization is a legitimate object to hand back, and a warm
        # start already at the fixed point can land here on float noise, which the magnitude shows.
        warnings.warn(
            "run_em() accepted no EM step -- %s -- so the returned model is the initial model "
            "unchanged, not a fit of the data (fit_provenance() reports converged=False). Check "
            "the initialization and the objective; pass monotone=False when the objective is a "
            "surrogate that legitimately decreases before it improves." % stall_reason,
            UserWarning,
            stacklevel=2,
        )
    return _attach_run_em_provenance(
        last_good,
        strategy=strategy,
        estimator=estimator,
        objective_label=objective_label,
        iterations=iterations,
        max_its=max_its,
        converged=converged,
        delta=delta,
        final_objective=final_objective,
        objective_gain=objective_gain,
        enc_data=enc_data,
    )


def observed_log_likelihood(enc_data: Any, engine: Any | None = None) -> Callable[[Any], float]:
    """Return a model objective over fixed encoded data."""

    def objective(model: SequenceEncodableProbabilityDistribution) -> float:
        if engine is None:
            return float(seq_log_density_sum(enc_data, model)[1])
        return float(_engine_seq_log_density_sum(enc_data, model, engine)[1])

    return objective


def _is_mixture_like(model: Any) -> bool:
    return hasattr(model, "components") and callable(getattr(model, "seq_posterior", None))


def _posterior_matrix(model: Any, enc: Any, engine: Any | None) -> np.ndarray:
    if engine is not None:
        kernel = model.kernel(engine=engine)
        if callable(getattr(kernel, "posteriors", None)):
            return np.asarray(engine.to_numpy(kernel.posteriors(enc)), dtype=np.float64)
    return np.asarray(model.seq_posterior(enc), dtype=np.float64)


def _mixture_stats_from_gamma(model: Any, estimator: ParameterEstimator, enc: Any, gamma: np.ndarray) -> Any:
    acc = estimator.accumulator_factory().make()
    if not hasattr(acc, "accumulators"):
        raise TypeError("Mixture posterior transforms require a MixtureEstimator accumulator.")
    comp_stats = []
    for i, child_acc in enumerate(acc.accumulators):
        child_acc.seq_update(enc, gamma[:, i], model.components[i])
        comp_stats.append(child_acc.value())
    return gamma.sum(axis=0), tuple(comp_stats)


def _split_suff_stat(sampled: Any) -> Any:
    if isinstance(sampled, SampledSufficientStatistics):
        return sampled.nobs, sampled.suff_stat
    return None, sampled
