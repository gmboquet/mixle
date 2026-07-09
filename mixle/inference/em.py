"""Expectation-maximization strategy helpers.

The strategies in this module are deliberately orchestration-level objects:
they move encoded data through existing estimators/kernels and never contain
distribution-specific likelihood math.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mixle.inference.estimation import _engine_seq_estimate, _engine_seq_log_density_sum, _local_encoded_chunks
from mixle.stats.compute.pdist import ParameterEstimator, SequenceEncodableProbabilityDistribution
from mixle.stats.compute.sequence import seq_estimate, seq_log_density_sum


@dataclass
class EMStepResult:
    """Result from one EM-family strategy step."""

    model: SequenceEncodableProbabilityDistribution
    objective: float | None = None
    accepted: bool = True
    metadata: dict | None = None


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


class PosteriorTransformEM:
    """EM update that transforms mixture posteriors before the M-step.

    ``temperature=1`` gives the usual soft EM responsibilities. ``hard=True``
    gives classification/hard EM. Intermediate temperatures implement a simple
    deterministic-annealing style generalized EM update.
    """

    def __init__(self, temperature: float = 1.0, hard: bool = False) -> None:
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        self.temperature = float(temperature)
        self.hard = bool(hard)

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
        self.temperatures = tuple(float(t) for t in temperatures)
        if any(t < 0.0 for t in self.temperatures):
            raise ValueError("temperatures must be non-negative.")
        self.hard_final = bool(hard_final)
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
        self.require_improvement = bool(require_improvement)

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
        self.require_improvement = bool(require_improvement)
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
        try:
            base_result = self.base_strategy.step(enc_data, estimator, model, engine=engine, objective=objective)
            candidate = base_result.model
            new_value = objective(candidate) if base_result.objective is None else base_result.objective
        except (np.linalg.LinAlgError, FloatingPointError, ValueError, RuntimeError):
            # M-step blew up (e.g. a singular covariance slipped through): keep the last good model.
            return EMStepResult(model, old_value, False, metadata={"rejected": "exception"})

        if not np.isfinite(new_value):
            return EMStepResult(model, old_value, False, metadata={"rejected": "nonfinite"})
        if self.require_improvement and new_value + self.tolerance < old_value:
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
        self.require_improvement = bool(require_improvement)

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
    ``fn(enc_data, estimator, model, rng, num_samples, engine)``.  It may return
    either ``suff_stat`` or ``(nobs, suff_stat)`` for ``estimator.estimate``.
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
    ) -> None:
        self.schedule = schedule
        self.init_estimator = init_estimator
        self.init_p = init_p
        self.rng = rng
        self.encoder = encoder
        self.num_chunks = num_chunks
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
        self.require_improvement = bool(require_improvement)
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
        for initial in self.initial_models:
            candidate = run_em(
                enc_data,
                estimator,
                initial,
                strategy=self.strategy,
                max_its=self.max_its,
                delta=self.delta,
                engine=engine,
                objective=objective,
            )
            value = objective(candidate)
            if best_model is None or value > best_value:
                best_model = candidate
                best_value = value
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
) -> SequenceEncodableProbabilityDistribution:
    """Run an EM-family strategy until convergence or ``max_its``.

    ``objective`` takes the same values as :func:`~mixle.inference.estimation.optimize`: ``None`` (MLE),
    a selection string (``'auto'`` / ``'mle'`` / ``'map'`` / ``'vb'``), or a ready ``model -> float``
    callable. ``max_its`` is the canonical iteration-cap spelling (matching ``optimize`` / ``fit`` /
    ``best_of``); ``max_iter`` is accepted as a back-compat alias and overrides ``max_its`` when given.
    """
    if max_iter is not None:
        max_its = max_iter
    strategy = StandardEM() if strategy is None else strategy
    objective = _resolve_run_em_objective(objective, enc_data, estimator, initial_model, engine)
    model = initial_model
    last_good = model
    old_value = objective(model)
    for _ in range(max(1, int(max_its))):
        result = strategy.step(enc_data, estimator, model, engine=engine, objective=objective)
        candidate = result.model
        value = objective(candidate) if result.objective is None else result.objective
        # NaN/inf guard: never propagate a non-finite step; roll back to the last good model.
        if not np.isfinite(value):
            return last_good
        model = candidate
        last_good = model
        # converge on a small *improvement* only; abs() would also stop on a decrease (the wrong model)
        if delta is not None and 0.0 <= value - old_value < delta:
            break
        old_value = value
    return model


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
    if isinstance(sampled, tuple) and len(sampled) == 2:
        return sampled
    return None, sampled
