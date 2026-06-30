"""The *propose* contract: a uniform :class:`ImprovementOperator` over existing fit mechanisms.

Every "improve" move -- a warm-start refit, an online update, an auto-select, a recalibration -- is one
operator with the same shape, so they become interchangeable proposal moves the driver can schedule and
the gate can compare. Each operator body is a thin shell over a verified-present API:

* :class:`Refit`         -> :func:`mixle.inference.estimation.optimize` warm-started from the champion.
* :class:`OnlineUpdate`  -> the streaming estimators (``StreamingEstimator`` / ``IncrementalEstimator``
                            / ``BayesianStreamingEstimator``) ``.update``.
* :class:`AutoSelect`    -> :func:`mixle.utils.automatic.get_estimator` -> ``optimize``.
* :class:`Recalibrate`   -> a post-hoc affine spread-temperature wrap that recalibrates the predictive
                            without refitting the base parameters.

Operators are registrable through a *scoped* registry (``register_operator`` / ``unregister_operator``)
that mirrors the "register, don't branch" pattern without polluting the global Detector registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mixle.capability import ConjugateUpdatable, supports
from mixle.inference.estimation import BayesianStreamingEstimator, optimize
from mixle.inference.streaming import IncrementalEstimator, StreamingEstimator


@dataclass(frozen=True)
class Candidate:
    """A proposed, already-fitted challenger plus the provenance of how it was made."""

    model: Any
    operator: str
    parent_hash: str | None = None
    meta: dict = field(default_factory=dict)


@runtime_checkable
class ImprovementOperator(Protocol):
    """A uniform proposal move: a cheap applicability pre-flight plus a fitted-challenger ``propose``."""

    name: str
    cost_hint: float

    def applicable(self, model: Any, data: Any, *, ctx: dict) -> bool:
        """Cheap structural gate -- can this operator even run on this model/data?"""
        ...

    def propose(self, model: Any, data: Any, *, ctx: dict) -> Candidate:
        """Return a fitted challenger (or raise if the proposal cannot be built)."""
        ...


# ---------------------------------------------------------------------------
# scoped operator registry (not the global Detector registry)
# ---------------------------------------------------------------------------
_OPERATOR_REGISTRY: dict[str, ImprovementOperator] = {}


def register_operator(operator: ImprovementOperator) -> ImprovementOperator:
    """Register ``operator`` in the scoped evolve operator registry (returns it for decorator use)."""
    name = getattr(operator, "name", None)
    if not name:
        raise ValueError("operator must have a non-empty .name to be registered.")
    _OPERATOR_REGISTRY[name] = operator
    return operator


def unregister_operator(name: str) -> None:
    """Remove a previously-registered operator by name (no-op if absent)."""
    _OPERATOR_REGISTRY.pop(name, None)


def registered_operators() -> dict[str, ImprovementOperator]:
    """A copy of the current scoped operator registry."""
    return dict(_OPERATOR_REGISTRY)


def _quiet(kwargs: dict) -> dict:
    """Silence ``optimize`` output unless the caller asked for it."""
    out = dict(kwargs)
    out.setdefault("out", None)
    return out


# ---------------------------------------------------------------------------
# Refit -- warm-start resume from the champion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Refit:
    """Re-fit the champion's family on fresh data, warm-started from the champion's parameters."""

    name: str = "refit"
    cost_hint: float = 1.0
    max_its: int = 20

    def applicable(self, model: Any, data: Any, *, ctx: dict) -> bool:
        return callable(getattr(model, "estimator", None)) and bool(len(list(data)))

    def propose(self, model: Any, data: Any, *, ctx: dict) -> Candidate:
        rows = list(data)
        estimator = model.estimator()
        fitted = optimize(rows, estimator, max_its=self.max_its, prev_estimate=model, out=None)
        return Candidate(fitted, self.name, ctx.get("parent_hash"), {"warm_start": True, "max_its": self.max_its})


# ---------------------------------------------------------------------------
# OnlineUpdate -- fold a batch into the champion via the streaming estimators
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OnlineUpdate:
    """Fold a fresh batch into the champion via a streaming estimator.

    ``mode``:
      * ``'streaming'``       -- decay-mode :class:`StreamingEstimator` (running-accumulator forgetting).
      * ``'incremental'``     -- Neal-Hinton :class:`IncrementalEstimator` (replace one chunk).
      * ``'posterior_carry'`` -- exact recursive-Bayes :class:`BayesianStreamingEstimator` (needs a
                                conjugate family; ``applicable`` honestly checks ``ConjugateUpdatable``).
      * ``'forgetting'``      -- power-prior :class:`BayesianStreamingEstimator`.
    """

    mode: str = "streaming"
    cost_hint: float = 0.2

    @property
    def name(self) -> str:
        return f"online_update[{self.mode}]"

    def applicable(self, model: Any, data: Any, *, ctx: dict) -> bool:
        if not (callable(getattr(model, "estimator", None)) and len(list(data))):
            return False
        if self.mode in ("posterior_carry", "forgetting"):
            # honest conjugacy pre-flight: the Bayesian carry/forgetting paths need a conjugate family.
            return bool(supports(model, ConjugateUpdatable))
        if self.mode in ("streaming", "incremental"):
            return True
        return False

    def propose(self, model: Any, data: Any, *, ctx: dict) -> Candidate:
        rows = list(data)
        estimator = model.estimator()
        if self.mode == "streaming":
            driver = StreamingEstimator(estimator, model=model)
            updated = driver.update(rows)
        elif self.mode == "incremental":
            driver = IncrementalEstimator(estimator, model=model)
            updated = driver.update(rows, chunk_id="batch")
        elif self.mode in ("posterior_carry", "forgetting"):
            driver = BayesianStreamingEstimator(estimator, mode=self.mode, model=model)
            updated = driver.update(rows)
        else:
            raise ValueError(f"unknown OnlineUpdate mode {self.mode!r}.")
        return Candidate(updated, self.name, ctx.get("parent_hash"), {"mode": self.mode, "nobs": len(rows)})


# ---------------------------------------------------------------------------
# AutoSelect -- infer a family from the data and fit it
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AutoSelect:
    """Infer an estimator from the raw data (``get_estimator``) and fit it -- a possible family swap."""

    name: str = "auto_select"
    cost_hint: float = 3.0
    max_its: int = 20

    def applicable(self, model: Any, data: Any, *, ctx: dict) -> bool:
        return bool(len(list(data)))

    def propose(self, model: Any, data: Any, *, ctx: dict) -> Candidate:
        from mixle.utils.automatic import get_estimator

        rows = list(data)
        estimator = get_estimator(rows)
        fitted = optimize(rows, estimator, max_its=self.max_its, out=None)
        return Candidate(
            fitted,
            self.name,
            ctx.get("parent_hash"),
            {"family": type(fitted).__name__, "family_swap": type(fitted).__name__ != type(model).__name__},
        )


# ---------------------------------------------------------------------------
# Recalibrate -- post-hoc affine spread temperature (no parameter refit)
# ---------------------------------------------------------------------------
class _RecalibratedModel:
    """A base distribution recalibrated by an exact affine spread map ``y -> c + (y - c) / T``.

    The recalibration inflates (``T > 1``) or deflates (``T < 1``) the predictive spread about a center
    ``c`` (the predictive mean) *without* refitting the base parameters. It is exact: under the
    change of variables ``u = c + (y - c) / T`` the density is
    ``p_T(y) = p_base(u) / T`` and a sample ``y = c + T (s - c)`` for a base draw ``s``. Scoring and
    encoding delegate to the base model, so this stays family-agnostic for scalar continuous leaves.
    """

    def __init__(self, base: Any, temperature: float, center: float) -> None:
        if temperature <= 0.0:
            raise ValueError("recalibration temperature must be positive.")
        self.base = base
        self.temperature = float(temperature)
        self.center = float(center)

    # -- scoring -------------------------------------------------------------
    def _transform(self, x: np.ndarray) -> np.ndarray:
        return self.center + (np.asarray(x, dtype=float) - self.center) / self.temperature

    def log_density(self, x: float) -> float:
        u = self.center + (float(x) - self.center) / self.temperature
        return float(self.base.log_density(u) - np.log(self.temperature))

    def dist_to_encoder(self):
        return self.base.dist_to_encoder()

    def seq_log_density_raw(self, rows: Any) -> np.ndarray:
        """Exact per-observation log density on *raw* rows (the stateless, split-safe path).

        ``log p_T(y) = log p_base(c + (y - c) / T) - log T``. Computed by re-encoding the transformed
        values through the base encoder, so it is correct on any split (no cached-row assumption).
        """
        u = self._transform(np.asarray(rows, dtype=float))
        enc_u = self.base.dist_to_encoder().seq_encode(list(u))
        return np.asarray(self.base.seq_log_density(enc_u), dtype=float) - float(np.log(self.temperature))

    def seq_log_density(self, enc: Any) -> np.ndarray:
        # The encoded handle does not expose raw y, so the change-of-variables transform cannot be
        # applied here in general. The split-safe path is ``seq_log_density_raw``; the objective bridge
        # routes recalibrated models through it. This delegating fallback applies only the Jacobian.
        base_ld = np.asarray(self.base.seq_log_density(enc), dtype=float)
        return base_ld - float(np.log(self.temperature))

    # -- sampling ------------------------------------------------------------
    def sampler(self, seed: int | None = None):
        return _RecalibratedSampler(self, seed)

    def estimator(self, *args: Any, **kwargs: Any):
        # Recalibration is a post-hoc wrap; refitting it falls back to the base family's estimator.
        return self.base.estimator(*args, **kwargs)

    def __repr__(self) -> str:
        return f"_RecalibratedModel(T={self.temperature:.4g}, base={type(self.base).__name__})"


class _RecalibratedSampler:
    def __init__(self, model: _RecalibratedModel, seed: int | None) -> None:
        self.model = model
        self.base_sampler = model.base.sampler(seed)

    def sample(self, size: int | None = None):
        s = self.base_sampler.sample(size)
        c, t = self.model.center, self.model.temperature
        return c + t * (np.asarray(s, dtype=float) - c)


@dataclass(frozen=True)
class Recalibrate:
    """Learn a predictive spread temperature ``T`` that flattens the PIT, no parameter refit.

    ``applicable`` requires a sampler (used both to estimate the predictive center and to evaluate PIT
    calibration). The temperature is chosen on the train split by minimising the PIT calibration error
    over a small grid; ``T == 1`` (the identity) is always in the grid, so the recalibrated model can
    never be worse-calibrated than the base on the fitting data.
    """

    name: str = "recalibrate"
    cost_hint: float = 0.5
    ensemble: int = 256
    seed: int = 0
    grid: tuple[float, ...] = (0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)

    def applicable(self, model: Any, data: Any, *, ctx: dict) -> bool:
        if isinstance(model, _RecalibratedModel):
            return False  # don't stack recalibrations
        return callable(getattr(model, "sampler", None)) and bool(len(list(data)))

    def propose(self, model: Any, data: Any, *, ctx: dict) -> Candidate:
        from mixle.inference.calibration import pit_calibration_error, pit_ensemble

        rows = np.asarray(list(data), dtype=float).reshape(-1)
        sampler = model.sampler(self.seed)
        ref = np.asarray(sampler.sample(self.ensemble), dtype=float).reshape(-1)
        center = float(np.mean(ref))

        best_t, best_err = 1.0, np.inf
        for t in self.grid:
            cand = _RecalibratedModel(model, t, center)
            f = np.broadcast_to(
                center + t * (ref - center), (rows.shape[0], ref.shape[0])
            )
            pit = pit_ensemble(rows, f, seed=self.seed)
            err = pit_calibration_error(pit)
            if err < best_err:
                best_err, best_t = err, t

        recal = _RecalibratedModel(model, best_t, center)
        return Candidate(
            recal,
            self.name,
            ctx.get("parent_hash"),
            {"temperature": best_t, "center": center, "pit_error": float(best_err)},
        )


def default_operators() -> list[ImprovementOperator]:
    """The Phase-1 default operator set: refit, online update, auto-select, recalibrate."""
    return [Refit(), OnlineUpdate(mode="streaming"), AutoSelect(), Recalibrate()]


__all__ = [
    "ImprovementOperator",
    "Candidate",
    "Refit",
    "OnlineUpdate",
    "AutoSelect",
    "Recalibrate",
    "register_operator",
    "unregister_operator",
    "registered_operators",
    "default_operators",
]
