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
    """A uniform proposal move: an applicability pre-flight plus a fitted-challenger ``propose``."""

    name: str
    cost_hint: float

    def applicable(self, model: Any, data: Any, *, ctx: dict) -> bool:
        """Structural gate: whether this operator can run on the model and data."""
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


# Every driver in this codebase (closed_loop.py's step(), improve.py's improve(), population.py's
# Population.step()) calls `applicable(model, data, ctx=ctx)` immediately followed by
# `propose(model, data, ctx=ctx)` on the SAME `data` argument and the SAME `ctx` dict -- and the
# population.py / improve.py per-generation loops thread that SAME `data` (and `ctx`) through EVERY
# operator considered in the loop, not just one. A list/tuple is safely reiterable, but a generator (or
# any other one-shot iterable) is not: `applicable`'s own non-empty check used to fully consume it via
# `len(list(data))`, so `propose`'s later `list(data)` (or the NEXT operator sharing the same generator)
# silently got nothing back. `_materialize` fixes this by caching the first materialization on `ctx`
# (keyed by `data`'s identity), so every later touch -- this operator's own `propose`, or a different
# operator later in the same driver loop -- reuses the already-materialized rows instead of re-iterating
# a spent iterator into an empty list.
_MATERIALIZED_DATA_KEY = "__evolve_operators_materialized_data__"


def _materialize(data: Any, ctx: dict) -> list:
    """Return ``data`` as a fresh list, consuming a one-shot iterable (e.g. a generator) at most once.

    See the comment above :data:`_MATERIALIZED_DATA_KEY` for why this has to cache through ``ctx``
    rather than just returning ``list(data)`` freshly on every call.
    """
    cached = ctx.get(_MATERIALIZED_DATA_KEY)
    if cached is not None and cached[0] is data:
        return list(cached[1])
    rows = list(data)
    ctx[_MATERIALIZED_DATA_KEY] = (data, rows)
    return list(rows)


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
        """Return whether ``model`` can be re-fit on a non-empty batch."""
        return callable(getattr(model, "estimator", None)) and bool(len(_materialize(data, ctx)))

    def propose(self, model: Any, data: Any, *, ctx: dict) -> Candidate:
        """Fit the model family on ``data`` using ``model`` as the warm start."""
        rows = _materialize(data, ctx)
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
                                conjugate family; ``applicable`` checks ``ConjugateUpdatable``).
      * ``'forgetting'``      -- power-prior :class:`BayesianStreamingEstimator`.
    """

    mode: str = "streaming"
    cost_hint: float = 0.2

    @property
    def name(self) -> str:
        """Registry name including the selected online-update mode."""
        return f"online_update[{self.mode}]"

    def applicable(self, model: Any, data: Any, *, ctx: dict) -> bool:
        """Return whether the selected update mode is legal for ``model`` and ``data``."""
        if not (callable(getattr(model, "estimator", None)) and len(_materialize(data, ctx))):
            return False
        if self.mode in ("posterior_carry", "forgetting"):
            # Bayesian carry/forgetting paths need a conjugate family.
            return bool(supports(model, ConjugateUpdatable))
        if self.mode in ("streaming", "incremental"):
            return True
        return False

    def propose(self, model: Any, data: Any, *, ctx: dict) -> Candidate:
        """Apply the selected streaming update and return the updated challenger."""
        rows = _materialize(data, ctx)
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
        """Return whether there is data available for automatic family selection."""
        return bool(len(_materialize(data, ctx)))

    def propose(self, model: Any, data: Any, *, ctx: dict) -> Candidate:
        """Infer an estimator from ``data``, fit it, and return the fitted challenger."""
        from mixle.utils.automatic import get_estimator

        rows = _materialize(data, ctx)
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
    encoding delegate to the base model, so this stays family-agnostic for scalar continuous leaves --
    ``dist_to_encoder()`` returns a :class:`_RecalibratedEncoder` that applies the ``u`` transform at
    encode time (rather than the base's own encoder unmodified), so the public
    ``dist_to_encoder().seq_encode(...)`` -> ``seq_log_density(...)`` route scores the same value the
    scalar ``log_density`` path does.
    """

    def __init__(self, base: Any, temperature: float, center: float) -> None:
        # `temperature <= 0.0` alone let NaN and infinity straight through (both comparisons are
        # False), and center was unchecked. The affine map y -> c + (y - c) / T is simply undefined
        # for a non-finite T or c: NaN gave NaN densities and NaN samples, and T = inf collapsed
        # every observation onto the center and returned -inf density with NaN samples -- a
        # confidently-reported predictive model that is not a distribution at all. Neither state is
        # something recalibration legitimately produces (the searched grid is finite and positive,
        # and the center is a predictive mean), and no healing path downstream repairs them.
        t = float(temperature)
        if not np.isfinite(t) or t <= 0.0:
            raise ValueError(f"recalibration temperature must be finite and positive, got {temperature!r}")
        c = float(center)
        if not np.isfinite(c):
            raise ValueError(f"recalibration center must be finite, got {center!r}")
        self.base = base
        self.temperature = t
        self.center = c

    # -- scoring -------------------------------------------------------------
    def _transform(self, x: np.ndarray) -> np.ndarray:
        return self.center + (np.asarray(x, dtype=float) - self.center) / self.temperature

    def log_density(self, x: float) -> float:
        u = self.center + (float(x) - self.center) / self.temperature
        return float(self.base.log_density(u) - np.log(self.temperature))

    def dist_to_encoder(self):
        # NOT self.base.dist_to_encoder(): that would hand out an encoder whose seq_encode(raw) encodes
        # the UNTRANSFORMED raw observation, so seq_log_density(enc) below -- which only ever adds the
        # Jacobian term -- would silently score log p_base(raw) - log T instead of the correct
        # log p_base(transform(raw)) - log T. _RecalibratedEncoder applies the same transform this
        # class's own log_density/seq_log_density_raw apply, so the public encoded/batch route
        # (dist_to_encoder().seq_encode(...) -> seq_log_density(...), the contract every other
        # distribution in this codebase honors) agrees with the scalar path for the same observation.
        return _RecalibratedEncoder(self)

    def seq_log_density_raw(self, rows: Any) -> np.ndarray:
        """Exact per-observation log density on *raw* rows (the stateless, split-safe path).

        ``log p_T(y) = log p_base(c + (y - c) / T) - log T``. Computed by re-encoding the transformed
        values through the base encoder, so it is correct on any split (no cached-row assumption).
        """
        u = self._transform(np.asarray(rows, dtype=float))
        enc_u = self.base.dist_to_encoder().seq_encode(list(u))
        return np.asarray(self.base.seq_log_density(enc_u), dtype=float) - float(np.log(self.temperature))

    def seq_log_density(self, enc: Any) -> np.ndarray:
        # `enc` is assumed to have come from THIS model's own dist_to_encoder() (_RecalibratedEncoder),
        # which already applied the center/temperature transform at encode time -- so, unlike a plain
        # delegating fallback, adding the Jacobian here on top of the base's density at the (already
        # transformed) encoded value is the complete, correct computation, not an approximation of it.
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


class _RecalibratedEncoder:
    """Encoder companion for :class:`_RecalibratedModel`.

    Applies the SAME ``u = c + (y - c) / T`` transform ``log_density``/``seq_log_density_raw`` apply,
    at encode time, before delegating to the base family's own encoder -- so a caller using the normal
    ``model.dist_to_encoder().seq_encode(raw)`` -> ``model.seq_log_density(enc)`` contract (the one every
    other distribution in this codebase honors) gets a density consistent with the scalar
    ``model.log_density(raw)`` path, instead of one that silently skipped the value transform and only
    carried the Jacobian forward.

    Duck-types the ``DataSequenceEncoder`` contract (``seq_encode``, ``nbytes``, ``__eq__``, ``__str__``)
    without importing it, matching :class:`_RecalibratedModel`'s own dependency-light, family-agnostic
    style.
    """

    def __init__(self, model: _RecalibratedModel) -> None:
        self._model = model
        self._base_encoder = model.base.dist_to_encoder()

    def __str__(self) -> str:
        return f"_RecalibratedEncoder(T={self._model.temperature:.4g}, base={self._base_encoder})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _RecalibratedEncoder)
            and self._model.center == other._model.center
            and self._model.temperature == other._model.temperature
            and self._base_encoder == other._base_encoder
        )

    def seq_encode(self, x: Any) -> Any:
        """Transform ``x`` by the recalibration map, then encode it through the base encoder."""
        return self._base_encoder.seq_encode(self._model._transform(x))

    def nbytes(self, x: Any) -> int:
        """Delegate to the base encoder: ``x`` here is already-encoded, same shape either way."""
        return self._base_encoder.nbytes(x)


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

    def __post_init__(self) -> None:
        """Validate the search controls at the declaration boundary.

        An empty grid used to leave the initial ``best_t = 1.0`` in place with ``best_err`` still
        infinite -- a temperature reported as chosen that no candidate evaluation ever produced --
        and a non-positive or non-finite grid entry only failed much later, inside the base family's
        own density, as an error that never mentions the grid.
        """
        if isinstance(self.ensemble, bool) or not isinstance(self.ensemble, int) or self.ensemble <= 0:
            raise ValueError(f"ensemble must be a positive integer sample budget, got {self.ensemble!r}")
        grid = tuple(self.grid)
        if not grid:
            raise ValueError("grid must contain at least one candidate temperature")
        if any(not np.isfinite(float(t)) or float(t) <= 0.0 for t in grid):
            raise ValueError(f"grid temperatures must all be finite and positive, got {self.grid!r}")

    def applicable(self, model: Any, data: Any, *, ctx: dict) -> bool:
        """Return whether ``model`` can be sampled and the batch is non-empty."""
        if isinstance(model, _RecalibratedModel):
            return False  # don't stack recalibrations
        return callable(getattr(model, "sampler", None)) and bool(len(_materialize(data, ctx)))

    def propose(self, model: Any, data: Any, *, ctx: dict) -> Candidate:
        """Search the temperature grid and wrap ``model`` in the best calibration transform."""
        from mixle.inference.calibration import pit_calibration_error, pit_ensemble

        rows = np.asarray(_materialize(data, ctx), dtype=float).reshape(-1)
        if rows.size == 0 or not np.isfinite(rows).all():
            raise ValueError("recalibration needs a non-empty batch of finite observations")
        sampler = model.sampler(self.seed)
        ref = np.asarray(sampler.sample(self.ensemble), dtype=float).reshape(-1)
        if ref.size == 0 or not np.isfinite(ref).all():
            raise ValueError("the model's reference draws must be a non-empty finite ensemble")
        center = float(np.mean(ref))

        best_t: float | None = None
        best_err = np.inf
        for t in self.grid:
            f = np.broadcast_to(center + t * (ref - center), (rows.shape[0], ref.shape[0]))
            pit = pit_ensemble(rows, f, seed=self.seed)
            err = float(pit_calibration_error(pit))
            if err < best_err:
                best_err, best_t = err, float(t)
        # never report a temperature the search did not actually evaluate: `best_t` stays None iff
        # every candidate's calibration error was non-finite (or the loop never ran).
        if best_t is None or not np.isfinite(best_err):
            raise ValueError("no grid temperature produced a finite PIT calibration error")

        recal = _RecalibratedModel(model, best_t, center)
        return Candidate(
            recal,
            self.name,
            ctx.get("parent_hash"),
            {"temperature": best_t, "center": center, "pit_error": float(best_err)},
        )


@dataclass(frozen=True)
class Recompose:
    """Propose a richer two-component mixture structure for the champion family.

    Each component is warm-fit on a different half of the data so EM starts
    away from the identical-component solution. The normal verification gate
    decides whether the additional structure improves held-out evidence enough
    to promote. The operator is registered but omitted from the default set
    because it is intentionally more expensive than the conservative updates.
    """

    name: str = "recompose"
    cost_hint: float = 4.0
    max_its: int = 30

    def applicable(self, model: Any, data: Any, *, ctx: dict) -> bool:
        """Return whether the model can be re-estimated on enough rows for a split."""
        return callable(getattr(model, "estimator", None)) and len(_materialize(data, ctx)) >= 8

    def propose(self, model: Any, data: Any, *, ctx: dict) -> Candidate:
        """Fit a two-component mixture challenger initialized from data splits."""
        import numpy as np

        from mixle.ops import mixture

        rows = _materialize(data, ctx)
        rng = np.random.RandomState(int(ctx.get("seed", 0)))
        perm = rng.permutation(len(rows))
        half = max(1, len(rows) // 2)
        left = [rows[i] for i in perm[:half]]
        right = [rows[i] for i in perm[half:]] or left
        estimator = model.estimator()
        comp_a = optimize(left, estimator, max_its=15, prev_estimate=model, out=None)
        comp_b = optimize(right, estimator, max_its=15, prev_estimate=model, out=None)
        proto = mixture([comp_a, comp_b], [0.5, 0.5])
        fitted = optimize(rows, proto.estimator(), max_its=self.max_its, prev_estimate=proto, out=None)
        return Candidate(fitted, self.name, ctx.get("parent_hash"), {"components": 2})


@dataclass(frozen=True)
class Mutate:
    """Apply a random structural mutation to the champion and refit.

    Available moves are ``grow`` for adding a bootstrap-fit component,
    ``shrink`` for dropping the lowest-weight component from an existing
    mixture, and ``perturb`` for a bootstrap re-fit. Repeated use inside a
    :class:`~mixle.evolve.population.Population` gives the search driver a
    structure-induction move, while the verification gate remains responsible
    for promotion. The operator is registered but omitted from the default set
    because it is comparatively expensive.
    """

    name: str = "mutate"
    cost_hint: float = 4.0
    max_its: int = 30

    def applicable(self, model: Any, data: Any, *, ctx: dict) -> bool:
        """Return whether the model can be structurally mutated and re-fit."""
        return callable(getattr(model, "estimator", None)) and len(_materialize(data, ctx)) >= 8

    def propose(self, model: Any, data: Any, *, ctx: dict) -> Candidate:
        """Sample one structural move, refit it, and return the resulting challenger."""
        import numpy as np

        from mixle.ops import mixture

        rng = np.random.RandomState(int(ctx.get("seed", 0)))
        rows = _materialize(data, ctx)
        components = getattr(model, "components", None)
        is_mixture = isinstance(components, (list, tuple)) and len(components) >= 1

        moves = ["grow", "perturb"]
        if is_mixture and len(components) > 1:
            moves.append("shrink")
        move = moves[rng.randint(len(moves))]

        def _bootstrap() -> list:
            idx = rng.randint(0, len(rows), len(rows))
            return [rows[int(i)] for i in idx]

        if move == "shrink":
            weights = np.asarray(model.w, dtype=float)
            drop = int(np.argmin(weights))
            keep = [i for i in range(len(components)) if i != drop]
            # proto is just an EM init for the re-fit below, so a common-scale rescale here changes
            # nothing about the optimization outcome (posterior responsibilities are scale-invariant)
            # -- but MixtureDistribution requires a simplex, and the surviving weights alone (having
            # lost the dropped component's share of the mass) usually don't sum to 1 on their own.
            w = np.asarray([weights[i] for i in keep], dtype=float)
            residual = float(w.sum())
            if not np.isfinite(w).all() or not np.isfinite(residual) or residual <= 0.0:
                # Degenerate residual mass: e.g. every surviving component happened to carry zero
                # weight, or a non-finite weight reached this operator from an unvalidated duck-typed
                # `model` (this operator only requires `.components`/`.w`, not a validated
                # MixtureDistribution). Dividing by a zero or non-finite sum would silently manufacture
                # a NaN/inf weight vector instead of a clear failure, so refuse outright instead.
                raise ValueError(
                    "Mutate 'shrink' move cannot renormalize surviving component weights to a "
                    f"simplex: residual mass sum={residual!r} (need finite, strictly positive mass)."
                )
            proto = mixture([components[i] for i in keep], (w / residual).tolist())
        elif move == "grow":
            leaf = components[0] if is_mixture else model
            extra = optimize(_bootstrap(), leaf.estimator(), max_its=12, prev_estimate=leaf, out=None)
            base = list(components) if is_mixture else [model]
            base_w = list(np.asarray(model.w, dtype=float)) if is_mixture else [1.0]
            # proto is just an EM init for the re-fit below, so a common-scale rescale here changes
            # nothing about the optimization outcome (posterior responsibilities are scale-invariant)
            # -- but MixtureDistribution requires a simplex, and the raw base_w + [0.5] usually isn't one.
            w = np.asarray(base_w + [0.5], dtype=float)
            proto = mixture(base + [extra], (w / w.sum()).tolist())
        else:  # perturb
            proto = optimize(_bootstrap(), model.estimator(), max_its=12, prev_estimate=model, out=None)

        fitted = optimize(rows, proto.estimator(), max_its=self.max_its, prev_estimate=proto, out=None)
        return Candidate(fitted, self.name, ctx.get("parent_hash"), {"move": move})


def default_operators() -> list[ImprovementOperator]:
    """The Phase-1 default operator set: refit, online update, auto-select, recalibrate. (Recompose + Mutate are
    structural + expensive, so they are available via the registry but not enabled by default.)"""
    return [Refit(), OnlineUpdate(mode="streaming"), AutoSelect(), Recalibrate()]


register_operator(Recompose())  # discoverable via the registry, off by default
register_operator(Mutate())


__all__ = [
    "ImprovementOperator",
    "Candidate",
    "Refit",
    "OnlineUpdate",
    "AutoSelect",
    "Recalibrate",
    "Recompose",
    "Mutate",
    "register_operator",
    "unregister_operator",
    "registered_operators",
    "default_operators",
]
