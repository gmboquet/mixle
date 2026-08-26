"""Gaussian-process Bayesian optimization over a bounded input space (WS-E).

A sequential model-based optimization loop: fit a GP surrogate to the observed points, score
Latin-hypercube candidates with an acquisition function, and evaluate the best candidate next.
Reuses :class:`mixle.models.gaussian_process.GaussianProcessRegressor` (torch) as the surrogate;
the acquisition functions themselves are torch-free numpy.

Acquisitions are looked up through a small registry (``register_acquisition`` / ``acq=`` name) --
the same "register, don't branch" pattern as the engines and encoded-data backends -- so a new
acquisition plugs in without editing the proposal loop. Built in: expected improvement (``"ei"``),
probability of improvement (``"pi"``), and the upper/lower confidence bound (``"ucb"``). Each takes
``(mean, std, best, *, maximize, **params)`` and returns a *merit* array that is maximized over the
candidate set. Batch proposals (``propose_batch``) use the kriging-believer heuristic: fantasize the
posterior mean at each pick, refit, and repeat, giving a spatially diverse batch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import RandomState
from scipy.special import erfcx, ndtr

from mixle.doe._contracts import Acquisition, Surrogate
from mixle.doe.designs import Bounds, _as_bounds, _as_rng, _require_exact_positive_int, latin_hypercube


def _require_finite_scalar(value: Any, name: str) -> float:
    """Coerce ``value`` to a Python ``float`` and require it to be finite; raise ``ValueError`` if not.

    Catches both non-finite scalars (NaN/Inf) and non-scalars (arrays of length != 1, strings, ``None``,
    ...) that ``float()`` itself rejects with a bare ``TypeError``/``ValueError`` -- renamed here to a
    message that identifies which argument (``name``) and what was actually passed.
    """
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar, got {value!r}.") from exc
    if not np.isfinite(as_float):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return as_float


def _validate_acquisition_inputs(
    mean: Any,
    std: Any,
    best: float,
    *,
    xi: float | None = None,
    kappa: float | None = None,
    rng: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the common posterior-moment / acquisition-parameter contract; return float64 ``(mean, std)``.

    Every built-in acquisition (EI, log-EI, PI, UCB, Thompson) calls this first, so a malformed input is
    rejected with a clear message instead of silently accepted or broadcast (MXR-080-0169):

    * ``mean``/``std`` convert to float64 arrays that must have the SAME shape -- no silent broadcasting
      of a mismatched pair.
    * Every ``mean``/``std`` entry must be finite (no NaN/Inf).
    * ``std`` must be non-negative. A negative std is an invalid upstream input, not a legitimate
      deterministic (``std == 0``) one -- :func:`probability_of_improvement` used to test ``std <=
      threshold`` and so silently treated negative std as deterministic; this closes that off for every
      acquisition, not just PI.
    * ``best`` must be a finite scalar.
    * ``xi`` / ``kappa`` (only checked when the caller passes one -- not every acquisition takes both)
      must be a finite, non-negative scalar: both are exploration/exploitation margins, never
      meaningfully negative.
    * ``rng`` (only checked when the caller passes one) must be a genuine ``numpy.random`` generator
      (``RandomState`` or ``Generator``) when given, not an arbitrary object.
    """
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if mean.shape != std.shape:
        raise ValueError(f"mean and std must have the same shape; got {mean.shape} and {std.shape}.")
    if not np.all(np.isfinite(mean)):
        raise ValueError("mean must be finite (no NaN/Inf).")
    if not np.all(np.isfinite(std)):
        raise ValueError("std must be finite (no NaN/Inf).")
    if np.any(std < 0.0):
        raise ValueError(
            "std must be non-negative; a negative predictive std is an invalid input, not a "
            "deterministic (std == 0) one."
        )
    _require_finite_scalar(best, "best")
    if xi is not None and _require_finite_scalar(xi, "xi") < 0.0:
        raise ValueError(f"xi must be non-negative, got {xi!r}.")
    if kappa is not None and _require_finite_scalar(kappa, "kappa") < 0.0:
        raise ValueError(f"kappa must be non-negative, got {kappa!r}.")
    if rng is not None and not isinstance(rng, (np.random.RandomState, np.random.Generator)):
        raise ValueError(f"rng must be a numpy RandomState or Generator instance, got {type(rng).__name__!r}.")
    return mean, std


def expected_improvement(
    mean: Any, std: Any, best: float, xi: float = 0.0, *, maximize: bool = False, **_: Any
) -> np.ndarray:
    """Return the expected-improvement acquisition at points with surrogate ``mean`` and ``std``.

    For minimization the improvement over the incumbent ``best`` is ``best - mean - xi``; for
    maximization it is ``mean - best - xi``. ``xi >= 0`` trades exploration for exploitation.
    Points with (near-)zero predictive ``std`` get the DETERMINISTIC LIMIT ``max(improve, 0)`` -- a
    point-mass posterior has no uncertainty to average over, so the "expected" improvement is exactly
    the guaranteed one, not 0. Higher is better (maximized over candidates).

    Raises ``ValueError`` if ``std`` is negative/non-finite, ``mean``/``std`` shapes disagree,
    ``mean``/``best`` are non-finite, or ``xi`` is negative/non-finite.
    """
    mean, std = _validate_acquisition_inputs(mean, std, best, xi=xi)
    improve = (mean - best - xi) if maximize else (best - mean - xi)
    ei = np.zeros_like(std)
    pos = std > 1.0e-12
    z = np.zeros_like(std)
    z[pos] = improve[pos] / std[pos]
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    ei[pos] = improve[pos] * ndtr(z[pos]) + std[pos] * pdf[pos]
    # sigma -> 0 (including exactly 0): the standard sigma*(z*Phi(z) + phi(z)) formula's own limit,
    # by the Mills-ratio tail expansion of Phi/phi as z -> +/-inf, is exactly max(improve, 0) -- the
    # regression tests' continuity sweep confirms the formula above already converges smoothly to this
    # as std shrinks toward the threshold, so this is a continuous extension, not a discontinuous patch.
    ei[~pos] = np.maximum(improve[~pos], 0.0)
    return np.maximum(ei, 0.0)


def log_expected_improvement(
    mean: Any, std: Any, best: float, xi: float = 0.0, *, maximize: bool = False, **_: Any
) -> np.ndarray:
    """Return the log expected-improvement acquisition -- a numerically stable EI (Ament et al. 2023).

    Mathematically ``log(EI)``, but computed so it stays finite and informative deep in the
    no-improvement tail where ``EI`` itself underflows to 0 (and ``log EI`` to ``-inf``), keeping the
    optimizer's ordering and gradients usable. Same argmax as :func:`expected_improvement`; points with
    (near-)zero predictive ``std`` get the deterministic limit ``log(max(improve, 0))`` -- ``-inf`` when
    the point-mass outcome does not improve on ``best``, and the log of the guaranteed improvement when
    it does (see :func:`expected_improvement`). Higher is better. The ``z >= 0`` branch is the direct
    well-conditioned form; the ``z < 0`` tail uses the scaled complementary error function ``erfcx``
    (``Phi(z)/phi(z) = sqrt(pi/2) erfcx(-z/sqrt2)``), which is bounded there and so never under/overflows.

    Raises ``ValueError`` if ``std`` is negative/non-finite, ``mean``/``std`` shapes disagree,
    ``mean``/``best`` are non-finite, or ``xi`` is negative/non-finite.
    """
    mean, std = _validate_acquisition_inputs(mean, std, best, xi=xi)
    improve = (mean - best - xi) if maximize else (best - mean - xi)
    out = np.full(std.shape, -np.inf, dtype=np.float64)
    pos = std > 1.0e-300
    z = improve[pos] / std[pos]
    log_h = np.empty_like(z)
    neg = z < 0.0
    zp = z[~neg]
    log_h[~neg] = np.log(zp * ndtr(zp) + np.exp(-0.5 * zp * zp) / np.sqrt(2.0 * np.pi))
    zn = z[neg]
    mills = np.sqrt(np.pi / 2.0) * erfcx(-zn / np.sqrt(2.0))
    log_h[neg] = -0.5 * zn * zn - 0.5 * np.log(2.0 * np.pi) + np.log1p(zn * mills)
    out[pos] = np.log(std[pos]) + log_h
    # sigma -> 0 deterministic limit (matching expected_improvement's max(improve, 0)): out is already
    # -inf everywhere from the initialization above, which is already correct when improve <= 0 (no
    # guaranteed improvement); only overwrite the entries that DO have a positive guaranteed improvement,
    # computing log() only there so a non-improving point never triggers a log(0)/log(negative) warning.
    det = ~pos
    improving = det & (improve > 0.0)
    out[improving] = np.log(improve[improving])
    return out


def probability_of_improvement(
    mean: Any, std: Any, best: float, *, maximize: bool = False, xi: float = 0.0, **_: Any
) -> np.ndarray:
    """Return the probability-of-improvement acquisition.

    The probability that a candidate improves on the incumbent ``best`` by at least ``xi``:
    ``P(f < best - xi)`` for minimization, ``P(f > best + xi)`` for maximization. Where the
    predictive ``std`` is zero the improvement is deterministic (1.0 if it improves, else 0.0). A
    NEGATIVE ``std`` is rejected outright (see :func:`_validate_acquisition_inputs`) rather than
    treated as this same deterministic case -- it is an invalid input, not a zero-variance one.
    Higher is better.

    Raises ``ValueError`` if ``std`` is negative/non-finite, ``mean``/``std`` shapes disagree,
    ``mean``/``best`` are non-finite, or ``xi`` is negative/non-finite.
    """
    mean, std = _validate_acquisition_inputs(mean, std, best, xi=xi)
    improve = (mean - best - xi) if maximize else (best - mean - xi)
    pi = np.zeros_like(std)
    pos = std > 1.0e-12
    pi[pos] = ndtr(improve[pos] / std[pos])
    pi[~pos] = (improve[~pos] > 0.0).astype(np.float64)
    return pi


def upper_confidence_bound(
    mean: Any, std: Any, best: float = 0.0, *, maximize: bool = False, kappa: float = 1.96, **_: Any
) -> np.ndarray:
    """Return the confidence-bound acquisition (UCB for maximization, LCB for minimization).

    Returns a merit that is maximized over candidates: the optimistic bound ``mean + kappa * std``
    when maximizing, and ``kappa * std - mean`` when minimizing (so picking the largest merit
    selects the most promising low-objective point). ``kappa >= 0`` trades exploration for
    exploitation; ``best`` is ignored. Higher is better.

    Raises ``ValueError`` if ``std`` is negative/non-finite, ``mean``/``std`` shapes disagree,
    ``mean``/``best`` are non-finite, or ``kappa`` is negative/non-finite.
    """
    mean, std = _validate_acquisition_inputs(mean, std, best, kappa=kappa)
    return (mean + kappa * std) if maximize else (kappa * std - mean)


def thompson_sampling(
    mean: Any, std: Any, best: float = 0.0, *, maximize: bool = False, rng: Any = None, **_: Any
) -> np.ndarray:
    """Thompson-sampling acquisition: one marginal posterior draw ``N(mean, std)`` per candidate.

    Returns a merit (maximized over candidates) equal to the drawn value when maximizing and its
    negation when minimizing, so the selected point is the optimum of the *sampled* objective. A
    randomized, exploration-aware acquisition -- repeated proposals explore competing optima in
    proportion to posterior probability, with no exploration knob to tune. This is the low-cost *marginal*
    variant (independent per-candidate draws, ignoring the GP's cross-candidate correlation). The proposal
    loop (:func:`_propose_one`, and so :func:`propose_next` / :func:`propose_batch` /
    :class:`~mixle.doe.optimizer.BayesianOptimizer`) automatically threads its own candidate-generation
    ``rng`` through to this acquisition, so a seeded caller gets reproducible draws with no extra wiring;
    pass ``rng`` explicitly via ``acq_kwargs`` only to override that with a different generator. Called
    directly with no ``rng`` at all, this draws from a fresh, unseeded generator (matching ``_as_rng``'s
    own ``None`` convention) -- reproducible only through the proposal loop's threading above. ``best``
    is ignored.

    Raises ``ValueError`` if ``std`` is negative/non-finite, ``mean``/``std`` shapes disagree,
    ``mean``/``best`` are non-finite, or ``rng`` is neither ``None`` nor a ``numpy.random``
    ``RandomState``/``Generator``.
    """
    mean, std = _validate_acquisition_inputs(mean, std, best, rng=rng)
    # Match designs._as_rng's None convention (a fresh, OS-entropy-seeded RandomState) without routing
    # a non-None rng through _as_rng itself: _as_rng only special-cases RandomState, and constructing
    # RandomState(a_generator) raises -- np.random.Generator is a first-class rng type here too (see
    # _validate_acquisition_inputs above), so an already-given rng (RandomState or Generator) must pass
    # through untouched.
    rng = rng if rng is not None else _as_rng(None)
    draw = mean + std * rng.standard_normal(mean.shape)
    return draw if maximize else -draw


# --- acquisition registry ("register, don't branch") --------------------------------------------
# An acquisition is ``fn(mean, std, best, *, maximize, **params) -> merit`` where ``merit`` is
# maximized over the candidate set. Built-ins are registered below; third parties register their own.
_ACQUISITIONS: dict[str, Acquisition] = {}


def register_acquisition(name: str, fn: Acquisition, aliases: tuple[str, ...] = ()) -> None:
    """Register an acquisition ``fn`` under ``name`` (and any ``aliases``).

    ``fn`` is called as ``fn(mean, std, best, *, maximize, **params)`` and must return a merit array
    that the proposal loop maximizes over candidates. This is the extension point for new
    acquisitions -- registering is all that is needed, no edits to ``propose_next``/``minimize``.
    """
    if not callable(fn):
        raise TypeError("acquisition must be callable.")
    _ACQUISITIONS[name.lower()] = fn
    for alias in aliases:
        _ACQUISITIONS[alias.lower()] = fn


def available_acquisitions() -> list[str]:
    """Return the sorted names (and aliases) of all registered acquisitions."""
    return sorted(_ACQUISITIONS)


def _get_acquisition(acq: str | Acquisition) -> Acquisition:
    if callable(acq):
        return acq
    fn = _ACQUISITIONS.get(str(acq).lower())
    if fn is None:
        raise ValueError("unknown acquisition %r; registered: %s" % (acq, ", ".join(available_acquisitions())))
    return fn


register_acquisition("expected_improvement", expected_improvement, aliases=("ei",))
register_acquisition("log_expected_improvement", log_expected_improvement, aliases=("logei", "log_ei"))
register_acquisition("probability_of_improvement", probability_of_improvement, aliases=("pi",))
register_acquisition("upper_confidence_bound", upper_confidence_bound, aliases=("ucb", "lcb", "confidence_bound", "cb"))
register_acquisition("thompson_sampling", thompson_sampling, aliases=("thompson", "ts"))


@dataclass(frozen=True)
class OptimizationResult:
    """Common outcome of a model-based optimization run: the full evaluation history.

    ``x`` is the ``(N, d)`` matrix of evaluated points and ``y`` the corresponding objective
    values (an ``(N,)`` vector for single-objective runs, an ``(N, M)`` matrix for multi-objective).
    Concrete result types extend this with their best-point / Pareto-front fields.
    """

    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class BayesOptResult(OptimizationResult):
    """Outcome of a Bayesian-optimization run."""

    best_x: np.ndarray | None
    best_y: float | None
    n_evaluations: int
    failed_evaluations: tuple[dict[str, Any], ...]
    stopped_reason: str


def _validate_observations(x: np.ndarray, y: np.ndarray, *, context: str) -> None:
    """Require every entry of the training observations ``x``/``y`` to be finite before fitting.

    A non-finite entry (a NaN/Inf leaking into ``y`` from a bad objective evaluation, or from a
    fantasized kriging-believer value computed from a broken surrogate prediction) would otherwise
    propagate silently into the surrogate's fit and every downstream posterior mean/covariance --
    named here, at the fit boundary shared by every proposal path, instead of surfacing as an opaque
    numerical failure deep inside the GP (MXR-080-0170).
    """
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{context}: x contains non-finite values (NaN/Inf); cannot fit the surrogate.")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{context}: y contains non-finite values (NaN/Inf); cannot fit the surrogate.")


def _validate_prediction(mean: Any, cov: Any, n: int, *, context: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate a surrogate's posterior prediction against the candidate contract.

    ``mean`` must be a finite, length-``n`` vector -- one prediction per point queried. A duck-typed
    ``gp=`` surrogate that silently returns the wrong length, or whose fit diverged to NaN, is caught
    here instead of corrupting the acquisition merit / argmax that follows. When ``cov`` is not
    ``None`` it must additionally be a finite, symmetric ``(n, n)`` matrix (MXR-080-0170). Returns the
    validated ``(mean, cov)`` as float64 arrays (``cov`` is ``None`` through unchanged).
    """
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    if mean.shape != (n,):
        raise ValueError(f"{context}: surrogate predicted mean has shape {mean.shape}, expected ({n},).")
    if not np.all(np.isfinite(mean)):
        raise ValueError(f"{context}: surrogate predicted mean contains non-finite values (NaN/Inf).")
    if cov is None:
        return mean, None
    cov = np.asarray(cov, dtype=np.float64)
    if cov.shape != (n, n):
        raise ValueError(f"{context}: surrogate covariance has shape {cov.shape}, expected ({n}, {n}).")
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"{context}: surrogate covariance contains non-finite values (NaN/Inf).")
    scale = (
        max(float(np.linalg.norm(cov, ord=np.inf)), np.finfo(np.float64).tiny)
        if cov.size
        else np.finfo(np.float64).tiny
    )
    tolerance = 64.0 * np.finfo(np.float64).eps * scale * max(n, 1)
    asymmetry = float(np.linalg.norm(cov - cov.T, ord=np.inf))
    if asymmetry > tolerance:
        raise ValueError(
            f"{context}: surrogate covariance is not symmetric within {tolerance:.6g}; asymmetry is {asymmetry:.6g}."
        )
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    if eigvals[0] < -tolerance:
        raise ValueError(
            f"{context}: surrogate covariance is not positive-semidefinite; smallest eigenvalue is {eigvals[0]:.6g}."
        )
    if eigvals[0] < 0.0:
        cov = (eigvecs * np.clip(eigvals, 0.0, None)) @ eigvecs.T
    return mean, cov


def _select_index(values: Any, n: int, *, largest: bool, context: str) -> int:
    """Validate a per-candidate score array and return its arg-best (largest or smallest) index.

    ``values`` must have length ``n`` and at least one finite entry. Neither ``np.argmax`` nor
    ``np.argmin`` skips NaN -- an array containing one can make either return that NaN's index rather
    than the true best-scoring candidate's -- so non-finite entries are masked out before selecting,
    and a ``values`` that is entirely non-finite raises a clear error instead of returning an
    arbitrary index (MXR-080-0170).
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape != (n,):
        raise ValueError(f"{context}: expected {n} scores, got shape {values.shape}.")
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError(f"{context}: no candidate has a finite score (all {n} are NaN/Inf).")
    fill = -np.inf if largest else np.inf
    masked = np.where(finite, values, fill)
    return int(np.argmax(masked) if largest else np.argmin(masked))


def _require_default_surrogate() -> None:
    """Refuse cleanly, with the extra named, when the default GP surrogate cannot run (base install).

    The default surrogate is the torch :class:`~mixle.models.gaussian_process.GaussianProcessRegressor`.
    On a base install its lazy ``import torch`` used to surface only after real objective budget had
    been spent -- ``minimize`` evaluated its full ``n_init`` design (expensive black-box calls, the
    exact resource this module exists to conserve) and then crashed with an unexplained
    ``ImportError``. Callers check this BEFORE any evaluation is spent.
    """
    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Bayesian optimization's default Gaussian-process surrogate requires the optional torch "
            'dependency, which is not installed. Install it with pip install "mixle[torch]". '
            "Refusing before any objective evaluations are spent."
        ) from error


def _fit_surrogate(x: np.ndarray, y: np.ndarray, gp: Surrogate | None, fit_kwargs: dict[str, Any] | None) -> Surrogate:
    _validate_observations(x, y, context="_fit_surrogate")
    default_surrogate = gp is None
    if gp is None:
        from mixle.models.gaussian_process import GaussianProcessRegressor

        # np.std([]) is nan (with a RuntimeWarning), and `nan or 1.0` evaluates to nan -- nan is
        # truthy, so the `or` fallback only ever caught the exact-zero-variance case, not the
        # empty-y case. An empty y is a real, documented path: BayesianOptimizer.ask(q) with
        # q > n_init before any tell() calls propose_batch with zero observations.
        std = float(np.std(y)) if y.size > 0 else 0.0
        scale = std if std > 0.0 else 1.0
        gp = GaussianProcessRegressor(lengthscale=1.0, amplitude=scale, noise=0.1 * scale + 1.0e-6)
    kwargs = {"out": None, **(fit_kwargs or {})}
    if default_surrogate:
        # name the missing extra for the DEFAULT surrogate only: a caller-supplied surrogate's
        # ImportError is its own contract and must not be re-labeled as torch advice
        _require_default_surrogate()
    # With no observations there is no log marginal likelihood to maximize, and GaussianProcessRegressor
    # .fit validates its inputs first, so it raises "x must contain at least one row" -- turning the
    # documented empty path noted above into a crash instead of the NaN it used to produce. Fitting is
    # hyperparameter tuning only (predict takes the training data as arguments, so the model carries no
    # fitted data state), and the prior amplitude/noise chosen just above are already the right answer
    # for zero evidence: the prior IS the posterior. Skip the fit and hand that GP back.
    if np.asarray(y).size == 0:
        return gp
    gp.fit(x, y, **kwargs)
    return gp


def _propose_one(
    x: np.ndarray,
    y: np.ndarray,
    b: np.ndarray,
    rng: RandomState,
    *,
    maximize: bool,
    acq_fn: Acquisition,
    acq_kwargs: dict[str, Any],
    n_candidates: int,
    gp: Surrogate | None,
    fit_kwargs: dict[str, Any] | None,
) -> tuple[np.ndarray, float, Surrogate]:
    """Fit the surrogate, score Latin-hypercube candidates, return (best point, its merit, fitted gp).

    ``rng`` is used both to draw the Latin-hypercube candidate set AND -- unless ``acq_kwargs`` already
    supplies its own ``"rng"`` -- is forwarded to ``acq_fn`` itself, so a randomized acquisition (e.g.
    ``thompson_sampling``) draws from the SAME caller-seeded stream as everything else in the proposal
    loop, instead of silently falling back to its own fresh, unseeded generator. Every built-in
    acquisition accepts and ignores an ``rng`` it does not use (the shared ``**params`` contract
    documented at this module's acquisition registry), so this is safe to pass unconditionally.

    Raises ``ValueError`` if the surrogate's predicted mean/covariance don't match the candidate
    contract (wrong shape, non-finite, asymmetric covariance) or if every candidate's acquisition
    merit is non-finite (see :func:`_validate_prediction` / :func:`_select_index`).
    """
    n_candidates = _require_exact_positive_int(n_candidates, "n_candidates")
    if y.size == 0:
        # np.min/np.max on an empty y crashes with an opaque "zero-size array" ValueError. This is a
        # real, reachable path: BayesianOptimizer.ask(q) with q > n_init before any tell() calls
        # propose_next/propose_batch with zero observations -- there is no incumbent to score
        # acquisition against yet, so name that clearly instead of a generic numpy crash.
        raise ValueError("cannot propose an acquisition-based point with zero observations; call tell() first.")
    gp = _fit_surrogate(x, y, gp, fit_kwargs)
    candidates = latin_hypercube(b, n_candidates, rng)
    n_cand = candidates.shape[0]
    mean, cov = gp.predict(x, y, candidates, return_cov=True)
    mean, cov = _validate_prediction(mean, cov, n_cand, context="_propose_one")
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    best = float(np.max(y)) if maximize else float(np.min(y))
    # {"rng": rng, **acq_kwargs}: an explicit rng in acq_kwargs (a caller-supplied override) wins over
    # this ambient one -- dict-literal duplicate keys resolve to the later value, so an acq_kwargs
    # entry always overrides the "rng": rng default that comes before it.
    call_kwargs = {"rng": rng, **acq_kwargs}
    merit = np.asarray(acq_fn(mean, std, best, maximize=maximize, **call_kwargs), dtype=np.float64)
    idx = _select_index(merit, n_cand, largest=True, context="_propose_one acquisition merit")
    return candidates[idx], float(merit[idx]), gp


def _kg_inner(a: np.ndarray, b: np.ndarray) -> float:
    """``E[max_i (a_i + b_i Z)] - max_i a_i`` for ``Z ~ N(0, 1)`` (Frazier 2009, Algorithm 1)."""
    order = np.lexsort((a, b))  # sort by slope b ascending, ties broken by intercept a
    a = a[order]
    b = b[order]
    keep = np.append(np.diff(b) > 0.0, True)  # equal-slope lines: keep only the highest intercept
    a = a[keep]
    b = b[keep]
    idx = [0]
    cross = [-np.inf]
    for i in range(1, len(b)):
        while True:
            j = idx[-1]
            c = (a[j] - a[i]) / (b[i] - b[j])
            if len(idx) > 1 and c <= cross[-1]:
                idx.pop()
                cross.pop()
            else:
                break
        idx.append(i)
        cross.append(c)
    a = a[idx]
    b = b[idx]
    c = np.array([*cross, np.inf])
    cl, cr = c[:-1], c[1:]
    pdf = np.exp(-0.5 * c * c) / np.sqrt(2.0 * np.pi)
    return float(np.sum(a * (ndtr(cr) - ndtr(cl)) + b * (pdf[:-1] - pdf[1:])) - a.max())


def knowledge_gradient(mean: Any, cov: Any, noise: float = 1.0e-6) -> np.ndarray:
    """Knowledge-gradient acquisition value of one observation at each candidate (Frazier et al. 2009).

    Given the Gaussian-process posterior ``mean`` and joint ``cov`` over a candidate set, returns, for
    each candidate ``x``, the expected increase in the best posterior mean after fantasizing one
    (noisy) observation there: ``KG(x) = E_y[max_x' mu_{n+1}(x')] - max_x' mu_n(x')``. Maximizing KG is
    the one-step Bayes-optimal, look-ahead choice (it values *information*, not just immediate
    improvement), so it explores where an observation would most change the believed optimum. Assumes a
    *maximization* objective; negate ``mean`` for minimization. Computed exactly via the piecewise-linear
    epigraph of the fantasized posterior means, and ``>= 0`` by construction.
    """
    mean_array = np.asarray(mean, dtype=np.float64)
    if mean_array.ndim == 2 and mean_array.shape[1] == 1:
        n = mean_array.shape[0]
    elif mean_array.ndim == 1:
        n = mean_array.shape[0]
    else:
        raise ValueError(f"knowledge_gradient mean must be one-dimensional, got shape {mean_array.shape}.")
    if n == 0:
        raise ValueError("knowledge_gradient requires at least one candidate.")
    mean_array, covariance = _validate_prediction(mean_array, cov, n, context="knowledge_gradient")
    noise = _require_finite_scalar(noise, "noise")
    if noise < 0.0:
        raise ValueError(f"noise must be nonnegative, got {noise!r}.")
    assert covariance is not None
    out = np.empty(n, dtype=np.float64)
    for candidate_index in range(n):
        predictive_variance = covariance[candidate_index, candidate_index] + noise
        if not np.isfinite(predictive_variance) or predictive_variance < 0.0:
            raise ValueError("knowledge_gradient predictive variance must be finite and nonnegative.")
        if predictive_variance == 0.0:
            out[candidate_index] = 0.0
            continue
        sigma_x = np.sqrt(predictive_variance)
        value = _kg_inner(mean_array.copy(), covariance[:, candidate_index] / sigma_x)
        tolerance = 64.0 * np.finfo(np.float64).eps * max(float(np.max(np.abs(mean_array))), 1.0)
        if not np.isfinite(value) or value < -tolerance:
            raise ValueError(f"knowledge_gradient produced invalid merit {value!r}.")
        out[candidate_index] = max(value, 0.0)
    if not np.all(np.isfinite(out)) or np.any(out < 0.0):
        raise ValueError("knowledge_gradient must produce finite nonnegative merits.")
    return out


def propose_knowledge_gradient(
    x: Any,
    y: Any,
    bounds: Any,
    *,
    maximize: bool = False,
    n_candidates: int = 512,
    seed: int | None = None,
    gp: Surrogate | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    noise: float = 1.0e-6,
) -> np.ndarray:
    """Propose the next evaluation point by maximizing the knowledge gradient over a candidate set.

    Fits the GP surrogate to ``(x, y)``, draws ``n_candidates`` Latin-hypercube points, evaluates the
    joint posterior, and returns the candidate with the largest :func:`knowledge_gradient` -- the
    look-ahead Bayesian-optimization proposal. ``maximize`` selects the objective sense (the mean is
    negated for minimization).

    Raises ``ValueError`` if the surrogate's predicted mean/covariance don't match the candidate
    contract (wrong shape, non-finite, asymmetric covariance) or if every candidate's knowledge-gradient
    value is non-finite (see :func:`_validate_prediction` / :func:`_select_index`).
    """
    n_candidates = _require_exact_positive_int(n_candidates, "n_candidates")
    x, y = _validate_xy(x, y)
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    gp = _fit_surrogate(x, y, gp, fit_kwargs)
    candidates = latin_hypercube(b, n_candidates, rng)
    n_cand = candidates.shape[0]
    mean, cov = gp.predict(x, y, candidates, return_cov=True)
    mean, cov = _validate_prediction(mean, cov, n_cand, context="propose_knowledge_gradient")
    signed_mean = mean if maximize else -mean  # KG is defined for maximization
    kg = knowledge_gradient(signed_mean, cov, noise)
    idx = _select_index(kg, n_cand, largest=True, context="propose_knowledge_gradient")
    return candidates[idx]


def _validate_xy(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(
            f"x must be an explicit 2-D (n_observations, n_features) matrix, got shape {x.shape}; "
            "reshape one-dimensional data to (-1, 1)."
        )
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of observations.")
    return x, y


def propose_next(
    x: Any,
    y: Any,
    bounds: Bounds,
    n_candidates: int = 512,
    seed: int | RandomState | None = None,
    *,
    maximize: bool = False,
    xi: float = 0.0,
    acq: str | Acquisition = "ei",
    acq_kwargs: dict[str, Any] | None = None,
    gp: Surrogate | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    return_acquisition: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Propose the next point to evaluate by maximizing an acquisition function.

    Fits a GP to ``(x, y)``, scores ``n_candidates`` Latin-hypercube points by the ``acq`` acquisition
    (``"ei"`` / ``"pi"`` / ``"ucb"`` or any registered name / callable), and returns the best candidate
    (a ``(d,)`` array), optionally with its merit. ``xi`` is forwarded to acquisitions that use it
    (EI, PI); per-acquisition parameters such as ``kappa`` go in ``acq_kwargs``.

    Raises ``ValueError`` if the surrogate's predicted mean/covariance don't match the candidate
    contract (wrong shape, non-finite, asymmetric covariance) or if every candidate's acquisition
    merit is non-finite (validated at the shared surrogate boundary, see :func:`_propose_one`).
    """
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    x, y = _validate_xy(x, y)
    acq_fn = _get_acquisition(acq)
    kw = {"xi": xi, **(acq_kwargs or {})}
    point, merit, _ = _propose_one(
        x,
        y,
        b,
        rng,
        maximize=maximize,
        acq_fn=acq_fn,
        acq_kwargs=kw,
        n_candidates=n_candidates,
        gp=gp,
        fit_kwargs=fit_kwargs,
    )
    if return_acquisition:
        return point, merit
    return point


def propose_batch(
    x: Any,
    y: Any,
    bounds: Bounds,
    q: int,
    n_candidates: int = 512,
    seed: int | RandomState | None = None,
    *,
    maximize: bool = False,
    xi: float = 0.0,
    acq: str | Acquisition = "ei",
    acq_kwargs: dict[str, Any] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Propose a batch of ``q`` points to evaluate together, via the kriging-believer heuristic.

    Each pick maximizes the acquisition; the GP posterior mean at the chosen point is then appended
    as a fantasized observation and the surrogate refit, so the next pick is steered away from it.
    Returns a ``(q, d)`` array. This needs no true objective evaluations between picks, so it suits
    parallel/asynchronous experiment campaigns.

    Raises ``ValueError`` if the surrogate's predicted mean/covariance or fantasized single-point
    prediction don't match the candidate contract, or if every candidate's acquisition merit is
    non-finite at some step (see :func:`_validate_prediction` / :func:`_select_index`).
    """
    q = _require_exact_positive_int(q, "q")
    n_candidates = _require_exact_positive_int(n_candidates, "n_candidates")
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    xs, ys = _validate_xy(x, y)
    acq_fn = _get_acquisition(acq)
    kw = {"xi": xi, **(acq_kwargs or {})}
    picks = []
    for _ in range(q):
        point, _, gp = _propose_one(
            xs,
            ys,
            b,
            rng,
            maximize=maximize,
            acq_fn=acq_fn,
            acq_kwargs=kw,
            n_candidates=n_candidates,
            gp=None,
            fit_kwargs=fit_kwargs,
        )
        picks.append(point)
        # kriging-believer fantasy: a single-point prediction that becomes a real observation for the
        # NEXT iteration's fit -- validate it here too, since it bypasses _validate_xy (called once,
        # up front, not on each iteration's grown xs/ys).
        fantasy_pred = gp.predict(xs, ys, point[None, :], return_cov=False)
        fantasy_mean, _ = _validate_prediction(
            fantasy_pred, None, 1, context="propose_batch (kriging-believer fantasy)"
        )
        xs = np.vstack([xs, point[None, :]])
        ys = np.append(ys, float(fantasy_mean[0]))
    return np.asarray(picks, dtype=np.float64)


def minimize(
    objective: Callable[[np.ndarray], float],
    bounds: Bounds,
    n_init: int = 5,
    n_iter: int = 15,
    seed: int | RandomState | None = None,
    *,
    maximize: bool = False,
    xi: float = 0.0,
    acq: str | Acquisition = "ei",
    acq_kwargs: dict[str, Any] | None = None,
    n_candidates: int = 512,
    fit_kwargs: dict[str, Any] | None = None,
) -> BayesOptResult:
    """Run sequential GP Bayesian optimization of a scalar ``objective`` over ``bounds``.

    Seeds with an ``n_init``-point Latin-hypercube design, then runs ``n_iter`` acquisition-driven
    steps using ``acq`` (``"ei"`` by default; also ``"pi"`` / ``"ucb"`` or any registered acquisition).
    Minimizes by default; set ``maximize=True`` to maximize. ``objective`` takes a ``(d,)`` point and
    returns a float.

    A non-finite objective response is recorded in ``failed_evaluations`` and stops the run with
    ``stopped_reason='objective_failed'``. It is never admitted to the fitted ``x``/``y`` history.

    The GP surrogate needs the optional torch extra (``pip install "mixle[torch]"``). On a base
    install that is checked up front, before the first objective evaluation -- never after the
    ``n_init`` design has already spent expensive black-box calls.
    """
    b = _as_bounds(bounds)
    rng = _as_rng(seed)
    n_init = _require_exact_positive_int(n_init, "n_init")
    n_iter = _require_exact_positive_int(n_iter, "n_iter", minimum=0)
    n_candidates = _require_exact_positive_int(n_candidates, "n_candidates")
    if type(maximize) is not bool:
        raise TypeError(f"maximize must be a bool, got {type(maximize).__name__}.")
    if n_iter > 0:
        # the surrogate is only ever fit for acquisition-driven steps; a pure n_init design
        # (n_iter=0) is torch-free and stays runnable on a base install
        _require_default_surrogate()
    x_rows: list[np.ndarray] = []
    y_values: list[float] = []
    failed_evaluations: list[dict[str, Any]] = []
    n_evaluations = 0

    def evaluate(point: np.ndarray) -> float | None:
        nonlocal n_evaluations
        candidate = np.array(point, dtype=np.float64, copy=True)
        n_evaluations += 1
        value = float(objective(candidate.copy()))
        if not np.isfinite(value):
            failed_evaluations.append(
                {
                    "evaluation": n_evaluations,
                    "x": candidate,
                    "status": "nonfinite_observation",
                    "observation": value,
                }
            )
            return None
        return value

    def result(stopped_reason: str) -> BayesOptResult:
        x = np.asarray(x_rows, dtype=np.float64).reshape(-1, b.shape[0])
        y = np.asarray(y_values, dtype=np.float64)
        if y.size:
            best_idx = int(np.argmax(y) if maximize else np.argmin(y))
            best_x: np.ndarray | None = x[best_idx].copy()
            best_y: float | None = float(y[best_idx])
        else:
            best_x = None
            best_y = None
        return BayesOptResult(
            best_x=best_x,
            best_y=best_y,
            x=x,
            y=y,
            n_evaluations=n_evaluations,
            failed_evaluations=tuple(failed_evaluations),
            stopped_reason=stopped_reason,
        )

    for row in latin_hypercube(b, n_init, rng):
        value = evaluate(row)
        if value is None:
            return result("objective_failed")
        x_rows.append(np.array(row, dtype=np.float64, copy=True))
        y_values.append(value)

    for _ in range(n_iter):
        x = np.asarray(x_rows, dtype=np.float64)
        y = np.asarray(y_values, dtype=np.float64)
        nxt = propose_next(
            x,
            y,
            b,
            n_candidates=n_candidates,
            seed=rng,
            maximize=maximize,
            xi=xi,
            acq=acq,
            acq_kwargs=acq_kwargs,
            fit_kwargs=fit_kwargs,
        )
        nxt = np.asarray(nxt, dtype=np.float64)
        value = evaluate(nxt)
        if value is None:
            return result("objective_failed")
        x_rows.append(nxt.copy())
        y_values.append(value)

    return result("budget_exhausted")


__all__: Sequence[str] = [
    "Acquisition",
    "Surrogate",
    "expected_improvement",
    "log_expected_improvement",
    "probability_of_improvement",
    "upper_confidence_bound",
    "thompson_sampling",
    "knowledge_gradient",
    "propose_knowledge_gradient",
    "register_acquisition",
    "available_acquisitions",
    "propose_next",
    "propose_batch",
    "minimize",
    "OptimizationResult",
    "BayesOptResult",
]
