"""Real options & decision-under-uncertainty (work-plan Sec.7-J, J3).

A risk-neutral NPV distribution (J2's :class:`~mixle.analysis.valuation.NPVDistribution`) answers "what
is the project worth if we commit now?" It does not answer the question a real decision-maker actually
faces: "should we commit now, or is it worth paying for the *option* to wait, expand, or walk away once
more is known?" Two tools close that gap:

  * :func:`real_option_value` -- prices the option embedded in the decision (defer / expand / abandon)
    with a binomial lattice on the NPV process itself, American-exercised at every step. NPV -- unlike a
    commodity price -- is a signed quantity that must be able to cross zero for "wait and see" to mean
    anything, so the lattice is *additive* (an arithmetic random walk around ``npv_dist.mean``, scaled by
    ``volatility`` as a fraction of the NPV's own magnitude) rather than the usual multiplicative GBM
    lattice: a strictly-positive geometric process started above zero can never reach the ``max(V, 0)``
    kink that gives the option its value. Volatility is what makes the option valuable: by Jensen's
    inequality, an American option on a driftless process is worth strictly more than the naive point NPV
    whenever there is dispersion wide enough to reach the kink, and it collapses to the naive
    ``max(NPV, 0)`` as volatility falls to zero (no dispersion, nothing to wait for).
  * :func:`voi_dollars` -- the dollar value of a piece of information (e.g. a delineation drillhole): the
    expected value of the best decision *with* that information, minus the expected value of the best
    decision *without* it. By default this is a Gaussian-dispersion HEURISTIC approximation, not the
    textbook expected value of sample information (EVSI): a hypothetical drillhole's effect on the
    posterior is summarized as a fractional variance reduction (either supplied directly via
    ``drill_info["variance_reduction"]``, or -- when available -- via C8's
    ``mixle_pde.voi.expected_variance_reduction`` hook), and the pre-posterior simulation invents
    plausible future-posterior centers/spreads by RESCALING today's posterior draws rather than sampling a
    hypothetical observation from a real likelihood and conditioning on it -- exact only for a Gaussian,
    unimodal, unconstrained posterior, and flagged with a warning otherwise. For a genuine
    sample-then-condition preposterior, declare a linear-Gaussian :class:`GaussianObservationModel`
    (``observation_model=``): a hypothetical observation is then simulated from the latent's actual
    prior-predictive marginal and the posterior is conditioned on it analytically via closed-form
    conjugate-Gaussian updating -- the textbook EVSI construction, for that declared-likelihood case. A
    rational decision-maker never loses from more information *in expectation*, but a Monte Carlo estimate
    of that gain is noisy around zero when there is little or no information to gain; :func:`voi_estimate`
    reports the (unfloored) estimate together with its own standard error rather than clipping the noise
    away, which -- naively floored at zero -- was systematically biased upward rather than merely
    conservative (:class:`VoiEstimate`).

Repo-boundary note: J2's ``mixle.analysis.valuation`` module (``NPVDistribution`` / ``monte_carlo_npv``)
has since landed and is imported normally below; C8's ``mixle_pde`` VOI hooks have not, so
``voi_dollars`` still best-effort imports that hook and falls back to a self-contained
posterior-refinement simulation over the frozen IC-1 :class:`~mixle.reason.posterior_protocol.Posterior`
when it is unavailable. ``real_option_value``'s ``npv_dist`` parameter stays duck-typed against
anything exposing a ``.mean`` -- the hard import below is for the type annotation and documentation
value only (previously a ``TYPE_CHECKING``-only forward reference, which left the name unresolvable to
``typing.get_type_hints()`` -- a real, if rarely exercised, runtime introspection break); no
``isinstance`` check was added, so any duck-typed object still works exactly as before.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
from scipy.sparse.linalg import LinearOperator

from mixle.analysis.valuation import NPVDistribution
from mixle.reason.posterior_protocol import Posterior

__all__ = [
    "OptionValue",
    "real_option_value",
    "GaussianObservationModel",
    "VoiEstimate",
    "voi_estimate",
    "voi_dollars",
    "VoiStoppingDecision",
    "voi_stopping_decision",
]

_KINDS = ("defer", "expand", "abandon")


class OptionValue(NamedTuple):
    """The priced option, alongside the exercise policy that generated it.

    ``value`` is the total value of holding the option (project value + optionality). ``exercise_boundary``
    is the per-time-step critical underlying value at which immediate exercise first becomes optimal
    (``nan`` at any step where it is never optimal to exercise). ``premium_over_npv`` is ``value`` minus
    the naive ``npv_dist.mean`` -- the dollar amount the flexibility to defer/expand/abandon is worth
    over just committing to the point-estimate NPV today.
    """

    value: float
    exercise_boundary: np.ndarray
    premium_over_npv: float


def _intrinsic(v: np.ndarray, kind: str, expand_fraction: float) -> np.ndarray:
    """Exercise payoff at underlying value(s) ``v`` for the given option ``kind``."""
    if kind == "defer":
        # Option to wait, then invest only if the project is worth it: invest for max(V, 0), else walk.
        return np.maximum(v, 0.0)
    if kind == "abandon":
        # Already committed; option to abandon for salvage (assumed 0) rather than ride a negative NPV.
        return np.maximum(v, 0.0)
    if kind == "expand":
        # Option to scale up capacity by `expand_fraction` whenever doing so is profitable.
        return v + expand_fraction * np.maximum(v, 0.0)
    raise ValueError(f"real_option_value: kind must be one of {_KINDS}, got {kind!r}")


def real_option_value(
    npv_dist: NPVDistribution,
    *,
    volatility: float,
    horizon: int,
    kind: str = "defer",
    rate: float,
    n_steps: int | None = None,
    expand_fraction: float = 0.3,
) -> OptionValue:
    """Price the defer/expand/abandon option on a project via an additive binomial lattice on NPV.

    The NPV process is modeled as a driftless arithmetic random walk started at ``npv_dist.mean``: each
    of ``n_steps`` (default: ``max(horizon, 1)``) steps up to ``horizon`` moves the value by
    ``+-h`` with ``h = volatility * scale * sqrt(dt)``, ``scale = abs(npv_dist.mean)`` (or ``1.0`` if the
    mean is exactly zero, so ``volatility`` is never degenerate) -- i.e. ``volatility`` is a *fractional*
    per-sqrt-period dispersion relative to the project's own scale, matching how the parameter is usually
    quoted (e.g. ``0.3`` ~ "30% dispersion"), while still letting the walk go negative the way a signed
    NPV must be able to. The risk-neutral measure is taken driftless (an NPV is already the discounted
    expectation of a self-financing project, so there is no further risk-neutral drift to add); ``rate``
    only discounts the continuation value, which is what makes immediate exercise optimal once volatility
    vanishes. At every node the holder may exercise (payoff via :func:`_intrinsic`, depending on ``kind``)
    or continue holding the option; the American value is the larger of the two, by backward induction
    from the horizon.

    Args:
        npv_dist: anything exposing ``.mean`` (a float) -- typically J2's ``NPVDistribution``. Only the
            mean is used; ``real_option_value`` prices the *option on top of* the point estimate, not a
            re-derivation of the distribution itself.
        volatility: fractional (per-sqrt-period) dispersion of NPV around its mean. Must be ``>= 0``;
            ``0`` means no dispersion and the option collapses to immediate exercise of ``kind``'s own
            payoff on ``npv_dist.mean`` (:func:`_intrinsic`) -- ``max(npv_dist.mean, 0)`` for
            ``"defer"``/``"abandon"``, ``npv_dist.mean + expand_fraction * max(npv_dist.mean, 0)`` for
            ``"expand"``.
        horizon: number of periods over which the option may be exercised. ``0`` means "decide now".
        kind: one of ``"defer"``, ``"expand"``, ``"abandon"``.
        rate: per-period discount rate applied to the continuation value.
        n_steps: lattice steps (defaults to ``max(horizon, 1)``, i.e. one step per period). Must be a
            positive integer when given explicitly.
        expand_fraction: for ``kind="expand"``, the fractional capacity bonus applied to a positive
            underlying value when the expansion is exercised.

    Returns:
        An :class:`OptionValue`.
    """
    if kind not in _KINDS:
        raise ValueError(f"real_option_value: kind must be one of {_KINDS}, got {kind!r}")
    if volatility < 0.0:
        raise ValueError("real_option_value: volatility must be non-negative")
    if horizon < 0:
        raise ValueError("real_option_value: horizon must be non-negative")
    if n_steps is not None and n_steps <= 0:
        # n_steps=0 with horizon>0 divides by zero below; n_steps<0 builds an EMPTY lattice and then
        # indexes boundary[n] == boundary[-1] on it, raising a confusing IndexError three lines later.
        # One clear error at the boundary instead of two different internal crashes downstream.
        raise ValueError(f"real_option_value: n_steps must be a positive integer, got {n_steps}")

    npv_mean = float(npv_dist.mean)
    n = int(n_steps) if n_steps is not None else max(int(horizon), 1)
    dt = float(horizon) / n if horizon > 0 else 0.0
    scale = abs(npv_mean) if npv_mean != 0.0 else 1.0
    h = volatility * scale * float(np.sqrt(dt))

    if h == 0.0:
        # No dispersion (or no time to disperse in): waiting has no upside, and discounting only makes
        # it worse, so the optimal policy is immediate exercise -- the SAME payoff _intrinsic gives
        # every node of the lattice below, not a hardcoded max(npv_mean, 0.0) (which happens to match
        # _intrinsic for "defer"/"abandon" but silently ignores "expand"'s expand_fraction bonus).
        value = float(_intrinsic(np.array([npv_mean]), kind, expand_fraction)[0])
        boundary = np.full(n + 1, np.nan)
        return OptionValue(value=value, exercise_boundary=boundary, premium_over_npv=value - npv_mean)

    disc = float(np.exp(-rate * dt))
    boundary = np.full(n + 1, np.nan)

    j = np.arange(n + 1)
    v = npv_mean + (2 * j - n) * h  # j up-moves, (n - j) down-moves out of n steps
    option = _intrinsic(v, kind, expand_fraction)
    boundary[n] = 0.0  # at maturity, exercise iff the underlying is non-negative (kind-independent here)

    for step in range(n - 1, -1, -1):
        j = np.arange(step + 1)
        v = npv_mean + (2 * j - step) * h
        continuation = disc * 0.5 * (option[1 : step + 2] + option[0 : step + 1])
        intrinsic = _intrinsic(v, kind, expand_fraction)
        exercise = intrinsic > continuation
        option = np.where(exercise, intrinsic, continuation)
        if np.any(exercise):
            exercised_v = v[exercise]
            boundary[step] = float(np.min(exercised_v)) if kind != "abandon" else float(np.max(exercised_v))

    value = float(option[0])
    return OptionValue(value=value, exercise_boundary=boundary, premium_over_npv=value - npv_mean)


def _variance_reduction(posterior: Posterior, drill_info: dict) -> float:
    """Fraction of posterior variance a hypothetical drillhole is expected to remove, in ``[0, 1)``.

    Best-effort: if C8's ``mixle_pde.voi.expected_variance_reduction`` hook is importable and
    ``drill_info`` supplies the geometry/forward-operator it needs, use it. Otherwise fall back to a
    directly-supplied ``drill_info["variance_reduction"]`` (default ``0.5`` -- a generic "meaningfully
    informative" delineation hole).
    """
    candidate_geometry = drill_info.get("candidate_geometry")
    forward_op = drill_info.get("forward_op")
    if candidate_geometry is not None and forward_op is not None:
        try:
            from mixle_pde.voi import expected_variance_reduction  # C8 hook; mixle_pde is a soft dependency
        except ImportError:
            expected_variance_reduction = None
        if expected_variance_reduction is not None:
            reduction = float(
                expected_variance_reduction(
                    posterior,
                    candidate_geometry,
                    forward_op,
                    region=drill_info.get("region"),
                    cell_volumes=drill_info.get("cell_volumes"),
                )
            )
            return _clamp_variance_reduction(reduction)
    return _clamp_variance_reduction(float(drill_info.get("variance_reduction", 0.5)))


def _clamp_variance_reduction(reduction: float) -> float:
    """Clamp a fractional variance reduction to ``[0, 1)``.

    NaN is rejected explicitly, and first: plain ``min``/``max`` do not reliably filter it out
    (``max(float("nan"), 0.0)`` returns ``nan``, not ``0.0``, silently defeating the floor below) --
    the same NaN-comparisons-are-False trap that :func:`real_option_value`'s own validation guards
    against. Left unfiltered, a NaN reduction would also silently miss ``voi_estimate``'s
    ``reduction == 0.0`` fast path (NaN compares unequal to everything, itself included) and instead
    poison the general Monte Carlo path with NaN centers/scales.
    """
    if not np.isfinite(reduction):
        raise ValueError(f"voi_dollars: variance_reduction must be finite, got {reduction!r}")
    return min(max(reduction, 0.0), 1.0 - 1e-9)


class VoiEstimate(NamedTuple):
    """A Monte Carlo value-of-information estimate, reported honestly alongside its own sampling noise
    (MXR-080-0108: independent no-info/with-info draws made a should-be-zero difference noisy, and
    flooring that noise at zero upward-biased it; see :func:`voi_estimate`'s docstring for the fix).

    ``value`` is the point estimate, ``E[value | info] - E[value | no info]`` -- NOT floored at zero (see
    :func:`voi_estimate`'s docstring for why flooring a noisy estimate at zero systematically biases it
    upward, rather than merely rounding it). ``standard_error`` is the *paired-difference* Monte Carlo
    standard error of ``value`` across the ``n_outer`` hypothetical-information replicates (paired because
    each replicate's with-info and no-info evaluations share their underlying random draw -- see
    :func:`voi_estimate`). ``ci_low``/``ci_high`` are an approximate two-sided 95% confidence interval
    (``value +- 1.96 * standard_error``; a normal approximation, not exact for small ``n_outer``). When
    ``ci_low <= 0 <= ci_high``, ``value`` is statistically indistinguishable from "no additional value" --
    a more honest read than a bare point estimate that happens to be (noisily) positive or negative.

    ``method`` names which computation produced the estimate: ``"variance_rescaling_heuristic"`` (the
    default, always available -- a Gaussian-dispersion approximation, exact only when the posterior is
    close to Gaussian, unimodal, and unconstrained) or ``"gaussian_conjugate_evsi"`` (a genuine
    sample-then-condition preposterior computation under a declared linear-Gaussian ``observation_model``).
    """

    value: float
    standard_error: float
    ci_low: float
    ci_high: float
    method: str


_VOI_CI95_Z = 1.959963984540054  # two-sided 95% normal-approximation multiplier


def _voi_estimate_from_paired_diffs(diffs: np.ndarray, method: str) -> VoiEstimate:
    n = int(diffs.shape[0])
    value = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    if np.isfinite(se):
        ci_low, ci_high = value - _VOI_CI95_Z * se, value + _VOI_CI95_Z * se
    else:
        ci_low, ci_high = float("nan"), float("nan")
    return VoiEstimate(value=value, standard_error=se, ci_low=ci_low, ci_high=ci_high, method=method)


class GaussianObservationModel(NamedTuple):
    """A declared linear-Gaussian observation model: ``y = obs_matrix @ theta + noise``, ``noise ~
    N(0, obs_cov)``, for a latent ``theta`` distributed as the current posterior's ``(mean, cov)``.

    MXR-080-0109: the default heuristic invents future-posterior centers/spreads by rescaling draws from
    the CURRENT posterior; it never samples an observation from a likelihood/forward model and conditions
    on it, so it is not a real Bayesian preposterior for non-Gaussian, multimodal, or constrained beliefs.
    This is the simplest well-defined experiment likelihood for which the preposterior -- the distribution
    of what TOMORROW's posterior could look like, before the observation is actually taken -- has a closed
    form: conjugate-Gaussian updating. Passing this to :func:`voi_dollars` / :func:`voi_estimate` (as
    ``observation_model=``) is what makes them compute a genuine expected value of sample information
    (EVSI): simulate a hypothetical ``y`` from theta's actual prior-predictive marginal, analytically
    condition the CURRENT posterior on that ``y`` to get a real future posterior, decide against it, and
    average over many simulated ``y`` -- rather than the default heuristic's rescaling of today's draws
    (see :func:`voi_estimate`'s docstring for why that heuristic is not a real preposterior in general).

    ``obs_matrix`` (``H``, shape ``(k, d)``) and ``obs_cov`` (``R``, shape ``(k, k)``, symmetric positive
    definite) together say: a real experiment would return a ``k``-dimensional noisy linear readout of the
    ``d``-dimensional latent quantity the posterior is over (``d`` = ``posterior.mean``'s length). This
    path requires ``posterior.mean`` / ``posterior.cov`` to fully characterize the belief -- i.e. a
    genuinely Gaussian posterior -- since that is what makes the conjugate update exact; a dense
    ``posterior.cov`` is required (a matrix-free ``LinearOperator`` covariance is not supported here).
    ``obs_matrix = 0`` (or ``obs_cov`` large) declares a literally uninformative experiment -- VOI comes
    back exactly zero for that case, not just approximately (see :func:`voi_estimate`'s docstring).
    """

    obs_matrix: np.ndarray
    obs_cov: np.ndarray


_GAUSSIAN_SKEW_WARN = 0.75
_GAUSSIAN_EXCESS_KURTOSIS_WARN = 1.5


def _warn_if_not_gaussian_like(posterior: Posterior, rng: np.random.Generator, *, n_probe: int = 512) -> None:
    """Cheap empirical screen for the Gaussian/unimodal/unconstrained regime both VOI paths lean on.

    Not a proof: a finite probe sample can miss subtle non-Gaussianity, or (rarely) flag a legitimate
    Gaussian draw on an unlucky sample. It is a fast, honest diagnostic so a caller silently outside the
    assumed regime (multimodal, heavily skewed, bounded/constrained posteriors) gets a warning instead of
    a silently over-claimed textbook VOI number: both the default variance-rescaling heuristic and the
    Gaussian-conjugate ``observation_model`` path assume the posterior's mean/cov fully characterize its
    shape, which is only true for a genuinely Gaussian belief. Thresholds are generous (roughly 7 standard
    errors at ``n_probe=512`` for true Gaussian data) to keep the false-positive rate low.
    """
    probe = np.asarray(posterior.samples(n_probe, rng), dtype=np.float64)
    if probe.ndim == 1:
        probe = probe[:, None]
    sd = probe.std(axis=0, ddof=1)
    valid = sd > 1e-12
    if not np.any(valid):
        return
    centered = probe[:, valid] - probe[:, valid].mean(axis=0, keepdims=True)
    skew = np.mean(centered**3, axis=0) / sd[valid] ** 3
    excess_kurtosis = np.mean(centered**4, axis=0) / sd[valid] ** 4 - 3.0
    if np.any(np.abs(skew) > _GAUSSIAN_SKEW_WARN) or np.any(np.abs(excess_kurtosis) > _GAUSSIAN_EXCESS_KURTOSIS_WARN):
        warnings.warn(
            "voi_dollars/voi_estimate: posterior samples look non-Gaussian (skewness or excess kurtosis "
            "beyond this heuristic screen's threshold on a fresh probe sample). Both the default "
            "variance-rescaling heuristic and the Gaussian-conjugate observation_model path assume a "
            "Gaussian, unimodal, unconstrained posterior; treat this VOI estimate as approximate, or "
            "supply a non-Gaussian-aware decision procedure instead.",
            UserWarning,
            stacklevel=3,
        )


def _require_dense_finite_matrix(value: Any, name: str) -> np.ndarray:
    if isinstance(value, LinearOperator):
        raise TypeError(
            f"voi_dollars: {name} must be a dense array for the observation_model path (got a matrix-free "
            "LinearOperator) -- survey-scale field posteriors aren't supported by the closed-form "
            "Gaussian-conjugate path; use the default variance-rescaling heuristic instead."
        )
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"voi_dollars: {name} must be finite, got values containing NaN/inf")
    return arr


def _safe_cholesky(mat: np.ndarray, *, name: str) -> np.ndarray:
    """Cholesky factor of a symmetric PSD matrix, with a small jitter retry for numerical near-singularity."""
    sym = 0.5 * (mat + mat.T)
    try:
        return np.linalg.cholesky(sym)
    except np.linalg.LinAlgError:
        pass
    scale = float(np.trace(sym)) / max(sym.shape[0], 1)
    jitter = 1e-10 * (scale if scale > 0.0 else 1.0)
    try:
        return np.linalg.cholesky(sym + jitter * np.eye(sym.shape[0]))
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"voi_dollars: {name} is not (numerically) symmetric positive-definite, even after a small "
            "jitter -- the Gaussian-conjugate observation_model path requires a genuine covariance."
        ) from exc


def _voi_estimate_gaussian_conjugate(
    posterior: Posterior,
    decision_fn: Callable[[np.ndarray], float],
    observation_model: GaussianObservationModel,
    *,
    rng: np.random.Generator,
    n_outer: int,
    n_inner: int,
) -> VoiEstimate:
    """The genuine sample-then-condition EVSI path: see :class:`GaussianObservationModel`'s docstring."""
    method = "gaussian_conjugate_evsi"
    mean = np.asarray(posterior.mean, dtype=np.float64)
    d = mean.shape[0]
    cov = _require_dense_finite_matrix(posterior.cov, "posterior.cov")
    if cov.shape != (d, d):
        raise ValueError(f"voi_dollars: posterior.cov must have shape ({d}, {d}), got {cov.shape}")

    obs_matrix = _require_dense_finite_matrix(observation_model.obs_matrix, "observation_model.obs_matrix")
    if obs_matrix.ndim != 2 or obs_matrix.shape[1] != d:
        raise ValueError(f"voi_dollars: observation_model.obs_matrix must have shape (k, {d}), got {obs_matrix.shape}")
    k = obs_matrix.shape[0]
    obs_cov = _require_dense_finite_matrix(observation_model.obs_cov, "observation_model.obs_cov")
    if obs_cov.shape != (k, k):
        raise ValueError(f"voi_dollars: observation_model.obs_cov must have shape ({k}, {k}), got {obs_cov.shape}")

    _warn_if_not_gaussian_like(posterior, rng)

    # Prior-predictive covariance of the hypothetical observation, the Kalman gain, and the post-observation
    # covariance -- all independent of the observed VALUE, a standard closed-form property of conjugate-
    # Gaussian updating that lets these be computed once rather than per simulated observation.
    mean_y = obs_matrix @ mean
    cov_y = obs_matrix @ cov @ obs_matrix.T + obs_cov
    gain = np.linalg.solve(cov_y, obs_matrix @ cov).T  # (d, k); solved rather than inverting cov_y directly
    post_cov = cov - gain @ obs_matrix @ cov

    l_prior = _safe_cholesky(cov, name="posterior.cov")
    l_post = _safe_cholesky(post_cov, name="the post-observation covariance")
    l_y = _safe_cholesky(cov_y, name="the prior-predictive observation covariance")

    y_draws = mean_y + rng.standard_normal((n_outer, k)) @ l_y.T

    diffs = np.empty(n_outer, dtype=np.float64)
    for i in range(n_outer):
        # Shared innovations z: at obs_matrix = 0 (or obs_cov -> large), gain = 0 and post_cov = cov
        # EXACTLY (both independent of y_draws[i]), so with_info_samples == no_info_samples bit-for-bit
        # and diffs[i] == 0.0 exactly -- the same common-random-numbers guarantee as the heuristic path.
        z = rng.standard_normal((n_inner, d))
        no_info_samples = mean + z @ l_prior.T
        post_mean_i = mean + gain @ (y_draws[i] - mean_y)
        with_info_samples = post_mean_i + z @ l_post.T
        diffs[i] = float(decision_fn(with_info_samples)) - float(decision_fn(no_info_samples))

    return _voi_estimate_from_paired_diffs(diffs, method)


def voi_estimate(
    posterior: Posterior,
    decision_fn: Callable[[np.ndarray], float],
    drill_info: dict[str, Any],
    *,
    rng: np.random.Generator,
    n_outer: int = 64,
    n_inner: int = 256,
    observation_model: GaussianObservationModel | None = None,
) -> VoiEstimate:
    """Value of information, honestly: the point estimate together with its own Monte Carlo uncertainty.

    ``decision_fn`` maps a set of posterior draws (an ``(n, d)`` array, physical units) to the dollar
    value of the *single* best decision made using that belief state -- e.g. a risk-neutral go/no-go
    choice is ``max(samples.mean(), 0)``, not an average of a per-draw payoff: the latter implicitly
    assumes the realization is already known, which is exactly the perfect-information case this function
    is pricing the *gap* to, not the belief itself.

    Two methods are available, controlled by ``observation_model``:

    * ``observation_model=None`` (the default, ``method="variance_rescaling_heuristic"``): a pre-posterior
      Monte Carlo built on the law of total variance, NOT real Bayesian conditioning on a simulated
      observation. Today's posterior variance splits into "where the post-drill posterior will be
      centered" (unknown until the drillhole is actually put in) and "how much spread remains once it
      lands there." If the drillhole is expected to remove a fraction ``r`` of variance
      (:func:`_variance_reduction`), then for ``n_outer`` hypothetical drill outcomes a center is drawn
      with standard deviation scaled by ``sqrt(r)`` around today's posterior mean (how much the belief
      could plausibly shift), and a *shared* base draw (see "Common random numbers" below) is re-centered
      there and rescaled by ``sqrt(1 - r)`` for the remaining spread. This INVENTS future-belief "centers"
      by rescaling *today's* draws rather than drawing them from an actual forward/likelihood model, so it
      is accurate only when the posterior is (close to) Gaussian, unimodal, and unconstrained -- outside
      that regime it is flagged with a :class:`UserWarning` (see :func:`_warn_if_not_gaussian_like`), not
      silently trusted.
    * ``observation_model=`` a :class:`GaussianObservationModel` (``method="gaussian_conjugate_evsi"``): a
      genuine sample-then-condition preposterior. A hypothetical observation ``y`` is simulated from the
      latent's real prior-predictive marginal (``y = H @ theta + noise`` for ``theta ~ N(posterior.mean,
      posterior.cov)``), the CURRENT posterior is analytically conditioned on that ``y`` via closed-form
      conjugate-Gaussian updating to get a genuine future posterior, and the decision is re-made against
      it -- averaged over many simulated ``y`` for ``E[value | info]``. This is the textbook EVSI
      construction for the declared-likelihood case, but it still requires the PRIOR (``posterior.mean`` /
      ``posterior.cov``) to be genuinely Gaussian for the conjugate update to be exact -- also screened by
      the same warning.

    Common random numbers (MXR-080-0108): each of the ``n_outer`` replicates draws ONE base/innovation
    sample and uses it for BOTH the no-info decision and the with-info (refined) decision for that replicate, differing only
    by a deterministic transform (the heuristic's re-centering/rescaling, or the conjugate update's affine
    map) -- not independent draws for the two sides. This matters when there is truly no information to
    gain (``r = 0`` for the heuristic; ``obs_matrix = 0`` or ``obs_cov`` arbitrarily large for the
    declared-likelihood path): with independent draws, the no-info and with-info values would be two
    separately noisy estimates of the *same* quantity, so their difference would be noise centered at
    zero; flooring that noise at zero (the previous behavior) kept only its positive half, upward-biasing
    the reported VOI (a standard-normal fake posterior with a go/no-go decision spuriously returned
    positive VOI at zero information for most tested seeds). With a shared draw, the with-info value is
    bit-for-bit identical to the no-info value whenever there is truly no information (checked directly for
    the heuristic via an analytic short-circuit below, not just left to fall out of the Monte Carlo -- and
    provably exact by construction for the declared-likelihood path too), and the two stay correlated --
    hence lower-variance in their difference -- away from that boundary as well.

    Args:
        posterior: the current (pre-drill) belief, satisfying IC-1's ``Posterior`` protocol.
        decision_fn: belief state (``(n, d)`` draws) -> expected dollar value of the best decision.
        drill_info: describes the hypothetical drillhole. Recognized keys: ``variance_reduction`` (direct
            fractional variance reduction, default ``0.5``), or ``candidate_geometry`` + ``forward_op``
            (+ optional ``region`` / ``cell_volumes``) to route through the C8 VOI hook when available;
            ``n_outer_samples`` / ``n_inner_samples`` override the Monte Carlo sample counts. Ignored
            (aside from the sample-count overrides) when ``observation_model`` is given.
        rng: seeded random generator for reproducibility.
        n_outer: number of hypothetical-information replicates (overridden by ``drill_info``).
        n_inner: posterior draws per replicate (overridden by ``drill_info``).
        observation_model: when given, a declared :class:`GaussianObservationModel` routes the computation
            through the genuine ``"gaussian_conjugate_evsi"`` path instead of the default heuristic; see
            above. Requires a dense (non-``LinearOperator``) ``posterior.cov``.

    Returns:
        A :class:`VoiEstimate`: the (unfloored) point estimate plus its Monte Carlo uncertainty.
    """
    n_outer = int(drill_info.get("n_outer_samples", n_outer))
    n_inner = int(drill_info.get("n_inner_samples", n_inner))
    if n_outer < 1:
        raise ValueError(f"voi_dollars: n_outer_samples must be >= 1, got {n_outer}")
    if n_inner < 1:
        raise ValueError(f"voi_dollars: n_inner_samples must be >= 1, got {n_inner}")

    if observation_model is not None:
        return _voi_estimate_gaussian_conjugate(
            posterior, decision_fn, observation_model, rng=rng, n_outer=n_outer, n_inner=n_inner
        )

    method = "variance_rescaling_heuristic"
    reduction = _variance_reduction(posterior, drill_info)
    if reduction == 0.0:
        # Analytic equality at exactly zero reduction, not just an empirically-near-zero estimate: by
        # this heuristic's own construction (inner_scale = sqrt(1 - r) = 1, center_scale = sqrt(r) = 0
        # below), the with-info belief IS today's belief, so the two decisions coincide exactly -- no
        # Monte Carlo needed (and, since no regime approximation is actually being exercised in this
        # trivial case, no need for the Gaussian-regime warning either), and none of its sampling noise
        # to worry about flooring.
        return VoiEstimate(value=0.0, standard_error=0.0, ci_low=0.0, ci_high=0.0, method=method)

    _warn_if_not_gaussian_like(posterior, rng)

    center_scale = float(np.sqrt(reduction))
    inner_scale = float(np.sqrt(1.0 - reduction))
    mean = np.asarray(posterior.mean, dtype=np.float64)

    centers = mean + center_scale * (posterior.samples(n_outer, rng) - mean)
    diffs = np.empty(n_outer, dtype=np.float64)
    for i in range(n_outer):
        base = posterior.samples(n_inner, rng)  # shared by both sides of this replicate -- see docstring
        no_info = float(decision_fn(base))
        refined = centers[i] + inner_scale * (base - mean)
        with_info = float(decision_fn(refined))
        diffs[i] = with_info - no_info

    return _voi_estimate_from_paired_diffs(diffs, method)


def voi_dollars(
    posterior: Posterior,
    decision_fn: Callable[[np.ndarray], float],
    drill_info: dict[str, Any],
    *,
    rng: np.random.Generator,
    n_outer: int = 64,
    n_inner: int = 256,
    observation_model: GaussianObservationModel | None = None,
) -> float:
    """The value-of-information point estimate, in dollars: ``E[value | info] - E[value | no info]``.

    A thin convenience wrapper around :func:`voi_estimate` for callers that only want the point estimate;
    see its docstring for the full method (the default heuristic vs. the genuine ``observation_model``
    EVSI path, common random numbers, the zero-information analytic equality, and what ``method`` the
    estimate used) and for how to also get the Monte Carlo standard error / CI.

    Args:
        posterior, decision_fn, drill_info, rng, n_outer, n_inner, observation_model: forwarded to
            :func:`voi_estimate` unchanged.

    Returns:
        The value-of-information point estimate, in dollars. NOT floored at zero -- see
        :func:`voi_estimate`'s docstring for why a floor is dishonest, not merely conservative; this can
        be a small negative number when the true VOI is at or near zero, which is the honest reflection
        of Monte Carlo noise, not a defect. Use :func:`voi_estimate` if you need to tell that case apart
        from a real, larger difference.
    """
    return voi_estimate(
        posterior,
        decision_fn,
        drill_info,
        rng=rng,
        n_outer=n_outer,
        n_inner=n_inner,
        observation_model=observation_model,
    ).value


class VoiStoppingDecision(NamedTuple):
    """A real decision-theoretic answer to "should we sample again?": compare the value of one more
    sample against its cost, rather than an arbitrary uncertainty threshold picked by hand."""

    voi_dollars: float
    sample_cost: float
    net_value: float
    keep_sampling: bool


def voi_stopping_decision(
    posterior: Posterior,
    decision_fn: Callable[[np.ndarray], float],
    drill_info: dict[str, Any],
    *,
    sample_cost: float,
    rng: np.random.Generator,
    n_outer: int = 64,
    n_inner: int = 256,
) -> VoiStoppingDecision:
    """Should the next sample (drillhole, monitoring well, survey station, ...) actually be taken?

    A real, principled alternative to picking an uncertainty threshold by hand and telling an LLM
    loop-controller to stop when it's cleared (the pattern used, and flagged as a real gap, in
    ``experiments/adaptive-groundwater-monitoring`` and ``experiments/adaptive-gravity-survey-design``):
    sample again iff :func:`voi_dollars` -- the actual expected dollar value the next sample would add
    to the decision -- exceeds what that sample costs. As uncertainty tightens, ``voi_dollars`` shrinks
    toward zero (there's less left to learn that would change the decision), so this converges to a
    stopping rule on its own without any separately-chosen threshold; the only free parameter is
    ``sample_cost``, which is an actual real-world number (drilling/survey cost) rather than an
    arbitrary uncertainty width.

    Args:
        posterior, decision_fn, drill_info, rng, n_outer, n_inner: forwarded to :func:`voi_dollars`
            unchanged -- see its docstring for what each means.
        sample_cost: the real dollar cost of taking the next sample.

    Returns:
        A :class:`VoiStoppingDecision`: the computed VOI, the cost it was compared against, their
        difference, and whether sampling should continue (``voi_dollars > sample_cost``).
    """
    voi = voi_dollars(posterior, decision_fn, drill_info, rng=rng, n_outer=n_outer, n_inner=n_inner)
    return VoiStoppingDecision(
        voi_dollars=voi,
        sample_cost=float(sample_cost),
        net_value=voi - float(sample_cost),
        keep_sampling=voi > float(sample_cost),
    )
