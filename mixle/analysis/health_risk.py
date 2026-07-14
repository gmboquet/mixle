"""Dose-response and population health-risk models (K3, work-plan Workstream K).

Given an exposure/dose -- a bare number, an array of dose realisations, or an IC-1 `Posterior` over
a receptor field (K1/K2 transport output) -- these push it through a named dose-response curve into
an outcome-probability distribution, so downstream liability/constraint code (K6) always has a
distribution, never a bare point estimate:

  * :class:`DoseResponse` -- a named model (``loglinear`` / ``logit`` / ``hill`` /
    ``threshold_linear``) with caller-supplied ``params``; :meth:`DoseResponse.probability` maps a
    dose into an IC-1-shaped `DerivedQuantity` (samples + credible interval + the `prior_dominated`
    honesty flag). When ``dose`` is a `Posterior`, the pushforward runs through the posterior's own
    ``derived_quantity`` so the flag propagates correctly from the exposure uncertainty.
  * :func:`cumulative_exposure` -- trapezoidal time-integration of an exposure series, with an
    optional first-order biological-decay discount (older exposure counts less toward the current
    body burden) -- the intake feeding a chronic dose-response evaluation.
  * :func:`population_risk` -- aggregates per-receptor dose-response probabilities (one draw per
    posterior sample, or a single point evaluation for a bare array) into an expected-case-count
    `DerivedQuantity`.

This module supplies the dose-response *machinery*; it ships no regulatory/clinical dose-response
table -- ``params`` are always supplied by the caller or a knowledge lookup (see Non-goals).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mixle.reason.posterior_protocol import DerivedQuantity, Posterior

# `Posterior`/`DerivedQuantity` are imported lazily inside the functions that need them at runtime
# (rather than at module level) so that merely importing `mixle.analysis` does not force-load
# `mixle.reason`'s package `__init__` -- `mixle.analysis.extreme` sits on `mixle.inference.risk`'s
# import path, which `mixle.stats.bayes.dirichlet` pulls in while it is itself mid-initialization;
# a module-level import here would close that into a real circular-import failure of `mixle.stats`.

DOSE_RESPONSE_MODELS = ("loglinear", "logit", "hill", "threshold_linear")


@dataclass
class _SampleDerivedQuantity:
    """A concrete IC-1 `DerivedQuantity`: a draw matrix + the honesty flag, CI by empirical quantile.

    The same "samples + quantile-based `credible_interval` + `prior_dominated`" shape used by the
    frozen IC-1 conformance stub and the H4 stochastic-plan tests -- the repo's established idiom for
    a concrete derived quantity, rather than a bespoke one per caller.
    """

    samples: np.ndarray
    prior_dominated: bool = False

    def credible_interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        a = (1.0 - level) / 2.0
        return np.quantile(self.samples, a, axis=0), np.quantile(self.samples, 1.0 - a, axis=0)


def _dose_response_fn(model: str, params: dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """Return the elementwise dose -> outcome-probability map for the named ``model``."""
    if model == "loglinear":
        beta = float(params["beta"])
        return lambda d: 1.0 - np.exp(-beta * np.clip(np.asarray(d, dtype=float), 0.0, None))
    if model == "logit":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 0.0))
        return lambda d: 1.0 / (1.0 + np.exp(-(a * np.asarray(d, dtype=float) + b)))
    if model == "hill":
        emax = float(params.get("emax", 1.0))
        ec50 = float(params["ec50"])
        hill_n = float(params.get("n", 1.0))

        def _hill(d: np.ndarray) -> np.ndarray:
            x = np.clip(np.asarray(d, dtype=float), 0.0, None)
            xn = x**hill_n
            return emax * xn / (ec50**hill_n + xn)

        return _hill
    if model == "threshold_linear":
        slope = float(params["slope"])
        threshold = float(params.get("threshold", 0.0))
        return lambda d: np.clip(slope * (np.asarray(d, dtype=float) - threshold), 0.0, 1.0)
    raise ValueError(f"unknown dose-response model {model!r}; expected one of {DOSE_RESPONSE_MODELS}")


def _as_dose_samples(dose: Any, n: int, rng: np.random.Generator) -> np.ndarray:
    """Coerce a bare-array/scalar ``dose`` into ``n`` dose draws (Posterior doses take a separate path).

    A scalar (or length-1 array) is a degenerate point mass, replicated ``n`` times. A length-``n``
    array is treated as an already-drawn ensemble. Any other length is an "array-with-UQ" sample set
    of a different size, resampled with replacement to ``n`` draws.
    """
    arr = np.atleast_1d(np.asarray(dose, dtype=float))
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    if arr.shape[0] == n:
        return arr
    idx = rng.integers(0, arr.shape[0], size=n)
    return arr[idx]


@dataclass
class DoseResponse:
    """A named dose-response model: ``model`` selects the functional form, ``params`` its coefficients.

    ``model in {"loglinear", "logit", "hill", "threshold_linear"}``:

      * ``loglinear``: ``P = 1 - exp(-beta * dose)`` (``params: {"beta"}``) -- the EPA-style linear
        low-dose cancer/chronic form.
      * ``logit``: ``P = sigmoid(a * dose + b)`` (``params: {"a", "b"}``, both optional).
      * ``hill``: ``P = emax * dose^n / (ec50^n + dose^n)`` (``params: {"ec50"}``, ``"emax"``/``"n"``
        optional) -- saturating receptor-occupancy form.
      * ``threshold_linear``: ``P = clip(slope * (dose - threshold), 0, 1)`` (``params: {"slope"}``,
        ``"threshold"`` optional) -- no response below ``threshold``.

    No regulatory dose-response table ships here -- ``params`` are supplied by the caller (see the
    module Non-goals).
    """

    model: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model not in DOSE_RESPONSE_MODELS:
            raise ValueError(f"unknown dose-response model {self.model!r}; expected one of {DOSE_RESPONSE_MODELS}")

    def response_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """The elementwise dose -> outcome-probability function for this model + params."""
        return _dose_response_fn(self.model, self.params)

    def probability(self, dose: Any, *, n: int = 2000, rng: np.random.Generator) -> DerivedQuantity:
        """Push ``dose`` through the dose-response model into an outcome-probability `DerivedQuantity`.

        ``dose`` may be an IC-1 `Posterior` over exposure (the pushforward runs through the
        posterior's own ``derived_quantity``, so `prior_dominated` propagates from the exposure
        uncertainty), an array of dose draws/ensemble members ("array-with-UQ", resampled to ``n``
        if its length differs), or a bare scalar (a degenerate point mass -- the returned quantity
        still carries a (trivial) credible interval).
        """
        from mixle.reason.posterior_protocol import Posterior

        fn = self.response_fn()
        if isinstance(dose, Posterior):
            return dose.derived_quantity(fn, n, rng)
        draws = _as_dose_samples(dose, n, rng)
        return _SampleDerivedQuantity(samples=fn(draws), prior_dominated=False)


def cumulative_exposure(series: np.ndarray, dt: float, *, decay: float = 0.0) -> float:
    """Time-integrated exposure (trapezoidal rule), with optional first-order biological decay.

    ``decay=0`` is the plain trapezoidal integral of ``series`` over its ``dt``-spaced timesteps
    (area under the exposure-rate curve). ``decay > 0`` discounts each sample toward the *final*
    timestep by ``exp(-decay * (t_end - t))`` before integrating -- the way a biological half-life
    would -- so a spike long ago contributes less to the current cumulative body burden than an
    equally large spike near the end of the series. Feeds a chronic dose-response evaluation (e.g.
    via :meth:`DoseResponse.probability`).
    """
    x = np.asarray(series, dtype=float).ravel()
    if x.size == 0:
        return 0.0
    if x.size == 1:
        return float(x[0] * dt)
    if decay <= 0.0:
        return float(np.trapz(x, dx=dt))
    times = np.arange(x.size) * dt
    decayed = x * np.exp(-decay * (times[-1] - times))
    return float(np.trapz(decayed, dx=dt))


def population_risk(
    exposure: Posterior | np.ndarray, dr: DoseResponse, *, n: int, rng: np.random.Generator
) -> DerivedQuantity:
    """Aggregate per-receptor dose-response probabilities into an expected-case-count `DerivedQuantity`.

    ``exposure`` is a per-receptor dose: an IC-1 `Posterior` whose draws are ``(n, n_receptors)`` dose
    vectors (K1/K2 transport output propagated through UQ), or a plain ``(n_receptors,)`` array of
    point doses. Each posterior draw is pushed through ``dr``'s response function and summed over
    receptors, so the returned quantity carries the expected-case-count distribution (not just its
    mean); a bare array has no exposure uncertainty and yields a degenerate (constant) distribution.
    """
    from mixle.reason.posterior_protocol import Posterior

    fn = dr.response_fn()
    if isinstance(exposure, Posterior):
        return exposure.derived_quantity(lambda draws: fn(draws).sum(axis=-1), n, rng)
    arr = np.atleast_1d(np.asarray(exposure, dtype=float))
    expected_cases = float(np.sum(fn(arr)))
    return _SampleDerivedQuantity(samples=np.full(int(n), expected_cases), prior_dominated=False)


__all__ = [
    "DOSE_RESPONSE_MODELS",
    "DoseResponse",
    "cumulative_exposure",
    "population_risk",
]
