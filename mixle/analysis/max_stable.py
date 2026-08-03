"""Max-stable processes for spatial extremes: the Smith (Gaussian-storm) model.

Block maxima of a spatial field (annual flood peaks, peak seismic amplitude, extreme porosity) are
spatially *dependent*, and that dependence has its own limit law -- a max-stable process -- which the
ordinary GEV/GPD (treated independently per site) misses. The Smith model is the canonical one:
``Z(s) = max_i xi_i * phi_Sigma(s - U_i)`` over a Poisson storm process, giving unit-Frechet margins and a
closed-form pairwise dependence. The extremal coefficient ``theta(h) in [1, 2]`` summarizes it: 1 = full
dependence (extremes always co-occur), 2 = independence.

``extremal_coefficient`` and ``bivariate_cdf`` are closed-form and exact. The storm process itself is, in
principle, an infinite Poisson process over an unbounded domain; :meth:`SmithMaxStableSampler.sample`
necessarily truncates both the storm count and the domain, so simulated margins are only
*approximately* unit Frechet -- see that method's docstring for exactly what is and is not rigorously
controlled, and :meth:`SmithMaxStableSampler.approximation_diagnostic` to check it empirically.
"""

from __future__ import annotations

import operator
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from mixle.utils.vector import cholesky_logdet

__all__ = [
    "SmithMaxStable",
    "SmithMaxStableFit",
    "SmithMaxStableSampler",
    "FrechetApproximationDiagnostic",
    "fit_smith_maxstable",
]


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-Boolean positive integer.")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-Boolean positive integer.") from exc
    if integer <= 0:
        raise ValueError(f"{name} must be positive, got {integer}.")
    return int(integer)


def _positive_finite_scalar(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-Boolean finite positive scalar.")
    arr = np.asarray(value)
    if arr.ndim != 0 or arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite positive scalar.")
    scalar = float(arr)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive; got {value!r}.")
    return scalar


class SmithMaxStable:
    """The Smith max-stable process with Gaussian storm-profile covariance ``sigma`` (d x d, SPD).

    A *spatial process*, not an i.i.d. leaf distribution: its full likelihood is intractable, so it is not
    a ``SequenceEncodableProbabilityDistribution`` -- it exposes the things that do have closed forms
    (``extremal_coefficient``, ``bivariate_cdf``) plus a ``sampler``, and is fitted by the module-level
    :func:`fit_smith_maxstable` (composite/madogram estimation), mirroring the functional fit style of the
    other non-leaf spatial models. Margins are unit Frechet; spatial dependence grows with ``sigma``.
    """

    def __init__(self, sigma: np.ndarray):
        v = np.atleast_2d(np.asarray(sigma, dtype=float))
        if not np.all(np.isfinite(v)):
            raise ValueError("sigma must be finite.")
        if v.ndim != 2 or v.shape[0] != v.shape[1]:
            raise ValueError(f"sigma must be a square matrix; got shape {v.shape}.")
        if not np.allclose(v, v.T):
            # cholesky_logdet (like np.linalg.cholesky) factors from one triangle only and never
            # inspects the other, so an asymmetric matrix with a positive-definite-looking triangle
            # would otherwise pass straight through -- checked before that call, matching the
            # symmetry validation already established for WishartDistribution's scale matrix.
            raise ValueError("sigma must be symmetric.")
        logdet = cholesky_logdet(v)
        if logdet is None:
            raise ValueError("sigma must be positive definite.")
        sigma_owned = np.array(v, copy=True)
        inverse = np.linalg.inv(sigma_owned)
        sigma_owned.setflags(write=False)
        inverse.setflags(write=False)
        self._sigma = sigma_owned
        self._sigma_view = sigma_owned.view()
        self._sigma_view.setflags(write=False)
        self.dim = sigma_owned.shape[0]
        self._inv = inverse

    @property
    def sigma(self) -> np.ndarray:
        """Owned read-only covariance used consistently by dependence formulas and simulation."""
        return self._sigma_view

    def _mahalanobis(self, h: np.ndarray) -> float:
        h = np.atleast_1d(np.asarray(h, dtype=float))
        if h.shape != (self.dim,):
            raise ValueError(f"h must have shape ({self.dim},) to match sigma's dimension; got shape {h.shape}.")
        if not np.all(np.isfinite(h)):
            raise ValueError("h must contain only finite lag coordinates.")
        squared = float(h @ self._inv @ h)
        if not np.isfinite(squared) or squared < -1e-12:
            raise ValueError("Mahalanobis lag length must remain finite and nonnegative.")
        return float(np.sqrt(max(squared, 0.0)))

    def extremal_coefficient(self, h: np.ndarray) -> float:
        """``theta(h) = 2 * Phi(a/2)`` with ``a`` the Mahalanobis lag length -- 1 at h=0 (full dependence)
        rising to 2 as the lag grows (independence)."""
        coefficient = float(2.0 * norm.cdf(self._mahalanobis(h) / 2.0))
        if not np.isfinite(coefficient) or not 1.0 <= coefficient <= 2.0:
            raise ValueError("extremal coefficient must be finite and in [1, 2].")
        return coefficient

    def bivariate_cdf(self, z1: float, z2: float, h: np.ndarray) -> float:
        """``P(Z(s) <= z1, Z(s+h) <= z2) = exp(-V(z1, z2))`` -- the Smith bivariate distribution.

        ``z1``/``z2`` are thresholds on unit-Frechet margins and so must be strictly positive (the
        unit-Frechet support is ``(0, inf)``); the closed form below divides by ``z1``/``z2`` and takes
        ``log(z2/z1)``, which silently produces ``nan``/``inf`` rather than a real probability at or
        below 0.
        """
        if not (np.isfinite(z1) and z1 > 0 and np.isfinite(z2) and z2 > 0):
            raise ValueError(
                f"z1 and z2 must be finite and strictly positive (unit-Frechet support is (0, inf)); "
                f"got z1={z1}, z2={z2}."
            )
        a = self._mahalanobis(h)
        if a < 1e-12:
            probability = float(np.exp(-1.0 / min(z1, z2)))  # fully dependent limit
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("Smith bivariate CDF must be a finite probability in [0, 1].")
            return probability
        v = (1.0 / z1) * norm.cdf(a / 2.0 + np.log(z2 / z1) / a) + (1.0 / z2) * norm.cdf(a / 2.0 + np.log(z1 / z2) / a)
        probability = float(np.exp(-v))
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("Smith bivariate CDF must be a finite probability in [0, 1].")
        return probability

    def sampler(self, locations: np.ndarray, seed: int | None = None) -> SmithMaxStableSampler:
        """Return a sampler over the requested spatial locations."""
        loc = np.asarray(locations, dtype=float)
        if loc.ndim != 2 or loc.shape[0] == 0 or loc.shape[1] != self.dim:
            raise ValueError(
                f"locations must be a nonempty (n, {self.dim}) matrix to match sigma; got shape {loc.shape}."
            )
        if not np.all(np.isfinite(loc)):
            raise ValueError("locations must be finite.")
        return SmithMaxStableSampler(self, loc, seed)


@dataclass
class SmithMaxStableFit:
    """A fitted isotropic Smith max-stable process, plus a diagnostic of whether the pairwise
    dependence-scale match actually identified an interior optimum.

    ``status`` is one of:

    * ``"ok"``: the fitted scale is an interior point of the search bracket -- a genuine stationary
      point of the F-madogram least-squares objective.
    * ``"boundary"``: the fit landed at (or numerically indistinguishable from) the edge of the search
      bracket -- the signature of an objective with no interior minimum. This happens, for example,
      when replicates carry almost no independent information (every location's rank collapses
      together), which pushes the empirical extremal coefficient to the same value for every pair and
      drives the least-squares match monotonically toward one edge instead of a genuine optimum.
      ``model`` is still populated with the boundary value, but it is a search artifact, not a
      dependence-scale estimate, and should not be trusted.

    Attributes:
        model: the fitted :class:`SmithMaxStable`.
        n_locations: distinct spatial locations used.
        n_replicates: replicated fields used.
        n_pairs: location pairs the least-squares match was computed over (``n_locations choose 2``).
        residual: the F-madogram objective's value at the fitted scale (mean squared error between the
            empirical and model extremal coefficients across pairs; exactly 0 for a 2-location fit,
            which has one pair and one parameter).
        status: see above.

    Construction validates the record against its own documented domain (MXR-080-1900). It previously
    validated nothing at all, so ``SmithMaxStableFit(model=None, n_locations=-1, n_replicates=0,
    n_pairs=-5, residual=nan)`` constructed cleanly and reported ``converged is True`` -- a record
    claiming a converged fit that had no model, no data and no objective value. The invariants are
    exactly what :func:`fit_smith_maxstable` produces: a real :class:`SmithMaxStable`; at least two
    locations (a dependence scale needs a pair) and one replicate; ``n_pairs`` the ``n_locations
    choose 2`` the docstring above states; a finite, non-negative ``residual`` (it is a mean of
    squares); and a ``status`` this class actually defines.

    Deliberately NOT changed: the record stays a MUTABLE dataclass. Its frozen siblings in this
    package were converted under a separate finding, and freezing this one is a broader API change
    than the shape invariants this validates -- the fields can still be rebound after construction and
    escape the checks below.
    """

    model: SmithMaxStable
    n_locations: int
    n_replicates: int
    n_pairs: int
    residual: float
    status: str = "ok"

    def __post_init__(self) -> None:
        if not isinstance(self.model, SmithMaxStable):
            raise TypeError(f"SmithMaxStableFit.model must be a fitted SmithMaxStable, got {type(self.model).__name__}")
        for name, minimum in (("n_locations", 2), ("n_replicates", 1), ("n_pairs", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
                raise ValueError(
                    f"SmithMaxStableFit.{name} must be an exact non-Boolean integer >= {minimum}, got {value!r}"
                )
        expected_pairs = self.n_locations * (self.n_locations - 1) // 2
        if int(self.n_pairs) != expected_pairs:
            raise ValueError(
                f"SmithMaxStableFit.n_pairs must be n_locations choose 2 = {expected_pairs} for "
                f"{self.n_locations} locations, got {self.n_pairs!r}"
            )
        residual = float(self.residual)
        if not np.isfinite(residual) or residual < 0.0:
            raise ValueError(
                "SmithMaxStableFit.residual is a mean squared error and must be finite and non-negative, "
                f"got {self.residual!r}"
            )
        if self.status not in ("ok", "boundary"):
            raise ValueError(f"SmithMaxStableFit.status must be 'ok' or 'boundary', got {self.status!r}")

    @property
    def converged(self) -> bool:
        """``True`` only when the fit is a genuine interior optimum, not a search-bound artifact."""
        return self.status == "ok"


def fit_smith_maxstable(locations: np.ndarray, fields: np.ndarray) -> SmithMaxStableFit:
    """Fit an isotropic Smith max-stable process (``sigma = s^2 I``) to replicated spatial extremes.

    ``locations`` is ``(n_locations, d)`` and ``fields`` is ``(n_replicates, n_locations)`` of block
    maxima. Estimation matches the binned empirical extremal coefficient (from the F-madogram) to the
    model ``2 Phi(|h| / (2 s))``.

    This is a single-parameter (``s``) fit estimated from pairwise statistics, so it is only
    identifiable given enough structure to compute those statistics from:

    * at least 2 locations, so there is at least one pair (with at least one nonzero lag between some
      pair) to compute an empirical extremal coefficient from at all;
    * at least 2 replicates, so the per-location rank transform is not degenerate -- with exactly 1
      replicate every location's rank trivially collapses to a single point, making the empirical
      extremal coefficient identically 1 (full dependence) for every pair regardless of distance, which
      is not a real dependence signal.

    Returns a :class:`SmithMaxStableFit` (the fitted model plus fit-quality diagnostics); raises
    ``ValueError`` if the inputs are malformed or too small to attempt a fit at all.
    """
    from scipy.optimize import minimize_scalar

    loc = np.atleast_2d(np.asarray(locations, dtype=float))
    if not np.all(np.isfinite(loc)):
        raise ValueError("locations must be finite.")
    if loc.shape[0] < 2:
        raise ValueError(
            "fit_smith_maxstable needs at least 2 locations: the isotropic model's single "
            "dependence-scale parameter is estimated from pairwise empirical extremal coefficients, "
            f"and with fewer than 2 locations there is no pair to compute one from; got {loc.shape[0]}."
        )

    z = np.asarray(fields, dtype=float)  # (n_replicates, n_locations), unit-Frechet-ish
    if z.ndim != 2 or z.shape[1] != loc.shape[0]:
        raise ValueError(
            f"fields must have shape (n_replicates, n_locations={loc.shape[0]}) to match locations; got {z.shape}."
        )
    if not np.all(np.isfinite(z)):
        raise ValueError("fields must be finite (no NaN/Inf replicated maxima).")
    if z.shape[0] < 2:
        raise ValueError(
            "fit_smith_maxstable needs at least 2 replicates: the empirical extremal coefficient is a "
            "rank statistic across replicates, and with exactly 1 replicate every location's rank "
            f"trivially collapses to a single point (identically 'fully dependent' for every pair, "
            f"carrying no spatial information); got {z.shape[0]}."
        )

    d = loc.shape[1]
    # empirical extremal coefficient per pair via the F-madogram: theta = (1 + nu) / (1 - nu) with
    # nu = E|F(Z1) - F(Z2)| on uniform margins (nu = 1/3 at independence -> theta = 2).
    u = np.argsort(np.argsort(z, axis=0), axis=0) / (z.shape[0] + 1.0)  # rank-transform to uniform margins
    pairs = [(i, j) for i in range(len(loc)) for j in range(i + 1, len(loc))]
    lags = np.array([np.linalg.norm(loc[i] - loc[j]) for i, j in pairs])
    if lags.max() <= 0.0:
        raise ValueError(
            "fit_smith_maxstable needs at least one pair of distinct locations (nonzero separation) "
            "to estimate a spatial dependence scale; every provided location coincides."
        )
    nu = np.array([np.mean(np.abs(u[:, i] - u[:, j])) for i, j in pairs])
    theta_emp = np.clip((1 + nu) / (1 - nu + 1e-9), 1.0, 2.0)

    def obj(s):
        theta_model = 2.0 * norm.cdf(lags / (2.0 * max(s, 1e-3)))
        return np.mean((theta_model - theta_emp) ** 2)

    lo_bound, hi_bound = 0.05, 10 * (lags.max() + 1e-9)
    fitted = minimize_scalar(obj, bounds=(lo_bound, hi_bound), method="bounded")
    s = float(fitted.x)
    # A "bounded" Brent search can land extremely close to, but not bit-exactly at, the edge; treat
    # anything within a small fraction of the bracket width as a boundary (non-interior) solution -- see
    # SmithMaxStableFit for why that is a search artifact rather than a genuine estimate.
    edge_tol = 1e-3 * (hi_bound - lo_bound)
    status = "boundary" if (s - lo_bound < edge_tol or hi_bound - s < edge_tol) else "ok"

    return SmithMaxStableFit(
        model=SmithMaxStable(s**2 * np.eye(d)),
        n_locations=loc.shape[0],
        n_replicates=z.shape[0],
        n_pairs=len(pairs),
        residual=float(fitted.fun),
        status=status,
    )


@dataclass(frozen=True)
class FrechetApproximationDiagnostic:
    """Empirical check of how close a truncated :class:`SmithMaxStableSampler` draw is to true unit
    Frechet margins.

    At each site and each requested probability level ``p``, compares the *empirical* CDF of simulated
    draws (evaluated at the *true* unit-Frechet quantile for ``p``) against ``p`` itself -- a
    QQ/PP-plot-style discrepancy. Both summary statistics should shrink toward 0 as the sampler's
    truncation is loosened (smaller ``tol``, larger ``box_sigma``/``n_storms``) and should NOT shrink
    just from raising ``n`` alone (``n`` only reduces Monte Carlo noise in the estimate of a possibly
    nonzero bias).

    Attributes:
        n_replicates: number of simulated draws the diagnostic was computed from.
        probability_levels: the probability levels checked.
        max_abs_error: worst-case ``|empirical_cdf(q_p) - p|`` over every (site, level) pair.
        mean_abs_error: mean ``|empirical_cdf(q_p) - p|`` over every (site, level) pair.
        per_site_max_abs_error: worst-case error at each site, shape ``(n_locations,)``.
    """

    n_replicates: int
    probability_levels: tuple[float, ...]
    max_abs_error: float
    mean_abs_error: float
    per_site_max_abs_error: np.ndarray

    def __post_init__(self) -> None:
        n_replicates = _positive_integer(self.n_replicates, name="n_replicates")
        levels = np.asarray(self.probability_levels, dtype=float)
        if levels.ndim != 1 or levels.size == 0 or not np.all(np.isfinite(levels)):
            raise ValueError("probability_levels must be a nonempty finite one-dimensional sequence.")
        if np.any(levels <= 0.0) or np.any(levels >= 1.0):
            raise ValueError("probability_levels must lie strictly inside (0, 1).")
        maximum = float(self.max_abs_error)
        mean = float(self.mean_abs_error)
        per_site = np.asarray(self.per_site_max_abs_error, dtype=float)
        if per_site.ndim != 1 or per_site.size == 0 or not np.all(np.isfinite(per_site)):
            raise ValueError("per_site_max_abs_error must be a nonempty finite one-dimensional array.")
        if not (np.isfinite(maximum) and np.isfinite(mean) and 0.0 <= mean <= maximum <= 1.0):
            raise ValueError("diagnostic errors must be finite and satisfy 0 <= mean <= max <= 1.")
        if np.any(per_site < 0.0) or np.any(per_site > 1.0):
            raise ValueError("per-site diagnostic errors must lie in [0, 1].")
        if not np.isclose(maximum, float(per_site.max()), rtol=1e-12, atol=1e-15):
            raise ValueError("max_abs_error must equal the maximum per-site absolute error.")
        owned = np.array(per_site, copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "n_replicates", n_replicates)
        object.__setattr__(self, "probability_levels", tuple(float(level) for level in levels))
        object.__setattr__(self, "max_abs_error", maximum)
        object.__setattr__(self, "mean_abs_error", mean)
        object.__setattr__(self, "per_site_max_abs_error", owned)


class SmithMaxStableSampler:
    """Sampler for a fitted Smith max-stable process at fixed locations.

    :meth:`sample` truncates the (in principle unbounded, infinite-storm) Smith spectral
    representation to a finite spatial box and a tolerance-controlled number of storms; see its
    docstring for exactly what that truncation does and does not rigorously control, and
    :meth:`approximation_diagnostic` for an empirical check of the result.
    """

    def __init__(self, dist: SmithMaxStable, locations: np.ndarray, seed: int | None = None):
        self.dist = dist
        owned_locations = np.array(locations, dtype=float, copy=True)
        # At least one location, matching the process's own dimension. An empty (0, d) matrix used to
        # construct a perfectly normal-looking sampler and only fail later, deep inside `sample`, as
        # "zero-size array to reduction operation minimum which has no identity" -- a NumPy message
        # that says nothing about the actual mistake. A max-stable field with no locations is not a
        # degenerate draw, it is a caller error.
        if owned_locations.ndim != 2 or owned_locations.shape[0] == 0:
            raise ValueError(
                "SmithMaxStableSampler: locations must be a non-empty (n_locations, d) matrix, "
                f"got shape {owned_locations.shape}."
            )
        expected_d = int(np.shape(dist.sigma)[0])
        if owned_locations.shape[1] != expected_d:
            raise ValueError(
                f"SmithMaxStableSampler: locations must have {expected_d} coordinate(s) per site to match "
                f"the process covariance, got {owned_locations.shape[1]}."
            )
        if not np.all(np.isfinite(owned_locations)):
            raise ValueError("SmithMaxStableSampler: locations must be finite (no NaN/Inf).")
        owned_locations.setflags(write=False)
        self.loc = owned_locations
        self.rng = np.random.RandomState(seed)
        self._chol = np.linalg.cholesky(dist.sigma)
        self._logdet = 2.0 * np.sum(np.log(np.diag(self._chol)))

    def _storm(self, u: np.ndarray) -> np.ndarray:
        """Gaussian storm profile phi_Sigma(loc - u) at every location."""
        diff = self.loc - u
        sol = np.linalg.solve(self._chol, diff.T)
        d = self.loc.shape[1]
        return np.exp(-0.5 * np.sum(sol**2, axis=0) - 0.5 * self._logdet - 0.5 * d * np.log(2 * np.pi))

    def _peak_density(self) -> float:
        """The Gaussian storm kernel's peak value ``phi_Sigma(0)`` -- an exact upper bound on
        ``phi_Sigma(s - u)`` over every site ``s`` and every possible storm center ``u``, attained only
        when a storm is centered exactly on a site. Used by :meth:`sample` to bound how much any
        not-yet-drawn storm could possibly still contribute, regardless of where it lands.
        """
        d = self.loc.shape[1]
        return float(np.exp(-0.5 * self._logdet - 0.5 * d * np.log(2 * np.pi)))

    def sample(
        self,
        size: int | None = None,
        *,
        n_storms: int = 20_000,
        tol: float = 1e-3,
        box_sigma: float = 5.0,
    ) -> np.ndarray:
        """Draw max-stable field(s) at the locations via a boxed, tolerance-truncated storm sum.

        The true Smith representation is a max over an infinite Poisson process of storms
        ``(xi_i, U_i)`` spread over the whole of R^d; only then are margins exactly unit Frechet. This
        sampler approximates that in two distinct ways, one rigorously controlled and one not:

        1. **Storm-count truncation (rigorously controlled by ``tol``).** Storms are drawn in
           decreasing order of their Poisson intensity mark ``xi`` (``xi_i = 1/Gamma_i``, the natural
           arrival order of this construction). Because the Gaussian storm kernel is bounded by its
           peak density ``phi_Sigma(0)``, a storm with mark ``xi`` can contribute at most
           ``xi * vol * phi_Sigma(0)`` to *any* site, no matter where its center ``U_i`` lands (``vol``
           the sampling box's volume). Since ``xi`` is strictly decreasing, once that bound drops below
           ``tol`` for one storm it has also dropped below ``tol`` for every later one -- so it is exact,
           not heuristic, to stop there: every dropped storm's true contribution is provably below
           ``tol`` at every site, which bounds the total storm-count truncation error (relative to the
           boxed field's own true value) by ``tol``. ``n_storms`` is a safety cap on iterations, not the
           primary control; a :class:`UserWarning` is raised if that cap binds before ``tol`` is reached
           for some replicate, since then the ``tol`` guarantee was not actually achieved. Every storm
           drawn is used before its own ``tol`` check is applied, so at least one storm always
           contributes to the field regardless of how loose ``tol`` is -- an all-zero field (like the
           previous ``n_storms=0`` behavior) is just as invalid a "draw" as a negative one, since 0 is
           outside unit Frechet's ``(0, inf)`` support.
        2. **Spatial truncation (NOT rigorously controlled).** Storm centers are drawn only from a
           finite box around the query locations, expanded by ``box_sigma`` storm-kernel standard
           deviations in each dimension, rather than the true unbounded domain. This introduces a bias
           that empirically shrinks as ``box_sigma`` grows, but is not exactly bounded here -- use
           :meth:`approximation_diagnostic` to check it for your particular locations/sigma.

        Net effect: simulated margins are only *approximately*, not exactly, unit Frechet. The
        storm-count half of that approximation is controlled to within ``tol`` (when ``n_storms``
        doesn't bind first); the box half is not formally bounded and should be checked empirically.

        Args:
            size: number of independent replicate fields (``None`` for a single field).
            n_storms: safety cap on storms drawn per replicate; must be a positive integer. Storms
                stop earlier, as soon as ``tol`` is satisfied. Must be a positive integer -- with
                ``n_storms=0`` no storms are drawn at all, which cannot produce a valid unit-Frechet
                draw (0 is outside its ``(0, inf)`` support), so that case is rejected outright rather
                than silently returned as if it were one.
            tol: stop once every remaining storm's max-possible contribution to every site is below
                this (absolute, field-scale) tolerance. Must be finite and strictly positive.
            box_sigma: half-width, in storm-kernel standard deviations, of the box storm centers are
                drawn from beyond the query locations' own extent. Must be finite and strictly
                positive.

        Returns:
            Field(s) at the locations. Shape ``(len(locations),)`` if ``size`` is ``None``, else
            ``(size, len(locations))``.

        ``size`` (when supplied) and ``n_storms`` are exact non-Boolean positive integers; zero does
        not represent a valid request for a field or a storm process.
        """
        n_storms = _positive_integer(n_storms, name="n_storms")
        tol = _positive_finite_scalar(tol, name="tol")
        box_sigma = _positive_finite_scalar(box_sigma, name="box_sigma")

        n = 1 if size is None else _positive_integer(size, name="size")
        lo, hi = (
            self.loc.min(0) - box_sigma * np.sqrt(np.diag(self.dist.sigma)),
            self.loc.max(0) + box_sigma * np.sqrt(np.diag(self.dist.sigma)),
        )
        vol = float(np.prod(hi - lo))
        peak_contribution = vol * self._peak_density()  # a storm with mark xi=1 contributes <= this, anywhere
        d = self.loc.shape[1]

        out = np.zeros((n, len(self.loc)))
        any_unconverged = False
        for r in range(n):
            z = np.zeros(len(self.loc))
            gamma = 0.0
            converged_r = False
            for _ in range(n_storms):
                gamma += self.rng.exponential()  # Poisson arrival of storm intensity 1/gamma
                xi = 1.0 / gamma
                u = lo + self.rng.uniform(size=d) * (hi - lo)
                z = np.maximum(z, xi * vol * self._storm(u))
                # Every storm is included BEFORE this check (so at least one always is, even under a
                # loose tol -- a field with an exact 0 is just as invalid as n_storms=0's all-zero
                # field, since unit-Frechet's support excludes 0). Once xi's bound drops below tol,
                # every remaining (smaller-xi) storm's bound is below tol too, everywhere, so stopping
                # here is exact, not heuristic.
                if xi * peak_contribution < tol:
                    converged_r = True
                    break
            any_unconverged = any_unconverged or not converged_r
            out[r] = z
        if not np.all(np.isfinite(out)) or np.any(out <= 0.0):
            raise ValueError("Smith sampler produced values outside the finite positive unit-Frechet support.")
        if any_unconverged:
            warnings.warn(
                f"SmithMaxStableSampler.sample(): n_storms={n_storms} safety cap was reached before "
                f"tol={tol} was satisfied for at least one replicate; the storm-count truncation error "
                "for that replicate is not actually bounded by tol. Raise n_storms or loosen tol, or "
                "check approximation_diagnostic().",
                stacklevel=2,
            )
        return out[0] if size is None else out

    def approximation_diagnostic(
        self,
        n: int = 2000,
        probability_levels: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9),
        **sample_kwargs,
    ) -> FrechetApproximationDiagnostic:
        """Empirically estimate how close this sampler's truncated simulation is to true unit-Frechet
        margins.

        Draws ``n`` replicate fields (with ``sample_kwargs`` forwarded to :meth:`sample`, so a specific
        ``n_storms``/``tol``/``box_sigma`` setting can be checked) and compares their empirical CDF,
        at each site and each requested probability level, against the true unit-Frechet CDF -- see
        :class:`FrechetApproximationDiagnostic`. ``n`` is an exact non-Boolean positive integer and
        ``probability_levels`` is a nonempty finite one-dimensional sequence strictly inside
        ``(0, 1)``.
        """
        n = _positive_integer(n, name="n")
        levels = np.asarray(probability_levels, dtype=float)
        if levels.ndim != 1 or levels.size == 0 or not np.all(np.isfinite(levels)):
            raise ValueError("probability_levels must be a nonempty finite one-dimensional sequence.")
        if np.any(levels <= 0.0) or np.any(levels >= 1.0):
            raise ValueError("probability_levels must lie strictly inside (0, 1).")
        z = self.sample(n, **sample_kwargs)  # size=n is never None here, so always shape (n, n_locations)
        quantiles = -1.0 / np.log(levels)  # true unit-Frechet quantile: exp(-1/q) = p  =>  q = -1/log(p)
        empirical = np.mean(z[:, :, None] <= quantiles[None, None, :], axis=0)  # (n_locations, n_levels)
        abs_err = np.abs(empirical - levels[None, :])
        if not np.all(np.isfinite(abs_err)) or np.any(abs_err < 0.0) or np.any(abs_err > 1.0):
            raise ValueError("approximation diagnostics must be finite absolute probability errors in [0, 1].")
        return FrechetApproximationDiagnostic(
            n_replicates=n,
            probability_levels=tuple(probability_levels),
            max_abs_error=float(abs_err.max()),
            mean_abs_error=float(abs_err.mean()),
            per_site_max_abs_error=abs_err.max(axis=1),
        )
