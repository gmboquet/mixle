"""Geostatistics: variograms and kriging (best linear unbiased spatial prediction).

Kriging predicts a spatially correlated field at unobserved locations and -- unlike a black-box
regressor -- returns a *prediction variance* that grows with distance from data. The spatial
correlation is encoded by a variogram ``gamma(h)`` (how fast values decorrelate with separation ``h``):

  * :func:`empirical_variogram` / :func:`fit_variogram` -- estimate and fit a variogram model
    (spherical / exponential / gaussian / matern; the Gaussian model is also reachable as
    ``"squared_exponential"`` / ``"rbf"``, its covariance being the squared-exponential kernel) with
    **nugget** (measurement error / micro-scale variance), **sill** (total variance), and **range**
    (correlation length), plus geometric **anisotropy** (direction-dependent range).
  * :func:`ordinary_kriging` -- BLUP with an unknown constant mean; exact interpolation with no nugget,
    smoothing with one, and **heteroscedastic** (per-observation) noise.
  * :func:`universal_kriging` -- kriging with a polynomial trend / external drift.
  * :func:`calibrate_variance` -- rescale kriging variances so their predictive intervals hit a target
    coverage on held-out data (generic GP/kriging recalibration).

Coordinates are ``(n, d)`` arrays (typically ``d = 2``); values are the measured field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special
from scipy.spatial.distance import cdist

# Covariance models implemented by `_shape`; the Gaussian model is reachable under three aliases.
_MODELS = ("spherical", "exponential", "gaussian", "squared_exponential", "squared-exponential", "rbf", "matern")


def _shape(model: str, h: np.ndarray, rng: float, nu: float = 1.5) -> np.ndarray:
    """Correlation-decay shape in [0, 1]: 0 at h=0, ->1 as h->inf (the standardised variogram)."""
    h = np.asarray(h, dtype=float)
    r = np.where(rng <= 0, 1e-12, rng)
    if model == "spherical":
        s = np.where(h < rng, 1.5 * h / r - 0.5 * (h / r) ** 3, 1.0)
    elif model == "exponential":
        s = 1.0 - np.exp(-h / r)
    elif model in ("gaussian", "squared_exponential", "squared-exponential", "rbf"):
        # the Gaussian variogram; its covariance psill*exp(-(h/r)^2) is the squared-exponential (RBF) kernel
        s = 1.0 - np.exp(-((h / r) ** 2))
    elif model == "matern":
        sqrt2nu = np.sqrt(2.0 * nu) * h / r
        sqrt2nu = np.where(sqrt2nu == 0, 1e-12, sqrt2nu)
        corr = (2.0 ** (1.0 - nu) / special.gamma(nu)) * (sqrt2nu**nu) * special.kv(nu, sqrt2nu)
        s = 1.0 - np.where(h == 0, 1.0, corr)
    else:
        raise ValueError(f"model must be one of {_MODELS}, got {model!r}.")
    return np.clip(s, 0.0, 1.0)


@dataclass
class Variogram:
    """A fitted variogram model ``gamma(h) = nugget + psill * shape(h)``.

    Attributes:
        model: ``"spherical"``, ``"exponential"``, ``"gaussian"`` (aka ``"squared_exponential"`` /
            ``"rbf"`` -- covariance ``psill * exp(-(h/rng)**2)``), or ``"matern"``.
        nugget: discontinuity at ``h=0`` (measurement error / micro-scale variance).
        psill: partial sill (correlated variance); ``nugget + psill`` is the total sill.
        rng: range (correlation length).
        nu: Matern smoothness (ignored by other models).
        anisotropy: optional ``(angle_rad, ratio)`` geometric anisotropy -- coordinates are rotated by
            ``angle`` and the minor axis scaled by ``1/ratio`` before distances are taken.

    Raises:
        ValueError: if ``model`` is not implemented; ``nugget``/``psill`` are not finite and ``>= 0``;
            ``rng``/``nu`` are not finite and ``> 0``; or ``anisotropy`` is set with a non-finite angle
            or a ratio that is not finite and ``> 0`` (a zero or negative ratio collapses or flips the
            minor axis, which previously produced NaN predictions and variances downstream instead of
            an error).
    """

    model: str
    nugget: float
    psill: float
    rng: float
    nu: float = 1.5
    anisotropy: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.model not in _MODELS:
            raise ValueError(f"model must be one of {_MODELS}, got {self.model!r}.")
        if not (np.isfinite(self.nugget) and self.nugget >= 0):
            raise ValueError(f"nugget must be finite and >= 0, got {self.nugget!r}.")
        if not (np.isfinite(self.psill) and self.psill >= 0):
            raise ValueError(f"psill must be finite and >= 0, got {self.psill!r}.")
        if not (np.isfinite(self.rng) and self.rng > 0):
            raise ValueError(f"rng must be finite and > 0, got {self.rng!r}.")
        if not (np.isfinite(self.nu) and self.nu > 0):
            raise ValueError(f"nu (Matern smoothness) must be finite and > 0, got {self.nu!r}.")
        if self.anisotropy is not None:
            if len(self.anisotropy) != 2:
                raise ValueError(f"anisotropy must be an (angle, ratio) pair, got {self.anisotropy!r}.")
            angle, ratio = self.anisotropy
            if not np.isfinite(angle):
                raise ValueError(f"anisotropy angle must be finite, got {angle!r}.")
            if not (np.isfinite(ratio) and ratio > 0):
                raise ValueError(
                    f"anisotropy ratio must be finite and > 0, got {ratio!r} "
                    "(a zero or negative ratio collapses or flips the minor axis)."
                )

    def gamma(self, h: np.ndarray) -> np.ndarray:
        """Evaluate the semivariogram at lag distances."""
        return self.nugget * (np.asarray(h) > 0) + self.psill * _shape(self.model, h, self.rng, self.nu)

    def cov_field(self, h: np.ndarray) -> np.ndarray:
        """Covariance of the *correlated* field part (excludes the nugget): ``psill (1 - shape)``."""
        return self.psill * (1.0 - _shape(self.model, h, self.rng, self.nu))


def _transform(coords: np.ndarray, anisotropy: tuple[float, float] | None) -> np.ndarray:
    if anisotropy is None or coords.shape[1] != 2:
        return coords
    angle, ratio = anisotropy
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, s], [-s, c]])
    scaled = coords @ rot.T
    scaled[:, 1] /= ratio
    return scaled


def empirical_variogram(
    coords: np.ndarray, values: np.ndarray, *, n_bins: int = 15, max_dist: float | None = None
) -> dict[str, np.ndarray | bool | str]:
    """Binned empirical (semi-)variogram: mean ``0.5 (z_i - z_j)^2`` by separation distance.

    ``max_dist`` defaults to half the largest pairwise distance. Bins are ``n_bins`` equal-width
    intervals, left-closed and right-open (``[edges[b], edges[b+1])``) except the last, which is
    closed on both ends -- so a pair separated by exactly ``max_dist`` (routine, not an edge case:
    e.g. the endpoints of an evenly spaced line always sit exactly on the default cutoff) lands in
    the last bin instead of being classified as "beyond the last bin" and silently dropped.

    Returns:
        ``{'lag', 'semivariance', 'count', 'insufficient_evidence', 'reason'}`` -- ``lag`` /
        ``semivariance`` / ``count`` cover each non-empty distance bin. ``insufficient_evidence`` is
        ``True`` (with ``reason`` set and the other fields empty) only when no pair falls within
        ``[0, max_dist]`` at all -- e.g. a single input point, or a ``max_dist`` narrower than every
        pairwise distance -- so no bin is identifiable; it is ``False`` (with ``reason`` empty)
        whenever at least one bin is populated.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(values, dtype=float).ravel()
    d = cdist(coords, coords)
    iu = np.triu_indices_from(d, k=1)
    dist = d[iu]
    sv = 0.5 * (z[iu[0]] - z[iu[1]]) ** 2
    if max_dist is None:
        max_dist = dist.max() / 2.0 if dist.size else 0.0
    edges = np.linspace(0, max_dist, n_bins + 1)
    idx = np.digitize(dist, edges) - 1
    # np.digitize is half-open on the right, so a pair sitting exactly on the outer edge digitizes to
    # n_bins ("beyond the last bin") and would otherwise be dropped. Fold that boundary case into the
    # last bin; pairs genuinely beyond max_dist also digitize to n_bins but fail `dist <= max_dist`,
    # so they correctly stay excluded by the `range(n_bins)` loop below.
    idx = np.where((idx == n_bins) & (dist <= max_dist), n_bins - 1, idx)
    lag, semi, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if np.any(m):
            lag.append(0.5 * (edges[b] + edges[b + 1]))
            semi.append(float(sv[m].mean()))
            cnt.append(int(m.sum()))
    if not cnt:
        return {
            "lag": np.array([]),
            "semivariance": np.array([]),
            "count": np.array([], dtype=int),
            "insufficient_evidence": True,
            "reason": (
                "no point pair falls within [0, max_dist]: the empirical variogram is not "
                "identifiable from this coordinate/distance configuration."
            ),
        }
    return {
        "lag": np.asarray(lag),
        "semivariance": np.asarray(semi),
        "count": np.asarray(cnt, dtype=int),
        "insufficient_evidence": False,
        "reason": "",
    }


# nugget, partial sill, and range are 3 free parameters; fewer populated lag bins leaves the
# least-squares fit underdetermined (mirrors the >= 2 frequency classes good_turing needs to fit its
# own 2-parameter log-linear smoother).
_MIN_POPULATED_BINS_FOR_FIT = 3


def fit_variogram(
    coords: np.ndarray, values: np.ndarray, *, model: str = "spherical", n_bins: int = 15, nu: float = 1.5
) -> Variogram:
    """Fit a variogram model to data by least squares on the empirical variogram.

    Returns:
        A fitted :class:`Variogram` (nugget, partial sill, range).

    Raises:
        ValueError: if the empirical variogram has fewer than :data:`_MIN_POPULATED_BINS_FOR_FIT`
            populated bins, so nugget/partial sill/range are not identifiable from the data (see
            :func:`empirical_variogram`).
    """
    ev = empirical_variogram(coords, values, n_bins=n_bins)
    if ev["insufficient_evidence"]:
        raise ValueError(f"cannot fit a variogram: {ev['reason']}")
    lag, semi, cnt = ev["lag"], ev["semivariance"], ev["count"]
    if len(cnt) < _MIN_POPULATED_BINS_FOR_FIT:
        raise ValueError(
            f"cannot fit a variogram: only {len(cnt)} populated lag bin(s), but nugget/partial "
            f"sill/range (3 free parameters) need at least {_MIN_POPULATED_BINS_FOR_FIT} to be "
            "identifiable. Provide more points, a larger max_dist, or fewer n_bins."
        )
    var = float(np.var(np.asarray(values, dtype=float)))
    max_lag = float(lag.max())
    # weight bins by the square root of their pair count (more-populated lags are more reliable)
    wt = np.sqrt(cnt)

    def resid(p: np.ndarray) -> np.ndarray:
        nugget, psill, rng = p
        pred = nugget + psill * _shape(model, lag, rng, nu)
        return wt * (pred - semi)

    p0 = np.array([max(semi.min(), 1e-6), var, max_lag / 3.0])
    # bound the range to the observed lags so the fit can't run to infinity on a non-saturating cloud
    sol = optimize.least_squares(resid, p0, bounds=([0, 0, 1e-6], [var * 5 + 1e-9, var * 5, 3.0 * max_lag]))
    nugget, psill, rng = sol.x
    return Variogram(model, float(nugget), float(psill), float(rng), nu)


def _validate_krige_geometry(coords: np.ndarray, z: np.ndarray, query: np.ndarray) -> None:
    """Shape/finiteness validation shared by :func:`ordinary_kriging` and :func:`universal_kriging`."""
    if coords.ndim != 2 or coords.shape[0] == 0:
        raise ValueError(f"coords must be a nonempty (n, d) array, got shape {coords.shape}.")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite values.")
    n, d = coords.shape
    if z.shape != (n,):
        raise ValueError(f"values must have shape ({n},) to match coords, got {z.shape}.")
    if not np.all(np.isfinite(z)):
        raise ValueError("values must contain only finite values.")
    if query.ndim != 2 or query.shape[0] == 0:
        raise ValueError(f"query must be a nonempty (q, d) array, got shape {query.shape}.")
    if query.shape[1] != d:
        raise ValueError(f"query must have {d} column(s) to match coords, got shape {query.shape}.")
    if not np.all(np.isfinite(query)):
        raise ValueError("query must contain only finite values.")


def _clip_variance(var: np.ndarray, scale: float) -> np.ndarray:
    """Clip kriging variance to ``>= 0``, but only across a small numerical-roundoff tolerance.

    A well-posed covariance solve should never produce a negative predictive variance; a tiny
    negative value is ordinary floating-point roundoff from the linear solve (empirically, up to
    ~1e-14 relative to the covariance scale even on moderately ill-conditioned systems) and is safe
    to zero out. A materially negative value instead indicates the covariance solve itself is invalid
    or too ill-conditioned to trust (e.g. an inconsistent variogram fit or near-duplicate points), and
    should be surfaced rather than silently hidden by an unconditional clip.

    The tolerance mirrors ``numpy.allclose``'s ``atol + rtol * scale`` convention: an absolute floor
    for when the problem's own scale is ~0, plus a relative term several orders of magnitude above
    the roundoff actually observed, so it stays generous without masking a real solve failure.
    """
    tol = 1e-10 + 1e-8 * scale
    if np.any(var < -tol):
        raise ValueError(
            f"kriging variance is materially negative (worst value {float(np.min(var)):.6g}, "
            f"roundoff tolerance -{tol:.3g}); this indicates an invalid or ill-conditioned covariance "
            "solve rather than floating-point roundoff -- check the variogram parameters and point "
            "configuration (e.g. near-duplicate points or a range/nugget mismatch)."
        )
    return np.clip(var, 0.0, None)


def _krige_solve(
    coords: np.ndarray,
    z: np.ndarray,
    variogram: Variogram,
    query: np.ndarray,
    *,
    drift: np.ndarray | None,
    drift0: np.ndarray | None,
    noise: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    n = coords.shape[0]
    if noise is not None:
        noise = np.asarray(noise, dtype=float)
        if noise.shape != (n,):
            raise ValueError(f"noise must have shape ({n},) to match coords, got {noise.shape}.")
        if not np.all(np.isfinite(noise)):
            raise ValueError("noise must contain only finite values.")
        if np.any(noise < 0):
            raise ValueError("noise must be >= 0 (it represents a measurement variance).")
    if drift is not None:
        if drift.ndim != 2 or drift.shape[0] != n:
            raise ValueError(f"drift must have {n} row(s) to match coords, got shape {drift.shape}.")
        q_expected, p = query.shape[0], drift.shape[1]
        if drift0 is None or drift0.shape != (q_expected, p):
            got = None if drift0 is None else drift0.shape
            raise ValueError(f"drift0 must have shape ({q_expected}, {p}) to match query and drift, got {got}.")
        if not np.all(np.isfinite(drift)) or not np.all(np.isfinite(drift0)):
            raise ValueError("drift and drift0 must contain only finite values.")

    coords = _transform(np.atleast_2d(coords), variogram.anisotropy)
    query = _transform(np.atleast_2d(query), variogram.anisotropy)
    dd = cdist(coords, coords)
    K = variogram.cov_field(dd)
    nug = variogram.nugget if noise is None else noise
    K[np.diag_indices(n)] = variogram.psill + nug  # field variance + measurement error
    k0 = variogram.cov_field(cdist(coords, query))  # (n, q)

    if drift is None:
        # ordinary kriging: one unbiasedness constraint
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = K
        A[:n, n] = 1.0
        A[n, :n] = 1.0
        rhs = np.ones((n + 1, query.shape[0]))
        rhs[:n] = k0
        sol = np.linalg.solve(A, rhs)
        w = sol[:n]
        mu = sol[n]
        pred = w.T @ z
        var = variogram.psill - np.sum(w * k0, axis=0) - mu
    else:
        q = drift.shape[1]
        A = np.zeros((n + q, n + q))
        A[:n, :n] = K
        A[:n, n:] = drift
        A[n:, :n] = drift.T
        rhs = np.zeros((n + q, query.shape[0]))
        rhs[:n] = k0
        rhs[n:] = drift0.T
        sol = np.linalg.solve(A, rhs)
        w = sol[:n]
        lam = sol[n:]
        pred = w.T @ z
        var = variogram.psill - np.sum(w * k0, axis=0) - np.sum(lam * drift0.T, axis=0)
    diag_scale = float(np.max(np.abs(np.atleast_1d(variogram.psill + nug))))
    return pred, _clip_variance(var, diag_scale)


def ordinary_kriging(
    coords: np.ndarray,
    values: np.ndarray,
    variogram: Variogram,
    query: np.ndarray,
    *,
    noise: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Ordinary kriging: BLUP of the field at ``query`` under an unknown constant mean.

    Args:
        coords: ``(n, d)`` data locations.
        values: ``(n,)`` measured field.
        variogram: a fitted :class:`Variogram`.
        query: ``(q, d)`` prediction locations.
        noise: optional ``(n,)`` per-observation measurement variance (heteroscedastic nugget); if None
            the homoscedastic ``variogram.nugget`` is used on the diagonal.

    Returns:
        ``{'prediction', 'variance'}`` arrays of length ``q``.

    Raises:
        ValueError: if ``coords``/``values``/``query`` are empty, mismatched in shape, or contain
            non-finite values, or if ``noise`` is provided and is mismatched in shape, non-finite, or
            negative.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(values, dtype=float).ravel()
    query = np.atleast_2d(np.asarray(query, dtype=float))
    _validate_krige_geometry(coords, z, query)
    pred, var = _krige_solve(coords, z, variogram, query, drift=None, drift0=None, noise=noise)
    return {"prediction": pred, "variance": var}


def _poly_basis(coords: np.ndarray, degree: int) -> np.ndarray:
    n, d = coords.shape
    cols = [np.ones(n)]
    if degree >= 1:
        cols.extend(coords[:, j] for j in range(d))
    if degree >= 2:
        for j in range(d):
            for k in range(j, d):
                cols.append(coords[:, j] * coords[:, k])
    return np.column_stack(cols)


def universal_kriging(
    coords: np.ndarray,
    values: np.ndarray,
    variogram: Variogram,
    query: np.ndarray,
    *,
    degree: int = 1,
    noise: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Universal kriging: kriging with a polynomial spatial trend (drift) of the given ``degree``.

    ``degree=1`` removes a linear trend, ``degree=2`` a quadratic one. Use when the field has a
    large-scale drift on top of the stationary residual the variogram describes.

    Returns:
        ``{'prediction', 'variance'}``.

    Raises:
        ValueError: if ``coords``/``values``/``query`` are empty, mismatched in shape, or contain
            non-finite values, or if ``noise`` is provided and is mismatched in shape, non-finite, or
            negative.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(values, dtype=float).ravel()
    query = np.atleast_2d(np.asarray(query, dtype=float))
    _validate_krige_geometry(coords, z, query)
    drift = _poly_basis(coords, degree)
    drift0 = _poly_basis(query, degree)
    pred, var = _krige_solve(coords, z, variogram, query, drift=drift, drift0=drift0, noise=noise)
    return {"prediction": pred, "variance": var}


def calibrate_variance(predicted_var: np.ndarray, residuals: np.ndarray, *, target: float = 0.9) -> float:
    """Scale factor that makes kriging predictive intervals hit a target coverage.

    Finds ``c`` so that standardised residuals ``residual / sqrt(c * predicted_var)`` achieve the
    ``target`` central coverage under a Gaussian predictive. Returns the variance multiplier ``c``;
    multiply ``predicted_var`` by it to recalibrate (generic GP/kriging variance recalibration).

    Args:
        predicted_var: ``(m,)`` held-out kriging variances (must be strictly positive).
        residuals: ``(m,)`` held-out ``actual - predicted``.
        target: desired central coverage (e.g. 0.9); must be finite and strictly in ``(0, 1)``.

    Returns:
        The variance multiplier ``c`` (> 0).

    Raises:
        ValueError: if ``target`` is not finite and strictly in ``(0, 1)`` -- a target ``<= 0``,
            ``>= 1``, or NaN previously converged to a silent boundary scale factor (``1e-6`` or
            ``1e6``) instead of raising; if ``predicted_var`` and ``residuals`` do not have the same
            nonempty shape (a valid paired held-out sample is required); or if either contains
            non-finite values, or ``predicted_var`` is not strictly positive everywhere.
    """
    from scipy.stats import norm

    if not (np.isfinite(target) and 0.0 < target < 1.0):
        raise ValueError(f"target must be finite and strictly in (0, 1), got {target!r}.")
    pv = np.asarray(predicted_var, dtype=float)
    r = np.asarray(residuals, dtype=float)
    if pv.shape != r.shape:
        raise ValueError(f"predicted_var and residuals must have the same shape, got {pv.shape} and {r.shape}.")
    if pv.size == 0:
        raise ValueError("predicted_var and residuals must be nonempty: calibration needs a paired held-out sample.")
    if not np.all(np.isfinite(pv)):
        raise ValueError("predicted_var must contain only finite values.")
    if not np.all(np.isfinite(r)):
        raise ValueError("residuals must contain only finite values.")
    if not np.all(pv > 0):
        raise ValueError("predicted_var must be strictly positive (a variance of zero or less is not valid).")
    z = norm.ppf(0.5 + target / 2.0)

    def coverage(c: float) -> float:
        sd = np.sqrt(np.clip(c * pv, 1e-300, None))
        return float(np.mean(np.abs(r) <= z * sd))

    lo, hi = 1e-6, 1e6
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        if coverage(mid) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


__all__ = [
    "Variogram",
    "empirical_variogram",
    "fit_variogram",
    "ordinary_kriging",
    "universal_kriging",
    "calibrate_variance",
]
