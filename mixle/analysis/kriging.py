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
        raise ValueError(
            "model must be 'spherical', 'exponential', 'gaussian' (aka 'squared_exponential' / 'rbf'), or 'matern'."
        )
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
    """

    model: str
    nugget: float
    psill: float
    rng: float
    nu: float = 1.5
    anisotropy: tuple[float, float] | None = None

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
) -> dict[str, np.ndarray]:
    """Binned empirical (semi-)variogram: mean ``0.5 (z_i - z_j)^2`` by separation distance.

    Returns:
        ``{'lag', 'semivariance', 'count'}`` for each non-empty distance bin.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(values, dtype=float).ravel()
    d = cdist(coords, coords)
    iu = np.triu_indices_from(d, k=1)
    dist = d[iu]
    sv = 0.5 * (z[iu[0]] - z[iu[1]]) ** 2
    if max_dist is None:
        max_dist = dist.max() / 2.0
    edges = np.linspace(0, max_dist, n_bins + 1)
    idx = np.digitize(dist, edges) - 1
    lag, semi, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if np.any(m):
            lag.append(0.5 * (edges[b] + edges[b + 1]))
            semi.append(float(sv[m].mean()))
            cnt.append(int(m.sum()))
    return {"lag": np.asarray(lag), "semivariance": np.asarray(semi), "count": np.asarray(cnt, dtype=int)}


def fit_variogram(
    coords: np.ndarray, values: np.ndarray, *, model: str = "spherical", n_bins: int = 15, nu: float = 1.5
) -> Variogram:
    """Fit a variogram model to data by least squares on the empirical variogram.

    Returns:
        A fitted :class:`Variogram` (nugget, partial sill, range).
    """
    ev = empirical_variogram(coords, values, n_bins=n_bins)
    lag, semi, cnt = ev["lag"], ev["semivariance"], ev["count"]
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
    coords = _transform(np.atleast_2d(coords), variogram.anisotropy)
    query = _transform(np.atleast_2d(query), variogram.anisotropy)
    n = coords.shape[0]
    dd = cdist(coords, coords)
    K = variogram.cov_field(dd)
    nug = variogram.nugget if noise is None else np.asarray(noise, dtype=float)
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
    return pred, np.clip(var, 0.0, None)


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
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(values, dtype=float).ravel()
    query = np.atleast_2d(np.asarray(query, dtype=float))
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
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(values, dtype=float).ravel()
    query = np.atleast_2d(np.asarray(query, dtype=float))
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
        predicted_var: ``(m,)`` held-out kriging variances.
        residuals: ``(m,)`` held-out ``actual - predicted``.
        target: desired central coverage (e.g. 0.9).

    Returns:
        The variance multiplier ``c`` (> 0).
    """
    from scipy.stats import norm

    pv = np.asarray(predicted_var, dtype=float)
    r = np.asarray(residuals, dtype=float)
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
