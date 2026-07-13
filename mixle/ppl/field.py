"""Latent fields observed through many proxies, fit jointly, with a posterior readable off any node.

This is the dedicated builder behind the "Cox foundation" extension: a shared latent field (a vector
over an index grid) carrying a Gaussian-process / Gaussian-Markov-random-field prior, observed through an
arbitrary list of heterogeneous *proxy* likelihoods (each its own forward model + noise), all coupled to
the one field and fit in a single joint optimization. The Laplace posterior is then read off any node --
the field itself or any proxy's latent parameters -- because information is additive: the joint posterior
precision is the prior precision plus every proxy's Fisher information, evaluated at the joint MAP.

The motivating instance is the earth-field engine: a latent temperature curve ``T(t)`` inferred jointly
from a benthic delta18O forward model (Gaussian) and foram thermal niches (logistic occupancy), where
combining proxies sharpens the posterior on the shared climate field. A log-Gaussian Cox process (latent
log-intensity field -> Poisson counts) is the same pattern with a Poisson proxy.

Example::

    field = GaussianField(index=np.arange(50), kernel=RandomWalk(scale=0.3, ridge=3.0), name="T")
    post = fit_field(field, [
        GaussianProxy(d18O, index=obs_idx, slope=free, intercept=free, scale=0.15),
        LogisticNicheProxy(presence),                      # presence[taxon, bin]
    ], how="laplace")
    mean, sd = post.posterior("T")     # the field marginal
    post.summary()                     # mean/sd for every node
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import Any

import numpy as np

from mixle.models._kernels import (
    exponential_from_scaled_dist,
    matern32_from_scaled_dist,
    matern52_from_scaled_dist,
    rbf_from_scaled_sqdist,
)

__all__ = [
    "FieldKernel",
    "RandomWalk",
    "RBF",
    "AnisotropicRBF",
    "GreatCircleRBF",
    "GreatCircleMatern",
    "great_circle_distance",
    "GaussianField",
    "FieldSystem",
    "Proxy",
    "GaussianProxy",
    "LogisticNicheProxy",
    "PoissonProxy",
    "CustomProxy",
    "fit_field",
    "FieldPosterior",
    "GP",
    "Gaussian",
    "GaussianObs",
    "Niche",
    "Cox",
    "joint",
    "FieldModel",
    "multistart",
]


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover - torch is a hard dep of the field engine
        raise ImportError("fit_field requires PyTorch (the joint field optimizer is autograd-based).") from e
    return torch


# Plugins (e.g. mixle-pde) register a detector ``fn(reset: bool = False) -> bool`` reporting
# whether a sparse adjoint solve ran during the last forward evaluation. fit_field uses it to refuse
# how='laplace' on a sparse forward (whose dense double-backward Hessian would be silently wrong).
# Empty by default -> mixle has no PDE dependency and the guard is a no-op.
_SPARSE_SOLVE_DETECTORS: list = []


def register_sparse_solve_detector(fn) -> None:
    """Register a sparse-solve detector for the fit_field ``how='laplace'`` safety guard."""
    _SPARSE_SOLVE_DETECTORS.append(fn)


# --------------------------------------------------------------------------------------------------
# Field priors: a kernel turns an index grid into a prior precision matrix Lambda (field ~ N(0, Lambda^-1)).
# --------------------------------------------------------------------------------------------------
class FieldKernel:
    """A Gaussian field prior over an index grid, expressed as a precision matrix."""

    def precision(self, index: np.ndarray) -> np.ndarray:
        """Return the prior precision matrix for the supplied field index grid."""
        raise NotImplementedError

    def covariance(self, index: np.ndarray) -> np.ndarray | None:
        """The prior covariance matrix, when available directly (a kernel defined by its covariance). Used by
        the low-rank Gauss-Newton posterior to get field marginals without a dense precision inverse;
        ``None`` (the default) falls back to the dense path."""
        return None


@dataclass
class RandomWalk(FieldKernel):
    """A Gaussian-Markov-random-field smoothness prior via finite differences.

    ``order=1`` penalizes ``sum (f[i+1]-f[i])^2 / scale^2`` (a random-walk / integrated-noise prior);
    ``order=2`` penalizes the discrete curvature (an integrated-Wiener / thin-plate prior). ``ridge``
    adds ``I / ridge^2`` so the otherwise-improper prior is proper (it anchors the level/trend).
    """

    scale: float = 1.0
    order: int = 1
    ridge: float | None = None

    def precision(self, index: np.ndarray) -> np.ndarray:
        """Build the finite-difference precision matrix for the indexed random walk."""
        n = len(index)
        D = np.eye(n)
        for _ in range(self.order):
            D = np.diff(D, axis=0)
        lam = D.T @ D / float(self.scale) ** 2
        if self.ridge is not None:
            lam = lam + np.eye(n) / float(self.ridge) ** 2
        return lam


@dataclass
class RBF(FieldKernel):
    """A squared-exponential (RBF) GP prior: ``K[i,j] = amplitude^2 exp(-0.5 (d_ij/lengthscale)^2)``.

    The precision is ``inv(K + jitter*I)``. ``index`` may be 1-D (a grid) or 2-D (coordinates, one row
    per node) -- distances are Euclidean, so this is the spatial-field prior as well.
    """

    lengthscale: float = 1.0
    amplitude: float = 1.0
    jitter: float = 1e-6

    def covariance(self, index: np.ndarray) -> np.ndarray:
        """Return the RBF covariance matrix plus diagonal jitter for ``index``."""
        x = np.asarray(index, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        d2 = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
        k = rbf_from_scaled_sqdist(d2 / float(self.lengthscale) ** 2, float(self.amplitude))
        return k + self.jitter * np.eye(len(x))

    def precision(self, index: np.ndarray) -> np.ndarray:
        """Return the inverse of the jittered RBF covariance matrix."""
        return np.linalg.inv(self.covariance(index))


@dataclass
class AnisotropicRBF(FieldKernel):
    """A geometrically-anisotropic squared-exponential GP prior -- correlation that is longer along one
    direction than another (geological layering / bedding, faulted fabric, flow channels).

    The squared distance is the Mahalanobis form ``(x-y)^T M (x-y)`` with ``M`` built from per-axis
    correlation ``ranges`` after rotating the coordinates: in 2-D, ``angle`` (radians, counter-clockwise
    from the x-axis) sets the principal direction and ``ranges=(major, minor)`` the correlation length
    along/across it. For >2-D, supply ``ranges`` per axis (axis-aligned) or a full ``metric`` matrix
    ``M`` directly. Provably positive-definite -- it is an ordinary RBF on linearly-transformed
    coordinates.
    """

    ranges: Sequence[float] = (1.0, 1.0)
    angle: float = 0.0
    amplitude: float = 1.0
    jitter: float = 1e-6
    metric: np.ndarray | None = None

    def _transform(self, x: np.ndarray) -> np.ndarray:
        """Whiten the coordinates so Euclidean distance becomes the anisotropic distance."""
        if self.metric is not None:
            chol = np.linalg.cholesky(np.asarray(self.metric, dtype=float))  # M = L L^T ; u = L^T x
            return x @ chol
        r = np.asarray(self.ranges, dtype=float)
        if x.shape[1] == 2 and self.angle:
            c, s = np.cos(self.angle), np.sin(self.angle)
            rot = np.array([[c, s], [-s, c]])  # rotate into the principal frame
            x = x @ rot.T
        return x / r[: x.shape[1]]

    def covariance(self, index: np.ndarray) -> np.ndarray:
        """Return the anisotropic RBF covariance matrix plus diagonal jitter."""
        x = np.asarray(index, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        u = self._transform(x)
        d2 = np.sum((u[:, None, :] - u[None, :, :]) ** 2, axis=-1)
        k = rbf_from_scaled_sqdist(d2, float(self.amplitude))
        return k + self.jitter * np.eye(len(x))

    def precision(self, index: np.ndarray) -> np.ndarray:
        """Return the inverse of the anisotropic RBF covariance matrix."""
        return np.linalg.inv(self.covariance(index))


def _unit_vectors(latlon_deg: np.ndarray) -> np.ndarray:
    """Map ``(lat, lon)`` rows in degrees to unit vectors on the sphere (n, 3)."""
    a = np.atleast_2d(np.asarray(latlon_deg, dtype=float))
    if a.shape[1] != 2:
        raise ValueError("a spherical index needs two columns: latitude and longitude (degrees).")
    lat, lon = np.radians(a[:, 0]), np.radians(a[:, 1])
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=1)


def great_circle_distance(latlon_a: np.ndarray, latlon_b: np.ndarray | None = None, *, radius: float = 1.0):
    """Great-circle (geodesic) distance between ``(lat, lon)`` points in degrees.

    Returns ``radius * theta`` where ``theta`` is the central angle (radians). With ``radius=1`` the
    result is the angle itself; ``radius=6371.0088`` gives kilometres on Earth. ``latlon_b=None`` returns
    the full pairwise ``(n, n)`` matrix for ``latlon_a``; otherwise the ``(na, nb)`` cross matrix (a float
    when both are single points). Uses the haversine formula for accuracy at small angles.
    """
    a = np.atleast_2d(np.asarray(latlon_a, dtype=float))
    b = a if latlon_b is None else np.atleast_2d(np.asarray(latlon_b, dtype=float))
    lat_a, lon_a = np.radians(a[:, 0])[:, None], np.radians(a[:, 1])[:, None]
    lat_b, lon_b = np.radians(b[:, 0])[None, :], np.radians(b[:, 1])[None, :]
    dlat, dlon = lat_b - lat_a, lon_b - lon_a
    h = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    theta = 2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
    d = float(radius) * theta
    return float(d[0, 0]) if d.shape == (1, 1) else d


def _chordal_sq(index: np.ndarray, radius: float) -> np.ndarray:
    """Pairwise squared chord length between sphere points (physical units of ``radius``).

    ``chord^2 = 2 radius^2 (1 - cos theta)`` -- the Euclidean distance in the R^3 embedding. Kernels of
    the chord are positive-definite on the sphere for every lengthscale (they are ordinary R^3 kernels of
    the embedded points), unlike a kernel of the raw geodesic distance, which is PD only for some scales.
    """
    u = _unit_vectors(index)
    g = np.clip(u @ u.T, -1.0, 1.0)
    d2 = 2.0 * float(radius) ** 2 * (1.0 - g)
    np.fill_diagonal(d2, 0.0)
    return np.maximum(d2, 0.0)


@dataclass
class GreatCircleRBF(FieldKernel):
    """Squared-exponential GP prior on the sphere for a ``(lat, lon)`` index (degrees).

    ``K[i,j] = amplitude^2 exp(-0.5 chord_ij^2 / lengthscale^2)`` where ``chord`` is the straight-line
    distance through the R^3 embedding -- so the kernel is positive-definite on the sphere at every
    lengthscale (a geodesic-distance RBF is not). For nearby points the chord matches the great-circle
    distance, so ``lengthscale`` reads as a spatial correlation length in the same units as ``radius``
    (``radius=6371.0088`` -> kilometres on Earth; default unit sphere -> radians of arc).
    """

    lengthscale: float = 1.0
    amplitude: float = 1.0
    jitter: float = 1e-6
    radius: float = 1.0

    def covariance(self, index: np.ndarray) -> np.ndarray:
        """Return the chordal-distance RBF covariance on spherical coordinates."""
        d2 = _chordal_sq(index, self.radius)
        k = rbf_from_scaled_sqdist(d2 / float(self.lengthscale) ** 2, float(self.amplitude))
        return k + self.jitter * np.eye(len(d2))

    def precision(self, index: np.ndarray) -> np.ndarray:
        """Return the inverse of the spherical RBF covariance matrix."""
        return np.linalg.inv(self.covariance(index))


@dataclass
class GreatCircleMatern(FieldKernel):
    """Matern GP prior on the sphere for a ``(lat, lon)`` index (degrees) -- the geostatistics default.

    A Matern covariance of the R^3 chordal distance (so PD on the sphere at every lengthscale), giving
    rougher, more realistic spatial fields than the infinitely-smooth RBF. ``nu`` in {0.5, 1.5, 2.5}
    (0.5 = exponential / Ornstein-Uhlenbeck, 1.5 and 2.5 are the common once/twice-differentiable
    choices). ``lengthscale`` and ``radius`` carry the same spatial-units meaning as in
    :class:`GreatCircleRBF`.
    """

    lengthscale: float = 1.0
    amplitude: float = 1.0
    nu: float = 1.5
    jitter: float = 1e-6
    radius: float = 1.0

    def covariance(self, index: np.ndarray) -> np.ndarray:
        """Return the chordal-distance Matern covariance on spherical coordinates."""
        r = np.sqrt(_chordal_sq(index, self.radius))
        ls, amp = float(self.lengthscale), float(self.amplitude)
        r_scaled = r / ls  # lengthscale-scaled chordal distance, shared by all three Matern smoothnesses
        if self.nu == 0.5:
            k = exponential_from_scaled_dist(r_scaled, amp)
        elif self.nu == 1.5:
            k = matern32_from_scaled_dist(r_scaled, amp)
        elif self.nu == 2.5:
            k = matern52_from_scaled_dist(r_scaled, amp)
        else:
            raise ValueError("nu must be 0.5, 1.5, or 2.5")
        return k + self.jitter * np.eye(len(r))

    def precision(self, index: np.ndarray) -> np.ndarray:
        """Return the inverse of the spherical Matern covariance matrix."""
        return np.linalg.inv(self.covariance(index))


@dataclass
class GaussianField:
    """A latent field: an index grid plus a Gaussian (GP/GMRF) prior over its node values."""

    index: np.ndarray
    kernel: FieldKernel
    name: str = "field"

    def __post_init__(self):
        """Normalize the index and cache the kernel precision and optional covariance."""
        self.index = np.asarray(self.index)
        self.dim = len(self.index)
        self.precision = np.asarray(self.kernel.precision(self.index), dtype=float)
        if self.precision.shape != (self.dim, self.dim):
            raise ValueError(f"kernel precision is {self.precision.shape}, expected {(self.dim, self.dim)}.")
        cov = self.kernel.covariance(self.index)  # prior covariance, when the kernel provides it directly
        self.covariance = None if cov is None else np.asarray(cov, dtype=float)


@dataclass
class FieldSystem:
    """Several named latent fields fit *jointly* -- the spine for coupled multiphysics / multivariate
    earth-science models (e.g. an ore-grade field constrained by several geophysical surveys, or coupled
    paleo-environment fields like temperature + salinity + pCO2 read through many proxies).

    By default the prior is block-diagonal over the fields (each keeps its own kernel precision) and
    cross-field dependence is expressed in the proxy likelihoods: a proxy ``loglik(field_t, params,
    torch)`` receives the full ``params`` dict, so it can read *any* field by name and couple them (a
    coupling PDE residual, a grade that depends on density + susceptibility, ...). Attach a proxy to a
    particular field with ``proxy.on('name')``; an unattached proxy defaults to the first field.

    Pass ``coregion=B`` (a ``K x K`` between-field covariance for the ``K`` fields) for *prior-level*
    coregionalization -- the intrinsic coregionalization model, joint prior precision ``B^-1 (x) Lambda``,
    where the fields share one spatial structure ``Lambda`` (the first field's kernel) and are correlated
    a priori through ``B`` (e.g. temperature and salinity covary before any data). Requires all fields to
    share one index/dim. ``B`` must be symmetric positive-definite.
    """

    fields: Sequence[GaussianField]
    coregion: np.ndarray | None = None

    def __post_init__(self):
        self.fields = list(self.fields)
        if not self.fields:
            raise ValueError("a FieldSystem needs at least one field.")
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"field names must be unique; got {names}.")
        self.names = names
        if self.coregion is not None:
            b = np.asarray(self.coregion, dtype=float)
            k = len(self.fields)
            if b.shape != (k, k):
                raise ValueError(f"coregion must be {k}x{k} for {k} fields; got {b.shape}.")
            if not np.allclose(b, b.T):
                raise ValueError("coregion must be symmetric.")
            if np.linalg.eigvalsh(b).min() <= 0:
                raise ValueError("coregion must be positive-definite.")
            if any(f.dim != self.fields[0].dim for f in self.fields):
                raise ValueError("coregionalization (coregion=) requires all fields to share one index/dim.")
            self.coregion = b


# --------------------------------------------------------------------------------------------------
# Proxies: each contributes a torch log-likelihood given the field tensor and its own latent params.
# A proxy declares its parameters as (name, shape, support, init); fit_field assembles them into the
# joint parameter vector and hands each proxy a dict {name: tensor} plus the field tensor.
# --------------------------------------------------------------------------------------------------
@dataclass
class _ParamSpec:
    name: str
    shape: tuple
    support: str  # 'real' | 'positive'
    init: np.ndarray


class Proxy:
    """A likelihood hung off the shared field. Subclasses declare params and score the data in torch."""

    prefix: str = "proxy"
    field: str | None = None  # which field in a FieldSystem this proxy observes (None -> the first field)

    def on(self, field_name: str) -> Proxy:
        """Attach this proxy to a named field of a :class:`FieldSystem`; returns ``self`` for chaining."""
        self.field = field_name
        return self

    def params(self) -> list[_ParamSpec]:
        """Return free parameter specifications used by the joint field fit."""
        return []

    def loglik(self, field_t: Any, params: dict, torch) -> Any:
        """Return this proxy's torch log-likelihood contribution."""
        raise NotImplementedError

    def residual(self, field_t: Any, params: dict, torch) -> Any:
        """The standardized Gaussian residual ``(y - prediction) / scale`` (1-D), or ``None`` if this
        proxy is not a Gaussian-misfit observation. Used by the Gauss-Newton posterior (``how='gauss_newton'``)."""
        return None


def _resolve(value, default_init, prefix, name, support):
    """A proxy coefficient is either a fixed float or the ``free`` token -> a scalar param to estimate."""
    from .core import _is_free  # local import to avoid a cycle at module load

    if _is_free(value):
        return None, _ParamSpec(f"{prefix}.{name}", (), support, np.array(float(default_init)))
    return float(value), None


@dataclass
class GaussianProxy(Proxy):
    """A linear-Gaussian forward model: ``y_j ~ N(intercept + slope * field[index_j], scale)``.

    ``slope``, ``intercept`` and ``scale`` are each a fixed float or the ``free`` token (estimated).
    With everything fixed this is the exact linear-Gaussian observation that makes the joint posterior
    exactly Gaussian -- the closed-form check on the Laplace covariance.
    """

    y: np.ndarray
    index: np.ndarray | None = None
    slope: Any = 1.0
    intercept: Any = 0.0
    scale: Any = 1.0
    prefix: str = "gauss"

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=float)
        self.idx = None if self.index is None else np.asarray(self.index, dtype=int)
        self._slope_v, self._slope_p = _resolve(self.slope, 1.0, self.prefix, "slope", "real")
        self._int_v, self._int_p = _resolve(self.intercept, float(np.mean(self.y)), self.prefix, "intercept", "real")
        self._scale_v, self._scale_p = _resolve(
            self.scale, float(np.std(self.y)) or 1.0, self.prefix, "scale", "positive"
        )

    def params(self) -> list[_ParamSpec]:
        """Return parameter specs for any free slope, intercept, or scale."""
        return [p for p in (self._slope_p, self._int_p, self._scale_p) if p is not None]

    def loglik(self, field_t, params, torch):
        """Return the linear-Gaussian observation log-likelihood."""
        f = field_t if self.idx is None else field_t[torch.as_tensor(self.idx)]
        slope = params[self._slope_p.name] if self._slope_p else self._slope_v
        intercept = params[self._int_p.name] if self._int_p else self._int_v
        scale = params[self._scale_p.name] if self._scale_p else self._scale_v
        y = torch.as_tensor(self.y)
        resid = (y - (intercept + slope * f)) / scale
        log_scale = torch.log(scale) if torch.is_tensor(scale) else float(np.log(scale))
        return -0.5 * torch.sum(resid * resid) - len(self.y) * (log_scale + 0.5 * np.log(2 * np.pi))

    def residual(self, field_t, params, torch):
        """Return standardized Gaussian residuals for Gauss-Newton covariance estimation."""
        f = field_t if self.idx is None else field_t[torch.as_tensor(self.idx)]
        slope = params[self._slope_p.name] if self._slope_p else self._slope_v
        intercept = params[self._int_p.name] if self._int_p else self._int_v
        scale = params[self._scale_p.name] if self._scale_p else self._scale_v
        return (torch.as_tensor(self.y) - (intercept + slope * f)) / scale


@dataclass
class LogisticNicheProxy(Proxy):
    """Thermal-niche occupancy: ``presence[i,t] ~ Bernoulli(sigmoid(b - 0.5 kappa_i (field[t] - mu_i)^2))``.

    Each row of ``presence`` is one taxon's presence/absence across the field's index bins; the latent
    niche location ``mu_i`` and precision ``kappa_i`` are co-estimated, and ``b`` is a shared baseline.
    The assemblage acts as a community thermometer -- a unimodal response peaked at the niche optimum.
    """

    presence: np.ndarray
    mu_scale: float = 2.0  # weakly-informative N(0, mu_scale^2) prior on niche locations
    prefix: str = "niche"

    def __post_init__(self):
        self.P = np.asarray(self.presence, dtype=float)
        self.S = self.P.shape[0]

    def params(self) -> list[_ParamSpec]:
        """Return niche-location, niche-precision, and baseline parameter specifications."""
        return [
            _ParamSpec(f"{self.prefix}.mu", (self.S,), "real", np.zeros(self.S)),
            _ParamSpec(f"{self.prefix}.logkappa", (self.S,), "real", np.full(self.S, -1.0)),
            _ParamSpec(f"{self.prefix}.b", (), "real", np.array(0.0)),
        ]

    def loglik(self, field_t, params, torch):
        """Return the Bernoulli niche-response log-likelihood with weak location regularization."""
        mu = params[f"{self.prefix}.mu"]
        kappa = torch.exp(torch.clamp(params[f"{self.prefix}.logkappa"], -4, 4))
        b = params[f"{self.prefix}.b"]
        P = torch.as_tensor(self.P)
        logit = b - 0.5 * kappa[:, None] * (field_t[None, :] - mu[:, None]) ** 2
        ll = torch.sum(P * torch.nn.functional.logsigmoid(logit) + (1 - P) * torch.nn.functional.logsigmoid(-logit))
        ll = ll - 0.5 * torch.sum((mu / self.mu_scale) ** 2)  # weak prior keeps niches identifiable
        return ll


@dataclass
class PoissonProxy(Proxy):
    """A log-Gaussian Cox process: ``counts[j] ~ Poisson(exp(offset_j + field[index_j]))``.

    The field is the latent log-intensity; counts are the point/aggregate observations. This is the
    canonical Cox-process proxy -- the namesake of the foundation.
    """

    counts: np.ndarray
    index: np.ndarray | None = None
    offset: Any = 0.0
    prefix: str = "cox"

    def __post_init__(self):
        self.c = np.asarray(self.counts, dtype=float)
        self.idx = None if self.index is None else np.asarray(self.index, dtype=int)
        self.off = np.asarray(self.offset, dtype=float)

    def loglik(self, field_t, params, torch):
        """Return the Cox-process Poisson log-likelihood up to the count-factorial constant."""
        f = field_t if self.idx is None else field_t[torch.as_tensor(self.idx)]
        log_rate = torch.as_tensor(self.off) + f
        c = torch.as_tensor(self.c)
        return torch.sum(c * log_rate - torch.exp(log_rate))  # Poisson up to the constant -lgamma(c+1)


@dataclass
class CustomProxy(Proxy):
    """An arbitrary proxy: supply a torch log-likelihood ``loglik_fn(field_t, params, torch)`` and the
    parameter specs it reads from ``params`` (each ``(name, shape, support, init)``)."""

    loglik_fn: Callable
    param_specs: Sequence[tuple] = ()  # each (name, support, init): init's shape sets the param shape
    prefix: str = "custom"

    def params(self) -> list[_ParamSpec]:
        """Return the custom parameter specifications declared by ``param_specs``."""
        out = []
        for name, support, init in self.param_specs:
            arr = np.asarray(init, dtype=float)
            out.append(_ParamSpec(name, arr.shape, support, arr))
        return out

    def loglik(self, field_t, params, torch):
        """Delegate log-likelihood evaluation to ``loglik_fn``."""
        return self.loglik_fn(field_t, params, torch)


# --------------------------------------------------------------------------------------------------
# Joint fit: assemble the field prior + every proxy likelihood into one torch log-target, MAP, Laplace.
# --------------------------------------------------------------------------------------------------
@dataclass
class FieldPosterior:
    """The joint posterior. ``posterior(node)`` returns ``(mean, sd)`` for the field or any proxy param.

    The Laplace covariance is the inverse of the joint negative-log-posterior Hessian at the MAP -- the
    prior precision plus every proxy's Fisher information. ``posterior(field, coupling=False)`` instead
    inverts only the field's own Hessian block (the posterior conditional on the other nodes at their
    MAP), which is the per-proxy additive-information picture.
    """

    map_values: dict
    _cov: np.ndarray
    _layout: dict
    _field_name: str
    _hessian: np.ndarray
    objective: float
    _field_prior: np.ndarray = _dc_field(default_factory=lambda: np.zeros((0, 0)))
    _proxy_info: dict = _dc_field(default_factory=dict)  # proxy label -> field Fisher-information block
    _supports: dict = _dc_field(default_factory=dict)  # node -> 'real' | 'positive'
    _marg_var: dict = _dc_field(default_factory=dict)  # node -> marginal variance (low-rank path, no full cov)

    def mean(self, node: str) -> np.ndarray:
        """Return the posterior mean/MAP value for ``node`` in its natural parameter space."""
        return self.map_values[node]

    def _slice(self, node: str) -> slice:
        if node not in self._layout:
            raise KeyError(f"unknown node {node!r}; nodes are {list(self._layout)}.")
        lo, hi = self._layout[node]
        return slice(lo, hi)

    def _jacobian(self, node: str) -> np.ndarray:
        """d(natural value)/d(unconstrained) for the delta method: the value itself for a log link, else 1."""
        if self._supports.get(node, "real") == "positive":
            return np.atleast_1d(np.asarray(self.map_values[node], dtype=float))
        lo, hi = self._layout[node]
        return np.ones(hi - lo)

    def cov(self, node: str) -> np.ndarray:
        """Posterior covariance of ``node`` in its natural space (delta method applied for positive nodes)."""
        if node in self._marg_var:
            raise ValueError(f"node {node!r} has only marginal variances (low-rank fit); use sd()/posterior().")
        s = self._slice(node)
        j = self._jacobian(node)
        return (j[:, None] * self._cov[s, s]) * j[None, :]

    def sd(self, node: str) -> np.ndarray:
        """Return posterior standard deviations for ``node`` in its natural parameter space."""
        if node in self._marg_var:  # low-rank (Woodbury) path: only marginal variances were formed
            j = self._jacobian(node)
            return np.sqrt(np.clip(self._marg_var[node], 1e-12, None)) * j
        return np.sqrt(np.clip(np.diag(np.atleast_2d(self.cov(node))), 1e-12, None))

    def posterior(self, node: str, *, coupling: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """``(mean, sd)`` for ``node`` in its natural space. ``coupling=True`` (default) marginalizes the
        other nodes; ``coupling=False`` fixes them at the MAP for an additive-information diagnostic."""
        m = self.map_values[node]
        if coupling:
            sd = self.sd(node)
        else:
            s = self._slice(node)
            block = self._hessian[s, s]
            j = self._jacobian(node)
            sd_u = np.sqrt(np.clip(np.diag(np.linalg.inv(np.atleast_2d(block))), 1e-12, None))
            sd = j * sd_u
        lo, hi = self._layout[node]
        if hi - lo == 1:  # a scalar node returns scalars, matching its scalar MAP value
            return m, float(sd[0])
        return m, sd

    def field_posterior(self, include: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """The field posterior ``(mean, sd)`` under a *subset* of proxies, evaluated at the one joint MAP.

        Because information is additive, the field posterior precision under any subset of proxies is the
        prior precision plus those proxies' Fisher-information blocks. ``include=None`` uses every proxy
        (the joint posterior); ``include=["gauss"]`` uses only that proxy -- so you can read how much each
        proxy sharpens the shared field without re-fitting (which would be ill-posed for a single proxy
        with a free forward-model gain). The mean is the joint MAP for every subset.
        """
        labels = list(self._proxy_info) if include is None else list(include)
        prec = np.array(self._field_prior, dtype=float)
        for lab in labels:
            if lab not in self._proxy_info:
                raise KeyError(f"unknown proxy {lab!r}; proxies are {list(self._proxy_info)}.")
            prec = prec + self._proxy_info[lab]
        sd = np.sqrt(np.clip(np.diag(np.linalg.inv(prec)), 1e-12, None))
        return self.map_values[self._field_name], sd

    def sample(self, size: int = 1, rng=None, *, nodes: Sequence[str] | None = None, given: dict | None = None) -> dict:
        """Draw joint samples from the Gaussian posterior, returned in each node's natural space.

        The Laplace / Gauss-Newton posterior is the Gaussian ``N(map, _cov)`` in the *unconstrained*
        parameter space; this draws via a Cholesky factor of that joint covariance (so cross-node and
        within-field correlations are preserved) and maps each draw back through the node's support
        transform (``exp`` for positive nodes). That back-transform is *exact* -- unlike :meth:`cov`,
        which linearizes it with the delta method -- so a positive node's draws are properly lognormal.
        A mean-field ``how='vi'`` (or low-rank Woodbury) fit exposes only per-node marginal variances, so
        there the draws are independent across nodes (which is exactly the mean-field assumption).

        ``given`` (a ``{node: value}`` dict) conditions the draw on fixed node values -- the closed-form
        Gaussian conditional of the remaining coordinates given the observed ones (e.g. draw the field
        consistent with a pinned measurement or a fixed proxy parameter). Conditioning couples nodes
        through the joint covariance, so it requires the full-covariance fit
        (``how='laplace'``/``'gauss_newton'``); the fixed nodes retain their given values.

        Args:
            size: number of joint draws.
            rng: a ``numpy.random.RandomState``, an integer seed, or ``None``.
            nodes: which nodes to return (default: all). The draw is always joint over the full vector.
            given: optional ``{node: value}`` to condition on (values in each node's natural space).

        Returns:
            ``{node: ndarray}`` with shape ``(size,)`` for scalar nodes and ``(size, dim)`` otherwise.
        """
        if rng is None or isinstance(rng, (int, np.integer)):
            rng = np.random.RandomState(None if rng is None else int(rng))
        has_full = self._cov is not None and np.asarray(self._cov).size > 0
        if not has_full and not self._marg_var:
            raise ValueError(
                "this posterior carries no covariance (how='map'); refit with how='laplace', "
                "'gauss_newton', or 'vi' to sample."
            )
        dim = max((hi for _, hi in self._layout.values()), default=0)

        def _to_unconstrained(node, value):
            v = np.atleast_1d(np.asarray(value, dtype=float))
            return np.log(np.clip(v, 1e-12, None)) if self._supports.get(node) == "positive" else v

        mu = np.zeros(dim)
        for node, (lo, hi) in self._layout.items():
            val = np.atleast_1d(np.asarray(self.map_values[node], dtype=float))
            if val.size == 0:
                continue
            mu[lo:hi] = _to_unconstrained(node, val)
        z = rng.standard_normal((size, dim))
        if given:
            if not has_full:
                raise ValueError(
                    "conditioning on given= needs the full joint covariance (how='laplace'/'gauss_newton')."
                )
            obs_pos: list[int] = []
            x_o: list[float] = []
            for node, value in given.items():
                lo, hi = self._slice(node).start, self._slice(node).stop
                uval = _to_unconstrained(node, value)
                if uval.size != hi - lo:
                    raise ValueError(f"given[{node!r}] has {uval.size} value(s), expected {hi - lo}.")
                obs_pos.extend(range(lo, hi))
                x_o.extend(uval.tolist())
            obs = np.array(obs_pos, dtype=int)
            unobs = np.array([i for i in range(dim) if i not in set(obs_pos)], dtype=int)
            if unobs.size == 0:
                raise ValueError("given= fixes every coordinate; leave at least one node free.")
            cov = np.atleast_2d(np.asarray(self._cov, dtype=float))
            s_oo, s_uo, s_uu = cov[np.ix_(obs, obs)], cov[np.ix_(unobs, obs)], cov[np.ix_(unobs, unobs)]
            solve = np.linalg.solve(s_oo, np.concatenate([(np.array(x_o) - mu[obs])[:, None], s_uo.T], axis=1))
            mu_u = mu[unobs] + s_uo @ solve[:, 0]
            cov_u = s_uu - s_uo @ solve[:, 1:]
            chol = np.linalg.cholesky(0.5 * (cov_u + cov_u.T) + 1e-12 * np.eye(unobs.size))
            draws = np.empty((size, dim))
            draws[:, obs] = np.array(x_o)[None, :]
            draws[:, unobs] = mu_u[None, :] + z[:, : unobs.size] @ chol.T
        elif has_full:
            cov = np.atleast_2d(np.asarray(self._cov, dtype=float))
            chol = np.linalg.cholesky(cov + 1e-12 * np.eye(dim))
            draws = mu[None, :] + z @ chol.T
        else:  # mean-field: independent per-node marginal draws
            sdv = np.zeros(dim)
            for node, (lo, hi) in self._layout.items():
                if node in self._marg_var:
                    sdv[lo:hi] = np.sqrt(np.clip(np.atleast_1d(self._marg_var[node]), 1e-12, None))
            draws = mu[None, :] + z * sdv[None, :]
        out = {}
        for node in self._layout if nodes is None else nodes:
            lo, hi = self._slice(node).start, self._slice(node).stop
            u = draws[:, lo:hi]
            v = np.exp(u) if self._supports.get(node) == "positive" else u
            out[node] = v[:, 0] if hi - lo == 1 else v
        return out

    def summary(self) -> dict:
        """Return posterior means and standard deviations for every non-empty node."""
        out = {}
        for node in self._layout:
            m = np.atleast_1d(self.map_values[node])
            if m.size == 0:  # the degenerate no-field node in pure-parameter inference
                continue
            sd = self.sd(node)
            out[node] = {"mean": m if m.size > 1 else float(m[0]), "sd": sd if sd.size > 1 else float(sd[0])}
        return out


# Register FieldPosterior with the core sampler so ``mixle.stats.sample(field_post, n)`` dispatches here
# without the core ``sampling_api`` importing this (ppl) module -- the dependency stays ppl -> core.
def _register_field_posterior_sampling() -> None:
    from mixle.stats.compute.sampling_api import SAMPLE_UNHANDLED, _resolve_rng, register_sample_dispatch

    @register_sample_dispatch
    def _sample_field_posterior(model, size, *, seed, rng, **kwargs):
        if isinstance(model, FieldPosterior):
            return model.sample(1 if size is None else size, rng=_resolve_rng(seed, rng), **kwargs)
        return SAMPLE_UNHANDLED


_register_field_posterior_sampling()


class _NoField:
    """A degenerate field (no shared latent) so fit_field also serves pure-parameter inference."""

    name = "_field"
    dim = 0
    precision = np.zeros((0, 0))


def fit_field(
    field: GaussianField | None,
    proxies: Sequence[Proxy],
    *,
    how: str = "laplace",
    max_iter: int = 500,
    lr: float = 0.4,
    init: dict | None = None,
    vi_steps: int = 400,
    vi_lr: float = 0.05,
    vi_samples: int = 4,
) -> FieldPosterior:
    """Fit a latent field jointly to a list of proxy likelihoods.

    ``field=None`` runs pure-parameter inference (the proxies carry all the latents, e.g. ODE/PDE
    coefficients) with no shared field. ``how='map'`` returns the joint MAP (no covariance);
    ``how='laplace'`` adds the Gaussian posterior (the inverse-Hessian covariance, exact when every
    factor is Gaussian; needs a twice-differentiable forward); ``how='gauss_newton'`` builds the posterior
    from ``J^T J + prior`` with ``J`` the Jacobian of the standardized residual -- first-order only, so it
    is the posterior for the sparse adjoint solve (where ``how='laplace'`` cannot run) and is exact for a
    linear forward; ``how='vi'`` fits a mean-field Gaussian variational posterior (reparameterized ADVI on
    the unconstrained vector), the calibrated approximation for genuinely non-Gaussian posteriors (e.g. with
    total-variation / Potts priors), and it too is first-order so it works through the sparse solve.
    Posteriors over any node are read off the returned :class:`FieldPosterior`.
    """
    if how not in ("map", "laplace", "gauss_newton", "vi"):
        raise ValueError("how must be 'map', 'laplace', 'gauss_newton', or 'vi'.")
    torch = _torch()
    if field is None:
        field = _NoField()

    # ----- normalize to a list of fields (one shared field, or a FieldSystem of several) -----
    field_list = list(field.fields) if isinstance(field, FieldSystem) else [field]
    field_name = field_list[0].name  # the primary field (single-field models reference this throughout)
    field_names = [fld.name for fld in field_list]
    primary = field_list[0]

    def _proxy_field(px):
        return getattr(px, "field", None) or field_name

    # ----- assemble the flat parameter layout: the field(s) first, then each proxy's params -----
    layout: dict[str, tuple[int, int]] = {}
    supports: list[str] = []
    init_vals: list[np.ndarray] = []
    pos = 0

    node_support: dict[str, str] = {}

    def add(name, size, support, value):
        nonlocal pos
        layout[name] = (pos, pos + size)
        supports.extend([support] * size)
        node_support[name] = support
        init_vals.append(np.atleast_1d(np.asarray(value, dtype=float)).ravel())
        pos += size

    for fld in field_list:
        add(fld.name, fld.dim, "real", np.zeros(fld.dim))
    for px in proxies:
        for spec in px.params():
            v = init[spec.name] if (init and spec.name in init) else spec.init
            add(spec.name, int(np.prod(spec.shape)) if spec.shape else 1, spec.support, v)

    u0 = np.concatenate(init_vals) if init_vals else np.zeros(0)
    supports_arr = supports
    field_dim = {fld.name: fld.dim for fld in field_list}

    # ----- joint field prior precision over the contiguous field block (fields are added first) -----
    n_field = sum(fld.dim for fld in field_list)
    coregion = field.coregion if isinstance(field, FieldSystem) else None
    if coregion is not None:
        # intrinsic coregionalization: Lambda_joint = B^-1 (x) Lambda_shared (the first field's precision)
        field_prec_joint = np.kron(np.linalg.inv(coregion), field_list[0].precision)
    else:
        blocks = [fld.precision for fld in field_list if fld.dim]
        field_prec_joint = np.zeros((0, 0))
        if blocks:
            field_prec_joint = np.zeros((n_field, n_field))
            o = 0
            for m in blocks:  # block-diagonal: independent fields
                d = m.shape[0]
                field_prec_joint[o : o + d, o : o + d] = m
                o += d
    Lambda_joint = torch.as_tensor(field_prec_joint)

    def unpack(u_t):
        """Map the unconstrained vector to {node: constrained tensor}."""
        out = {}
        for name, (lo, hi) in layout.items():
            seg = u_t[lo:hi]
            if hi > lo and supports_arr[lo] == "positive":
                seg = torch.exp(seg)
            out[name] = seg if (hi - lo) != 1 else seg[0]  # vector/empty -> array, scalar -> 0-d
        return out

    def neg_log_post(u_t):
        vals = unpack(u_t)
        # Gaussian field prior over the contiguous, real-support field block (up to a constant). The joint
        # precision is block-diagonal (independent fields) or B^-1 (x) Lambda (coregionalized).
        ff = u_t[:n_field]
        nlp = 0.5 * ff @ (Lambda_joint @ ff) if n_field else 0.0
        for px in proxies:
            nlp = nlp - px.loglik(vals[_proxy_field(px)], vals, torch)
        return nlp

    # ----- joint MAP via L-BFGS (strong-Wolfe), as in the earth-field engine -----
    u = torch.tensor(u0, dtype=torch.double, requires_grad=True)
    opt = torch.optim.LBFGS([u], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = neg_log_post(u)
        loss.backward()
        return loss

    opt.step(closure)
    # The sparse-forward instrumentation lives with the PDE solver plugin (mixle-pde), which
    # registers a detector. Without the plugin there are no sparse forwards, so the laplace guard is a
    # no-op. Reset every detector, run one forward, then check whether any sparse solve fired.
    for _detect in _SPARSE_SOLVE_DETECTORS:
        _detect(reset=True)
    obj = float(neg_log_post(u).detach())  # one eval to detect whether the forward uses the sparse solve
    if how == "laplace" and any(_detect() for _detect in _SPARSE_SOLVE_DETECTORS):
        raise ValueError(
            "how='laplace' builds a dense Hessian by double-backward, which the adjoint sparse solve does "
            "not support (the Hessian would be silently wrong). Use how='gauss_newton' for sparse forwards."
        )

    # ----- read constrained MAP values per node -----
    u_np = u.detach().numpy()
    map_values: dict[str, np.ndarray] = {}
    for name, (lo, hi) in layout.items():
        seg = u_np[lo:hi]
        if hi > lo and supports_arr[lo] == "positive":
            seg = np.exp(seg)
        map_values[name] = seg if (hi - lo) != 1 else float(seg[0])

    if how == "map":
        return FieldPosterior(
            map_values, np.zeros((0, 0)), layout, field_name, np.zeros((0, 0)), obj, _supports=node_support
        )

    if how == "gauss_newton":
        # Gauss-Newton posterior: H = J^T J + prior precision, J the Jacobian of the standardized residual.
        # Uses only first-order gradients (one adjoint solve per residual), so it works with the sparse
        # adjoint solve where the dense Hessian (how='laplace') cannot, and is exact for a linear forward.
        def residual_vec(u_t):
            vals = unpack(u_t)
            parts = []
            for px in proxies:
                r = px.residual(vals[_proxy_field(px)], vals, torch)
                if r is None:
                    raise ValueError(
                        f"how='gauss_newton' needs Gaussian-misfit observations; {type(px).__name__} has no residual()."
                    )
                parts.append(torch.atleast_1d(r))
            return torch.cat(parts)

        u_star = u.detach().clone().requires_grad_(True)
        jac = torch.autograd.functional.jacobian(residual_vec, u_star).detach().numpy()

        # Low-rank (Woodbury) fast path for the field marginals: when a single field is the only latent and
        # its prior covariance K is available (e.g. an RBF kernel), the posterior covariance
        # (K^-1 + J^T J)^-1 = K - K J^T (I + J K J^T)^-1 J K is formed without any dense n_field inverse --
        # only an n_obs x n_obs solve -- so the marginal sds scale to large fields. n_resid is the sensor count.
        field_only = len(field_list) == 1 and primary.dim > 0 and pos == primary.dim and coregion is None
        if field_only and primary.covariance is not None:
            K = primary.covariance
            M = K @ jac.T  # n_field x n_resid  (= K J^T)
            C = np.linalg.inv(np.eye(jac.shape[0]) + jac @ M)  # n_resid x n_resid
            marg_var = np.diag(K) - np.einsum("ij,jk,ik->i", M, C, M)
            return FieldPosterior(
                map_values,
                np.zeros((0, 0)),
                layout,
                field_name,
                np.zeros((0, 0)),
                obj,
                _supports=node_support,
                _marg_var={field_name: marg_var},
            )

        H = jac.T @ jac
        if n_field:  # add the joint field prior precision (block-diagonal or coregionalized) to J^T J
            H[:n_field, :n_field] += field_prec_joint
        H = 0.5 * (H + H.T) + 1e-10 * np.eye(H.shape[0])
        cov = np.linalg.inv(H)
        return FieldPosterior(
            map_values, cov, layout, field_name, H, obj, _field_prior=primary.precision, _supports=node_support
        )

    if how == "vi":
        # Mean-field Gaussian variational posterior over the unconstrained vector, by reparameterized ADVI
        # (initialized at the MAP). First-order only, so it works through the sparse solve, and -- unlike
        # Laplace -- it is a genuine variational fit for non-Gaussian posteriors (e.g. TV / Potts priors).
        n_u = u0.shape[0]
        mu = u.detach().clone().requires_grad_(True)
        log_sigma = torch.full((n_u,), -2.0, dtype=torch.double, requires_grad=True)
        opt2 = torch.optim.Adam([mu, log_sigma], lr=vi_lr)
        for _ in range(vi_steps):
            opt2.zero_grad()
            sigma = torch.exp(log_sigma)
            neg = 0.0
            for _ in range(vi_samples):
                neg = neg + neg_log_post(mu + sigma * torch.randn(n_u, dtype=torch.double))
            loss = neg / vi_samples - log_sigma.sum()  # -ELBO = E_q[neg_log_post] - entropy(q)
            loss.backward()
            opt2.step()
        mu_np = mu.detach().numpy()
        var_np = np.exp(2.0 * log_sigma.detach().numpy())
        vi_values: dict[str, np.ndarray] = {}
        vi_marg: dict[str, np.ndarray] = {}
        for name, (lo, hi) in layout.items():
            seg = mu_np[lo:hi]
            if hi > lo and supports_arr[lo] == "positive":
                seg = np.exp(seg)
            vi_values[name] = seg if (hi - lo) != 1 else float(seg[0])
            if hi > lo:
                vi_marg[name] = var_np[lo:hi]
        return FieldPosterior(
            vi_values,
            np.zeros((0, 0)),
            layout,
            field_name,
            np.zeros((0, 0)),
            float(neg_log_post(mu).detach()),
            _supports=node_support,
            _marg_var=vi_marg,
        )

    # ----- per-proxy field Fisher information at the MAP (information is additive across proxies) -----
    map_t = {
        name: torch.as_tensor(np.atleast_1d(np.asarray(val, dtype=float)).ravel() if np.ndim(val) else float(val))
        for name, val in map_values.items()
    }
    f_star = torch.as_tensor(np.atleast_1d(map_values[field_name]).astype(float))
    proxy_info: dict[str, np.ndarray] = {}
    labels_seen: dict[str, int] = {}

    def _proxy_label(px):
        base = getattr(px, "prefix", "proxy")
        labels_seen[base] = labels_seen.get(base, 0) + 1
        return base if labels_seen[base] == 1 else f"{base}{labels_seen[base]}"

    for px in proxies:
        label = _proxy_label(px)
        # Attribute information only for proxies on the primary field (the field_posterior(include=)
        # ablation decomposes that field's precision). Proxies on other fields don't sharpen it.
        if field_dim.get(field_name, 0) == 0 or _proxy_field(px) != field_name:
            continue

        def negll(fv, _px=px):
            vals = dict(map_t)
            vals[field_name] = fv
            return -_px.loglik(fv, vals, torch)

        try:
            Hk = torch.autograd.functional.hessian(negll, f_star).detach().numpy()
        except RuntimeError as e:
            raise ValueError(
                "how='laplace' needs a twice-differentiable forward model; the adjoint sparse solve does "
                "not qualify. Use how='map' for sparse forward models."
            ) from e
        proxy_info[label] = 0.5 * (Hk + Hk.T)

    # ----- joint Laplace covariance: invert the full negative-log-posterior Hessian at the MAP -----
    u_star = u.detach().clone().requires_grad_(True)
    try:
        H = torch.autograd.functional.hessian(neg_log_post, u_star).detach().numpy()
    except RuntimeError as e:  # e.g. a forward using the adjoint sparse solve is not twice-differentiable
        raise ValueError(
            "how='laplace' needs a twice-differentiable forward model (dense solves and ODE integration "
            "qualify); the adjoint sparse solve does not. Use how='map' for sparse forward models."
        ) from e
    H = 0.5 * (H + H.T)
    cov = np.linalg.inv(H + 1e-10 * np.eye(H.shape[0]))
    return FieldPosterior(
        map_values,
        cov,
        layout,
        field_name,
        H,
        obj,
        _field_prior=primary.precision,
        _proxy_info=proxy_info,
        _supports=node_support,
    )


# --------------------------------------------------------------------------------------------------
# PPL-native surface: write the forward model as an equation in a GP node and let joint() discover the
# field and lower each likelihood to a proxy. A thin, ergonomic layer over fit_field for the common
# shapes (affine-Gaussian, logistic niche, log-Gaussian Cox); the dedicated builder remains the escape
# hatch for arbitrary forward models.
# --------------------------------------------------------------------------------------------------
class GP:
    """A latent field node for the equation-style surface: ``T = GP("T", index=grid, kernel=...)``.

    Supports affine algebra (``c0 - c1*T``, ``T + b``) so a linear forward model reads as math; the
    result carries the field, the gain and the offset to :func:`joint`.
    """

    def __init__(self, name: str, index: np.ndarray, kernel: FieldKernel):
        self.field = GaussianField(index, kernel, name)
        self.name = name

    def _affine(self) -> _GPAffine:
        return _GPAffine(self, 1.0, 0.0)

    def __mul__(self, a):
        return self._affine() * a

    __rmul__ = __mul__

    def __add__(self, b):
        return self._affine() + b

    __radd__ = __add__

    def __sub__(self, b):
        return self._affine() - b

    def __rsub__(self, b):
        return b - self._affine()

    def __neg__(self):
        return -self._affine()


@dataclass
class _GPAffine:
    """``gain * field + offset`` for one GP field -- the linear forward model carried into joint()."""

    gp: GP
    gain: float
    offset: float

    def __mul__(self, a):
        return _GPAffine(self.gp, self.gain * float(a), self.offset * float(a))

    __rmul__ = __mul__

    def __add__(self, b):
        return _GPAffine(self.gp, self.gain, self.offset + float(b))

    __radd__ = __add__

    def __sub__(self, b):
        return _GPAffine(self.gp, self.gain, self.offset - float(b))

    def __rsub__(self, b):
        return _GPAffine(self.gp, -self.gain, float(b) - self.offset)

    def __neg__(self):
        return _GPAffine(self.gp, -self.gain, -self.offset)


def _as_affine(x) -> _GPAffine:
    if isinstance(x, _GPAffine):
        return x
    if isinstance(x, GP):
        return x._affine()
    raise TypeError(f"expected a GP field (or an affine in one), got {type(x).__name__}.")


def Gaussian(y, *, mean, sd) -> tuple:
    """A linear-Gaussian observation ``y ~ N(gain*field + offset, sd)``; ``mean`` is an affine in a GP.

    This is an *observation helper* for :func:`joint` -- it returns a ``(field, proxy)`` pair, not a
    distribution. For the Gaussian distribution as a random variable (the ``stats`` ``Gaussian``
    analogue), use :func:`mixle.ppl.Normal`. Also exported as :data:`GaussianObs` to keep the
    distinction visible at the call site.
    """
    aff = _as_affine(mean)
    return aff.gp, GaussianProxy(y, slope=aff.gain, intercept=aff.offset, scale=sd, prefix=aff.gp.name + "_gauss")


#: Alias of :func:`Gaussian`, named for what it is -- a linear-Gaussian *observation* term for
#: :func:`joint`, not the Gaussian distribution (that is :func:`mixle.ppl.Normal`).
GaussianObs = Gaussian


def Niche(presence, *, over: GP, mu_scale: float = 2.0) -> tuple:
    """Logistic thermal-niche occupancy of ``presence`` over the field ``over`` (a community thermometer)."""
    if not isinstance(over, GP):
        raise TypeError("Niche(over=...) takes a GP field.")
    return over, LogisticNicheProxy(presence, mu_scale=mu_scale, prefix=over.name + "_niche")


def Cox(counts, *, log_intensity, offset=0.0) -> tuple:
    """A log-Gaussian Cox process ``counts ~ Poisson(exp(offset + field))``; ``log_intensity`` is a GP."""
    aff = _as_affine(log_intensity)
    if aff.gain != 1.0 or aff.offset != 0.0:
        # fold an affine log-intensity into the per-observation offset; the field stays the latent log-rate
        raise ValueError("Cox(log_intensity=...) takes the field directly; put any gain/offset in `offset`.")
    return aff.gp, PoissonProxy(counts, offset=offset, prefix=aff.gp.name + "_cox")


@dataclass
class FieldModel:
    """A latent-field model built from equation-style observations; fit with ``.fit(how=...)``.

    The one fit verb, matching the rest of ``mixle.ppl``: ``joint([...]).fit(how='map'|'laplace')`` returns
    a :class:`FieldPosterior` with a posterior over any node. Delegates to :func:`fit_field`.
    """

    field: GaussianField | None
    proxies: list

    def fit(self, *, how: str = "laplace", **kw) -> FieldPosterior:
        """Fit the assembled field/proxy model with :func:`fit_field` and return its posterior."""
        return fit_field(self.field, self.proxies, how=how, **kw)


def joint(observations: Sequence[tuple]) -> FieldModel:
    """Assemble a latent-field model from equation-style observations; call ``.fit(how=...)`` to fit.

    Each item is the ``(field, proxy)`` pair returned by :func:`Gaussian`, :func:`Niche`, :func:`Cox` or
    :func:`mixle.ppl.Differential`. Observations sharing a field must name the same one (this surface targets
    one shared field); field-free observations (a pure-parameter ODE inverse problem) are allowed.
    """
    if not observations:
        raise ValueError("joint() needs at least one observation.")
    fields = {}
    for f, _ in observations:
        if f is None:
            continue
        gf = f.field if hasattr(f, "field") else f  # accept a GP (has .field) or a GaussianField directly
        fields[gf.name] = gf
    if len(fields) > 1:
        raise ValueError(f"joint() targets one shared field; saw {sorted(fields)}. Use fit_field for several.")
    field = next(iter(fields.values())) if fields else None
    return FieldModel(field, [proxy for _, proxy in observations])


def multistart(model: FieldModel, inits: Sequence[dict], *, how: str = "map", **kw) -> FieldPosterior:
    """Fit ``model`` from several initializations and keep the best (lowest-objective) fit.

    For the multimodal posteriors of nonlinear inverse problems (e.g. scattering / FWI cycle-skipping), a
    single optimization can land in a poor local mode; ``inits`` is a list of ``{node: value}`` dicts and
    the fit with the smallest ``objective`` is returned. (Frequency continuation -- fit a coarse/low-
    frequency model, then seed a finer one via ``fit(init=...)`` -- is the complementary strategy.)
    """
    best = None
    for init in inits:
        post = model.fit(how=how, init=init, **kw)
        if best is None or post.objective < best.objective:
            best = post
    if best is None:
        raise ValueError("multistart needs at least one initialization.")
    return best
