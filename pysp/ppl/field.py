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

__all__ = [
    "FieldKernel",
    "RandomWalk",
    "RBF",
    "GaussianField",
    "Proxy",
    "GaussianProxy",
    "LogisticNicheProxy",
    "PoissonProxy",
    "CustomProxy",
    "fit_field",
    "FieldPosterior",
    "GP",
    "Gaussian",
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


# --------------------------------------------------------------------------------------------------
# Field priors: a kernel turns an index grid into a prior precision matrix Lambda (field ~ N(0, Lambda^-1)).
# --------------------------------------------------------------------------------------------------
class FieldKernel:
    """A Gaussian field prior over an index grid, expressed as a precision matrix."""

    def precision(self, index: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def covariance(self, index: np.ndarray) -> np.ndarray | None:
        """The prior covariance matrix, when available cheaply (a kernel defined by its covariance). Used by
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
        x = np.asarray(index, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        d2 = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
        k = float(self.amplitude) ** 2 * np.exp(-0.5 * d2 / float(self.lengthscale) ** 2)
        return k + self.jitter * np.eye(len(x))

    def precision(self, index: np.ndarray) -> np.ndarray:
        return np.linalg.inv(self.covariance(index))


@dataclass
class GaussianField:
    """A latent field: an index grid plus a Gaussian (GP/GMRF) prior over its node values."""

    index: np.ndarray
    kernel: FieldKernel
    name: str = "field"

    def __post_init__(self):
        self.index = np.asarray(self.index)
        self.dim = len(self.index)
        self.precision = np.asarray(self.kernel.precision(self.index), dtype=float)
        if self.precision.shape != (self.dim, self.dim):
            raise ValueError(f"kernel precision is {self.precision.shape}, expected {(self.dim, self.dim)}.")
        cov = self.kernel.covariance(self.index)  # prior covariance, when the kernel provides it cheaply
        self.covariance = None if cov is None else np.asarray(cov, dtype=float)


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

    def params(self) -> list[_ParamSpec]:
        return []

    def loglik(self, field_t: Any, params: dict, torch) -> Any:
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
        return [p for p in (self._slope_p, self._int_p, self._scale_p) if p is not None]

    def loglik(self, field_t, params, torch):
        f = field_t if self.idx is None else field_t[torch.as_tensor(self.idx)]
        slope = params[self._slope_p.name] if self._slope_p else self._slope_v
        intercept = params[self._int_p.name] if self._int_p else self._int_v
        scale = params[self._scale_p.name] if self._scale_p else self._scale_v
        y = torch.as_tensor(self.y)
        resid = (y - (intercept + slope * f)) / scale
        log_scale = torch.log(scale) if torch.is_tensor(scale) else float(np.log(scale))
        return -0.5 * torch.sum(resid * resid) - len(self.y) * (log_scale + 0.5 * np.log(2 * np.pi))

    def residual(self, field_t, params, torch):
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
        return [
            _ParamSpec(f"{self.prefix}.mu", (self.S,), "real", np.zeros(self.S)),
            _ParamSpec(f"{self.prefix}.logkappa", (self.S,), "real", np.full(self.S, -1.0)),
            _ParamSpec(f"{self.prefix}.b", (), "real", np.array(0.0)),
        ]

    def loglik(self, field_t, params, torch):
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
        out = []
        for name, support, init in self.param_specs:
            arr = np.asarray(init, dtype=float)
            out.append(_ParamSpec(name, arr.shape, support, arr))
        return out

    def loglik(self, field_t, params, torch):
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
        if node in self._marg_var:  # low-rank (Woodbury) path: only marginal variances were formed
            j = self._jacobian(node)
            return np.sqrt(np.clip(self._marg_var[node], 1e-12, None)) * j
        return np.sqrt(np.clip(np.diag(np.atleast_2d(self.cov(node))), 1e-12, None))

    def posterior(self, node: str, *, coupling: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """``(mean, sd)`` for ``node`` in its natural space. ``coupling=True`` (default) marginalizes the
        other nodes (the honest marginal); ``coupling=False`` fixes them at the MAP (additive information)."""
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
        (``how='laplace'``/``'gauss_newton'``); the fixed nodes come back at their given values.

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
                raise ValueError("conditioning on given= needs the full joint covariance (how='laplace'/'gauss_newton').")
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
        out = {}
        for node in self._layout:
            m = np.atleast_1d(self.map_values[node])
            if m.size == 0:  # the degenerate no-field node in pure-parameter inference
                continue
            sd = self.sd(node)
            out[node] = {"mean": m if m.size > 1 else float(m[0]), "sd": sd if sd.size > 1 else float(sd[0])}
        return out


def _support_transforms(support: str):
    if support == "positive":
        return (lambda u, t: t.exp(u) if hasattr(t, "exp") else np.exp(u), lambda v: np.log(np.clip(v, 1e-12, None)))
    return (lambda u, t: u, lambda v: v)


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

    # ----- assemble the flat parameter layout: the field first, then each proxy's params -----
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

    field_name = field.name
    add(field_name, field.dim, "real", np.zeros(field.dim))
    for px in proxies:
        for spec in px.params():
            v = init[spec.name] if (init and spec.name in init) else spec.init
            add(spec.name, int(np.prod(spec.shape)) if spec.shape else 1, spec.support, v)

    u0 = np.concatenate(init_vals) if init_vals else np.zeros(0)
    supports_arr = supports
    Lambda = torch.as_tensor(field.precision)

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
        f = vals[field_name]
        nlp = 0.0 if field.dim == 0 else 0.5 * f @ (Lambda @ f)  # Gaussian field prior, up to a constant
        for px in proxies:
            nlp = nlp - px.loglik(f, vals, torch)
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
    from pysp.ppl.pde_solve import sparse_used_since

    sparse_used_since(reset=True)
    obj = float(neg_log_post(u).detach())  # one eval to detect whether the forward uses the sparse solve
    if how == "laplace" and sparse_used_since():
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
            f = vals[field_name]
            parts = []
            for px in proxies:
                r = px.residual(f, vals, torch)
                if r is None:
                    raise ValueError(
                        f"how='gauss_newton' needs Gaussian-misfit observations; {type(px).__name__} has no residual()."
                    )
                parts.append(torch.atleast_1d(r))
            return torch.cat(parts)

        u_star = u.detach().clone().requires_grad_(True)
        jac = torch.autograd.functional.jacobian(residual_vec, u_star).detach().numpy()

        # Low-rank (Woodbury) fast path for the field marginals: when the field is the only latent and its
        # prior covariance K is available (e.g. an RBF kernel), the posterior covariance
        # (K^-1 + J^T J)^-1 = K - K J^T (I + J K J^T)^-1 J K is formed without any dense n_field inverse --
        # only an n_obs x n_obs solve -- so the marginal sds scale to large fields. n_resid is the sensor count.
        field_only = field.dim > 0 and pos == field.dim  # layout holds the field and nothing else
        if field_only and field.covariance is not None:
            K = field.covariance
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
        if field.dim > 0:
            lo, hi = layout[field_name]
            H[lo:hi, lo:hi] += field.precision  # the field prior precision (Gauss-Newton adds it to J^T J)
        H = 0.5 * (H + H.T) + 1e-10 * np.eye(H.shape[0])
        cov = np.linalg.inv(H)
        return FieldPosterior(
            map_values, cov, layout, field_name, H, obj, _field_prior=field.precision, _supports=node_support
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
        if field.dim == 0:  # no shared field -> no per-proxy field information to attribute
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
        _field_prior=field.precision,
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
    """A linear-Gaussian observation ``y ~ N(gain*field + offset, sd)``; ``mean`` is an affine in a GP."""
    aff = _as_affine(mean)
    return aff.gp, GaussianProxy(y, slope=aff.gain, intercept=aff.offset, scale=sd, prefix=aff.gp.name + "_gauss")


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

    The one fit verb, matching the rest of ``pysp.ppl``: ``joint([...]).fit(how='map'|'laplace')`` returns
    a :class:`FieldPosterior` with a posterior over any node. Delegates to :func:`fit_field`.
    """

    field: GaussianField | None
    proxies: list

    def fit(self, *, how: str = "laplace", **kw) -> FieldPosterior:
        return fit_field(self.field, self.proxies, how=how, **kw)


def joint(observations: Sequence[tuple]) -> FieldModel:
    """Assemble a latent-field model from equation-style observations; call ``.fit(how=...)`` to fit.

    Each item is the ``(field, proxy)`` pair returned by :func:`Gaussian`, :func:`Niche`, :func:`Cox` or
    :func:`pysp.ppl.Differential`. Observations sharing a field must name the same one (this surface targets
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
