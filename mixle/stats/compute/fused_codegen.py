"""Source-generated *fused* numba kernels for composite / mixture trees.

The generated-numba kernels in :mod:`declarations` lower one leaf at a time: each leaf materializes its
sufficient statistic in numpy and runs its own row loop, so a composite or mixture pays a Python<->C
boundary crossing and an intermediate allocation *per factor and per component*. For a deep model of
low-cost leaves that overhead dominates -- numpy itself is multi-pass for the same reason.

This module instead emits a SINGLE ``@njit`` function for the whole model structure: one pass over the
rows with the composite sum and the mixture log-sum-exp in nopython registers, no per-factor allocation.

Two leaf flavours, each using the right primitive (numba supports BOTH a SIMD-vectorized scalar loop and
BLAS via ``@``/``np.dot`` in nopython):

* **scalar leaves** (Gaussian, Exponential, Poisson, Gamma, Geometric, Bernoulli): univariate, O(1) per
  element. Their scalar log-density is inlined into the row loop -- numba SIMD-vectorizes it, and fusing
  every factor into one pass with no intermediate arrays beats numpy's multi-pass evaluation. A leaf may
  have ``arity > 1`` data arrays (e.g. Poisson carries ``(x, lgamma(x+1))``, Gamma ``(x, log x)``).
* **matrix leaves** (MultivariateGaussian, ...): the quadratic form ``(x-mu)' P (x-mu)`` and the E-step's
  weighted Gram ``X' (r .* X)`` are matmuls -- emitted as numba ``@`` (BLAS). Fusing the score, the
  responsibility soft-max and the BLAS accumulation into one njit beats numpy's per-component dispatch
  (~5x on a GMM E-step). A scalar triple-loop here would be ~15x SLOWER -- never hand-roll a matmul.

When a model mixes the two, the matrix quad-forms are precomputed via BLAS before the row loop, the row
loop scores + forms responsibilities (storing them only when a matrix leaf needs them downstream), and the
matrix sufficient statistics are accumulated via BLAS after it. Pure-scalar models never materialize the
responsibility matrix, so their fast path is byte-for-byte the original one.

A discrete leaf whose value-keyed category counts are the sufficient statistic (Categorical) fuses too --
the **categorical** kind scores from a precomputed (K, C) log-prob table and accumulates a (K, C) weighted
count histogram. The general bar for fusion is a *map-reducible* sufficient statistic (a sum/count, a
min/max, or a histogram -- not necessarily additive): that is what lets Binomial (n from max x) and
NegativeBinomial (iterative dispersion over the count histogram) fuse despite non-additive/iterative MLEs.

Generated kernels are compiled once and disk-cached (see :func:`_njit`), so the compile cost is paid once
per structure *ever*, not per process. A leaf with no template (e.g. Laplace, whose weighted-median MLE
keeps the raw observations) -> :func:`fusible` is False -> numpy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

# --- leaf templates -------------------------------------------------------------------------------


@dataclass(frozen=True)
class LeafTemplate:
    """How to fuse one leaf family.

    ``kind="scalar"`` leaves use ``expr`` / ``acc_stmt`` over ``arity`` 1-D data arrays (the row values
    are passed as a list ``vals`` of numba expressions). ``kind="matrix"`` leaves use the ``mat_*`` hooks
    (a single 2-D data array + BLAS precompute/accumulate). ``params`` stacks the leaf's parameters across
    the K mixture components; ``to_value`` packs a component's accumulated statistics into the estimator's
    ``value()`` tuple.
    """

    name: str
    matches: Callable[[Any], bool]
    data: Callable[[Any], tuple[np.ndarray, ...]]  # leaf encoding -> tuple of (N,...) data arrays
    params: Callable[[list[Any]], dict[str, np.ndarray]]  # K leaf dists -> {pname: (K,...) array}
    to_value: Callable[[tuple, float], tuple] | None = None  # (stats for comp k, count_k) -> leaf value
    arity: int = 1  # number of data arrays this leaf consumes
    kind: str = "scalar"
    # scalar hooks (vals = [numba expr per data array], p/a = {name: arg}, r = responsibility var)
    expr: Callable[[list[str], dict[str, str]], str] | None = None
    acc_names: tuple[str, ...] = ()  # per-leaf weighted-statistic accumulator arrays, shape (K,)
    acc_stmt: Callable[[list[str], dict[str, str], str], str] | None = None
    # matrix hooks (emit numba source lines; leaf index `i`, {pname: arg} map)
    mat_precompute: Callable[[int, dict[str, str]], list[str]] | None = None  # -> fill q{i} (K,N)
    mat_row: Callable[[int, dict[str, str]], str] | None = None  # -> per-(i,k) scoring expr
    mat_accumulate: Callable[[int], list[str]] | None = None  # -> fill S1_{i} (K,D), S2_{i} (K,D,D) via BLAS
    # vector hooks (2-D data, inline per-dim loop, (K,D) accumulators; emit source lines)
    vec_row: Callable[[int, dict[str, str]], list[str]] | None = None  # -> add the leaf score to `acc`
    vec_accumulate: Callable[[int, dict[str, str]], list[str]] | None = None  # -> per-dim weighted-stat update
    # tabulated hooks (low-cardinality integer x: precompute a (K, max+1) log-pmf table, look it up per row;
    # accumulate sum_wx and -- when ``tab_hist`` -- a per-component weighted histogram (a map-reducible
    # bincount). This is how discrete families whose MLE needs a max/min reduction (Binomial) or the full
    # count distribution (NegativeBinomial's iterative dispersion) still fuse in one pass.)
    tab_table: Callable[[list[Any], int], np.ndarray] | None = None  # (comps, max_x) -> (K, max_x+1) log-pmf
    tab_hist: bool = False  # also accumulate a (K, max_x+1) weighted histogram
    tab_to_value: Callable[[float, float, np.ndarray | None, int, int], tuple] | None = (
        None  # (sx,count,hist_k,min,max)
    )
    # global min/max over x (a map-reducible reduction) for a scalar leaf whose value() needs it (Pareto's xm)
    wants_minmax: bool = False
    to_value_g: Callable[[tuple, float, float, float], tuple] | None = None  # (stats, count, min, max) -> value
    # categorical hooks (data is already a category index; score from a (K,C) log-prob table, accumulate a
    # (K,C) weighted count histogram -- the only sufficient statistic). Generalizes the tabulated pattern to
    # value-keyed categories (the table/value depend on the encoding's category list, so they get ``enc``).
    cat_table: Callable[[list[Any], Any], np.ndarray] | None = None  # (comps, enc) -> (K, C) log-prob table
    cat_to_value: Callable[[np.ndarray, Any, float], Any] | None = None  # (hist_k, enc, count) -> leaf value
    dtype: str = "float64"


_TEMPLATES: list[LeafTemplate] = []


def register_leaf_template(t: LeafTemplate) -> None:
    """Register a leaf template for fused scoring and E-step generation."""
    _TEMPLATES.append(t)


def _template_for(dist: Any) -> LeafTemplate | None:
    for t in _TEMPLATES:
        if t.matches(dist):
            return t
    return None


def _arr(enc: Any, dtype: Any = np.float64) -> np.ndarray:
    return np.ascontiguousarray(enc, dtype=dtype)


# --- scalar leaves --------------------------------------------------------------------------------
def _gaussian_params(comps: list[Any]) -> dict[str, np.ndarray]:
    mu = np.array([c.mu for c in comps], dtype=np.float64)
    s2 = np.array([c.sigma2 for c in comps], dtype=np.float64)
    return {"mu": mu, "inv2s2": 0.5 / s2, "lognorm": -0.5 * np.log(2.0 * np.pi * s2)}


register_leaf_template(
    LeafTemplate(
        name="gaussian",
        matches=lambda d: type(d).__name__ == "GaussianDistribution",
        data=lambda enc: (_arr(enc),),
        params=_gaussian_params,
        expr=lambda v, p: f"{p['lognorm']}[k] - ({v[0]} - {p['mu']}[k]) * ({v[0]} - {p['mu']}[k]) * {p['inv2s2']}[k]",
        acc_names=("sx", "sx2"),
        acc_stmt=lambda v, a, r: f"{a['sx']}[k] += {r} * {v[0]}; {a['sx2']}[k] += {r} * {v[0]} * {v[0]}",
        to_value=lambda s, count: (s[0], s[1], count, count),  # (sum_wx, sum_wx2, sum_w, sum_w)
    )
)


register_leaf_template(
    LeafTemplate(
        name="exponential",
        matches=lambda d: type(d).__name__ == "ExponentialDistribution",
        data=lambda enc: (_arr(enc),),
        params=lambda comps: (lambda beta: {"rate": 1.0 / beta, "lograte": -np.log(beta)})(
            np.array([c.beta for c in comps], dtype=np.float64)
        ),
        expr=lambda v, p: f"{p['lograte']}[k] - {p['rate']}[k] * {v[0]}",
        acc_names=("sx",),
        acc_stmt=lambda v, a, r: f"{a['sx']}[k] += {r} * {v[0]}",
        to_value=lambda s, count: (count, s[0]),  # (sum_w, sum_wx)
    )
)


register_leaf_template(
    LeafTemplate(
        name="geometric",
        matches=lambda d: type(d).__name__ == "GeometricDistribution",
        data=lambda enc: (_arr(enc),),  # support {1,2,...}: log p((x)) = (x-1) ln(1-p) + ln p
        params=lambda comps: (lambda p: {"logp": np.log(p), "log1mp": np.log1p(-p)})(
            np.array([c.p for c in comps], dtype=np.float64)
        ),
        expr=lambda v, p: f"{p['logp']}[k] + ({v[0]} - 1.0) * {p['log1mp']}[k]",
        acc_names=("sx",),
        acc_stmt=lambda v, a, r: f"{a['sx']}[k] += {r} * {v[0]}",
        to_value=lambda s, count: (count, s[0]),  # (sum_w, sum_wx)
    )
)


register_leaf_template(
    LeafTemplate(
        name="bernoulli",
        matches=lambda d: type(d).__name__ == "BernoulliDistribution",
        data=lambda enc: (_arr(enc),),  # log p(x) = x*logit + ln(1-p)
        params=lambda comps: (lambda p: {"logit": np.log(p) - np.log1p(-p), "log1mp": np.log1p(-p)})(
            np.array([c.p for c in comps], dtype=np.float64)
        ),
        expr=lambda v, p: f"{p['log1mp']}[k] + {v[0]} * {p['logit']}[k]",
        acc_names=("sx",),
        acc_stmt=lambda v, a, r: f"{a['sx']}[k] += {r} * {v[0]}",
        to_value=lambda s, count: (count, s[0]),  # (sum_w, sum_wx)
    )
)


register_leaf_template(
    LeafTemplate(
        name="poisson",
        matches=lambda d: type(d).__name__ == "PoissonDistribution",
        data=lambda enc: (_arr(enc[0]), _arr(enc[1])),  # (x, lgamma(x+1));  log p = -lam + x ln lam - lgamma(x+1)
        params=lambda comps: (lambda lam: {"lam": lam, "loglam": np.log(lam)})(
            np.array([c.lam for c in comps], dtype=np.float64)
        ),
        arity=2,
        expr=lambda v, p: f"{p['loglam']}[k] * {v[0]} - {p['lam']}[k] - {v[1]}",
        acc_names=("sx",),
        acc_stmt=lambda v, a, r: f"{a['sx']}[k] += {r} * {v[0]}",
        to_value=lambda s, count: (count, s[0]),  # (sum_w, sum_wx)
    )
)


def _gamma_params(comps: list[Any]) -> dict[str, np.ndarray]:
    k = np.array([c.k for c in comps], dtype=np.float64)
    theta = np.array([c.theta for c in comps], dtype=np.float64)
    from scipy.special import gammaln

    return {"km1": k - 1.0, "inv_theta": 1.0 / theta, "norm": -k * np.log(theta) - gammaln(k)}


register_leaf_template(
    LeafTemplate(
        name="gamma",
        matches=lambda d: type(d).__name__ == "GammaDistribution",
        data=lambda enc: (_arr(enc[0]), _arr(enc[1])),  # (x, log x); log p = (k-1)lnx - x/th - k ln th - lgamma(k)
        params=_gamma_params,
        arity=2,
        expr=lambda v, p: f"{p['norm']}[k] + {p['km1']}[k] * {v[1]} - {p['inv_theta']}[k] * {v[0]}",
        acc_names=("sx", "slogx"),
        acc_stmt=lambda v, a, r: f"{a['sx']}[k] += {r} * {v[0]}; {a['slogx']}[k] += {r} * {v[1]}",
        to_value=lambda s, count: (count, s[0], s[1]),  # (sum_w, sum_wx, sum_w_logx)
    )
)


def _arr0(enc: Any) -> tuple[np.ndarray]:
    """Raw x of a univariate encoding (enc[0] for the precomputed-transform tuples, else the array)."""
    return (_arr(enc[0] if isinstance(enc, tuple) else enc),)


# Each of these scalar leaves takes raw x and computes its transforms inline (numba has log/log1p/cos/sin),
# so only the per-component normalizer is precomputed in numpy. Verified bit-accurate vs seq_log_density.
register_leaf_template(
    LeafTemplate(
        name="halfnormal",
        matches=lambda d: type(d).__name__ == "HalfNormalDistribution",
        data=_arr0,
        params=lambda comps: (lambda s2: {"a": 0.5 / s2, "lognorm": 0.5 * np.log(2.0 / np.pi) - 0.5 * np.log(s2)})(
            np.array([c.sigma**2 for c in comps], dtype=np.float64)
        ),
        expr=lambda v, p: f"{p['lognorm']}[k] - {v[0]} * {v[0]} * {p['a']}[k]",
        acc_names=("sx2",),
        acc_stmt=lambda v, a, r: f"{a['sx2']}[k] += {r} * {v[0]} * {v[0]}",
        to_value=lambda s, count: (count, s[0]),  # (n, sum_w x^2)
    )
)


register_leaf_template(
    LeafTemplate(
        name="rayleigh",
        matches=lambda d: type(d).__name__ == "RayleighDistribution",
        data=_arr0,
        params=lambda comps: (lambda s2: {"a": 0.5 / s2, "lognorm": -np.log(s2)})(
            np.array([c.sigma**2 for c in comps], dtype=np.float64)
        ),
        expr=lambda v, p: f"{p['lognorm']}[k] + np.log({v[0]}) - {v[0]} * {v[0]} * {p['a']}[k]",
        acc_names=("sx2",),
        acc_stmt=lambda v, a, r: f"{a['sx2']}[k] += {r} * {v[0]} * {v[0]}",
        to_value=lambda s, count: (count, s[0]),  # (n, sum_w x^2)
    )
)


def _inverse_gaussian_params(comps: list[Any]) -> dict[str, np.ndarray]:
    mu = np.array([c.mu for c in comps], dtype=np.float64)
    lam = np.array([c.lam for c in comps], dtype=np.float64)
    return {
        "c1": -lam / (2.0 * mu * mu),
        "c2": -lam / 2.0,
        "lognorm": 0.5 * (np.log(lam) - np.log(2.0 * np.pi)) + lam / mu,
    }


register_leaf_template(
    LeafTemplate(
        name="inversegaussian",
        matches=lambda d: type(d).__name__ == "InverseGaussianDistribution",
        data=_arr0,
        params=_inverse_gaussian_params,
        expr=lambda v, p: f"{p['lognorm']}[k] - 1.5 * np.log({v[0]}) + {p['c1']}[k] * {v[0]} + {p['c2']}[k] / {v[0]}",
        acc_names=("sx", "s1x"),
        acc_stmt=lambda v, a, r: f"{a['sx']}[k] += {r} * {v[0]}; {a['s1x']}[k] += {r} / {v[0]}",
        to_value=lambda s, count: (count, s[0], s[1]),  # (n, sum_w x, sum_w / x)
    )
)


def _beta_params(comps: list[Any]) -> dict[str, np.ndarray]:
    from scipy.special import betaln

    a = np.array([c.a for c in comps], dtype=np.float64)
    b = np.array([c.b for c in comps], dtype=np.float64)
    return {"am1": a - 1.0, "bm1": b - 1.0, "lognorm": -betaln(a, b)}


register_leaf_template(
    LeafTemplate(
        name="beta",
        matches=lambda d: type(d).__name__ == "BetaDistribution",
        data=lambda enc: (_arr(enc[0]), _arr(enc[1]), _arr(enc[2]), _arr(enc[3])),  # (log x, log(1-x), x, x^2)
        params=_beta_params,
        arity=4,
        expr=lambda v, p: f"{p['lognorm']}[k] + {p['am1']}[k] * {v[0]} + {p['bm1']}[k] * {v[1]}",
        acc_names=("slogx", "slog1mx", "sx", "sx2"),
        acc_stmt=lambda v, a, r: (
            f"{a['slogx']}[k] += {r} * {v[0]}; {a['slog1mx']}[k] += {r} * {v[1]}; "
            f"{a['sx']}[k] += {r} * {v[2]}; {a['sx2']}[k] += {r} * {v[3]}"
        ),
        to_value=lambda s, count: (count, s[0], s[1], s[2], s[3]),  # (n, sumlogx, sumlog1mx, sumx, sumx2)
    )
)


def _inverse_gamma_params(comps: list[Any]) -> dict[str, np.ndarray]:
    from scipy.special import gammaln

    a = np.array([c.alpha for c in comps], dtype=np.float64)
    b = np.array([c.beta for c in comps], dtype=np.float64)
    return {"ap1": a + 1.0, "beta": b, "lognorm": a * np.log(b) - gammaln(a)}


register_leaf_template(
    LeafTemplate(
        name="inversegamma",
        matches=lambda d: type(d).__name__ == "InverseGammaDistribution",
        data=lambda enc: (_arr(enc[0]), _arr(enc[1])),  # (log x, 1/x)
        params=_inverse_gamma_params,
        arity=2,
        expr=lambda v, p: f"{p['lognorm']}[k] - {p['ap1']}[k] * {v[0]} - {p['beta']}[k] * {v[1]}",
        acc_names=("s1x", "slogx"),
        acc_stmt=lambda v, a, r: f"{a['s1x']}[k] += {r} * {v[1]}; {a['slogx']}[k] += {r} * {v[0]}",
        to_value=lambda s, count: (count, s[0], -s[1]),  # (n, sum_w / x, sum_w log(1/x))  -- Gamma on 1/x
    )
)


def _vonmises_params(comps: list[Any]) -> dict[str, np.ndarray]:
    from scipy.special import i0

    mu = np.array([c.mu for c in comps], dtype=np.float64)
    kappa = np.array([c.kappa for c in comps], dtype=np.float64)
    return {"kcos": kappa * np.cos(mu), "ksin": kappa * np.sin(mu), "lognorm": -np.log(2.0 * np.pi * i0(kappa))}


register_leaf_template(
    LeafTemplate(
        name="vonmises",
        matches=lambda d: type(d).__name__ == "VonMisesDistribution",
        data=lambda enc: (_arr(enc[0]), _arr(enc[1])),  # (cos x, sin x)
        params=_vonmises_params,
        arity=2,
        expr=lambda v, p: f"{p['lognorm']}[k] + {p['kcos']}[k] * {v[0]} + {p['ksin']}[k] * {v[1]}",
        acc_names=("scos", "ssin"),
        acc_stmt=lambda v, a, r: f"{a['scos']}[k] += {r} * {v[0]}; {a['ssin']}[k] += {r} * {v[1]}",
        to_value=lambda s, count: (count, s[0], s[1]),  # (n, sum_w cos, sum_w sin)
    )
)


def _wrappedcauchy_params(comps: list[Any]) -> dict[str, np.ndarray]:
    mu = np.array([c.mu for c in comps], dtype=np.float64)
    rho = np.array([c.rho for c in comps], dtype=np.float64)
    return {
        "a": 1.0 + rho * rho,
        "rcos": 2.0 * rho * np.cos(mu),
        "rsin": 2.0 * rho * np.sin(mu),
        "lognorm": np.log1p(-rho * rho) - np.log(2.0 * np.pi),
    }


register_leaf_template(
    LeafTemplate(
        name="wrappedcauchy",
        matches=lambda d: type(d).__name__ == "WrappedCauchyDistribution",
        data=lambda enc: (_arr(enc[0]), _arr(enc[1])),  # (cos x, sin x)
        params=_wrappedcauchy_params,
        arity=2,
        expr=lambda v, p: (
            f"{p['lognorm']}[k] - np.log({p['a']}[k] - {p['rcos']}[k] * {v[0]} - {p['rsin']}[k] * {v[1]})"
        ),
        acc_names=("scos", "ssin"),
        acc_stmt=lambda v, a, r: f"{a['scos']}[k] += {r} * {v[0]}; {a['ssin']}[k] += {r} * {v[1]}",
        to_value=lambda s, count: (s[0], s[1], count),  # (sum_w cos, sum_w sin, n)
    )
)


register_leaf_template(
    LeafTemplate(
        name="pareto",
        matches=lambda d: type(d).__name__ == "ParetoDistribution",
        data=lambda enc: (_arr(enc[0]), _arr(enc[1])),  # (x, log x)
        params=lambda comps: (lambda al, xm: {"const": np.log(al) + al * np.log(xm), "am1": al + 1.0, "xm": xm})(
            np.array([c.alpha for c in comps], dtype=np.float64), np.array([c.xm for c in comps], dtype=np.float64)
        ),
        arity=2,
        # support x >= xm: log p = log a + a log xm - (a+1) log x; below the scale the component is impossible
        expr=lambda v, p: f"(({p['const']}[k] - {p['am1']}[k] * {v[1]}) if {v[0]} >= {p['xm']}[k] else -np.inf)",
        acc_names=("slogx",),
        acc_stmt=lambda v, a, r: f"{a['slogx']}[k] += {r} * {v[1]}",
        wants_minmax=True,  # the scale xm is estimated as the minimum observed x (a global reduction)
        to_value_g=lambda s, count, mn, mx: (count, s[0], mn),  # (n, sum_w logx, min x)
    )
)


register_leaf_template(
    LeafTemplate(
        name="logseries",
        matches=lambda d: type(d).__name__ == "LogSeriesDistribution",
        data=_arr0,
        params=lambda comps: (lambda p: {"logp": np.log(p), "lognorm": -np.log(-np.log1p(-p))})(
            np.array([c.p for c in comps], dtype=np.float64)
        ),
        expr=lambda v, p: f"{p['lognorm']}[k] + {v[0]} * {p['logp']}[k] - np.log({v[0]})",
        acc_names=("sx",),
        acc_stmt=lambda v, a, r: f"{a['sx']}[k] += {r} * {v[0]}",
        to_value=lambda s, count: (count, s[0]),  # (n, sum_w x)
    )
)


# Location-scale families with moment-matched estimators: value() = (sum_w x, sum_w x^2, n), exactly the
# Gaussian statistics; only the (closed-form) density differs. z = (x - loc) / scale, inlined.
def _locscale_params(comps: list[Any]) -> dict[str, np.ndarray]:
    loc = np.array([c.loc for c in comps], dtype=np.float64)
    scale = np.array([c.scale for c in comps], dtype=np.float64)
    return {"loc": loc, "inv_scale": 1.0 / scale, "nls": -np.log(scale)}


def _z(v: list[str], p: dict[str, str]) -> str:
    return f"(({v[0]} - {p['loc']}[k]) * {p['inv_scale']}[k])"


register_leaf_template(
    LeafTemplate(
        name="gumbel",
        matches=lambda d: type(d).__name__ == "GumbelDistribution",
        data=lambda enc: (_arr(enc),),
        params=_locscale_params,
        expr=lambda v, p: f"{p['nls']}[k] - {_z(v, p)} - np.exp(-{_z(v, p)})",
        acc_names=("sx", "sx2"),
        acc_stmt=lambda v, a, r: f"{a['sx']}[k] += {r} * {v[0]}; {a['sx2']}[k] += {r} * {v[0]} * {v[0]}",
        to_value=lambda s, count: (s[0], s[1], count),  # (sum_w x, sum_w x^2, n)
    )
)


register_leaf_template(
    LeafTemplate(
        name="logistic",
        matches=lambda d: type(d).__name__ == "LogisticDistribution",
        data=lambda enc: (_arr(enc),),
        params=_locscale_params,
        expr=lambda v, p: f"{p['nls']}[k] - {_z(v, p)} - 2.0 * np.log1p(np.exp(-{_z(v, p)}))",
        acc_names=("sx", "sx2"),
        acc_stmt=lambda v, a, r: f"{a['sx']}[k] += {r} * {v[0]}; {a['sx2']}[k] += {r} * {v[0]} * {v[0]}",
        to_value=lambda s, count: (s[0], s[1], count),  # (sum_w x, sum_w x^2, n)
    )
)


def _studentt_params(comps: list[Any]) -> dict[str, np.ndarray]:
    from scipy.special import gammaln

    loc = np.array([c.loc for c in comps], dtype=np.float64)
    scale = np.array([c.scale for c in comps], dtype=np.float64)
    df = np.array([c.df for c in comps], dtype=np.float64)
    lognorm = gammaln(0.5 * (df + 1.0)) - gammaln(0.5 * df) - 0.5 * np.log(df * np.pi) - np.log(scale)
    return {"loc": loc, "inv_scale": 1.0 / scale, "ndf": 0.5 * (df + 1.0), "inv_df": 1.0 / df, "lognorm": lognorm}


register_leaf_template(
    LeafTemplate(
        name="studentt",
        matches=lambda d: type(d).__name__ == "StudentTDistribution",
        data=lambda enc: (_arr(enc),),
        params=_studentt_params,
        expr=lambda v, p: f"{p['lognorm']}[k] - {p['ndf']}[k] * np.log1p({_z(v, p)} * {_z(v, p)} * {p['inv_df']}[k])",
        acc_names=("sx", "sx2"),
        acc_stmt=lambda v, a, r: f"{a['sx']}[k] += {r} * {v[0]}; {a['sx2']}[k] += {r} * {v[0]} * {v[0]}",
        to_value=lambda s, count: (s[0], s[1], count),  # (sum_w x, sum_w x^2, n)  -- moment-matched loc/scale
    )
)


def _loggaussian_params(comps: list[Any]) -> dict[str, np.ndarray]:
    mu = np.array([c.mu for c in comps], dtype=np.float64)
    s2 = np.array([c.sigma2 for c in comps], dtype=np.float64)
    return {"mu": mu, "inv2s2": 0.5 / s2, "lognorm": -0.5 * np.log(2.0 * np.pi * s2)}


register_leaf_template(
    LeafTemplate(
        name="loggaussian",
        matches=lambda d: type(d).__name__ == "LogGaussianDistribution",
        data=lambda enc: (_arr(enc),),  # the encoder already supplies log x; score = Gaussian(log x) - log x (Jacobian)
        params=_loggaussian_params,
        expr=lambda v, p: (
            f"{p['lognorm']}[k] - {v[0]} - ({v[0]} - {p['mu']}[k]) * ({v[0]} - {p['mu']}[k]) * {p['inv2s2']}[k]"
        ),
        acc_names=("sx", "sx2"),
        acc_stmt=lambda v, a, r: f"{a['sx']}[k] += {r} * {v[0]}; {a['sx2']}[k] += {r} * {v[0]} * {v[0]}",
        to_value=lambda s, count: (s[0], s[1], count, count),  # (sum_w_logx, sum_w_logx2, sum_w, sum_w)
    )
)


# --- vector leaves (2-D data, per-dim inline loop, (K,D) accumulators -- no matmul) ----------------
def _diaggaussian_params(comps: list[Any]) -> dict[str, np.ndarray]:
    mu = np.ascontiguousarray(np.stack([np.asarray(c.mu, dtype=np.float64) for c in comps]))
    s2 = np.ascontiguousarray(np.stack([np.asarray(c.covar, dtype=np.float64) for c in comps]))  # diagonal variances
    lognorm = -0.5 * np.sum(np.log(2.0 * np.pi * s2), axis=1)  # (K,)
    return {"mu": mu, "inv2s2": 0.5 / s2, "lognorm": lognorm}


def _diag_row(i: int, p: dict[str, str]) -> list[str]:
    return [
        f"acc += {p['lognorm']}[k]",
        f"for d in range(x{i}_0.shape[1]):",
        f"    diff{i} = x{i}_0[i, d] - {p['mu']}[k, d]",
        f"    acc -= diff{i} * diff{i} * {p['inv2s2']}[k, d]",
    ]


def _diag_accumulate(i: int, a: dict[str, str]) -> list[str]:
    return [
        f"for d in range(x{i}_0.shape[1]):",
        f"    {a['s1']}[k, d] += r * x{i}_0[i, d]",
        f"    {a['s2']}[k, d] += r * x{i}_0[i, d] * x{i}_0[i, d]",
    ]


register_leaf_template(
    LeafTemplate(
        name="diaggaussian",
        matches=lambda d: type(d).__name__ == "DiagonalGaussianDistribution",
        data=lambda enc: (_arr(enc),),
        params=_diaggaussian_params,
        kind="vector",
        vec_row=_diag_row,
        acc_names=("s1", "s2"),
        vec_accumulate=_diag_accumulate,
        to_value=lambda s, count: (s[0], s[1], count),  # (sum_wx (D,), sum_wxx (D,), sum_w)
    )
)


# --- matrix leaves --------------------------------------------------------------------------------
def _mvgaussian_params(comps: list[Any]) -> dict[str, np.ndarray]:
    mu = np.ascontiguousarray(np.stack([np.asarray(c.mu, dtype=np.float64) for c in comps]))
    prec = np.ascontiguousarray(np.stack([np.asarray(c.inv_covar, dtype=np.float64) for c in comps]))
    d = mu.shape[1]
    lognorm = np.empty(len(comps), dtype=np.float64)
    for k, c in enumerate(comps):
        _, logdet = np.linalg.slogdet(np.asarray(c.covar, dtype=np.float64))
        lognorm[k] = -0.5 * (d * np.log(2.0 * np.pi) + logdet)
    return {"mu": mu, "prec": prec, "lognorm": lognorm}


def _mv_precompute(i: int, p: dict[str, str]) -> list[str]:
    # q{i}[k, r] = (x_r - mu_k)' P_k (x_r - mu_k), the centered quadratic form, via BLAS gemm `C @ P`.
    return [
        f"q{i} = np.empty((kc, n))",
        "for k in range(kc):",
        f"    C{i} = x{i}_0 - {p['mu']}[k]",
        f"    Z{i} = C{i} @ {p['prec']}[k]",
        "    for ii in range(n):",
        f"        qq{i} = 0.0",
        f"        for jj in range(x{i}_0.shape[1]):",
        f"            qq{i} += Z{i}[ii, jj] * C{i}[ii, jj]",
        f"        q{i}[k, ii] = qq{i}",
    ]


def _mv_accumulate(i: int) -> list[str]:
    # MVGaussian weighted stats: S1_k = X' r_k (weighted sum), S2_k = X' (r_k .* X) (weighted Gram), via BLAS.
    return [
        "for k in range(kc):",
        f"    Rk{i} = R[:, k].copy()",
        f"    S1_{i}[k] = x{i}_0.T @ Rk{i}",
        f"    S2_{i}[k] = x{i}_0.T @ (Rk{i}.reshape(-1, 1) * x{i}_0)",
    ]


register_leaf_template(
    LeafTemplate(
        name="mvgaussian",
        matches=lambda d: type(d).__name__ == "MultivariateGaussianDistribution",
        data=lambda enc: (_arr(enc),),
        params=_mvgaussian_params,
        kind="matrix",
        mat_precompute=_mv_precompute,
        mat_row=lambda i, p: f"-0.5 * q{i}[k, i] + {p['lognorm']}[k]",
        mat_accumulate=_mv_accumulate,
        to_value=lambda s, count: (s[0], s[1], count),  # (sum_wx (D,), sum_wxx (D,D), sum_w)
    )
)


# --- tabulated leaves (low-cardinality integer x: table-scored + map-reducible histogram/min-max) --
def _binomial_table(comps: list[Any], max_x: int) -> np.ndarray:
    from scipy.special import gammaln

    xs = np.arange(max_x + 1, dtype=np.float64)
    tab = np.full((len(comps), max_x + 1), -np.inf, dtype=np.float64)
    for k, c in enumerate(comps):
        n = float(c.n)
        mv = float(getattr(c, "min_val", 0) or 0)
        xx = xs - mv
        valid = (xx >= 0) & (xx <= n)
        lp = np.log(c.p) if c.p > 0 else -np.inf
        l1p = np.log1p(-c.p) if c.p < 1 else -np.inf
        with np.errstate(invalid="ignore", divide="ignore"):
            row = gammaln(n + 1) - gammaln(xx + 1) - gammaln(n - xx + 1) + xx * lp + (n - xx) * l1p
        tab[k, valid] = row[valid]
    return tab


register_leaf_template(
    LeafTemplate(
        name="binomial",
        matches=lambda d: type(d).__name__ == "BinomialDistribution",
        data=lambda enc: (_arr(enc[2]),),  # enc = (unique, inverse, x, min, max); accumulate over raw x
        params=lambda comps: {},
        kind="tabulated",
        tab_table=_binomial_table,
        acc_names=("sx",),
        # value() = (count, sum, min_val, max_val); min/max are global reductions over x (weight-independent)
        tab_to_value=lambda sx, count, hist, mn, mx: (count, sx, mn, mx),
    )
)


def _negbinomial_table(comps: list[Any], max_x: int) -> np.ndarray:
    from scipy.special import gammaln

    xs = np.arange(max_x + 1, dtype=np.float64)
    tab = np.empty((len(comps), max_x + 1), dtype=np.float64)
    for k, c in enumerate(comps):  # NB(r,p): log p = lgamma(x+r) - lgamma(r) - lgamma(x+1) + r ln p + x ln(1-p)
        tab[k] = gammaln(xs + c.r) - gammaln(c.r) - gammaln(xs + 1) + c.r * np.log(c.p) + xs * np.log1p(-c.p)
    return tab


def _hist_to_dict(hist_k: np.ndarray) -> dict[int, float]:
    nz = np.nonzero(hist_k)[0]
    return {int(x): float(hist_k[x]) for x in nz}


register_leaf_template(
    LeafTemplate(
        name="negbinomial",
        matches=lambda d: type(d).__name__ == "NegativeBinomialDistribution",
        data=lambda enc: (_arr(enc[0]),),  # enc = (x, lgamma(x+1)); accumulate sum_wx + weighted histogram
        params=lambda comps: {},
        kind="tabulated",
        tab_table=_negbinomial_table,
        acc_names=("sx", "hist"),
        tab_hist=True,
        # value() = (count, sum, histogram); the iterative dispersion MLE needs the full weighted count dist
        tab_to_value=lambda sx, count, hist, mn, mx: (count, sx, _hist_to_dict(hist)),
    )
)


# --- categorical leaves (data is already a category index; (K,C) log-prob table + count histogram) ------
def _categorical_table(comps: list[Any], enc: Any) -> np.ndarray:
    values = enc[1]  # the shared category list the encoder assigned indices over
    tab = np.empty((len(comps), len(values)), dtype=np.float64)
    for k, c in enumerate(comps):
        tab[k] = [float(c.log_density(v)) for v in values]  # -inf where a value is outside this component
    return tab


register_leaf_template(
    LeafTemplate(
        name="categorical",
        matches=lambda d: type(d).__name__ == "CategoricalDistribution",
        data=lambda enc: (_arr(enc[0]),),  # enc = (index, values); index is already 0..C-1
        params=lambda comps: {},
        kind="categorical",
        acc_names=("hist",),
        cat_table=_categorical_table,
        # value() is the weighted count dict {value: weight}
        cat_to_value=lambda hist_k, enc, count: {
            enc[1][c]: float(hist_k[c]) for c in range(len(enc[1])) if hist_k[c] != 0.0
        },
    )
)


def _int_categorical_table(comps: list[Any], enc: Any) -> np.ndarray:
    mn = int(np.rint(np.asarray(enc).min())) if np.asarray(enc).size else 0
    width = int(np.rint(np.asarray(enc).max())) - mn + 1 if np.asarray(enc).size else 1
    tab = np.empty((len(comps), width), dtype=np.float64)
    for k, c in enumerate(comps):
        tab[k] = [float(c.log_density(mn + j)) for j in range(width)]
    return tab


register_leaf_template(
    LeafTemplate(
        name="intcategorical",
        matches=lambda d: type(d).__name__ == "IntegerCategoricalDistribution",
        data=lambda enc: (_arr(enc) - int(np.rint(np.asarray(enc).min())) if np.asarray(enc).size else _arr(enc),),
        params=lambda comps: {},
        kind="categorical",
        acc_names=("hist",),
        cat_table=_int_categorical_table,
        # value() = (min_val, weighted count vector over min_val .. max_val)
        cat_to_value=lambda hist_k, enc, count: (
            int(np.rint(np.asarray(enc).min())) if np.asarray(enc).size else 0,
            hist_k.copy(),
        ),
    )
)


# --- structure analysis ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FusedPlan:
    """A fusible model structure: K mixture components (1 if not a mixture), each a list of leaf factors."""

    num_components: int
    is_mixture: bool
    leaf_templates: tuple[LeafTemplate, ...]  # one per factor (composite factor order)
    signature: tuple
    component_is_composite: bool = False  # whether each leaf-bearing node is a Composite (vs a bare leaf)

    @property
    def has_matrix(self) -> bool:
        """Whether any leaf in the fused plan requires matrix BLAS handling."""
        return any(t.kind == "matrix" for t in self.leaf_templates)


def _node_factors(node: Any) -> list[Any] | None:
    """The leaf factors of a fusible *node* (an exact Composite, or a templated bare leaf), else None."""
    if type(node).__name__ == "CompositeDistribution":
        return list(node.dists)
    if _template_for(node) is not None:
        return [node]
    return None


def analyze(model: Any) -> FusedPlan | None:
    """Return a :class:`FusedPlan` if ``model`` is a fusible mixture/composite of templated leaves, else None.

    Handles: a single leaf, a Composite of leaves, a Mixture of leaves, and a Mixture of Composites
    (every component sharing the same leaf-type structure). Anything else -> None (fall back to numpy).

    The structural nodes are matched by EXACT type, not duck-typing: mixture-flavoured relatives
    (SemiSupervised / Joint / Hierarchical mixtures, Select, ...) carry ``components``/``log_w`` but have
    different E-step semantics, so they must keep their own kernels.
    """
    tname = type(model).__name__
    if tname == "MixtureDistribution":
        comps = list(model.components)
        per = [_node_factors(c) for c in comps]
        if any(p is None for p in per):  # a component is not a plain Composite / templated leaf
            return None
        templates = [_template_for(f) for f in per[0]]  # type: ignore[union-attr]
        if any(t is None for t in templates):  # a factor has no leaf template (e.g. a Categorical)
            return None
        # every component must have the same leaf types in the same order
        if any([_template_for(f) for f in p] != templates for p in per):  # type: ignore[union-attr]
            return None
        comp_is_composite = type(comps[0]).__name__ == "CompositeDistribution"
        sig = ("mix", tuple(t.name for t in templates), comp_is_composite)  # type: ignore[union-attr]
        return FusedPlan(len(comps), True, tuple(templates), sig, comp_is_composite)  # type: ignore[arg-type]
    factors = _node_factors(model)
    if factors is None:
        return None
    templates = [_template_for(f) for f in factors]
    if any(t is None for t in templates):  # a composite factor has no leaf template
        return None
    is_composite = tname == "CompositeDistribution"
    sig = ("comp", tuple(t.name for t in templates), is_composite)  # type: ignore[union-attr]
    return FusedPlan(1, False, tuple(templates), sig, is_composite)  # type: ignore[arg-type]


def fusible(model: Any) -> bool:
    """Return whether ``model`` can use a fused scoring kernel."""
    if analyze(model) is not None:
        return True
    from mixle.stats.compute.fused_nested import fusible_nested  # nested scalar trees (Mixture-of-Mixture, ...)

    return fusible_nested(model)


def _dummy(t: LeafTemplate) -> Any:
    """A throwaway leaf instance to read this template's parameter names (kept small and family-specific)."""
    import mixle.stats as stats

    return {
        "gaussian": stats.GaussianDistribution(0.0, 1.0),
        "exponential": stats.ExponentialDistribution(1.0),
        "geometric": stats.GeometricDistribution(0.5),
        "bernoulli": stats.BernoulliDistribution(0.5),
        "poisson": stats.PoissonDistribution(1.0),
        "gamma": stats.GammaDistribution(1.0, 1.0),
        "loggaussian": stats.LogGaussianDistribution(0.0, 1.0),
        "halfnormal": stats.HalfNormalDistribution(1.0),
        "rayleigh": stats.RayleighDistribution(1.0),
        "inversegaussian": stats.InverseGaussianDistribution(1.0, 1.0),
        "beta": stats.BetaDistribution(2.0, 2.0),
        "inversegamma": stats.InverseGammaDistribution(2.0, 1.0),
        "vonmises": stats.VonMisesDistribution(0.0, 1.0),
        "wrappedcauchy": stats.WrappedCauchyDistribution(0.0, 0.5),
        "logseries": stats.LogSeriesDistribution(0.5),
        "pareto": stats.ParetoDistribution(2.0, 1.0),
        "gumbel": stats.GumbelDistribution(0.0, 1.0),
        "logistic": stats.LogisticDistribution(0.0, 1.0),
        "studentt": stats.StudentTDistribution(5.0, 0.0, 1.0),
        "categorical": stats.CategoricalDistribution({"a": 0.5, "b": 0.5}),
        "intcategorical": stats.IntegerCategoricalDistribution(0, [0.5, 0.5]),
        "diaggaussian": stats.DiagonalGaussianDistribution([0.0, 0.0], [1.0, 1.0]),
        "binomial": stats.BinomialDistribution(0.5, 1),
        "negbinomial": stats.NegativeBinomialDistribution(1.0, 0.5),
        "mvgaussian": stats.MultivariateGaussianDistribution([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
    }[t.name]


def _argmap(i: int, t: LeafTemplate) -> dict[str, str]:
    return {pn: f"p{i}_{pn}" for pn in sorted(t.params([_dummy(t)]).keys())}


def _data_names(i: int, t: LeafTemplate) -> list[str]:
    return [f"x{i}_{j}" for j in range(t.arity)]


# --- code generation ------------------------------------------------------------------------------
_COMPILED: dict[tuple, Callable] = {}
_ESTEP_COMPILED: dict[tuple, Callable] = {}


def _emit(plan: FusedPlan) -> dict[str, list[str]]:
    """Per-leaf source fragments shared by the scorer and the E-step.

    Each leaf contributes (by ``kind``): ``data``/``param`` argument names; ``precompute`` (matrix-only,
    BLAS quad form before the row loop); ``row`` lines that add the leaf's score to ``acc`` inside the
    component loop (scalar = one ``acc +=`` expr, vector = an inline per-dim loop, matrix = a precomputed
    reference); E-step ``acc_args`` + ``acc`` lines (scalar/vector inline in the responsibility loop);
    and matrix-only ``post`` lines (BLAS Gram accumulation after the row loop). Pure-scalar plans produce
    empty ``precompute``/``post``, so their generated source is unchanged.
    """
    frag: dict[str, list[str]] = {
        k: [] for k in ("data_args", "param_args", "precompute", "row", "acc_args", "acc", "post")
    }
    for i, t in enumerate(plan.leaf_templates):
        frag["data_args"].extend(_data_names(i, t))
        amap = _argmap(i, t)
        frag["param_args"].extend(amap.values())
        if t.kind == "matrix":
            frag["precompute"].extend(t.mat_precompute(i, amap))  # type: ignore[misc]
            frag["row"].append("acc += " + t.mat_row(i, amap))  # type: ignore[misc]
            frag["acc_args"] += [f"S1_{i}", f"S2_{i}"]
            frag["post"].extend(t.mat_accumulate(i))  # type: ignore[misc]
            continue
        accmap = {an: f"a{i}_{an}" for an in t.acc_names}
        frag["acc_args"].extend(accmap.values())
        if t.kind == "vector":
            frag["row"].extend(t.vec_row(i, amap))  # type: ignore[misc]
            frag["acc"].extend(t.vec_accumulate(i, accmap))  # type: ignore[misc]
        elif t.kind == "tabulated":
            frag["param_args"].append(f"tab{i}")  # the (K, max+1) log-pmf table, looked up by int(x)
            frag["row"].append(f"acc += tab{i}[k, int(x{i}_0[i])]")
            frag["acc"].append(f"{accmap['sx']}[k] += r * x{i}_0[i]")
            if t.tab_hist:
                frag["acc"].append(f"{accmap['hist']}[k, int(x{i}_0[i])] += r")  # weighted count histogram
        elif t.kind == "categorical":
            frag["param_args"].append(f"cat{i}")  # the (K, C) log-prob table, looked up by the category index
            frag["row"].append(f"acc += cat{i}[k, int(x{i}_0[i])]")
            frag["acc"].append(f"{accmap['hist']}[k, int(x{i}_0[i])] += r")  # weighted per-category count
        else:
            vals = [f"{nm}[i]" for nm in _data_names(i, t)]
            frag["row"].append("acc += " + t.expr(vals, amap))  # type: ignore[misc]
            frag["acc"].append(t.acc_stmt(vals, accmap, "r"))  # type: ignore[misc]
    return frag


import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402

_CACHE_DIR = os.environ.get("MIXLE_FUSED_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "mixle_fused_cache")
_NJIT_LOCK = threading.Lock()


def _njit(src: str, fname: str) -> Callable:
    """Compile generated source to a disk-cached ``njit`` function.

    The source (``def _fused.../_estep...``) is written to a stable per-source module file and decorated
    with ``numba.njit(cache=True)`` so numba persists the compiled kernel to disk: a fresh process that
    fuses the same structure reuses it instead of paying the ~1-2s compile again (the matmul E-step kernel
    is the costly one). ``exec``'d ``<string>`` source cannot do this -- numba's cache needs a file locator
    -- so without the file the compile is paid every process. The module name is the source digest, so the
    same structure maps to the same file (and cache) and different structures never collide. Any
    filesystem/import problem falls back to an in-memory ``exec`` (no disk cache, identical result)."""
    import numba

    digest = hashlib.sha1(src.encode()).hexdigest()[:16]  # noqa: S324 -- cache key, not security
    modname = f"_pysp_fused_{digest}"
    with _NJIT_LOCK:
        if modname in sys.modules:
            return getattr(sys.modules[modname], fname)
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            path = os.path.join(_CACHE_DIR, modname + ".py")
            if not os.path.exists(path):
                module_src = f"import numpy as np\nimport numba\n\n\n@numba.njit(fastmath=True, cache=True)\n{src}\n"
                tmp = f"{path}.{os.getpid()}.tmp"
                with open(tmp, "w") as fh:
                    fh.write(module_src)
                os.replace(tmp, path)  # atomic publish -> concurrent processes never read a partial file
            spec = importlib.util.spec_from_file_location(modname, path)
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            sys.modules[modname] = mod
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return getattr(mod, fname)
        except Exception:  # noqa: BLE001
            ns: dict[str, Any] = {"np": np}
            exec(src, ns)  # noqa: S102 -- generated from fixed templates, no user input
            return numba.njit(fastmath=True)(ns[fname])


def _compile(plan: FusedPlan) -> Callable:
    cached = _COMPILED.get(plan.signature)
    if cached is not None:
        return cached
    f = _emit(plan)
    data_args = f["data_args"]
    args = ", ".join(data_args + f["param_args"] + ["logw", "out", "llbuf"])
    lines = [f"def _fused({args}):", f"    n = {data_args[0]}.shape[0]", "    kc = logw.shape[0]"]
    lines += ["    " + ln for ln in f["precompute"]]
    lines += ["    for i in range(n):", "        for k in range(kc):", "            acc = logw[k]"]
    lines += ["            " + rt for rt in f["row"]]
    lines += [
        "            llbuf[k] = acc",
        "        m = llbuf[0]",
        "        for k in range(1, kc):",
        "            if llbuf[k] > m:",
        "                m = llbuf[k]",
        "        s = 0.0",
        "        for k in range(kc):",
        "            s += np.exp(llbuf[k] - m)",
        "        out[i] = (m + np.log(s)) if m > -np.inf else -np.inf",  # all components -inf (out of support)
    ]
    fn = _njit("\n".join(lines), "_fused")
    _COMPILED[plan.signature] = fn
    return fn


def _compile_estep(plan: FusedPlan) -> Callable:
    cached = _ESTEP_COMPILED.get(plan.signature)
    if cached is not None:
        return cached
    f = _emit(plan)
    data_args = f["data_args"]
    args = ", ".join(
        data_args + f["param_args"] + ["weights", "logw", "comp_counts", *f["acc_args"], "llbuf", "out_ll"]
    )
    lines = [f"def _estep({args}):", f"    n = {data_args[0]}.shape[0]", "    kc = logw.shape[0]"]
    if plan.has_matrix:
        lines.append("    R = np.empty((n, kc))")  # responsibilities -- only matrix accumulation needs them
    lines += ["    " + ln for ln in f["precompute"]]
    lines += [
        "    for i in range(n):",
        "        wi = weights[i]",
        "        for k in range(kc):",
        "            acc = logw[k]",
    ]
    lines += ["            " + rt for rt in f["row"]]
    lines += [
        "            llbuf[k] = acc",
        "        m = llbuf[0]",
        "        for k in range(1, kc):",
        "            if llbuf[k] > m:",
        "                m = llbuf[k]",
        "        s = 0.0",
        "        for k in range(kc):",
        "            s += np.exp(llbuf[k] - m)",
        "        out_ll[0] += wi * (m + np.log(s))",  # data log-likelihood, free as the posterior normalizer
        "        for k in range(kc):",
        "            r = np.exp(llbuf[k] - m) / s * wi",
        "            comp_counts[k] += r",
    ]
    lines += ["            " + st for st in f["acc"]]
    if plan.has_matrix:
        lines.append("            R[i, k] = r")
    lines += ["    " + ln for ln in f["post"]]
    fn = _njit("\n".join(lines), "_estep")
    _ESTEP_COMPILED[plan.signature] = fn
    return fn


# --- marshalling ----------------------------------------------------------------------------------
def _component_factor_lists(model: Any, plan: FusedPlan) -> list[list[Any]]:
    if plan.is_mixture:
        return [_node_factors(c) for c in model.components]  # type: ignore[misc]
    return [_node_factors(model)]  # type: ignore[list-item]


def _data_and_params(
    model: Any, plan: FusedPlan, enc: Any, compute_dtype: Any = None
) -> tuple[list[np.ndarray], list[np.ndarray], dict[int, tuple[int, int]]]:
    """Return (flattened data arrays, param arrays, tabulated-leaf context {i: (min_x, max_x)}).

    Tabulated leaves additionally get a data-dependent ``(K, max_x+1)`` log-pmf table appended to the param
    arrays (right after their -- empty -- ``params()`` block, matching the order :func:`_emit` assigns), and
    their integer min/max over x recorded for ``to_value``.

    When ``compute_dtype`` is a reduced float (e.g. ``np.float32``), the *floating* data and parameter
    arrays are down-cast to it so the row arithmetic runs in that precision (DeepSeek-style low-precision
    compute), while every accumulator -- ``acc``/``llbuf``/``out``/``comp_counts`` and the E-step
    sufficient statistics -- stays float64 (high-precision accumulation), so the reduction does not drift
    with N or with the number of components. Integer index arrays (categorical / tabulated lookups) are
    left untouched. ``None`` keeps everything float64 (byte-identical to the legacy path).
    """
    factor_lists = _component_factor_lists(model, plan)
    # A Composite encodes as a per-factor tuple; a bare leaf encodes as its own payload (which may itself
    # be a tuple, e.g. Poisson's (x, lgamma(x+1))) -- so key on the structure, not isinstance(enc, tuple).
    factor_encs = enc if plan.component_is_composite else (enc,)
    data_arrays: list[np.ndarray] = []
    param_arrays: list[np.ndarray] = []
    tab_ctx: dict[int, tuple[int, int]] = {}
    for i, t in enumerate(plan.leaf_templates):
        arrs = t.data(factor_encs[i])
        data_arrays.extend(arrs)  # arity arrays, flattened in leaf order
        comps_i = [factor_lists[k][i] for k in range(plan.num_components)]
        pdict = t.params(comps_i)
        param_arrays.extend(np.ascontiguousarray(pdict[pn]) for pn in sorted(pdict.keys()))
        if t.kind == "tabulated":
            x = arrs[0]
            mx = int(np.rint(x.max())) if x.size else 0
            mn = int(np.rint(x.min())) if x.size else 0
            param_arrays.append(np.ascontiguousarray(t.tab_table(comps_i, mx)))  # type: ignore[misc]
            tab_ctx[i] = (mn, mx)
        elif t.wants_minmax:
            x = arrs[0]
            tab_ctx[i] = (float(x.min()) if x.size else 0.0, float(x.max()) if x.size else 0.0)
        elif t.kind == "categorical":
            table = np.ascontiguousarray(t.cat_table(comps_i, factor_encs[i]))  # type: ignore[misc]
            param_arrays.append(table)
            tab_ctx[i] = (factor_encs[i], table.shape[1])  # (encoding for to_value, C for histogram width)
    if compute_dtype is not None and np.dtype(compute_dtype) != np.float64:
        if np.dtype(compute_dtype) != np.float32:
            raise ValueError(
                "fused kernels support reduced precision only in float32: numba cannot compile float16 / "
                "bfloat16 or sub-byte formats on CPU. Got %r." % (compute_dtype,)
            )
        # Down-cast only the FLOATING arrays; integer index/lookup arrays must keep their dtype. The
        # generated kernels promote back to the float64 accumulators on every ``acc += ...``.
        cast = lambda a: np.ascontiguousarray(a, dtype=compute_dtype) if a.dtype.kind == "f" else a  # noqa: E731
        data_arrays = [cast(a) for a in data_arrays]
        param_arrays = [cast(a) for a in param_arrays]
    return data_arrays, param_arrays, tab_ctx


def fused_seq_log_density(model: Any, enc: Any, compute_dtype: Any = None) -> np.ndarray:
    """Per-row log densities of ``model`` over encoding ``enc`` via one fused numba pass.

    ``compute_dtype`` (e.g. ``np.float32``) runs the row arithmetic in reduced precision while the
    log-sum-exp accumulator and output stay float64; ``None`` keeps the byte-identical float64 path.

    Raises ``ValueError`` if ``model`` is not fusible -- callers should check :func:`fusible` first.
    """
    plan = analyze(model)
    if plan is None:
        from mixle.stats.compute.fused_nested import fused_nested_seq_log_density

        return fused_nested_seq_log_density(model, enc)  # nested scalar tree (raises if not that either)
    data_arrays, param_arrays, _ = _data_and_params(model, plan, enc, compute_dtype)
    logw = np.asarray(getattr(model, "log_w", np.zeros(1)), dtype=np.float64)
    out = np.empty(data_arrays[0].shape[0], dtype=np.float64)
    llbuf = np.empty(plan.num_components, dtype=np.float64)
    _compile(plan)(*data_arrays, *param_arrays, logw, out, llbuf)
    return out


# --- fused E-step (score + responsibilities + per-leaf weighted sufficient statistics, one njit) ----
def fusible_estep(model: Any) -> bool:
    """Return whether ``model`` can use a fused E-step accumulation kernel."""
    plan = analyze(model)
    if plan is None:
        from mixle.stats.compute.fused_nested import fusible_nested  # nested scalar trees fit the E-step too

        return fusible_nested(model)
    hook = {
        "scalar": lambda t: t.acc_stmt,
        "vector": lambda t: t.vec_accumulate,
        "matrix": lambda t: t.mat_accumulate,
        "tabulated": lambda t: t.tab_table,
        "categorical": lambda t: t.cat_table,
    }
    return all(hook[t.kind](t) is not None for t in plan.leaf_templates)


def fused_accumulate(
    model: Any, enc: Any, weights: np.ndarray, return_ll: bool = False, compute_dtype: Any = None
) -> Any:
    """Run one fused E-step and return the sufficient statistic in the estimator's ``value()`` format.

    The whole E-step -- component scoring, responsibility soft-max, and per-leaf weighted-statistic
    accumulation (scalar inline, matrix via BLAS) -- runs in a single nopython pass, then is packed into
    the exact tuple shape ``estimate(nobs, suff_stat)`` expects. With ``return_ll`` the weighted data
    log-likelihood (the posterior normalizer, computed for free in the same pass) is also returned as
    ``(suff_stat, ll)`` so the EM loop can skip a separate scoring pass. Raises ``ValueError`` if not
    fusible.
    """
    plan = analyze(model)
    if plan is None:
        from mixle.stats.compute.fused_nested import fused_nested_accumulate

        return fused_nested_accumulate(model, enc, weights, return_ll=return_ll)  # nested scalar tree
    if not fusible_estep(model):
        raise ValueError("%s is not a fusible E-step (an unsupported leaf)." % type(model).__name__)
    K = plan.num_components
    data_arrays, param_arrays, tab_ctx = _data_and_params(model, plan, enc, compute_dtype)

    # per-leaf accumulator arrays, in the same leaf order the generated signature expects. data_arrays is
    # flattened by arity, so track the running offset to find each leaf's first data array.
    scalar_acc: list[dict[str, np.ndarray]] = []
    matrix_acc: list[tuple[np.ndarray, np.ndarray]] = []
    acc_arrays: list[np.ndarray] = []
    offset = 0
    for i, t in enumerate(plan.leaf_templates):
        if t.kind == "matrix":
            d = data_arrays[offset].shape[1]
            s1 = np.zeros((K, d), dtype=np.float64)
            s2 = np.zeros((K, d, d), dtype=np.float64)
            matrix_acc.append((s1, s2))
            scalar_acc.append({})
            acc_arrays += [s1, s2]
        elif t.kind == "vector":
            d = data_arrays[offset].shape[1]  # (K,D) per-dim weighted statistics
            ad = {an: np.zeros((K, d), dtype=np.float64) for an in t.acc_names}
            scalar_acc.append(ad)
            matrix_acc.append((np.empty(0), np.empty(0)))
            acc_arrays.extend(ad.values())
        elif t.kind == "tabulated":
            width = tab_ctx[i][1] + 1  # max_x + 1; sx is (K,), the histogram (if any) is (K, max_x+1)
            ad = {an: np.zeros((K, width) if an == "hist" else K, dtype=np.float64) for an in t.acc_names}
            scalar_acc.append(ad)
            matrix_acc.append((np.empty(0), np.empty(0)))
            acc_arrays.extend(ad.values())
        elif t.kind == "categorical":
            width = tab_ctx[i][1]  # C categories; the only statistic is the (K, C) weighted count histogram
            ad = {"hist": np.zeros((K, width), dtype=np.float64)}
            scalar_acc.append(ad)
            matrix_acc.append((np.empty(0), np.empty(0)))
            acc_arrays.append(ad["hist"])
        else:
            ad = {an: np.zeros(K, dtype=np.float64) for an in t.acc_names}
            scalar_acc.append(ad)
            matrix_acc.append((np.empty(0), np.empty(0)))
            acc_arrays.extend(ad.values())
        offset += t.arity

    comp_counts = np.zeros(K, dtype=np.float64)
    logw = np.asarray(getattr(model, "log_w", np.zeros(1)), dtype=np.float64)
    llbuf = np.empty(K, dtype=np.float64)
    out_ll = np.zeros(1, dtype=np.float64)
    _compile_estep(plan)(
        *data_arrays,
        *param_arrays,
        np.asarray(weights, dtype=np.float64),
        logw,
        comp_counts,
        *acc_arrays,
        llbuf,
        out_ll,
    )

    def leaf_value(i: int, t: LeafTemplate, k: int) -> Any:
        if t.kind == "matrix":
            s1, s2 = matrix_acc[i]
            return t.to_value((s1[k], s2[k]), float(comp_counts[k]))  # type: ignore[misc]
        if t.kind == "tabulated":
            mn, mx = tab_ctx[i]
            hist_k = scalar_acc[i]["hist"][k] if t.tab_hist else None
            return t.tab_to_value(float(scalar_acc[i]["sx"][k]), float(comp_counts[k]), hist_k, mn, mx)  # type: ignore[misc]
        if t.kind == "categorical":
            enc_i, _ = tab_ctx[i]
            return t.cat_to_value(scalar_acc[i]["hist"][k], enc_i, float(comp_counts[k]))  # type: ignore[misc]
        stats_k = tuple(scalar_acc[i][an][k] for an in t.acc_names)
        if t.wants_minmax:
            mn, mx = tab_ctx[i]
            return t.to_value_g(stats_k, float(comp_counts[k]), mn, mx)  # type: ignore[misc]
        return t.to_value(stats_k, float(comp_counts[k]))  # type: ignore[misc]

    def node_value(k: int) -> Any:
        leaf_vals = [leaf_value(i, t, k) for i, t in enumerate(plan.leaf_templates)]
        return tuple(leaf_vals) if plan.component_is_composite else leaf_vals[0]

    suff = (comp_counts, tuple(node_value(k) for k in range(K))) if plan.is_mixture else node_value(0)
    return (suff, float(out_ll[0])) if return_ll else suff


# --- kernel wiring (used by optimize(..., engine=FUSED_NUMPY_ENGINE) for fusible models) ------------
class FusedKernel:
    """A duck-typed ``Kernel`` backed by the source-generated fused scorer and E-step."""

    def __init__(self, dist: Any, engine: Any, estimator: Any = None) -> None:
        self.dist = dist
        self.engine = engine
        self.estimator = estimator
        # Reduced-precision row arithmetic when the engine declares a non-float64 float dtype (e.g. a
        # ``NumpyEngine(dtype="float32", prefer_fused=True)``); None keeps the byte-identical float64 path.
        # The default fused engine carries dtype=None, so auto-fusion never silently lowers precision.
        edt = getattr(engine, "dtype", None)
        self.compute_dtype = edt if edt is not None and np.dtype(edt) != np.float64 else None
        self.last_ll: float | None = None  # data LL of the last accumulate() pass (posterior normalizer)

    def encode(self, data: Any) -> Any:
        """Encode raw data through the distribution's sequence encoder."""
        return self.dist.dist_to_encoder().seq_encode(data)

    def score(self, enc: Any) -> np.ndarray:
        """Score encoded data with the fused log-density kernel."""
        return fused_seq_log_density(self.dist, getattr(enc, "engine_payload", enc), self.compute_dtype)

    def accumulate(self, enc: Any, weights: Any) -> Any:
        """Accumulate weighted sufficient statistics with the fused E-step kernel."""
        w = np.asarray(self.engine.to_numpy(weights) if hasattr(self.engine, "to_numpy") else weights, dtype=np.float64)
        suff, self.last_ll = fused_accumulate(
            self.dist, getattr(enc, "engine_payload", enc), w, return_ll=True, compute_dtype=self.compute_dtype
        )
        return suff

    def refresh(self, dist: Any) -> None:
        """Refresh the kernel wrapper with a newly estimated distribution."""
        self.dist = dist


class FusedKernelFactory:
    """Build a :class:`FusedKernel` on a numba-capable engine for fusible models, else delegate.

    Scoring needs only :func:`fusible`; estimation also needs the fused E-step (:func:`fusible_estep`),
    so when an estimator is present we require the stronger check. Everything else falls through to the
    declaration/numba/generic factory (the previous behaviour), so registering this is never a regression.
    """

    def __init__(self, fallback: Any = None) -> None:
        self._fallback = fallback

    def _fallback_factory(self) -> Any:
        if self._fallback is None:
            from mixle.stats.compute.kernel import GeneratedNumbaKernelFactory

            self._fallback = GeneratedNumbaKernelFactory()
        return self._fallback

    def build(self, dist: Any, engine: Any, estimator: Any = None) -> Any:
        """Build a fused kernel when supported, otherwise delegate to the fallback factory."""
        ok = fusible_estep(dist) if estimator is not None else fusible(dist)
        if getattr(engine, "supports_numba", False) and ok:
            return FusedKernel(dist, engine, estimator=estimator)
        return self._fallback_factory().build(dist, engine, estimator=estimator)


__all__ = [
    "LeafTemplate",
    "register_leaf_template",
    "analyze",
    "fusible",
    "fusible_estep",
    "fused_seq_log_density",
    "fused_accumulate",
    "FusedKernel",
    "FusedKernelFactory",
    "FusedPlan",
]
