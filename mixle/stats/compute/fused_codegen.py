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

Beyond the templated kinds, coverage is COMPLETE for protocol-bearing factors:

* **chain leaves** (MarkovChain with the Null length model, Dirichlet priors included): scored from
  per-encoding init/transition log tables scatter-built before the row loop, statistics scattered from
  the responsibility matrix in the post pass.
* **bridged factors** (everything else inside a composite -- nested Mixtures, HMMs, nested Composites,
  length-model chains, untemplated leaves like Laplace): scored from a precomputed per-component table
  (each column the factor's own native ``seq_log_density``), estimated by the factor's own accumulator
  driven by the responsibility matrix -- native semantics, priors and all, while the softmax and every
  templated sibling stay in one nopython pass. Registered last; specific templates always outrank it.

Every scorer and E-step has a chunk-parallel prange variant (fixed chunking, fixed-order combine:
bit-stable across reruns and worker counts), honors reduced-precision compute (float32 rows, float64
accumulation), and the scorers optionally take the qlut quantized log-sum-exp (``lse_bits``; error
bounded by the grid half-step, compounding per mixture level on nested trees). The nested scalar-tree
kernels (:mod:`fused_nested`) carry the same contracts.

Principled exclusions, deliberate and documented rather than pending: E-steps always use EXACT exp (a
delta-grid perturbation could flip a monotone-gate accept); bare nested mixtures keep fused_nested's
in-kernel path (faster than bridging); GradLeaf M-steps stay eager torch (torch.compile measured 0.79-
0.93x on CPU -- see the GradLeaf docstring).

Generated kernels are compiled once and disk-cached (see :func:`_njit`), so the compile cost is paid once
per structure *ever*, not per process.
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
    # chain hooks (a Markov-chain factor: RAGGED per-row sequences, already encoded as scatter-ready
    # (row_idx, state_idx) arrays). Scoring precomputes a per-row per-component score table by
    # scatter-adding init/transition log-probabilities BEFORE the row loop (the same precompute slot the
    # matrix kind's BLAS quad forms use), the row fragment is a table lookup, and the E-step accumulates
    # weighted init/transition histograms from the responsibility matrix R in the post pass.
    chain_tables: Callable[[list[Any], Any], tuple[np.ndarray, np.ndarray]] | None = (
        None  # (comps, enc) -> (initT (K,S), transT (K,S,S)) in the ENCODER's state indexing
    )
    chain_to_value: Callable[[np.ndarray, np.ndarray, Any, float], Any] | None = (
        None  # (init_hist_k (S,), trans_hist_k (S,S), enc, count) -> leaf value
    )
    dtype: str = "float64"


_TEMPLATES: list[LeafTemplate] = []


def register_leaf_template(t: LeafTemplate) -> None:
    """Register a leaf template for fused scoring and E-step generation."""
    _TEMPLATES.append(t)


def _template_for(dist: Any, allow_bridge: bool = True) -> LeafTemplate | None:
    for t in _TEMPLATES:
        if (allow_bridge or t.kind != "bridge") and t.matches(dist):
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


def _markov_chain_data(enc: Any) -> tuple[np.ndarray, ...]:
    """The chain encoding's scatter arrays: (init row idx, init state, trans row idx, from, to)."""
    _, idx0, idx1, init_x, prev_x, next_x, _, _ = enc
    as_i8 = lambda a: np.ascontiguousarray(np.asarray(a, dtype=np.int64).reshape(-1))  # noqa: E731
    return (as_i8(idx0), as_i8(init_x), as_i8(idx1), as_i8(prev_x), as_i8(next_x))


def _markov_chain_tables(comps: list[Any], enc: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-component (K,S) init and (K,S,S) transition log tables in the ENCODER's state order.

    Exactly replicates MarkovChainDistribution.seq_log_density's lookup: each component maps the
    encoder's states through its own key_map (0 = out-of-support default) into its init/transition
    log vectors, with the component's log1p(default) normalizer folded into every entry.
    """
    inv_key_map = enc[6]
    S = len(inv_key_map)
    init_t = np.empty((len(comps), max(S, 1)), dtype=np.float64)
    trans_t = np.empty((len(comps), max(S, 1), max(S, 1)), dtype=np.float64)
    for k, c in enumerate(comps):
        loc = np.asarray([c.key_map.get(u, 0) for u in inv_key_map], dtype=np.int64)
        if S:
            init_t[k] = c.init_log_pvec[loc] - c.log1p_dv
            dense = np.asarray(c.trans_log_pvec.toarray(), dtype=np.float64)
            trans_t[k] = dense[np.ix_(loc, loc)] - c.log1p_dv
        else:  # every sequence empty: no states, no scatter entries, contribution 0
            init_t[k] = 0.0
            trans_t[k] = 0.0
    return init_t, trans_t


def _markov_chain_to_value(init_hist_k: np.ndarray, trans_hist_k: np.ndarray, enc: Any, count: float) -> Any:
    """Pack the weighted histograms into the accumulator's native (init map, trans map, len value) tuple,
    dropping zeros exactly as MarkovChainAccumulator.seq_update does."""
    inv_key_map = enc[6]
    init_map = {inv_key_map[s]: float(init_hist_k[s]) for s in range(len(inv_key_map)) if init_hist_k[s] != 0.0}
    trans_map: dict[Any, dict[Any, float]] = {}
    nz_a, nz_b = np.nonzero(trans_hist_k)
    for a, b in zip(nz_a.tolist(), nz_b.tolist()):
        trans_map.setdefault(inv_key_map[a], {})[inv_key_map[b]] = float(trans_hist_k[a, b])
    return (init_map, trans_map, None)  # None: the fusibility guard requires the Null length model


def _markov_chain_matches(d: Any) -> bool:
    """Fast-path chain configs: a MarkovChainDistribution whose length model is the Null distribution
    (its seq_log_density contributes exactly zero, so the tables carry the whole density). A Dirichlet
    prior is fine -- the sufficient statistics are the standard count maps either way, and the
    estimator applies its prior at estimate() exactly as on the host path (parity-tested). Chains
    with a REAL length distribution fall through to the bridge template (native scoring/estimation,
    length model included)."""
    return (
        type(d).__name__ == "MarkovChainDistribution"
        and type(getattr(d, "len_dist", None)).__name__ == "NullDistribution"
    )


register_leaf_template(
    LeafTemplate(
        name="markovchain",
        matches=_markov_chain_matches,
        data=_markov_chain_data,
        params=lambda comps: {},
        arity=5,
        kind="chain",
        chain_tables=_markov_chain_tables,
        chain_to_value=_markov_chain_to_value,
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


def _bridge_matches(d: Any) -> bool:
    """ANY factor speaking the sequence protocol, scored and estimated through its OWN native
    machinery (seq_log_density / accumulator seq_update) -- so every configuration the factor itself
    supports (priors, length models, arbitrary combinators) is supported here. Registered LAST, so
    every specific template outranks the bridge and this is purely the completion of coverage: a
    composite with ANY protocol-bearing factor now fuses, with the un-templated factors paying
    exactly their host cost while the softmax, responsibilities, and templated siblings stay in one
    nopython pass."""
    return (
        hasattr(d, "seq_log_density")
        and hasattr(d, "dist_to_encoder")
        and hasattr(d, "estimator")
        and callable(getattr(d, "estimator", None))
    )


register_leaf_template(
    LeafTemplate(
        name="bridge",
        matches=_bridge_matches,
        data=lambda enc: (),
        params=lambda comps: {},
        arity=0,
        kind="bridge",
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

    @property
    def has_bridge(self) -> bool:
        """Whether any factor is a BRIDGED combinator (scored via its own native seq_log_density; its
        sufficient statistics collected by its own accumulator, weighted by the responsibility matrix)."""
        return any(t.kind == "bridge" for t in self.leaf_templates)

    @property
    def needs_responsibilities(self) -> bool:
        """Whether the E-step must materialize the (n, K) responsibility matrix R (matrix BLAS
        accumulation and chain post-pass scatters read it after the row loop; bridged factors hand it
        back to the caller for their native weighted updates)."""
        return any(t.kind in ("matrix", "chain", "bridge") for t in self.leaf_templates)


def _node_factors(node: Any) -> list[Any] | None:
    """The leaf factors of a fusible *node* (an exact Composite, or a templated bare leaf), else None.

    The bare-node check deliberately excludes the bridge template: a bare Mixture-of-Mixtures (no
    Composite anywhere) belongs to fused_nested's in-kernel nested-tree path, which is faster than
    bridging. Bridge factors exist for combinators sitting INSIDE a composite, where no in-kernel
    path exists at all."""
    if type(node).__name__ == "CompositeDistribution":
        return list(node.dists)
    if _template_for(node, allow_bridge=False) is not None:
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
        "markovchain": stats.MarkovChainDistribution({"a": 1.0}, {"a": {"a": 1.0}}),
        "bridge": stats.GaussianDistribution(0.0, 1.0),  # params() ignores it (bridge has no static params)
    }[t.name]


def _argmap(i: int, t: LeafTemplate) -> dict[str, str]:
    return {pn: f"p{i}_{pn}" for pn in sorted(t.params([_dummy(t)]).keys())}


def _data_names(i: int, t: LeafTemplate) -> list[str]:
    return [f"x{i}_{j}" for j in range(t.arity)]


# --- code generation ------------------------------------------------------------------------------
_COMPILED: dict[tuple, Callable] = {}
_ESTEP_COMPILED: dict[tuple, Callable] = {}

# Parallel (prange) kernel policy. Chunk count is a PURE function of n -- never of the worker count --
# and per-chunk partials combine in fixed order, so parallel results are bit-identical across runs and
# across NUMBA_NUM_THREADS (probe: fingerprints equal at 1/4/8 threads; 4.6x at 8 threads on a 2M-row
# 8-component E-step, ~1.0x at 1 thread; 3.1x end-to-end through fused_accumulate at 400k rows). Parallel
# output differs from the SEQUENTIAL kernel by float re-association: chunk-boundary reassociation in the
# E-step reductions (~1e-8 relative measured), and 1-2 ULP in the scorer -- the parallel scorer has no
# cross-row reduction, but sequential and parallel are DIFFERENT fastmath binaries, and LLVM vectorizes
# each loop nest differently (measured max 4.7e-16 relative). Bit-identity holds within each variant,
# never across variants.
_PARALLEL_MIN_OBS = 262_144  # below this the fork/join overhead eats the win
_PARALLEL_MAX_CHUNKS = 256
_PARALLEL_CHUNK_TARGET = 16_384


def _n_chunks(n: int) -> int:
    return max(1, min(_PARALLEL_MAX_CHUNKS, n // _PARALLEL_CHUNK_TARGET))


def _auto_parallel(n: int) -> bool:
    if n < _PARALLEL_MIN_OBS:
        return False
    import numba

    return numba.get_num_threads() > 1


def _emit(plan: FusedPlan, acc_suffix: str = "") -> dict[str, list[str]]:
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
        accmap = {an: f"a{i}_{an}{acc_suffix}" for an in t.acc_names}
        frag["acc_args"].extend(f"a{i}_{an}" for an in t.acc_names)  # signature names never carry the suffix
        if t.kind == "vector":
            frag["row"].extend(t.vec_row(i, amap))  # type: ignore[misc]
            frag["acc"].extend(t.vec_accumulate(i, accmap))  # type: ignore[misc]
        elif t.kind == "tabulated":
            frag["param_args"].append(f"tab{i}")  # the (K, max+1) log-pmf table, looked up by int(x)
            frag["row"].append(f"acc += tab{i}[k, int(x{i}_0[i])]")
            frag["acc"].append(f"{accmap['sx']}[k] += r * x{i}_0[i]")
            if t.tab_hist:
                frag["acc"].append(f"{accmap['hist']}[k, int(x{i}_0[i])] += r")  # weighted count histogram
        elif t.kind == "bridge":
            # scored from a precomputed (n, K) table (each column: the factor's OWN seq_log_density for
            # that component); no in-kernel statistics -- the wrapper drives the factor's native
            # accumulator with the R column instead.
            frag["param_args"].append(f"bt{i}")
            frag["row"].append(f"acc += bt{i}[i, k]")
        elif t.kind == "chain":
            frag["param_args"] += [f"cinit{i}", f"ctrans{i}"]  # (K,S) init and (K,S,S) transition log tables
            frag["acc_args"] += [f"ih{i}", f"th{i}"]  # (K,S) init and (K,S,S) transition weighted histograms
            # per-row per-component chain scores, scatter-built once before the row loop:
            # data arrays are x{i}_0=init row idx, x{i}_1=init state, x{i}_2=trans row idx, x{i}_3/4=from/to
            frag["precompute"].extend(
                [
                    f"cs{i} = np.zeros((n, kc))",
                    f"for t{i} in range(x{i}_0.shape[0]):",
                    "    for k in range(kc):",
                    f"        cs{i}[x{i}_0[t{i}], k] += cinit{i}[k, x{i}_1[t{i}]]",
                    f"for t{i} in range(x{i}_2.shape[0]):",
                    "    for k in range(kc):",
                    f"        cs{i}[x{i}_2[t{i}], k] += ctrans{i}[k, x{i}_3[t{i}], x{i}_4[t{i}]]",
                ]
            )
            frag["row"].append(f"acc += cs{i}[i, k]")
            frag["post"].extend(
                [
                    f"for t{i} in range(x{i}_0.shape[0]):",
                    "    for k in range(kc):",
                    f"        ih{i}[k, x{i}_1[t{i}]] += R[x{i}_0[t{i}], k]",
                    f"for t{i} in range(x{i}_2.shape[0]):",
                    "    for k in range(kc):",
                    f"        th{i}[k, x{i}_3[t{i}], x{i}_4[t{i}]] += R[x{i}_2[t{i}], k]",
                ]
            )
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
import stat  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402


def _default_cache_dir() -> str:
    """A per-user cache directory. A single shared ``/tmp/mixle_fused_cache`` lets another user pre-place the
    module the loader imports and ``exec``s -- arbitrary code execution -- so isolate the cache by uid."""
    suffix = f"_{os.getuid()}" if hasattr(os, "getuid") else ""
    return os.path.join(tempfile.gettempdir(), f"mixle_fused_cache{suffix}")


_CACHE_DIR = os.environ.get("MIXLE_FUSED_CACHE_DIR") or _default_cache_dir()
_NJIT_LOCK = threading.Lock()


def _owned_privately(path: str, *, require_dir: bool) -> bool:
    """True iff ``path`` is a real dir/file we own that no other user can write (no symlink, no g/o-write).

    ``lstat`` (not ``stat``) so a symlink can never pass as its -- possibly attacker-owned -- target.
    """
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if require_dir and not stat.S_ISDIR(info.st_mode):
        return False
    if not require_dir and not stat.S_ISREG(info.st_mode):
        return False
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        return False
    return not (info.st_mode & (stat.S_IWGRP | stat.S_IWOTH))


def _private_cache_dir() -> str | None:
    """Return the cache dir once it is guaranteed private to us (0700, we own it), else ``None``.

    ``None`` tells the caller to compile in memory only -- correct result, no disk cache, no chance of
    importing code from a directory another user can write.
    """
    try:
        os.makedirs(_CACHE_DIR, mode=0o700, exist_ok=True)
        info = os.lstat(_CACHE_DIR)
        # Heal a directory we own that an older (insecure) build may have created group/other-writable.
        own = not hasattr(os, "getuid") or info.st_uid == os.getuid()
        if stat.S_ISDIR(info.st_mode) and own and (info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)):
            os.chmod(_CACHE_DIR, 0o700)
    except OSError:
        return None
    return _CACHE_DIR if _owned_privately(_CACHE_DIR, require_dir=True) else None


def _njit(src: str, fname: str, parallel: bool = False) -> Callable:
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
            cache_dir = _private_cache_dir()
            if cache_dir is None:
                raise OSError("fused cache dir is not private to this user; compiling in memory")
            path = os.path.join(cache_dir, modname + ".py")
            # Write our own source unless a file we privately own is already there. Never import a
            # pre-existing file we do not own: it could be attacker-planted (arbitrary code execution).
            # os.replace over a symlink replaces the link itself, so this cannot be redirected outside.
            if not _owned_privately(path, require_dir=False):
                deco = (
                    "@numba.njit(fastmath=True, cache=True, parallel=True)"
                    if parallel
                    else "@numba.njit(fastmath=True, cache=True)"
                )
                module_src = f"import numpy as np\nimport numba\n\n\n{deco}\n{src}\n"
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
            ns: dict[str, Any] = {"np": np, "numba": numba}
            exec(src, ns)  # noqa: S102 -- generated from fixed templates, no user input
            return numba.njit(fastmath=True, parallel=parallel)(ns[fname])


def _score_body(indent: str, row_lines: list[str], llbuf_name: str) -> list[str]:
    """The per-row score+log-sum-exp block shared by the sequential and parallel scorers."""
    lines = [f"{indent}for k in range(kc):", f"{indent}    acc = logw[k]"]
    lines += [f"{indent}    {rt}" for rt in row_lines]
    lines += [
        f"{indent}    {llbuf_name}[k] = acc",
        f"{indent}m = {llbuf_name}[0]",
        f"{indent}for k in range(1, kc):",
        f"{indent}    if {llbuf_name}[k] > m:",
        f"{indent}        m = {llbuf_name}[k]",
        f"{indent}s = 0.0",
        f"{indent}for k in range(kc):",
        f"{indent}    s += np.exp({llbuf_name}[k] - m)",
        f"{indent}out[i] = (m + np.log(s)) if m > -np.inf else -np.inf",  # all components -inf (out of support)
    ]
    return lines


def _score_tail_quantized(indent: str, row_lines: list[str], llbuf_name: str) -> list[str]:
    """The per-row block with the log-sum-exp EXP replaced by one LUT gather over a delta-grid --
    mixle.engines.qlut.quantized_logsumexp's exact semantics (round to grid, clamp the deep tail into
    the bottom bin), emitted inline. Error bound: half a grid step (lse_error_bound), plus the
    negligible clipped-tail term the qlut docstring quantifies."""
    lines = [f"{indent}for k in range(kc):", f"{indent}    acc = logw[k]"]
    lines += [f"{indent}    {rt}" for rt in row_lines]
    lines += [
        f"{indent}    {llbuf_name}[k] = acc",
        f"{indent}m = {llbuf_name}[0]",
        f"{indent}for k in range(1, kc):",
        f"{indent}    if {llbuf_name}[k] > m:",
        f"{indent}        m = {llbuf_name}[k]",
        f"{indent}s = 0.0",
        f"{indent}for k in range(kc):",
        f"{indent}    qi = int(np.rint(({llbuf_name}[k] - m) * lse_inv_delta)) + lse_levm1",
        f"{indent}    if qi < 0:",
        f"{indent}        qi = 0",  # scores more than `span` below the max clip into the bottom bin
        f"{indent}    s += lse_lut[qi]",
        f"{indent}out[i] = (m + np.log(s)) if m > -np.inf else -np.inf",
    ]
    return lines


def _compile(plan: FusedPlan, parallel: bool = False, quantized_lse: bool = False) -> Callable:
    cached = _COMPILED.get((plan.signature, parallel, quantized_lse))
    if cached is not None:
        return cached
    f = _emit(plan)
    data_args = f["data_args"]
    lse_args = ["lse_lut", "lse_inv_delta", "lse_levm1"] if quantized_lse else []
    tail = _score_tail_quantized if quantized_lse else _score_body
    if not parallel:
        args = ", ".join(data_args + f["param_args"] + lse_args + ["logw", "out", "llbuf"])
        lines = ["def _fused(%s):" % args, "    n = out.shape[0]", "    kc = logw.shape[0]"]
        lines += ["    " + ln for ln in f["precompute"]]
        lines += ["    for i in range(n):"]
        lines += tail("        ", f["row"], "llbuf")
        fn = _njit("\n".join(lines), "_fused")
    else:
        # prange over fixed chunks; out[i] rows are disjoint and each row's arithmetic is identical to
        # the sequential kernel's, so the parallel scorer is BIT-IDENTICAL to it (asserted in tests).
        args = ", ".join(data_args + f["param_args"] + lse_args + ["logw", "out", "n_chunks"])
        lines = ["def _fused_par(%s):" % args, "    n = out.shape[0]", "    kc = logw.shape[0]"]
        lines += ["    " + ln for ln in f["precompute"]]
        lines += [
            "    step = (n + n_chunks - 1) // n_chunks",
            "    for c in numba.prange(n_chunks):",
            "        llbuf_c = np.empty(kc)",
            "        for i in range(c * step, min(n, (c + 1) * step)):",
        ]
        lines += tail("            ", f["row"], "llbuf_c")
        fn = _njit("\n".join(lines), "_fused_par", parallel=True)
    _COMPILED[(plan.signature, parallel, quantized_lse)] = fn
    return fn


def _compile_estep(plan: FusedPlan, parallel: bool = False) -> Callable:
    cached = _ESTEP_COMPILED.get((plan.signature, parallel))
    if cached is not None:
        return cached
    f = _emit(plan, acc_suffix="_c" if parallel else "")
    data_args = f["data_args"]
    if not parallel:
        extra = ["R"] if plan.has_bridge else []
        args = ", ".join(
            data_args + f["param_args"] + ["weights", "logw", "comp_counts", *f["acc_args"], *extra, "llbuf", "out_ll"]
        )
        lines = ["def _estep(%s):" % args, "    n = weights.shape[0]", "    kc = logw.shape[0]"]
        if plan.needs_responsibilities and not plan.has_bridge:
            lines.append("    R = np.empty((n, kc))")  # read back by the matrix BLAS / chain post passes
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
        if plan.needs_responsibilities:
            lines.append("            R[i, k] = r")
        lines += ["    " + ln for ln in f["post"]]
        fn = _njit("\n".join(lines), "_estep")
    else:
        # Chunk-parallel E-step. Every accumulator written inside the responsibility loop gains a leading
        # n_chunks axis (allocated by the caller); chunk c only ever touches row c, so there are NO
        # parallel-region reductions at all -- numba's parfor reduction analysis has nothing to reorder,
        # and the caller's fixed-order sum over the chunk axis makes results bit-stable across runs and
        # worker counts. Matrix leaves are untouched here: their statistics come from the sequential BLAS
        # post-pass over the (row-disjoint) responsibility matrix R, exactly as in the sequential kernel.
        chunked = [
            f"a{i}_{an}"
            for i, lt in enumerate(plan.leaf_templates)
            if lt.kind not in ("matrix", "chain")
            for an in lt.acc_names
        ]
        extra = ["R"] if plan.has_bridge else []
        args = ", ".join(
            data_args
            + f["param_args"]
            + ["weights", "logw", "comp_counts", *f["acc_args"], *extra, "out_ll", "n_chunks"]
        )
        lines = ["def _estep_par(%s):" % args, "    n = weights.shape[0]", "    kc = logw.shape[0]"]
        if plan.needs_responsibilities and not plan.has_bridge:
            lines.append("    R = np.empty((n, kc))")
        lines += ["    " + ln for ln in f["precompute"]]
        lines += [
            "    step = (n + n_chunks - 1) // n_chunks",
            "    for c in numba.prange(n_chunks):",
            "        llbuf_c = np.empty(kc)",
        ]
        lines += [f"        {nm}_c = {nm}[c]" for nm in chunked]
        lines += [
            "        for i in range(c * step, min(n, (c + 1) * step)):",
            "            wi = weights[i]",
            "            for k in range(kc):",
            "                acc = logw[k]",
        ]
        lines += ["                " + rt for rt in f["row"]]
        lines += [
            "                llbuf_c[k] = acc",
            "            m = llbuf_c[0]",
            "            for k in range(1, kc):",
            "                if llbuf_c[k] > m:",
            "                    m = llbuf_c[k]",
            "            s = 0.0",
            "            for k in range(kc):",
            "                s += np.exp(llbuf_c[k] - m)",
            "            out_ll[c] += wi * (m + np.log(s))",
            "            for k in range(kc):",
            "                r = np.exp(llbuf_c[k] - m) / s * wi",
            "                comp_counts[c, k] += r",
        ]
        lines += ["                " + st for st in f["acc"]]
        if plan.needs_responsibilities:
            lines.append("                R[i, k] = r")
        lines += ["    " + ln for ln in f["post"]]
        fn = _njit("\n".join(lines), "_estep_par", parallel=True)
    _ESTEP_COMPILED[(plan.signature, parallel)] = fn
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
    n_rows: int | None = None  # chain data arrays are transition-length, so rows come from another source
    for i, t in enumerate(plan.leaf_templates):
        arrs = t.data(factor_encs[i])
        data_arrays.extend(arrs)  # arity arrays, flattened in leaf order
        comps_i = [factor_lists[k][i] for k in range(plan.num_components)]
        pdict = t.params(comps_i)
        param_arrays.extend(np.ascontiguousarray(pdict[pn]) for pn in sorted(pdict.keys()))
        if t.kind == "bridge":
            cols = [np.asarray(c.seq_log_density(factor_encs[i]), dtype=np.float64) for c in comps_i]
            table = np.ascontiguousarray(np.stack(cols, axis=1))  # (n, K)
            if n_rows is None:
                n_rows = int(table.shape[0])
            param_arrays.append(table)
            tab_ctx[i] = (factor_encs[i], 0)  # the factor's own encoding, for the wrapper's native updates
        if t.kind == "chain":
            if n_rows is None:
                n_rows = int(factor_encs[i][0])  # the chain encoding carries the row count directly
            init_table, trans_table = t.chain_tables(comps_i, factor_encs[i])  # type: ignore[misc]
            param_arrays.append(np.ascontiguousarray(init_table))
            param_arrays.append(np.ascontiguousarray(trans_table))
            tab_ctx[i] = (factor_encs[i], init_table.shape[1])  # (encoding for to_value, S states)
        elif n_rows is None:
            n_rows = int(arrs[0].shape[0])
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
    return data_arrays, param_arrays, tab_ctx, int(n_rows if n_rows is not None else 0)


def fused_seq_log_density(
    model: Any,
    enc: Any,
    compute_dtype: Any = None,
    parallel: bool | None = None,
    lse_bits: int | None = None,
    lse_span: float = 24.0,
) -> np.ndarray:
    """Per-row log densities of ``model`` over encoding ``enc`` via one fused numba pass.

    ``compute_dtype`` (e.g. ``np.float32``) runs the row arithmetic in reduced precision while the
    log-sum-exp accumulator and output stay float64; ``None`` keeps the byte-identical float64 path.

    ``parallel``: ``None`` auto-engages the chunked prange scorer for large inputs (>= 262144 rows and
    more than one numba worker); ``True``/``False`` force it. Rows are disjoint (no cross-row
    reduction), so the parallel scorer is bit-identical across runs and worker counts; against the
    sequential kernel it agrees to 1-2 ULP (different fastmath binaries vectorize differently --
    measured max 4.7e-16 relative), not bit-for-bit.

    ``lse_bits`` (OPT-IN, banded): replace the per-row log-sum-exp's exp calls with one gather from a
    ``2**lse_bits``-entry table over a ``lse_span/2**lse_bits`` grid -- the qlut quantized-LSE kernel
    inlined. Per-row error is bounded by ``mixle.engines.qlut.lse_error_bound(lse_bits, lse_span)``
    (12 bits ~ 2.9e-3 bound, ~6e-5 measured), so this is for scoring/audit paths whose compute band
    tolerates a delta-grid; E-steps deliberately keep exact exp (a perturbed objective could flip
    monotone-gate accepts). ``None`` (default) is the exact path, byte-identical to before.

    Raises ``ValueError`` if ``model`` is not fusible -- callers should check :func:`fusible` first.
    """
    plan = analyze(model)
    if plan is None:
        from mixle.stats.compute.fused_nested import fused_nested_seq_log_density

        # nested scalar tree (raises if not that either); same dtype/parallel/quantized-LSE contract
        # (nested error bound compounds per mixture level -- see the nested wrapper's docstring)
        return fused_nested_seq_log_density(
            model, enc, compute_dtype=compute_dtype, parallel=parallel, lse_bits=lse_bits, lse_span=lse_span
        )
    data_arrays, param_arrays, _, n = _data_and_params(model, plan, enc, compute_dtype)
    logw = np.asarray(getattr(model, "log_w", np.zeros(1)), dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    if parallel is None:
        parallel = _auto_parallel(n)
    lse_extra: tuple = ()
    quantized = lse_bits is not None
    if quantized:
        if not 1 <= int(lse_bits) <= 24:
            raise ValueError(f"need 1 <= lse_bits <= 24, got {lse_bits}")
        if lse_span <= 0:
            raise ValueError(f"lse_span must be positive, got {lse_span}")
        levels = 1 << int(lse_bits)
        delta = float(lse_span) / levels
        table = np.exp((np.arange(levels, dtype=np.float64) - (levels - 1)) * delta)
        lse_extra = (table, 1.0 / delta, levels - 1)
    if parallel:
        _compile(plan, parallel=True, quantized_lse=quantized)(
            *data_arrays, *param_arrays, *lse_extra, logw, out, _n_chunks(n)
        )
    else:
        llbuf = np.empty(plan.num_components, dtype=np.float64)
        _compile(plan, quantized_lse=quantized)(*data_arrays, *param_arrays, *lse_extra, logw, out, llbuf)
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
        "chain": lambda t: t.chain_to_value,
        "bridge": lambda t: t.matches,  # support == matching: statistics come from the factor's own accumulator
    }
    return all(hook[t.kind](t) is not None for t in plan.leaf_templates)


def fused_accumulate(
    model: Any,
    enc: Any,
    weights: np.ndarray,
    return_ll: bool = False,
    compute_dtype: Any = None,
    parallel: bool | None = None,
) -> Any:
    """Run one fused E-step and return the sufficient statistic in the estimator's ``value()`` format.

    The whole E-step -- component scoring, responsibility soft-max, and per-leaf weighted-statistic
    accumulation (scalar inline, matrix via BLAS) -- runs in a single nopython pass, then is packed into
    the exact tuple shape ``estimate(nobs, suff_stat)`` expects. With ``return_ll`` the weighted data
    log-likelihood (the posterior normalizer, computed for free in the same pass) is also returned as
    ``(suff_stat, ll)`` so the EM loop can skip a separate scoring pass. Raises ``ValueError`` if not
    fusible.

    ``parallel``: ``None`` auto-engages the chunked prange E-step for large inputs (>= 262144 rows and
    more than one numba worker); ``True``/``False`` force it. Chunk boundaries are a pure function of n
    and chunk partials combine in fixed order, so parallel results are bit-identical across runs AND
    across worker counts; they differ from the sequential kernel only by float re-association across
    chunk boundaries (~1e-7 relative). Matrix leaves accumulate through the same sequential BLAS
    post-pass in both variants.
    """
    plan = analyze(model)
    if plan is None:
        from mixle.stats.compute.fused_nested import fused_nested_accumulate

        return fused_nested_accumulate(
            model, enc, weights, return_ll=return_ll, compute_dtype=compute_dtype, parallel=parallel
        )  # nested scalar tree
    if not fusible_estep(model):
        raise ValueError("%s is not a fusible E-step (an unsupported leaf)." % type(model).__name__)
    K = plan.num_components
    data_arrays, param_arrays, tab_ctx, n = _data_and_params(model, plan, enc, compute_dtype)
    if parallel is None:
        parallel = _auto_parallel(n)
    nc = _n_chunks(n) if parallel else 0
    chunk = (lambda shape: (nc, *shape)) if parallel else (lambda shape: shape)

    # per-leaf accumulator arrays, in the same leaf order the generated signature expects. data_arrays is
    # flattened by arity, so track the running offset to find each leaf's first data array.
    scalar_acc: list[dict[str, np.ndarray]] = []
    matrix_acc: list[tuple[np.ndarray, np.ndarray]] = []
    chain_acc: list[tuple[np.ndarray, np.ndarray]] = []
    acc_arrays: list[np.ndarray] = []
    offset = 0
    for i, t in enumerate(plan.leaf_templates):
        if t.kind == "bridge":  # no in-kernel statistics; arity 0
            scalar_acc.append({})
            matrix_acc.append((np.empty(0), np.empty(0)))
            chain_acc.append((np.empty(0), np.empty(0)))
            continue
        if t.kind == "chain":
            S = tab_ctx[i][1]
            # written by the sequential post-pass over R in BOTH kernel variants, so never chunked
            ih = np.zeros((K, S), dtype=np.float64)
            th = np.zeros((K, S, S), dtype=np.float64)
            chain_acc.append((ih, th))
            scalar_acc.append({})
            matrix_acc.append((np.empty(0), np.empty(0)))
            acc_arrays += [ih, th]
            offset += t.arity
            continue
        chain_acc.append((np.empty(0), np.empty(0)))
        if t.kind == "matrix":
            d = data_arrays[offset].shape[1]
            s1 = np.zeros((K, d), dtype=np.float64)
            s2 = np.zeros((K, d, d), dtype=np.float64)
            matrix_acc.append((s1, s2))
            scalar_acc.append({})
            acc_arrays += [s1, s2]
        elif t.kind == "vector":
            d = data_arrays[offset].shape[1]  # (K,D) per-dim weighted statistics
            ad = {an: np.zeros(chunk((K, d)), dtype=np.float64) for an in t.acc_names}
            scalar_acc.append(ad)
            matrix_acc.append((np.empty(0), np.empty(0)))
            acc_arrays.extend(ad.values())
        elif t.kind == "tabulated":
            width = tab_ctx[i][1] + 1  # max_x + 1; sx is (K,), the histogram (if any) is (K, max_x+1)
            ad = {
                an: np.zeros(chunk((K, width)) if an == "hist" else chunk((K,)), dtype=np.float64) for an in t.acc_names
            }
            scalar_acc.append(ad)
            matrix_acc.append((np.empty(0), np.empty(0)))
            acc_arrays.extend(ad.values())
        elif t.kind == "categorical":
            width = tab_ctx[i][1]  # C categories; the only statistic is the (K, C) weighted count histogram
            ad = {"hist": np.zeros(chunk((K, width)), dtype=np.float64)}
            scalar_acc.append(ad)
            matrix_acc.append((np.empty(0), np.empty(0)))
            acc_arrays.append(ad["hist"])
        else:
            ad = {an: np.zeros(chunk((K,)), dtype=np.float64) for an in t.acc_names}
            scalar_acc.append(ad)
            matrix_acc.append((np.empty(0), np.empty(0)))
            acc_arrays.extend(ad.values())
        offset += t.arity

    logw = np.asarray(getattr(model, "log_w", np.zeros(1)), dtype=np.float64)
    bridge_R = [np.empty((n, K), dtype=np.float64)] if plan.has_bridge else []
    if parallel:
        comp_counts = np.zeros((nc, K), dtype=np.float64)
        out_ll = np.zeros(nc, dtype=np.float64)
        _compile_estep(plan, parallel=True)(
            *data_arrays,
            *param_arrays,
            np.asarray(weights, dtype=np.float64),
            logw,
            comp_counts,
            *acc_arrays,
            *bridge_R,
            out_ll,
            nc,
        )
        # fixed-order combine over the chunk axis (numpy's single-threaded pairwise reduce): the SAME
        # partials in the SAME order regardless of how many workers computed them -> bit-stable results.
        comp_counts = comp_counts.sum(axis=0)
        for i, t in enumerate(plan.leaf_templates):
            if t.kind != "matrix":
                for an in list(scalar_acc[i]):
                    scalar_acc[i][an] = scalar_acc[i][an].sum(axis=0)
    else:
        comp_counts = np.zeros(K, dtype=np.float64)
        llbuf = np.empty(K, dtype=np.float64)
        out_ll = np.zeros(1, dtype=np.float64)
        _compile_estep(plan)(
            *data_arrays,
            *param_arrays,
            np.asarray(weights, dtype=np.float64),
            logw,
            comp_counts,
            *acc_arrays,
            *bridge_R,
            llbuf,
            out_ll,
        )

    bridge_values: dict[int, list[Any]] = {}
    if plan.has_bridge:
        # Each bridged factor's sufficient statistics come from its OWN accumulator, weighted by the
        # responsibility column the kernel just computed -- byte-for-byte the host mixture E-step's
        # semantics for that factor (priors and all), with everything else still fused.
        factor_lists = _component_factor_lists(model, plan)
        for i, t in enumerate(plan.leaf_templates):
            if t.kind != "bridge":
                continue
            enc_i = tab_ctx[i][0]
            vals = []
            for k in range(K):
                factor_k = factor_lists[k][i]
                acc = factor_k.estimator().accumulator_factory().make()
                acc.seq_update(enc_i, np.ascontiguousarray(bridge_R[0][:, k]), factor_k)
                vals.append(acc.value())
            bridge_values[i] = vals

    def leaf_value(i: int, t: LeafTemplate, k: int) -> Any:
        if t.kind == "bridge":
            return bridge_values[i][k]
        if t.kind == "chain":
            ih, th = chain_acc[i]
            enc_i, _ = tab_ctx[i]
            return t.chain_to_value(ih[k], th[k], enc_i, float(comp_counts[k]))  # type: ignore[misc]
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
    return (suff, float(out_ll.sum())) if return_ll else suff


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
