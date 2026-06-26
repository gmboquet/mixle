"""Source-generated *fused* numba kernels for composite / mixture trees.

The generated-numba kernels in :mod:`declarations` lower one leaf at a time: each leaf materializes its
sufficient statistic in numpy and runs its own row loop, so a composite or mixture pays a Python<->C
boundary crossing and an intermediate allocation *per factor and per component*. For a deep model of
cheap leaves that overhead dominates -- numpy itself is multi-pass for the same reason.

This module instead emits a SINGLE ``@njit`` function for the whole model structure: one pass over the
rows with the composite sum and the mixture log-sum-exp in nopython registers, no per-factor allocation.

Two leaf flavours, each using the right primitive (numba supports BOTH a SIMD-vectorized scalar loop and
BLAS via ``@``/``np.dot`` in nopython):

* **scalar leaves** (Gaussian, Exponential, ...): univariate, O(1) per element. Their scalar log-density
  is inlined into the row loop -- numba SIMD-vectorizes it, and fusing every factor into one pass with no
  intermediate arrays beats numpy's multi-pass evaluation (~1.2-2.7x, growing with depth).
* **matrix leaves** (MultivariateGaussian, ...): the quadratic form ``(x-mu)' P (x-mu)`` and the E-step's
  weighted Gram ``X' (r .* X)`` are matmuls -- emitted as numba ``@`` (BLAS). Fusing the score, the
  responsibility soft-max and the BLAS accumulation into one njit beats numpy's per-component dispatch
  (~5x on a GMM E-step). A scalar triple-loop here would be ~15x SLOWER -- never hand-roll a matmul.

When a model mixes the two, the matrix quad-forms are precomputed via BLAS before the row loop, the row
loop scores + forms responsibilities (storing them only when a matrix leaf needs them downstream), and the
matrix sufficient statistics are accumulated via BLAS after it. Pure-scalar models never materialize the
responsibility matrix, so their fast path is byte-for-byte the original one.

Anything with a leaf that has no template (e.g. Categorical) -> :func:`fusible` is False -> numpy.
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

    ``kind="scalar"`` leaves use ``expr`` / ``acc_stmt`` (inlined scalar work over a 1-D data array).
    ``kind="matrix"`` leaves use the ``mat_emit_*`` hooks (a 2-D data array + BLAS precompute/accumulate).
    ``params`` stacks the leaf's parameters across the K mixture components; ``to_value`` packs a
    component's accumulated statistics into the estimator's ``value()`` tuple.
    """

    name: str
    matches: Callable[[Any], bool]
    data: Callable[[Any], np.ndarray]  # leaf encoding -> data array ((N,) scalar / (N,D) matrix)
    params: Callable[[list[Any]], dict[str, np.ndarray]]  # K leaf dists -> {pname: (K,...) array}
    to_value: Callable[[tuple, float], tuple] | None = None  # (stats for comp k, count_k) -> leaf value
    kind: str = "scalar"
    # scalar hooks
    expr: Callable[[str, dict[str, str]], str] | None = None  # (value_var, {pname: arg}) -> scoring expr
    acc_names: tuple[str, ...] = ()  # per-leaf weighted-statistic accumulator arrays, shape (K,)
    acc_stmt: Callable[[str, dict[str, str], str], str] | None = None  # (value_var, {acc: arg}, resp) -> stmts
    # matrix hooks (emit numba source lines; leaf index `i`, {pname: arg} map)
    mat_precompute: Callable[[int, dict[str, str]], list[str]] | None = None  # -> fill q{i} (K,N)
    mat_row: Callable[[int, dict[str, str]], str] | None = None  # -> per-(i,k) scoring expr
    mat_accumulate: Callable[[int], list[str]] | None = None  # -> fill S1_{i} (K,D), S2_{i} (K,D,D) via BLAS
    dtype: str = "float64"


_TEMPLATES: list[LeafTemplate] = []


def register_leaf_template(t: LeafTemplate) -> None:
    _TEMPLATES.append(t)


def _template_for(dist: Any) -> LeafTemplate | None:
    for t in _TEMPLATES:
        if t.matches(dist):
            return t
    return None


# --- scalar leaves --------------------------------------------------------------------------------
def _gaussian_params(comps: list[Any]) -> dict[str, np.ndarray]:
    mu = np.array([c.mu for c in comps], dtype=np.float64)
    s2 = np.array([c.sigma2 for c in comps], dtype=np.float64)
    return {"mu": mu, "inv2s2": 0.5 / s2, "lognorm": -0.5 * np.log(2.0 * np.pi * s2)}


register_leaf_template(
    LeafTemplate(
        name="gaussian",
        matches=lambda d: type(d).__name__ == "GaussianDistribution",
        data=lambda enc: np.asarray(enc, dtype=np.float64),
        params=_gaussian_params,
        expr=lambda x, p: f"{p['lognorm']}[k] - ({x} - {p['mu']}[k]) * ({x} - {p['mu']}[k]) * {p['inv2s2']}[k]",
        acc_names=("sx", "sx2"),
        acc_stmt=lambda x, a, r: f"{a['sx']}[k] += {r} * {x}; {a['sx2']}[k] += {r} * {x} * {x}",
        to_value=lambda s, count: (s[0], s[1], count, count),  # (sum_wx, sum_wx2, sum_w, sum_w)
    )
)


def _exponential_params(comps: list[Any]) -> dict[str, np.ndarray]:
    beta = np.array([c.beta for c in comps], dtype=np.float64)  # mean; rate = 1/beta
    return {"rate": 1.0 / beta, "lograte": -np.log(beta)}


register_leaf_template(
    LeafTemplate(
        name="exponential",
        matches=lambda d: type(d).__name__ == "ExponentialDistribution",
        data=lambda enc: np.asarray(enc, dtype=np.float64),
        params=_exponential_params,
        expr=lambda x, p: f"{p['lograte']}[k] - {p['rate']}[k] * {x}",
        acc_names=("sx",),
        acc_stmt=lambda x, a, r: f"{a['sx']}[k] += {r} * {x}",
        to_value=lambda s, count: (count, s[0]),  # (sum_w, sum_wx)
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
        f"    C{i} = x{i} - {p['mu']}[k]",
        f"    Z{i} = C{i} @ {p['prec']}[k]",
        "    for ii in range(n):",
        f"        qq{i} = 0.0",
        f"        for jj in range(x{i}.shape[1]):",
        f"            qq{i} += Z{i}[ii, jj] * C{i}[ii, jj]",
        f"        q{i}[k, ii] = qq{i}",
    ]


def _mv_accumulate(i: int) -> list[str]:
    # MVGaussian weighted stats: S1_k = X' r_k (weighted sum), S2_k = X' (r_k .* X) (weighted Gram), via BLAS.
    return [
        "for k in range(kc):",
        f"    Rk{i} = R[:, k].copy()",
        f"    S1_{i}[k] = x{i}.T @ Rk{i}",
        f"    S2_{i}[k] = x{i}.T @ (Rk{i}.reshape(-1, 1) * x{i})",
    ]


register_leaf_template(
    LeafTemplate(
        name="mvgaussian",
        matches=lambda d: type(d).__name__ == "MultivariateGaussianDistribution",
        data=lambda enc: np.ascontiguousarray(enc, dtype=np.float64),
        params=_mvgaussian_params,
        kind="matrix",
        mat_precompute=_mv_precompute,
        mat_row=lambda i, p: f"-0.5 * q{i}[k, i] + {p['lognorm']}[k]",
        mat_accumulate=_mv_accumulate,
        to_value=lambda s, count: (s[0], s[1], count),  # (sum_wx (D,), sum_wxx (D,D), sum_w)
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
    return analyze(model) is not None


def _dummy(t: LeafTemplate) -> Any:
    """A throwaway leaf instance to read this template's parameter names (kept tiny and family-specific)."""
    import pysp.stats as stats

    return {
        "gaussian": stats.GaussianDistribution(0.0, 1.0),
        "exponential": stats.ExponentialDistribution(1.0),
        "mvgaussian": stats.MultivariateGaussianDistribution([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
    }[t.name]


def _argmap(i: int, t: LeafTemplate) -> dict[str, str]:
    return {pn: f"p{i}_{pn}" for pn in sorted(t.params([_dummy(t)]).keys())}


# --- code generation ------------------------------------------------------------------------------
_COMPILED: dict[tuple, Callable] = {}
_ESTEP_COMPILED: dict[tuple, Callable] = {}


def _leaf_io(plan: FusedPlan) -> tuple[list[str], list[str], list[str], list[str]]:
    """Per-leaf code fragments shared by the scorer and the E-step.

    Returns (data_args, param_args, precompute_lines, row_terms). ``precompute_lines`` and the matrix
    ``row_terms`` are empty for a pure-scalar plan, so its generated source is unchanged.
    """
    data_args = [f"x{i}" for i in range(len(plan.leaf_templates))]
    param_args: list[str] = []
    precompute: list[str] = []
    row_terms: list[str] = []
    for i, t in enumerate(plan.leaf_templates):
        amap = _argmap(i, t)
        param_args.extend(amap.values())
        if t.kind == "matrix":
            precompute.extend(t.mat_precompute(i, amap))  # type: ignore[misc]
            row_terms.append("acc += " + t.mat_row(i, amap))  # type: ignore[misc]
        else:
            row_terms.append("acc += " + t.expr(f"x{i}[i]", amap))  # type: ignore[misc]
    return data_args, param_args, precompute, row_terms


def _njit(src: str, fname: str) -> Callable:
    import numba

    ns: dict[str, Any] = {"np": np}
    exec(src, ns)  # noqa: S102 -- source is generated from fixed templates, no user input
    return numba.njit(fastmath=True)(ns[fname])


def _compile(plan: FusedPlan) -> Callable:
    cached = _COMPILED.get(plan.signature)
    if cached is not None:
        return cached
    data_args, param_args, precompute, row_terms = _leaf_io(plan)
    args = ", ".join(data_args + param_args + ["logw", "out", "llbuf"])
    lines = [f"def _fused({args}):", "    n = x0.shape[0]", "    kc = logw.shape[0]"]
    lines += ["    " + ln for ln in precompute]
    lines += ["    for i in range(n):", "        for k in range(kc):", "            acc = logw[k]"]
    lines += ["            " + rt for rt in row_terms]
    lines += [
        "            llbuf[k] = acc",
        "        m = llbuf[0]",
        "        for k in range(1, kc):",
        "            if llbuf[k] > m:",
        "                m = llbuf[k]",
        "        s = 0.0",
        "        for k in range(kc):",
        "            s += np.exp(llbuf[k] - m)",
        "        out[i] = m + np.log(s)",
    ]
    fn = _njit("\n".join(lines), "_fused")
    _COMPILED[plan.signature] = fn
    return fn


def _compile_estep(plan: FusedPlan) -> Callable:
    cached = _ESTEP_COMPILED.get(plan.signature)
    if cached is not None:
        return cached
    data_args, param_args, precompute, row_terms = _leaf_io(plan)
    scalar_acc: list[str] = []
    matrix_acc: list[str] = []
    acc_args: list[str] = []
    for i, t in enumerate(plan.leaf_templates):
        if t.kind == "matrix":
            acc_args += [f"S1_{i}", f"S2_{i}"]
            matrix_acc += t.mat_accumulate(i)  # type: ignore[misc]
        else:
            amap = {an: f"a{i}_{an}" for an in t.acc_names}
            acc_args.extend(amap.values())
            scalar_acc.append(t.acc_stmt(f"x{i}[i]", amap, "r"))  # type: ignore[misc]
    args = ", ".join(data_args + param_args + ["weights", "logw", "comp_counts", *acc_args, "llbuf"])
    lines = [f"def _estep({args}):", "    n = x0.shape[0]", "    kc = logw.shape[0]"]
    if plan.has_matrix:
        lines.append("    R = np.empty((n, kc))")  # responsibilities -- only matrix accumulation needs them
    lines += ["    " + ln for ln in precompute]
    lines += [
        "    for i in range(n):",
        "        wi = weights[i]",
        "        for k in range(kc):",
        "            acc = logw[k]",
    ]
    lines += ["            " + rt for rt in row_terms]
    lines += [
        "            llbuf[k] = acc",
        "        m = llbuf[0]",
        "        for k in range(1, kc):",
        "            if llbuf[k] > m:",
        "                m = llbuf[k]",
        "        s = 0.0",
        "        for k in range(kc):",
        "            s += np.exp(llbuf[k] - m)",
        "        for k in range(kc):",
        "            r = np.exp(llbuf[k] - m) / s * wi",
        "            comp_counts[k] += r",
    ]
    lines += ["            " + st for st in scalar_acc]
    if plan.has_matrix:
        lines.append("            R[i, k] = r")
    lines += ["    " + ln for ln in matrix_acc]
    fn = _njit("\n".join(lines), "_estep")
    _ESTEP_COMPILED[plan.signature] = fn
    return fn


# --- marshalling ----------------------------------------------------------------------------------
def _component_factor_lists(model: Any, plan: FusedPlan) -> list[list[Any]]:
    if plan.is_mixture:
        return [_node_factors(c) for c in model.components]  # type: ignore[misc]
    return [_node_factors(model)]  # type: ignore[list-item]


def _data_and_params(model: Any, plan: FusedPlan, enc: Any) -> tuple[list[np.ndarray], list[np.ndarray]]:
    factor_lists = _component_factor_lists(model, plan)
    factor_encs = enc if isinstance(enc, tuple) else (enc,)
    data_arrays = [t.data(factor_encs[i]) for i, t in enumerate(plan.leaf_templates)]
    param_arrays: list[np.ndarray] = []
    for i, t in enumerate(plan.leaf_templates):
        pdict = t.params([factor_lists[k][i] for k in range(plan.num_components)])
        param_arrays.extend(np.ascontiguousarray(pdict[pn]) for pn in sorted(pdict.keys()))
    return data_arrays, param_arrays


def fused_seq_log_density(model: Any, enc: Any) -> np.ndarray:
    """Per-row log densities of ``model`` over encoding ``enc`` via one fused numba pass.

    Raises ``ValueError`` if ``model`` is not fusible -- callers should check :func:`fusible` first.
    """
    plan = analyze(model)
    if plan is None:
        raise ValueError("%s is not a fusible composite/mixture." % type(model).__name__)
    data_arrays, param_arrays = _data_and_params(model, plan, enc)
    logw = np.asarray(getattr(model, "log_w", np.zeros(1)), dtype=np.float64)
    out = np.empty(data_arrays[0].shape[0], dtype=np.float64)
    llbuf = np.empty(plan.num_components, dtype=np.float64)
    _compile(plan)(*data_arrays, *param_arrays, logw, out, llbuf)
    return out


# --- fused E-step (score + responsibilities + per-leaf weighted sufficient statistics, one njit) ----
def fusible_estep(model: Any) -> bool:
    plan = analyze(model)
    if plan is None:
        return False
    return all(
        (t.acc_stmt is not None) if t.kind == "scalar" else (t.mat_accumulate is not None) for t in plan.leaf_templates
    )


def fused_accumulate(model: Any, enc: Any, weights: np.ndarray) -> Any:
    """Run one fused E-step and return the sufficient statistic in the estimator's ``value()`` format.

    The whole E-step -- component scoring, responsibility soft-max, and per-leaf weighted-statistic
    accumulation (scalar inline, matrix via BLAS) -- runs in a single nopython pass, then is packed into
    the exact tuple shape ``estimate(nobs, suff_stat)`` expects. Raises ``ValueError`` if not fusible.
    """
    plan = analyze(model)
    if plan is None or not fusible_estep(model):
        raise ValueError("%s is not a fusible E-step (an unsupported leaf)." % type(model).__name__)
    K = plan.num_components
    data_arrays, param_arrays = _data_and_params(model, plan, enc)

    # per-leaf accumulator arrays, in the same leaf order the generated signature expects
    scalar_acc: list[dict[str, np.ndarray]] = []
    matrix_acc: list[tuple[np.ndarray, np.ndarray]] = []
    acc_arrays: list[np.ndarray] = []
    for i, t in enumerate(plan.leaf_templates):
        if t.kind == "matrix":
            d = data_arrays[i].shape[1]
            s1 = np.zeros((K, d), dtype=np.float64)
            s2 = np.zeros((K, d, d), dtype=np.float64)
            matrix_acc.append((s1, s2))
            scalar_acc.append({})
            acc_arrays += [s1, s2]
        else:
            d = {an: np.zeros(K, dtype=np.float64) for an in t.acc_names}
            scalar_acc.append(d)
            matrix_acc.append((np.empty(0), np.empty(0)))
            acc_arrays.extend(d.values())

    comp_counts = np.zeros(K, dtype=np.float64)
    logw = np.asarray(getattr(model, "log_w", np.zeros(1)), dtype=np.float64)
    llbuf = np.empty(K, dtype=np.float64)
    _compile_estep(plan)(
        *data_arrays, *param_arrays, np.asarray(weights, dtype=np.float64), logw, comp_counts, *acc_arrays, llbuf
    )

    def leaf_value(i: int, t: LeafTemplate, k: int) -> Any:
        if t.kind == "matrix":
            s1, s2 = matrix_acc[i]
            return t.to_value((s1[k], s2[k]), float(comp_counts[k]))  # type: ignore[misc]
        stats_k = tuple(scalar_acc[i][an][k] for an in t.acc_names)
        return t.to_value(stats_k, float(comp_counts[k]))  # type: ignore[misc]

    def node_value(k: int) -> Any:
        leaf_vals = [leaf_value(i, t, k) for i, t in enumerate(plan.leaf_templates)]
        return tuple(leaf_vals) if plan.component_is_composite else leaf_vals[0]

    if plan.is_mixture:
        return comp_counts, tuple(node_value(k) for k in range(K))
    return node_value(0)


# --- kernel wiring (used by optimize(..., engine=FUSED_NUMPY_ENGINE) for fusible models) ------------
class FusedKernel:
    """A duck-typed ``Kernel`` backed by the source-generated fused scorer and E-step."""

    def __init__(self, dist: Any, engine: Any, estimator: Any = None) -> None:
        self.dist = dist
        self.engine = engine
        self.estimator = estimator

    def encode(self, data: Any) -> Any:
        return self.dist.dist_to_encoder().seq_encode(data)

    def score(self, enc: Any) -> np.ndarray:
        return fused_seq_log_density(self.dist, getattr(enc, "engine_payload", enc))

    def accumulate(self, enc: Any, weights: Any) -> Any:
        w = np.asarray(self.engine.to_numpy(weights) if hasattr(self.engine, "to_numpy") else weights, dtype=np.float64)
        return fused_accumulate(self.dist, getattr(enc, "engine_payload", enc), w)

    def refresh(self, dist: Any) -> None:
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
            from pysp.stats.compute.kernel import GeneratedNumbaKernelFactory

            self._fallback = GeneratedNumbaKernelFactory()
        return self._fallback

    def build(self, dist: Any, engine: Any, estimator: Any = None) -> Any:
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
