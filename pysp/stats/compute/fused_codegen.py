"""Source-generated *fused* numba kernels for composite / mixture trees of cheap leaves.

The generated-numba kernels in :mod:`declarations` lower one leaf at a time: each leaf materializes its
sufficient statistic in numpy and runs its own row loop, so a composite or mixture pays a Python<->C
boundary crossing and an intermediate allocation *per factor and per component*. For a deep model of
cheap leaves that overhead dominates -- numpy itself is multi-pass for the same reason.

This module instead emits a SINGLE ``@njit`` function for the whole model structure: one pass over the
rows, every leaf's scalar log-density inlined, the composite sum and the mixture log-sum-exp all in
nopython registers with no intermediate arrays. Measured ~1.2-2.7x over the numpy path for
mixture-of-composite Gaussian models, growing with depth.

Scope (by design): cheap scalar leaves whose per-element work is comparable to numpy's, where avoiding
passes/allocations wins. BLAS-bound leaves (MVGaussian's quadratic form) are deliberately NOT fused --
numpy's gemm beats a scalar loop there -- so :func:`fusible` returns False and callers fall back to numpy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

# --- leaf templates -------------------------------------------------------------------------------
# A leaf template knows, for one cheap-leaf family: how to pull its per-row data array out of the
# leaf's encoding, how to stack its scalar parameters across the K mixture components, and the numba
# expression (a string) for its scalar log-density given the row value and the component-indexed params.


@dataclass(frozen=True)
class LeafTemplate:
    name: str
    matches: Callable[[Any], bool]
    data: Callable[[Any], np.ndarray]  # leaf encoding -> (N,) data array
    params: Callable[[list[Any]], dict[str, np.ndarray]]  # K leaf dists -> {pname: (K,) array}
    expr: Callable[[str, dict[str, str]], str]  # (value_var, {pname: arg_name}) -> numba scoring expression
    # --- E-step accumulation (optional; a leaf without these can be scored but not fit via the fused path)
    acc_names: tuple[str, ...] = ()  # per-leaf weighted-statistic accumulator arrays (shape (K,))
    acc_stmt: Callable[[str, dict[str, str], str], str] | None = None  # (value_var, {acc: arg}, resp_var) -> stmts
    to_value: Callable[[tuple, float], tuple] | None = None  # (accumulated stats for comp k, count_k) -> leaf value
    dtype: str = "float64"


_TEMPLATES: list[LeafTemplate] = []


def register_leaf_template(t: LeafTemplate) -> None:
    _TEMPLATES.append(t)


def _template_for(dist: Any) -> LeafTemplate | None:
    for t in _TEMPLATES:
        if t.matches(dist):
            return t
    return None


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


# --- structure analysis ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FusedPlan:
    """A fusible model structure: K mixture components (1 if not a mixture), each a list of leaf factors."""

    num_components: int
    is_mixture: bool
    leaf_templates: tuple[LeafTemplate, ...]  # one per factor (composite factor order)
    signature: tuple
    component_is_composite: bool = False  # whether each leaf-bearing node is a Composite (vs a bare leaf)


def _node_factors(node: Any) -> list[Any] | None:
    """The leaf factors of a fusible *node* (an exact Composite, or a templated bare leaf), else None."""
    if type(node).__name__ == "CompositeDistribution":
        return list(node.dists)
    if _template_for(node) is not None:
        return [node]
    return None


def analyze(model: Any) -> FusedPlan | None:
    """Return a :class:`FusedPlan` if ``model`` is a fusible mixture/composite of cheap leaves, else None.

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
        if any(t is None for t in templates):  # a factor has no leaf template (e.g. an MVGaussian)
            return None
        # every component must have the same leaf types in the same order
        if any([_template_for(f) for f in p] != templates for p in per):  # type: ignore[union-attr]
            return None
        comp_is_composite = type(comps[0]).__name__ == "CompositeDistribution"
        sig = ("mix", tuple(t.name for t in templates), comp_is_composite)
        return FusedPlan(len(comps), True, tuple(templates), sig, comp_is_composite)  # type: ignore[arg-type]
    factors = _node_factors(model)
    if factors is None:
        return None
    templates = [_template_for(f) for f in factors]
    if any(t is None for t in templates):  # a composite factor has no leaf template
        return None
    is_composite = tname == "CompositeDistribution"
    sig = ("comp", tuple(t.name for t in templates), is_composite)
    return FusedPlan(1, False, tuple(templates), sig, is_composite)  # type: ignore[arg-type]


def fusible(model: Any) -> bool:
    return analyze(model) is not None


# --- code generation ------------------------------------------------------------------------------
_COMPILED: dict[tuple, Callable] = {}


def _compile(plan: FusedPlan) -> Callable:
    cached = _COMPILED.get(plan.signature)
    if cached is not None:
        return cached
    import numba

    L = len(plan.leaf_templates)
    data_args = [f"x{i}" for i in range(L)]
    param_args: list[str] = []
    body_terms = []
    for i, t in enumerate(plan.leaf_templates):
        pnames = sorted(t.params([_dummy(t)]).keys())  # stable arg names per leaf
        argmap = {pn: f"p{i}_{pn}" for pn in pnames}
        param_args.extend(argmap[pn] for pn in pnames)
        body_terms.append("acc += " + t.expr(f"x{i}[i]", argmap))
    args = ", ".join(data_args + param_args + ["logw", "out", "llbuf"])
    inner = "\n            ".join(body_terms)
    src = f"""
def _fused({args}):
    n = x0.shape[0]
    kc = logw.shape[0]
    for i in range(n):
        for k in range(kc):
            acc = logw[k]
            {inner}
            llbuf[k] = acc
        m = llbuf[0]
        for k in range(1, kc):
            if llbuf[k] > m:
                m = llbuf[k]
        s = 0.0
        for k in range(kc):
            s += np.exp(llbuf[k] - m)
        out[i] = m + np.log(s)
"""
    ns: dict[str, Any] = {"np": np}
    exec(src, ns)  # noqa: S102  -- source is generated from a fixed template, no user input
    fn = numba.njit(fastmath=True)(ns["_fused"])
    _COMPILED[plan.signature] = fn
    return fn


def _dummy(t: LeafTemplate) -> Any:
    """A throwaway leaf instance to read this template's parameter names (kept tiny and family-specific)."""
    import pysp.stats as stats

    return {"gaussian": stats.GaussianDistribution(0.0, 1.0), "exponential": stats.ExponentialDistribution(1.0)}[t.name]


def _component_factor_lists(model: Any, plan: FusedPlan) -> list[list[Any]]:
    if plan.is_mixture:
        return [_node_factors(c) for c in model.components]  # type: ignore[misc]
    return [_node_factors(model)]  # type: ignore[list-item]


def fused_seq_log_density(model: Any, enc: Any) -> np.ndarray:
    """Per-row log densities of ``model`` over encoding ``enc`` via one fused numba pass.

    ``enc`` is the model's ordinary ``dist_to_encoder().seq_encode`` payload (a composite tuple, or a
    bare leaf encoding). Raises ``ValueError`` if ``model`` is not fusible -- callers should check
    :func:`fusible` first or catch and fall back to numpy.
    """
    plan = analyze(model)
    if plan is None:
        raise ValueError("%s is not a fusible composite/mixture of cheap leaves." % type(model).__name__)
    factor_lists = _component_factor_lists(model, plan)
    # per-factor data arrays (the composite encoding is a tuple; a bare leaf encoding is the array itself)
    factor_encs = enc if isinstance(enc, tuple) else (enc,)
    data_arrays = [t.data(factor_encs[i]) for i, t in enumerate(plan.leaf_templates)]
    # per-factor parameter arrays stacked over the K components
    param_arrays: list[np.ndarray] = []
    for i, t in enumerate(plan.leaf_templates):
        leaf_dists = [factor_lists[k][i] for k in range(plan.num_components)]
        pdict = t.params(leaf_dists)
        param_arrays.extend(pdict[pn] for pn in sorted(pdict.keys()))
    logw = np.asarray(getattr(model, "log_w", np.zeros(1)), dtype=np.float64)
    out = np.empty(data_arrays[0].shape[0], dtype=np.float64)
    llbuf = np.empty(plan.num_components, dtype=np.float64)
    _compile(plan)(*data_arrays, *param_arrays, logw, out, llbuf)
    return out


# --- fused E-step (score + responsibilities + per-leaf weighted sufficient statistics, one njit) ----
def fusible_estep(model: Any) -> bool:
    plan = analyze(model)
    return plan is not None and all(t.acc_stmt is not None for t in plan.leaf_templates)


_ESTEP_COMPILED: dict[tuple, Callable] = {}


def _compile_estep(plan: FusedPlan) -> Callable:
    cached = _ESTEP_COMPILED.get(plan.signature)
    if cached is not None:
        return cached
    import numba

    data_args = [f"x{i}" for i in range(len(plan.leaf_templates))]
    param_args: list[str] = []
    acc_args: list[str] = []
    ll_terms: list[str] = []
    acc_stmts: list[str] = []
    for i, t in enumerate(plan.leaf_templates):
        pmap = {pn: f"p{i}_{pn}" for pn in sorted(t.params([_dummy(t)]).keys())}
        param_args.extend(pmap.values())
        ll_terms.append("acc += " + t.expr(f"x{i}[i]", pmap))
        amap = {an: f"a{i}_{an}" for an in t.acc_names}
        acc_args.extend(amap.values())
        acc_stmts.append(t.acc_stmt(f"x{i}[i]", amap, "r"))  # type: ignore[misc]
    args = ", ".join(data_args + param_args + ["weights", "logw", "comp_counts", *acc_args, "llbuf"])
    ll = "\n            ".join(ll_terms)
    accs = "\n            ".join(acc_stmts)  # aligns with the 12-space indent of {accs} in the template
    src = f"""
def _estep({args}):
    n = x0.shape[0]
    kc = logw.shape[0]
    for i in range(n):
        wi = weights[i]
        for k in range(kc):
            acc = logw[k]
            {ll}
            llbuf[k] = acc
        m = llbuf[0]
        for k in range(1, kc):
            if llbuf[k] > m:
                m = llbuf[k]
        s = 0.0
        for k in range(kc):
            s += np.exp(llbuf[k] - m)
        for k in range(kc):
            r = np.exp(llbuf[k] - m) / s * wi
            comp_counts[k] += r
            {accs}
"""
    ns: dict[str, Any] = {"np": np}
    exec(src, ns)  # noqa: S102 -- generated from a fixed template
    fn = numba.njit(fastmath=True)(ns["_estep"])
    _ESTEP_COMPILED[plan.signature] = fn
    return fn


def fused_accumulate(model: Any, enc: Any, weights: np.ndarray) -> Any:
    """Run one fused E-step and return the sufficient statistic in the estimator's ``value()`` format.

    The whole E-step -- component scoring, responsibility softmax, and per-leaf weighted statistic
    accumulation -- runs in a single nopython pass; the result is then packed into the exact tuple shape
    the corresponding ``estimate(nobs, suff_stat)`` expects (leaf / Composite / Mixture). Raises
    ``ValueError`` if the model is not fused-E-step capable (a leaf lacks accumulation support)."""
    plan = analyze(model)
    if plan is None or any(t.acc_stmt is None for t in plan.leaf_templates):
        raise ValueError("%s is not a fusible E-step (an unsupported leaf)." % type(model).__name__)
    K = plan.num_components
    factor_lists = _component_factor_lists(model, plan)
    factor_encs = enc if isinstance(enc, tuple) else (enc,)
    data_arrays = [t.data(factor_encs[i]) for i, t in enumerate(plan.leaf_templates)]
    param_arrays: list[np.ndarray] = []
    for i, t in enumerate(plan.leaf_templates):
        pdict = t.params([factor_lists[k][i] for k in range(K)])
        param_arrays.extend(pdict[pn] for pn in sorted(pdict.keys()))
    per_leaf_acc = [{an: np.zeros(K, dtype=np.float64) for an in t.acc_names} for t in plan.leaf_templates]
    acc_arrays = [arr for d in per_leaf_acc for arr in d.values()]
    comp_counts = np.zeros(K, dtype=np.float64)
    logw = np.asarray(getattr(model, "log_w", np.zeros(1)), dtype=np.float64)
    llbuf = np.empty(K, dtype=np.float64)
    _compile_estep(plan)(
        *data_arrays, *param_arrays, np.asarray(weights, dtype=np.float64), logw, comp_counts, *acc_arrays, llbuf
    )

    def node_value(k: int) -> Any:
        leaf_vals = [
            t.to_value(tuple(per_leaf_acc[i][an][k] for an in t.acc_names), float(comp_counts[k]))  # type: ignore[misc]
            for i, t in enumerate(plan.leaf_templates)
        ]
        return tuple(leaf_vals) if plan.component_is_composite else leaf_vals[0]

    if plan.is_mixture:
        return comp_counts, tuple(node_value(k) for k in range(K))
    return node_value(0)


# --- kernel wiring (used by optimize(..., engine=<numba-capable>) for fusible models) --------------
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
