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
    expr: Callable[[str, dict[str, str]], str]  # (value_var, {pname: arg_name}) -> numba expression string
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


def _factors(dist: Any) -> list[Any]:
    return list(getattr(dist, "dists", None) or [dist])  # a bare leaf is a 1-factor "composite"


def analyze(model: Any) -> FusedPlan | None:
    """Return a :class:`FusedPlan` if ``model`` is a fusible mixture/composite of cheap leaves, else None.

    Handles: a single leaf, a Composite of leaves, a Mixture of leaves, and a Mixture of Composites
    (every component sharing the same leaf-type structure). Anything else -> None (fall back to numpy).
    """
    comps = getattr(model, "components", None)
    log_w = getattr(model, "log_w", None)
    if comps is not None and log_w is not None:  # a mixture
        comps = list(comps)
        per = [_factors(c) for c in comps]
        widths = {len(p) for p in per}
        if len(widths) != 1:
            return None  # heterogeneous component structure
        templates = [_template_for(f) for f in per[0]]
        if any(t is None for t in templates):
            return None
        # every component must have the same leaf types in the same order
        for p in per:
            if [_template_for(f) for f in p] != templates:
                return None
        sig = ("mix", tuple(t.name for t in templates))
        return FusedPlan(len(comps), True, tuple(templates), sig)  # type: ignore[arg-type]
    factors = _factors(model)
    templates = [_template_for(f) for f in factors]
    if any(t is None for t in templates):
        return None
    sig = ("comp", tuple(t.name for t in templates))
    return FusedPlan(1, False, tuple(templates), sig)  # type: ignore[arg-type]


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
        return [_factors(c) for c in model.components]
    return [_factors(model)]


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


__all__ = ["LeafTemplate", "register_leaf_template", "analyze", "fusible", "fused_seq_log_density", "FusedPlan"]
