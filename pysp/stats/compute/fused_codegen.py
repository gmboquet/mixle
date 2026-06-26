"""Source-generated *fused* numba kernels for composite / mixture trees.

The generated-numba kernels in :mod:`declarations` lower one leaf at a time: each leaf materializes its
sufficient statistic in numpy and runs its own row loop, so a composite or mixture pays a Python<->C
boundary crossing and an intermediate allocation *per factor and per component*. For a deep model of
cheap leaves that overhead dominates -- numpy itself is multi-pass for the same reason.

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

Generated kernels are compiled once and disk-cached (see :func:`_njit`), so the compile cost is paid once
per structure *ever*, not per process. Anything with a leaf that has no template (e.g. Categorical) ->
:func:`fusible` is False -> numpy.
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
    dtype: str = "float64"


_TEMPLATES: list[LeafTemplate] = []


def register_leaf_template(t: LeafTemplate) -> None:
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
        "geometric": stats.GeometricDistribution(0.5),
        "bernoulli": stats.BernoulliDistribution(0.5),
        "poisson": stats.PoissonDistribution(1.0),
        "gamma": stats.GammaDistribution(1.0, 1.0),
        "mvgaussian": stats.MultivariateGaussianDistribution([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
    }[t.name]


def _argmap(i: int, t: LeafTemplate) -> dict[str, str]:
    return {pn: f"p{i}_{pn}" for pn in sorted(t.params([_dummy(t)]).keys())}


def _data_names(i: int, t: LeafTemplate) -> list[str]:
    return [f"x{i}_{j}" for j in range(t.arity)]


# --- code generation ------------------------------------------------------------------------------
_COMPILED: dict[tuple, Callable] = {}
_ESTEP_COMPILED: dict[tuple, Callable] = {}


def _leaf_io(plan: FusedPlan) -> tuple[list[str], list[str], list[str], list[str]]:
    """Per-leaf code fragments shared by the scorer and the E-step.

    Returns (data_args, param_args, precompute_lines, row_terms). ``precompute_lines`` and the matrix
    ``row_terms`` are empty for a pure-scalar plan, so its generated source is unchanged.
    """
    data_args: list[str] = []
    param_args: list[str] = []
    precompute: list[str] = []
    row_terms: list[str] = []
    for i, t in enumerate(plan.leaf_templates):
        data_args.extend(_data_names(i, t))
        amap = _argmap(i, t)
        param_args.extend(amap.values())
        if t.kind == "matrix":
            precompute.extend(t.mat_precompute(i, amap))  # type: ignore[misc]
            row_terms.append("acc += " + t.mat_row(i, amap))  # type: ignore[misc]
        else:
            vals = [f"{nm}[i]" for nm in _data_names(i, t)]
            row_terms.append("acc += " + t.expr(vals, amap))  # type: ignore[misc]
    return data_args, param_args, precompute, row_terms


import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402

_CACHE_DIR = os.environ.get("PYSP_FUSED_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "pysp_fused_cache")
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
        except Exception:
            ns: dict[str, Any] = {"np": np}
            exec(src, ns)  # noqa: S102 -- generated from fixed templates, no user input
            return numba.njit(fastmath=True)(ns[fname])


def _compile(plan: FusedPlan) -> Callable:
    cached = _COMPILED.get(plan.signature)
    if cached is not None:
        return cached
    data_args, param_args, precompute, row_terms = _leaf_io(plan)
    args = ", ".join(data_args + param_args + ["logw", "out", "llbuf"])
    lines = [f"def _fused({args}):", f"    n = {data_args[0]}.shape[0]", "    kc = logw.shape[0]"]
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
            vals = [f"{nm}[i]" for nm in _data_names(i, t)]
            scalar_acc.append(t.acc_stmt(vals, amap, "r"))  # type: ignore[misc]
    args = ", ".join(data_args + param_args + ["weights", "logw", "comp_counts", *acc_args, "llbuf"])
    lines = [f"def _estep({args}):", f"    n = {data_args[0]}.shape[0]", "    kc = logw.shape[0]"]
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
    # A Composite encodes as a per-factor tuple; a bare leaf encodes as its own payload (which may itself
    # be a tuple, e.g. Poisson's (x, lgamma(x+1))) -- so key on the structure, not isinstance(enc, tuple).
    factor_encs = enc if plan.component_is_composite else (enc,)
    data_arrays: list[np.ndarray] = []
    for i, t in enumerate(plan.leaf_templates):
        data_arrays.extend(t.data(factor_encs[i]))  # arity arrays, flattened in leaf order
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

    # per-leaf accumulator arrays, in the same leaf order the generated signature expects. data_arrays is
    # flattened by arity, so track the running offset to find each matrix leaf's (N,D) data array.
    scalar_acc: list[dict[str, np.ndarray]] = []
    matrix_acc: list[tuple[np.ndarray, np.ndarray]] = []
    acc_arrays: list[np.ndarray] = []
    offset = 0
    for _i, t in enumerate(plan.leaf_templates):
        if t.kind == "matrix":
            d = data_arrays[offset].shape[1]
            s1 = np.zeros((K, d), dtype=np.float64)
            s2 = np.zeros((K, d, d), dtype=np.float64)
            matrix_acc.append((s1, s2))
            scalar_acc.append({})
            acc_arrays += [s1, s2]
        else:
            ad = {an: np.zeros(K, dtype=np.float64) for an in t.acc_names}
            scalar_acc.append(ad)
            matrix_acc.append((np.empty(0), np.empty(0)))
            acc_arrays.extend(ad.values())
        offset += t.arity

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
