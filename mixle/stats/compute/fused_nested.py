"""Recursive fusion for arbitrarily nested Composite / Mixture trees of scalar leaves.

The flat :mod:`fused_codegen` handles depth-2 (Mixture -> Composite -> leaf) with a component loop. This
module handles arbitrary nesting -- a Composite factor that is itself a Mixture, a Mixture of Mixtures, a
Mixture of Composites whose factors nest, ... -- by UNROLLING the static tree into straight-line numba:

* forward: every node's score, bottom-up (a composite is the sum of its children, a mixture is the
  log-sum-exp of ``log_w_j + child_j``);
* E-step backward: the responsibility reaching a node (the product of the mixture posteriors down its path,
  times the observation weight) is pushed to its children; each leaf accumulates its weighted sufficient
  statistic, and each mixture its per-component counts.

It reuses the :class:`~mixle.stats.compute.fused_codegen.LeafTemplate` machinery by giving every leaf node a
``(1,)``-shaped parameter block indexed at ``k = 0`` (the templates are written for ``[k]`` indexing).
Scope: scalar leaves (the common case for nested mixtures); a nested model containing a matrix / tabulated
/ categorical leaf returns ``None`` here and falls back to numpy. It is consulted only when the flat
:func:`~mixle.stats.compute.fused_codegen.analyze` declines, so the flat fast path is never perturbed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.stats.compute.fused_codegen import LeafTemplate, _auto_parallel, _n_chunks, _njit, _template_for


@dataclass
class _Leaf:
    node_id: int
    template: LeafTemplate
    slot: int  # data slot (shared by all mixture components at the same composite path)
    dist: Any


@dataclass
class _Composite:
    node_id: int
    children: list[Any]


@dataclass
class _Mixture:
    node_id: int
    children: list[Any]
    log_w: np.ndarray


@dataclass
class _Ctx:
    next_id: int = 0
    slots: dict[tuple, tuple[LeafTemplate, int]] = field(default_factory=dict)
    slot_data: dict[int, tuple[np.ndarray, ...]] = field(default_factory=dict)
    slot_template: dict[int, LeafTemplate] = field(default_factory=dict)

    def fresh(self) -> int:
        nid = self.next_id
        self.next_id += 1
        return nid


def _shape(node: Any) -> Any:
    """Structural signature of a BUILT tree -- the kernel cache key. Captures ALL children so distinct
    heterogeneous mixtures never share a compiled kernel."""
    if isinstance(node, _Leaf):
        return ("leaf", node.template.name)
    if isinstance(node, _Composite):
        return ("comp", tuple(_shape(c) for c in node.children))
    return ("mix", tuple(_shape(c) for c in node.children))


def _model_shape(dist: Any) -> Any:
    """Structural signature of a MODEL node (available before building). A Mixture is homogeneous (its
    components share one encoder) iff all component shapes are equal; otherwise it is heterogeneous and
    encodes each component separately (``_HeteroMixtureEncoded``)."""
    tname = type(dist).__name__
    if tname == "CompositeDistribution":
        return ("comp", tuple(_model_shape(c) for c in dist.dists))
    if tname == "MixtureDistribution":
        return ("mix", tuple(_model_shape(c) for c in dist.components))
    t = _template_for(dist)
    return ("leaf", t.name if t is not None else tname)


def _homogeneous(model: Any) -> bool:
    shapes = [_model_shape(c) for c in model.components]
    return all(s == shapes[0] for s in shapes)


def _build(model: Any, path: tuple, ctx: _Ctx) -> Any | None:
    """Walk the model tree (no encoding needed -- the tree is the structure); the per-slot data arrays are
    filled later by :func:`_fill_slots` from the real encoding. Heterogeneous mixtures (different-typed
    components) unroll naturally -- each component just gets its own data slot (path branch ``("h", j)``)."""
    tname = type(model).__name__
    if tname == "CompositeDistribution":
        children = [_build(c, path + (("c", i),), ctx) for i, c in enumerate(model.dists)]
        if any(c is None for c in children):
            return None
        return _Composite(ctx.fresh(), children)
    if tname == "MixtureDistribution":
        homo = _homogeneous(model)  # homogeneous components share the encoding (same slot/path); else split
        children = []
        for j, c in enumerate(model.components):
            child = _build(c, path if homo else path + (("h", j),), ctx)
            if child is None:
                return None
            children.append(child)
        return _Mixture(ctx.fresh(), children, np.asarray(model.log_w, dtype=np.float64))
    t = _template_for(model)
    if t is None or t.kind != "scalar":  # only scalar leaves nest cleanly (no precompute / BLAS / tables)
        return None
    if path not in ctx.slots:
        slot = len(ctx.slots)
        ctx.slots[path] = (t, slot)
        ctx.slot_template[slot] = t
    return _Leaf(ctx.fresh(), t, ctx.slots[path][1], model)


def analyze_nested(model: Any) -> tuple[Any, _Ctx] | None:
    """Return (tree, ctx) for a nested scalar-leaf Composite/Mixture model, or None to fall back.

    Restricted to *genuinely nested* models -- depth-2 flat mixtures/composites are handled (faster) by the
    flat :func:`~mixle.stats.compute.fused_codegen.analyze`, so this only fires when that one declines.
    """
    if type(model).__name__ not in ("CompositeDistribution", "MixtureDistribution"):
        return None
    ctx = _Ctx()
    root = _build(model, (), ctx)
    if root is None or not ctx.slots:
        return None
    return root, ctx


def _argmap(node: _Leaf) -> dict[str, str]:
    return {pn: f"p{node.node_id}_{pn}" for pn in sorted(node.template.params([node.dist]).keys())}


def _vals(node: _Leaf) -> list[str]:
    return [f"x{node.slot}_{j}[i]" for j in range(node.template.arity)]


# --- code generation (recursive emit over the static tree) ----------------------------------------
def _emit_score(node: Any, lines: list[str]) -> str:
    """Append forward-pass lines computing ``node``'s score; return the score expression."""
    if isinstance(node, _Leaf):
        return node.template.expr(_vals(node), _argmap(node))  # type: ignore[misc]
    if isinstance(node, _Composite):
        return "(" + " + ".join("(" + _emit_score(c, lines) + ")" for c in node.children) + ")"
    cvars = []
    for j, c in enumerate(node.children):
        e = _emit_score(c, lines)
        lines.append(f"    s{node.node_id}_{j} = logw{node.node_id}[{j}] + ({e})")
        cvars.append(f"s{node.node_id}_{j}")
    lines.append(f"    mx{node.node_id} = {cvars[0]}")
    for v in cvars[1:]:
        lines.append(f"    mx{node.node_id} = max(mx{node.node_id}, {v})")
    lines.append(f"    sm{node.node_id} = 0.0")
    for v in cvars:
        lines.append(f"    sm{node.node_id} += np.exp({v} - mx{node.node_id})")
    lines.append(f"    ns{node.node_id} = mx{node.node_id} + np.log(sm{node.node_id})")
    return f"ns{node.node_id}"


def _emit_backward(node: Any, rho: str, lines: list[str]) -> None:
    """Append E-step lines: accumulate leaf stats / mixture counts under responsibility ``rho``."""
    if isinstance(node, _Leaf):
        accmap = {an: f"a{node.node_id}_{an}" for an in node.template.acc_names}
        lines.append(f"    ct{node.node_id}[0] += {rho}")
        lines.append("    " + node.template.acc_stmt(_vals(node), accmap, rho))  # type: ignore[misc]
        return
    if isinstance(node, _Composite):
        for c in node.children:
            _emit_backward(c, rho, lines)
        return
    rv = f"rho{node.node_id}"
    lines.append(f"    {rv} = {rho}")
    for j, c in enumerate(node.children):
        crv = f"{rv}_{j}"
        lines.append(f"    {crv} = {rv} * np.exp(s{node.node_id}_{j} - ns{node.node_id})")
        lines.append(f"    cc{node.node_id}[{j}] += {crv}")
        _emit_backward(c, crv, lines)


def _leaves(node: Any) -> list[_Leaf]:
    if isinstance(node, _Leaf):
        return [node]
    return [lf for c in node.children for lf in _leaves(c)]


def _mixtures(node: Any) -> list[_Mixture]:
    if isinstance(node, _Leaf):
        return []
    out = [node] if isinstance(node, _Mixture) else []
    return out + [m for c in node.children for m in _mixtures(c)]


def _data_args(ctx: _Ctx) -> list[str]:
    return [f"x{s}_{j}" for s in range(len(ctx.slots)) for j in range(ctx.slot_template[s].arity)]


def _param_args(root: Any) -> list[str]:
    args: list[str] = []
    for lf in _leaves(root):
        args.extend(_argmap(lf).values())
    for mx in _mixtures(root):
        args.append(f"logw{mx.node_id}")
    return args


_SCORE_CACHE: dict[Any, Any] = {}
_ESTEP_CACHE: dict[Any, Any] = {}


def _cast_reduced(arrays: list, compute_dtype: Any) -> list:
    """Down-cast FLOAT arrays to the reduced compute dtype (float32 only, as fused_codegen validates);
    integer index arrays and the float64 accumulators/outputs are never touched, so accumulation
    precision is unchanged -- the same discipline as fused_codegen's _data_and_params."""
    if compute_dtype is None or np.dtype(compute_dtype) == np.float64:
        return arrays
    if np.dtype(compute_dtype) != np.float32:
        raise ValueError("nested fused kernels support reduced precision only in float32. Got %r." % (compute_dtype,))
    return [np.ascontiguousarray(a, dtype=np.float32) if a.dtype.kind == "f" else a for a in arrays]


def _sig(root: Any, ctx: _Ctx) -> tuple:
    return ("nested", _shape(root), tuple(ctx.slot_template[s].name for s in range(len(ctx.slots))))


def _compile_score(root: Any, ctx: _Ctx, sig: tuple, parallel: bool = False) -> Any:
    cached = _SCORE_CACHE.get((sig, parallel))
    if cached is not None:
        return cached
    body: list[str] = []
    expr = _emit_score(root, body)
    if not parallel:
        args = ", ".join(_data_args(ctx) + _param_args(root) + ["out"])
        # k = 0: leaf templates index their (1,)-stacked params as p[k] (fused_codegen's component-loop
        # convention); in the nested emitter every leaf node is its own single-component stack.
        lines = [f"def _ns({args}):", "    n = out.shape[0]", "    k = 0", "    for i in range(n):"]
        lines += ["    " + ln for ln in body]  # body already indented to one level; add the loop indent
        lines.append(f"        out[i] = {expr}")
        fn = _njit("\n".join(lines), "_ns")
    else:
        # same fixed-chunk prange design as fused_codegen: rows are disjoint (out[i] only), so the
        # parallel scorer is bit-stable across reruns and worker counts, and agrees with the
        # sequential kernel to 1-2 ULP (different fastmath binaries).
        args = ", ".join(_data_args(ctx) + _param_args(root) + ["out", "n_chunks"])
        lines = [
            f"def _ns_par({args}):",
            "    n = out.shape[0]",
            "    step = (n + n_chunks - 1) // n_chunks",
            "    for c in numba.prange(n_chunks):",
            "        k = 0",
            "        for i in range(c * step, min(n, (c + 1) * step)):",
        ]
        lines += ["        " + ln for ln in body]
        lines.append(f"            out[i] = {expr}")
        fn = _njit("\n".join(lines), "_ns_par", parallel=True)
    _SCORE_CACHE[(sig, parallel)] = fn
    return fn


def _acc_names(root: Any) -> tuple[list[str], list[str]]:
    """(mixture-count arg names, leaf-accumulator arg names) in emission order."""
    cc_args = [f"cc{m.node_id}" for m in _mixtures(root)]
    leaf_acc: list[str] = []
    for lf in _leaves(root):
        leaf_acc += [f"a{lf.node_id}_{an}" for an in lf.template.acc_names] + [f"ct{lf.node_id}"]
    return cc_args, leaf_acc


def _compile_estep(root: Any, ctx: _Ctx, sig: tuple, parallel: bool = False) -> Any:
    cached = _ESTEP_CACHE.get((sig, parallel))
    if cached is not None:
        return cached
    fwd: list[str] = []
    root_expr = _emit_score(root, fwd)
    bwd: list[str] = []
    _emit_backward(root, "wi", bwd)
    cc_args, leaf_acc = _acc_names(root)
    if not parallel:
        args = ", ".join(_data_args(ctx) + _param_args(root) + ["weights", *cc_args, *leaf_acc, "out_ll"])
        lines = [
            f"def _es({args}):",
            "    n = weights.shape[0]",
            "    k = 0",
            "    for i in range(n):",
            "        wi = weights[i]",
        ]
        lines += ["    " + ln for ln in fwd]
        lines.append(f"        out_ll[0] += wi * ({root_expr})")
        lines += ["    " + ln for ln in bwd]
        fn = _njit("\n".join(lines), "_es")
    else:
        # Chunk-parallel with the fused_codegen design: every accumulator gains a leading n_chunks
        # axis (allocated by the caller), chunk c writes only row c through the _c views bound below,
        # and the caller's fixed-order sum over the chunk axis keeps results bit-stable across runs
        # and worker counts. The emitted statements are IDENTICAL to the sequential kernel's -- the
        # rebinding is pure text substitution on the accumulator names.
        acc_all = [*cc_args, *leaf_acc]
        args = ", ".join(_data_args(ctx) + _param_args(root) + ["weights", *acc_all, "out_ll", "n_chunks"])
        lines = [
            f"def _es_par({args}):",
            "    n = weights.shape[0]",
            "    step = (n + n_chunks - 1) // n_chunks",
            "    for c in numba.prange(n_chunks):",
        ]
        lines += [f"        {nm}_c = {nm}[c]" for nm in acc_all]
        lines += [
            "        k = 0",
            "        for i in range(c * step, min(n, (c + 1) * step)):",
            "            wi = weights[i]",
        ]
        lines += ["        " + ln for ln in fwd]
        lines.append(f"            out_ll[c] += wi * ({root_expr})")
        bwd_sub = bwd
        for nm in acc_all:
            bwd_sub = [ln.replace(f"{nm}[", f"{nm}_c[") for ln in bwd_sub]
        lines += ["        " + ln for ln in bwd_sub]
        fn = _njit("\n".join(lines), "_es_par", parallel=True)
    _ESTEP_CACHE[(sig, parallel)] = fn
    return fn


# --- marshalling ----------------------------------------------------------------------------------
def _marshal(model: Any, root: Any, ctx: _Ctx, enc: Any) -> tuple[list, list]:
    """Fill the per-slot data arrays (from the real encoding) and the per-leaf / per-mixture params."""
    _fill_slots(model, (), enc, ctx)
    data = [ctx.slot_data[s][j] for s in range(len(ctx.slots)) for j in range(ctx.slot_template[s].arity)]
    params: list = []
    for lf in _leaves(root):
        pdict = lf.template.params([lf.dist])
        params += [np.ascontiguousarray(pdict[pn]) for pn in sorted(pdict)]
    for mx in _mixtures(root):
        params.append(np.ascontiguousarray(mx.log_w))
    return data, params


def _fill_slots(model: Any, path: tuple, enc: Any, ctx: _Ctx) -> None:
    tname = type(model).__name__
    if tname == "CompositeDistribution":
        encs = enc if isinstance(enc, tuple) else (enc,)
        for i, c in enumerate(model.dists):
            _fill_slots(c, path + (("c", i),), encs[i], ctx)
    elif tname == "MixtureDistribution":
        if _homogeneous(model):
            for c in model.components:  # components share the encoding
                _fill_slots(c, path, enc, ctx)
        else:
            sub = enc.encodings  # _HeteroMixtureEncoded: one encoding per component
            for j, c in enumerate(model.components):
                _fill_slots(c, path + (("h", j),), sub[j], ctx)
    elif path in ctx.slots:
        slot = ctx.slots[path][1]
        ctx.slot_data[slot] = ctx.slots[path][0].data(enc)


def fused_nested_seq_log_density(
    model: Any, enc: Any, compute_dtype: Any = None, parallel: bool | None = None
) -> np.ndarray:
    """Score encoded observations with the nested scalar fused kernel.

    ``compute_dtype``/``parallel`` follow fused_codegen's contract exactly: float32 runs the row
    arithmetic reduced while accumulation and output stay float64; ``parallel=None`` auto-engages
    the chunked prange kernel on large inputs, bit-stable across reruns and worker counts.
    """
    built = analyze_nested(model)
    if built is None:
        raise ValueError("%s is not a fusible nested scalar tree." % type(model).__name__)
    root, ctx = built
    data, params = _marshal(model, root, ctx, enc)
    data = _cast_reduced(data, compute_dtype)
    params = _cast_reduced(params, compute_dtype)
    n = data[0].shape[0]
    out = np.empty(n, dtype=np.float64)
    if parallel is None:
        parallel = _auto_parallel(n)
    if parallel:
        _compile_score(root, ctx, _sig(root, ctx), parallel=True)(*data, *params, out, _n_chunks(n))
    else:
        _compile_score(root, ctx, _sig(root, ctx))(*data, *params, out)
    return out


def _node_value(node: Any, accs: dict) -> Any:
    if isinstance(node, _Leaf):
        stats_k = tuple(accs[f"a{node.node_id}_{an}"][0] for an in node.template.acc_names)
        return node.template.to_value(stats_k, float(accs[f"ct{node.node_id}"][0]))  # type: ignore[misc]
    children = [_node_value(c, accs) for c in node.children]
    if isinstance(node, _Composite):
        return tuple(children)
    return accs[f"cc{node.node_id}"], tuple(children)


def fused_nested_accumulate(
    model: Any,
    enc: Any,
    weights: np.ndarray,
    return_ll: bool = False,
    compute_dtype: Any = None,
    parallel: bool | None = None,
) -> Any:
    """Accumulate nested scalar sufficient statistics with the fused E-step kernel.

    ``compute_dtype``/``parallel`` follow fused_codegen's contract: reduced-precision row arithmetic
    with float64 accumulation; the chunk-parallel variant is bit-stable across reruns and worker
    counts (fixed chunking, fixed-order combine) and differs from the sequential kernel only by
    chunk-boundary float re-association.
    """
    built = analyze_nested(model)
    if built is None:
        raise ValueError("%s is not a fusible nested scalar tree." % type(model).__name__)
    root, ctx = built
    data, params = _marshal(model, root, ctx, enc)
    data = _cast_reduced(data, compute_dtype)
    params = _cast_reduced(params, compute_dtype)
    n = data[0].shape[0] if data else int(np.asarray(weights).shape[0])
    if parallel is None:
        parallel = _auto_parallel(n)
    nc = _n_chunks(n) if parallel else 0
    lead = (lambda shape: (nc, *shape)) if parallel else (lambda shape: shape)
    accs: dict = {}
    for lf in _leaves(root):
        for an in lf.template.acc_names:
            accs[f"a{lf.node_id}_{an}"] = np.zeros(lead((1,)), dtype=np.float64)
        accs[f"ct{lf.node_id}"] = np.zeros(lead((1,)), dtype=np.float64)
    for mx in _mixtures(root):
        accs[f"cc{mx.node_id}"] = np.zeros(lead((len(mx.children),)), dtype=np.float64)
    cc_arrays = [accs[f"cc{m.node_id}"] for m in _mixtures(root)]
    leaf_acc: list = []
    for lf in _leaves(root):
        leaf_acc += [accs[f"a{lf.node_id}_{an}"] for an in lf.template.acc_names] + [accs[f"ct{lf.node_id}"]]
    if parallel:
        out_ll = np.zeros(nc, dtype=np.float64)
        _compile_estep(root, ctx, _sig(root, ctx), parallel=True)(
            *data, *params, np.asarray(weights, dtype=np.float64), *cc_arrays, *leaf_acc, out_ll, nc
        )
        accs = {name: arr.sum(axis=0) for name, arr in accs.items()}  # fixed-order combine
    else:
        out_ll = np.zeros(1, dtype=np.float64)
        _compile_estep(root, ctx, _sig(root, ctx))(
            *data, *params, np.asarray(weights, dtype=np.float64), *cc_arrays, *leaf_acc, out_ll
        )
    suff = _node_value(root, accs)
    return (suff, float(out_ll.sum())) if return_ll else suff


def fusible_nested(model: Any) -> bool:
    """Return whether ``model`` can use nested scalar fusion."""
    return analyze_nested(model) is not None
