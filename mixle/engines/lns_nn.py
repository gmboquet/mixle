"""Neural-network operations in a logarithmic number system.

Built on :class:`mixle.engines.lns.LogNumberSystem`. Two families of op benefit from LNS, both because
they fight transcendentals (``exp``/``log``), not BLAS:

* **softmax / cross-entropy / log-softmax**: the normalizer is a ``logsumexp`` over logits (the LM head
  over the vocabulary, a classifier over classes, an attention/MoE-router softmax). The integer
  ``logsumexp`` replaces the ``exp``+``log`` with integer ``max`` + a LUT (~2x measured). The model's
  logits are quantized by the same log step; only softmax-back-to-linear still needs an ``exp``.
* **sum-product circuits / probabilistic circuits**: the whole forward pass is sums and products of
  probabilities, so in LNS every product node is an integer ADD and every sum node an integer ``logadd``.
  The entire network runs in integer log-space, not only the normalizer.

These are inference and scoring operations. The gradient path stays in floating
autograd, while LNS provides a compact integer representation for log-space
math.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.engines.lns import LOG_ZERO_CODE, LogNumberSystem

# MXR-080-0142: tolerance for a SumProductCircuit sum-node's (exponentiated) weights summing to 1 --
# same values as mixle.stats.latent.mixture.MixtureDistribution's own simplex-weight check, for the same
# reason (a fitted/user-supplied weight vector is not guaranteed float64-exact-sums-to-1, but should be
# close; not re-imported from mixture.py since the two modules are otherwise independent).
_SUM_NODE_WEIGHT_RTOL = 1.0e-5
_SUM_NODE_WEIGHT_ATOL = 1.0e-8


def _lse_keepdims(lns: LogNumberSystem, k: np.ndarray, axis: int) -> np.ndarray:
    lse = lns.logsumexp(k, axis=axis)
    return np.expand_dims(lse, axis)


def log_softmax(logits: Any, lns: LogNumberSystem, axis: int = -1) -> np.ndarray:
    """Compute log-softmax through the integer log-partition."""
    k = lns.quantize(logits)
    normalizer = _lse_keepdims(lns, k, axis)
    if np.any(normalizer == LOG_ZERO_CODE):
        raise ValueError("log_softmax is undefined for an all-impossible slice")
    ordinary = (k - normalizer).astype(np.float64) * lns.step
    return np.where(k == LOG_ZERO_CODE, -np.inf, ordinary)


def softmax(logits: Any, lns: LogNumberSystem, axis: int = -1) -> np.ndarray:
    """Softmax with the normalizer computed in LNS; back-to-linear needs one ``exp`` (for attention.V etc.).

    Renormalized (MXR-080-0141): exponentiating the LNS-approximate log-softmax does not, by itself, sum
    to exactly 1 -- each class's rounding error is independent of the others, so they do not cancel (100
    equal logits summed to ``1.00518`` pre-fix), which breaks the basic probability-simplex contract a
    function named ``softmax`` has to satisfy. Dividing by the actual computed sum costs one reduction
    and trades a sliver of the raw approximation's "purity" for an exactly-normalized guarantee.
    """
    p = np.exp(log_softmax(logits, lns, axis=axis))
    total = np.sum(p, axis=axis, keepdims=True)
    if np.any(~np.isfinite(total)) or np.any(total <= 0):
        raise ArithmeticError("softmax normalization mass must be finite and positive")
    return p / total


def _validate_cross_entropy_targets(k: np.ndarray, targets: np.ndarray, axis: int) -> np.ndarray:
    """Validate/normalize ``targets`` for :func:`cross_entropy` (MXR-080-0140, MXR-080-0141).

    Applied identically regardless of which path :func:`cross_entropy` ends up taking, so the compiled
    fused path and the numpy fallback can never disagree about what counts as a valid target. Enforces:

    * classes are nonempty along ``axis`` (a zero-width row has no valid target, and in the compiled
      path would read never-initialized scratch memory -- MXR-080-0140),
    * ``targets.shape`` exactly matches ``k.shape`` with ``axis`` removed (a mismatched/too-short
      ``targets`` would, in the compiled path, read past its own buffer -- MXR-080-0140),
    * every target is an exact integer -- no silent fractional truncation (MXR-080-0141),
    * every target is in ``[0, n_classes)`` -- no silent negative-index wraparound (MXR-080-0141), and,
      in the compiled path (boundscheck disabled), no out-of-bounds read (MXR-080-0140).

    Returns ``targets`` as a validated ``int64`` array, safe to hand to either path.
    """
    ndim = k.ndim
    ax = axis if axis >= 0 else axis + ndim
    if not (0 <= ax < ndim):
        raise ValueError(f"cross_entropy: axis {axis} is out of range for {ndim}-d logits")
    n_classes = k.shape[ax]
    if n_classes == 0:
        raise ValueError("cross_entropy: logits has zero classes along `axis`; classes must be nonempty")
    expected_shape = k.shape[:ax] + k.shape[ax + 1 :]
    targets = np.asarray(targets)
    if targets.shape != expected_shape:
        raise ValueError(
            f"cross_entropy: targets shape {targets.shape} does not match the logits shape {k.shape} "
            f"with axis {axis} removed (expected {expected_shape})"
        )
    if targets.size == 0:
        raise ValueError("cross_entropy: got an empty batch (0 rows); the mean loss is undefined")
    if np.issubdtype(targets.dtype, np.floating):
        finite = np.isfinite(targets)
        if not finite.all() or np.any(targets[finite] != np.trunc(targets[finite])):
            raise ValueError("cross_entropy: targets must be exact integers; got a non-finite or fractional value")
    elif not (np.issubdtype(targets.dtype, np.integer) or np.issubdtype(targets.dtype, np.bool_)):
        raise ValueError(f"cross_entropy: targets must be integer-valued, got dtype {targets.dtype}")
    targets_i64 = targets.astype(np.int64)
    out_of_range = (targets_i64 < 0) | (targets_i64 >= n_classes)
    if np.any(out_of_range):
        bad = int(targets_i64[out_of_range].flat[0])
        raise ValueError(f"cross_entropy: target class index {bad} out of range [0, {n_classes})")
    return targets_i64


def cross_entropy(logits: Any, targets: Any, lns: LogNumberSystem, axis: int = -1) -> float:
    """Mean negative log-likelihood ``mean(logsumexp(logits) - logit[target])`` via the integer normalizer.

    The LM / classifier loss: the log-partition over the vocab/classes is an integer ``logsumexp``; the
    target logit is gathered from the same quantized logits, so the loss is integer until the final scale.

    Targets must be exact, in-range integer class indices (MXR-080-0141: no negative-index wraparound,
    no silent fractional truncation) -- validated identically whether or not the compiled fused path is
    taken (MXR-080-0140: the compiled path has boundscheck disabled, so an unvalidated target is a
    genuine out-of-bounds read, not merely a wrong answer). A target whose own logit quantized to the
    reserved "log of an exact zero" sentinel has a mathematically infinite loss -- and so does the batch
    mean -- so that is returned directly rather than ever subtracting the sentinel as a plain code
    (MXR-080-0157: that subtraction is exactly the kind of extreme-code-range operation that overflows
    int64, in both this path and the compiled one).
    """
    logits = np.asarray(logits, dtype=np.float64)
    k = lns.quantize(logits)
    targets = _validate_cross_entropy_targets(k, targets, axis)

    target_codes = np.take_along_axis(k, np.expand_dims(targets, axis), axis=axis).squeeze(axis)
    if np.any(target_codes == LOG_ZERO_CODE):
        return float("inf")

    if k.ndim == 2 and axis in (-1, 1):
        from mixle.engines.lns import _HAS_LNS_KERNEL

        if _HAS_LNS_KERNEL:  # fused one pass: tree log-partition + target gather, no temporaries (~14x vs fp64)
            from mixle.engines._lns_kernel import cross_entropy_rows

            total = cross_entropy_rows(np.ascontiguousarray(k), np.ascontiguousarray(targets), lns.lut, lns.dmax)
            return float(total * lns.step / k.shape[0])
    lse_k = lns.logsumexp(k, axis=axis)  # integer log-partition per row
    return float(np.mean((lse_k - target_codes).astype(np.float64) * lns.step))


def _canonical_circuit_children(i: int, children: Any) -> tuple[int, ...]:
    """MXR-080-0142: every product/sum node must have >=1 child, each an earlier, in-range node index.

    Requiring ``0 <= child < i`` (strictly earlier than the referencing node's own index) in one
    comparison rules out forward references, self-references, and out-of-range indices together -- and
    since a child index can then never be >= its parent's, no chain of child references can ever loop
    back on itself, making a cycle structurally impossible without a separate graph search.
    """
    if not isinstance(children, (list, tuple, np.ndarray)):
        raise ValueError(f"SumProductCircuit: node {i} children must be a list/tuple of indices, got {children!r}")
    if len(children) == 0:
        raise ValueError(f"SumProductCircuit: node {i} has no children (empty product/sum node)")
    canonical = []
    for c in children:
        if isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer)):
            raise ValueError(f"SumProductCircuit: node {i} has a non-integer child reference {c!r}")
        if c < 0 or c >= i:
            raise ValueError(
                f"SumProductCircuit: node {i} references child {c}, which must be an earlier node "
                f"(0 <= child < {i}); forward, self, and out-of-range references are not a valid DAG"
            )
        canonical.append(int(c))
    return tuple(canonical)


def _canonical_circuit_sum_weights(i: int, children: Any, log_weights: Any) -> tuple[float, ...]:
    """MXR-080-0142: a sum node's weight count must match its child count, and the (exponentiated)
    weights must be finite and sum to ~1 -- a genuine probability simplex, the same contract
    :class:`mixle.stats.latent.mixture.MixtureDistribution` enforces on its own component weights.
    """
    raw = np.asarray(log_weights)
    if raw.ndim != 1 or raw.shape != (len(children),):
        raise ValueError(
            f"SumProductCircuit: sum node {i} weights must have exact shape {(len(children),)}, got {raw.shape}"
        )
    log_w = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(log_w).all():
        raise ValueError(f"SumProductCircuit: sum node {i} has non-finite weight(s): {list(log_weights)!r}")
    w = np.exp(log_w)
    total = float(w.sum())
    if not np.isclose(total, 1.0, rtol=_SUM_NODE_WEIGHT_RTOL, atol=_SUM_NODE_WEIGHT_ATOL):
        raise ValueError(
            f"SumProductCircuit: sum node {i} weights must sum to 1.0 (simplex weights), got sum={total!r}"
        )
    return tuple(float(value) for value in log_w)


def _canonical_leaf_id(value: Any) -> Any:
    """Own an immutable leaf identifier rather than retaining a mutable caller object."""
    if isinstance(value, (str, bytes, int, bool, type(None))):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("SumProductCircuit leaf ids must not be NaN/Inf")
        return value
    if isinstance(value, tuple):
        return tuple(_canonical_leaf_id(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_canonical_leaf_id(item) for item in value)
    raise ValueError("SumProductCircuit leaf ids must be immutable scalar/tuple/frozenset values, got %r" % (value,))


class SumProductCircuit:
    """A probabilistic circuit evaluated entirely in integer log-space (product=add, sum=logadd).

    ``nodes`` is a topologically ordered list (children before parents), each a tuple:

      * ``("leaf", leaf_id)``           -- an input whose log-value is supplied at evaluation,
      * ``("product", [child indices])`` -- log-output = sum of children (integer ADD),
      * ``("sum", [child indices], [log_weights])`` -- log-output = logsumexp of weighted children.

    The root is the last node. ``evaluate_lns`` runs the whole forward pass on integers; ``evaluate_float``
    is the float64 reference. Leaf values may be scalars or arrays (the forward broadcasts).
    """

    def __init__(self, nodes: list[tuple]) -> None:
        """Validate the complete typed DAG at construction (MXR-080-0142) instead of deferring to
        evaluation: nonempty circuit; every node a recognized ``leaf``/``product``/``sum`` tuple of the
        right arity; every product/sum node has >=1 child, each an earlier in-range node (see
        :func:`_validate_circuit_children` for why this also rules out cycles); every sum node's weight
        count matches its child count and its (exponentiated) weights are finite and sum to ~1. The leaf
        ids declared by ``leaf`` nodes are recorded so :meth:`evaluate_lns`/:meth:`evaluate_float` can
        validate the ``leaf_values`` contract up front too -- the actual values aren't known until then,
        only which ids the circuit is entitled to expect.
        """
        if len(nodes) == 0:
            raise ValueError("SumProductCircuit requires at least one node; got an empty circuit")
        leaf_ids: set = set()
        canonical_nodes = []
        for i, node in enumerate(nodes):
            if not isinstance(node, (tuple, list)) or len(node) == 0:
                raise ValueError(f"SumProductCircuit: node {i} must be a nonempty (kind, ...) tuple, got {node!r}")
            kind = node[0]
            if kind == "leaf":
                if len(node) != 2:
                    raise ValueError(f"SumProductCircuit: leaf node {i} must be ('leaf', leaf_id), got {node!r}")
                leaf_id = _canonical_leaf_id(node[1])
                leaf_ids.add(leaf_id)
                canonical_nodes.append(("leaf", leaf_id))
            elif kind == "product":
                if len(node) != 2:
                    raise ValueError(f"SumProductCircuit: product node {i} must be ('product', children), got {node!r}")
                canonical_nodes.append(("product", _canonical_circuit_children(i, node[1])))
            elif kind == "sum":
                if len(node) != 3:
                    raise ValueError(
                        f"SumProductCircuit: sum node {i} must be ('sum', children, log_weights), got {node!r}"
                    )
                children = _canonical_circuit_children(i, node[1])
                log_weights = _canonical_circuit_sum_weights(i, children, node[2])
                canonical_nodes.append(("sum", children, log_weights))
            else:
                raise ValueError(f"SumProductCircuit: node {i} has an unrecognized kind {kind!r}")
        self.nodes = tuple(canonical_nodes)
        self.leaf_ids = frozenset(leaf_ids)

    def _check_leaf_values(self, method: str, leaf_values: dict[Any, Any]) -> None:
        missing = self.leaf_ids - leaf_values.keys()
        if missing:
            raise ValueError(f"SumProductCircuit.{method}: leaf_values is missing id(s) {sorted(map(repr, missing))}")

    def evaluate_lns(self, lns: LogNumberSystem, leaf_values: dict[Any, Any]) -> np.ndarray:
        """Evaluate the circuit with log-number-system arithmetic."""
        self._check_leaf_values("evaluate_lns", leaf_values)
        vals: list[Any] = [None] * len(self.nodes)
        for i, node in enumerate(self.nodes):
            if node[0] == "leaf":
                vals[i] = lns.quantize(leaf_values[node[1]])
            elif node[0] == "product":
                acc = vals[node[1][0]]
                for c in node[1][1:]:
                    acc = lns.multiply(acc, vals[c])  # MXR-080-0142: safe LNS product (was raw `+`)
                vals[i] = acc
            elif node[0] == "sum":
                children, log_w = node[1], np.asarray(node[2], dtype=np.float64)
                wk = lns.quantize(log_w)
                # each term is log(child_prob * weight) = child_code (+) weight_code -- an LNS product,
                # so it must go through the same overflow-safe multiply() (MXR-080-0142), not raw `+`:
                # a child that quantized to LOG_ZERO_CODE (a legitimate zero-probability value) combined
                # with an ordinary weight code would otherwise overflow int64 exactly like the product
                # node case above.
                terms = np.stack([lns.multiply(vals[c], wk[j]) for j, c in enumerate(children)], axis=0)
                vals[i] = lns.logsumexp(terms, axis=0)
            else:  # pragma: no cover - unreachable: __init__ already rejects any other node kind
                raise ValueError("unknown node %r" % (node[0],))
        return lns.dequantize(vals[-1])

    def evaluate_float(self, leaf_values: dict[Any, Any]) -> np.ndarray:
        """Evaluate the circuit with float64 reference arithmetic."""
        self._check_leaf_values("evaluate_float", leaf_values)
        vals: list[Any] = [None] * len(self.nodes)
        for i, node in enumerate(self.nodes):
            if node[0] == "leaf":
                vals[i] = np.asarray(leaf_values[node[1]], dtype=np.float64)
            elif node[0] == "product":
                acc = vals[node[1][0]]
                for c in node[1][1:]:
                    acc = acc + vals[c]
                vals[i] = acc
            elif node[0] == "sum":
                children, log_w = node[1], np.asarray(node[2], dtype=np.float64)
                terms = np.stack([vals[c] + log_w[j] for j, c in enumerate(children)], axis=0)
                vals[i] = np.logaddexp.reduce(terms, axis=0)
            else:  # pragma: no cover - unreachable: __init__ already rejects any other node kind
                raise ValueError("unknown node %r" % (node[0],))
        return np.asarray(vals[-1], dtype=np.float64)
