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


def _lse_keepdims(lns: LogNumberSystem, k: np.ndarray, axis: int) -> np.ndarray:
    lse = lns.logsumexp(k, axis=axis)
    return np.expand_dims(lse, axis)


def log_softmax(logits: Any, lns: LogNumberSystem, axis: int = -1) -> np.ndarray:
    """Compute log-softmax through the integer log-partition."""
    k = lns.quantize(logits)
    return (k - _lse_keepdims(lns, k, axis)).astype(np.float64) * lns.step


def softmax(logits: Any, lns: LogNumberSystem, axis: int = -1) -> np.ndarray:
    """Softmax with the normalizer computed in LNS; back-to-linear needs one ``exp`` (for attention.V etc.)."""
    return np.exp(log_softmax(logits, lns, axis=axis))


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
        self.nodes = nodes

    def evaluate_lns(self, lns: LogNumberSystem, leaf_values: dict[Any, Any]) -> np.ndarray:
        """Evaluate the circuit with log-number-system arithmetic."""
        vals: list[Any] = [None] * len(self.nodes)
        for i, node in enumerate(self.nodes):
            if node[0] == "leaf":
                vals[i] = lns.quantize(leaf_values[node[1]])
            elif node[0] == "product":
                acc = vals[node[1][0]]
                for c in node[1][1:]:
                    acc = acc + vals[c]  # log-product = integer add
                vals[i] = acc
            elif node[0] == "sum":
                children, log_w = node[1], np.asarray(node[2], dtype=np.float64)
                wk = lns.quantize(log_w)
                terms = np.stack([np.add(vals[c], wk[j]) for j, c in enumerate(children)], axis=0)
                vals[i] = lns.logsumexp(terms, axis=0)
            else:  # pragma: no cover
                raise ValueError("unknown node %r" % (node[0],))
        return lns.dequantize(vals[-1])

    def evaluate_float(self, leaf_values: dict[Any, Any]) -> np.ndarray:
        """Evaluate the circuit with float64 reference arithmetic."""
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
            else:  # pragma: no cover
                raise ValueError("unknown node %r" % (node[0],))
        return np.asarray(vals[-1], dtype=np.float64)
