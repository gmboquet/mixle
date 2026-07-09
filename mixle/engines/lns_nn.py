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

from mixle.engines.lns import LogNumberSystem


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


def cross_entropy(logits: Any, targets: Any, lns: LogNumberSystem, axis: int = -1) -> float:
    """Mean negative log-likelihood ``mean(logsumexp(logits) - logit[target])`` via the integer normalizer.

    The LM / classifier loss: the log-partition over the vocab/classes is an integer ``logsumexp``; the
    target logit is gathered from the same quantized logits, so the loss is integer until the final scale.
    """
    logits = np.asarray(logits, dtype=np.float64)
    k = lns.quantize(logits)
    targets = np.asarray(targets)
    if k.ndim == 2 and axis in (-1, 1):
        from mixle.engines.lns import _HAS_LNS_KERNEL

        if _HAS_LNS_KERNEL:  # fused one pass: tree log-partition + target gather, no temporaries (~14x vs fp64)
            from mixle.engines._lns_kernel import cross_entropy_rows

            total = cross_entropy_rows(
                np.ascontiguousarray(k), np.ascontiguousarray(targets.astype(np.int64)), lns.lut, lns.dmax
            )
            return float(total * lns.step / k.shape[0])
    lse_k = lns.logsumexp(k, axis=axis)  # integer log-partition per row
    tgt_k = np.take_along_axis(k, np.expand_dims(targets, axis), axis=axis).squeeze(axis)
    return float(np.mean((lse_k - tgt_k).astype(np.float64) * lns.step))


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
