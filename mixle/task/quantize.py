"""Post-training quantization of distilled MLP students: int8/int4 weights, numpy-only inference.

An fp32 MLP student costs ``4 bytes x params`` and needs torch at inference. Quantizing to int8
(per-tensor symmetric: ``W ~ round(W / s) * s`` with one fp32 scale per layer) cuts the weight bytes
4x; int4 packs two weights per byte for 8x -- and because the dequantized forward pass is three
numpy matmuls, the quantized student needs **no torch at all** on the device: it joins the
structured students in the torch-free deployable class, while keeping the MLP's shape. Accuracy is
whatever it *measures* after quantization -- the edge search
(:func:`mixle.task.edge.distill_for_edge`) scores the quantized model's real agreement, so the bits
axis trades measured bytes against measured fidelity, never assumed ones.

``quantize_mlp(student, bits=8|4)`` converts a trained torch student in place of retraining; the
result stores quantized arrays (``payload="arrays"``, int4 stored nibble-packed), round-trips
through the artifact as an ``.npz``, and reports its true byte size.

LNS (log-number-system; ``mixle.engines.lns``) is wired where it is a *complete* fit: the
structured student, whose inference is sums of factor log-densities -- :func:`lns_classifier`
re-executes it on integers (add / max / LUT, no transcendentals above the leaf boundary). Signed
MLP matmuls in LNS would need signed-logadd kernels and stay out of scope.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.task.model import (
    HashedNGram,
    HashedRecord,
    StructuredClassifierIO,
    TaskModel,
    _ClassifierIO,
    register_adapter,
)

__all__ = [
    "QuantizedMLP",
    "QuantizedClassifierIO",
    "quantize_mlp",
    "quantize_dequantize_array",
    "LNSStructuredClassifierIO",
    "lns_classifier",
    "dequantize_symmetric",
]

_QMAX = {8: 127, 4: 7}  # symmetric integer range per weight precision


def quantize_dequantize_array(
    w: np.ndarray, *, bits: int = 8, clip_percentile: float | None = None
) -> tuple[np.ndarray, float]:
    """The per-tensor symmetric quantize step shared by PTQ (:func:`quantize_mlp`) and QAT's
    straight-through fake-quant (:mod:`mixle.models.qat`): ``scale = max|W| / qmax`` (or a
    percentile of ``|W|`` when ``clip_percentile`` is set), ``Wq = clip(round(W / scale), -qmax,
    qmax)``. Returns ``(Wq int8, scale)``; the dequantized value is ``Wq.astype(float) * scale`` --
    callers that only need the round-tripped float (QAT's fake-quant) do that multiply themselves,
    callers that need the deployable integer payload (PTQ) keep ``Wq`` and ``scale`` separate.
    """
    if bits not in _QMAX:
        raise ValueError(f"bits must be one of {sorted(_QMAX)}, got {bits}")
    if clip_percentile is not None and not (0.0 < clip_percentile <= 100.0):
        raise ValueError("clip_percentile must be in (0, 100]")
    qmax = _QMAX[bits]
    w = np.asarray(w, dtype=np.float64)
    if clip_percentile is None:
        wmax = float(np.max(np.abs(w))) if w.size else 0.0
    else:  # scale off a high percentile so outliers saturate instead of dictating the scale
        wmax = float(np.percentile(np.abs(w), clip_percentile)) if w.size else 0.0
    scale = (wmax / qmax) or 1.0
    wq = np.clip(np.round(w / scale), -qmax, qmax).astype(np.int8)
    return wq, scale


def dequantize_symmetric(wq: np.ndarray, scale: float) -> np.ndarray:
    """Inverse of :func:`quantize_dequantize_array`: ``wq * scale`` as float64."""
    return np.asarray(wq, dtype=np.float64) * float(scale)


def _pack_nibbles(w: np.ndarray) -> np.ndarray:
    """Pack an int8 array with values in [-7, 7] into two-per-byte uint8 (offset-8 nibbles)."""
    flat = (np.asarray(w, dtype=np.int16).reshape(-1) + 8).astype(np.uint8)  # [-7,7] -> [1,15]
    if flat.size % 2:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])  # pad nibble (0 = unused code)
    return (flat[0::2] << 4) | flat[1::2]


def _unpack_nibbles(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Inverse of :func:`_pack_nibbles`: uint8 pairs -> int8 values in [-7, 7] with ``shape``."""
    p = np.asarray(packed, dtype=np.uint8)
    flat = np.empty(p.size * 2, dtype=np.int16)
    flat[0::2] = p >> 4
    flat[1::2] = p & 0x0F
    n = int(np.prod(shape))
    return (flat[:n] - 8).astype(np.int8).reshape(shape)


class QuantizedMLP:
    """A quantized-weight MLP with a pure-numpy forward pass.

    ``layers`` is ``[(W_int (out, in), scale fp32, bias fp32 (out,)), ...]`` with weights in the
    symmetric ``bits`` range (int8: [-127, 127]; int4: [-7, 7], stored nibble-packed on disk); the
    forward is ``x @ (W * s).T + b`` with ReLU between layers -- exactly the dequantized version of
    the trained torch stack, so its logits match torch-on-dequantized-weights to float tolerance.
    """

    def __init__(self, layers: list[tuple[np.ndarray, float, np.ndarray]], *, bits: int = 8) -> None:
        if not layers:
            raise ValueError("QuantizedMLP needs at least one layer")
        if bits not in _QMAX:
            raise ValueError(f"bits must be one of {sorted(_QMAX)}, got {bits}")
        self.bits = int(bits)
        self.layers = [(np.asarray(w, dtype=np.int8), float(s), np.asarray(b, dtype=np.float32)) for w, s, b in layers]
        qmax = _QMAX[self.bits]
        for w, _s, _b in self.layers:
            if int(np.abs(w).max(initial=0)) > qmax:
                raise ValueError(f"weight magnitude exceeds the int{self.bits} range [-{qmax}, {qmax}]")

    def logits(self, feats: np.ndarray) -> np.ndarray:
        """Compute dequantized logits for a feature matrix."""
        x = np.asarray(feats, dtype=np.float32)
        last = len(self.layers) - 1
        for i, (w, s, b) in enumerate(self.layers):
            x = x @ (w.astype(np.float32) * s).T + b
            if i != last:
                x = np.maximum(x, 0.0)
        return x

    def nbytes(self) -> int:
        """Deployable payload bytes: packed weights (1 B/weight at int8, 1/2 B at int4) + fp32
        biases + one fp32 scale per layer."""
        per_w = 1.0 if self.bits == 8 else 0.5
        return int(sum(int(np.ceil(w.size * per_w)) + b.nbytes + 4 for w, _s, b in self.layers))

    def macs(self) -> int:
        """Per-inference multiply-accumulates (integer x fp32 dequant multiplies count the same)."""
        return int(sum(w.shape[0] * w.shape[1] for w, _s, _b in self.layers))

    # -- artifact arrays payload --
    def to_arrays(self) -> dict[str, np.ndarray]:
        """Serialize the quantized layers into artifact-ready NumPy arrays."""
        out: dict[str, np.ndarray] = {
            "n_layers": np.asarray(len(self.layers), dtype=np.int64),
            "bits": np.asarray(self.bits, dtype=np.int64),
        }
        for i, (w, s, b) in enumerate(self.layers):
            if self.bits == 4:
                out[f"w{i}"] = _pack_nibbles(w)  # true 4-bit storage, two weights per byte
                out[f"shape{i}"] = np.asarray(w.shape, dtype=np.int64)
            else:
                out[f"w{i}"] = w
            out[f"s{i}"] = np.asarray(s, dtype=np.float32)
            out[f"b{i}"] = b
        return out

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> QuantizedMLP:
        """Reconstruct a quantized MLP from artifact array payloads."""
        k = int(np.asarray(arrays["n_layers"]).reshape(()))
        bits = int(np.asarray(arrays.get("bits", 8)).reshape(()))
        layers = []
        for i in range(k):
            w = arrays[f"w{i}"]
            if bits == 4:
                w = _unpack_nibbles(w, tuple(int(d) for d in np.asarray(arrays[f"shape{i}"]).reshape(-1)))
            layers.append((w, float(np.asarray(arrays[f"s{i}"]).reshape(())), arrays[f"b{i}"]))
        return cls(layers, bits=bits)


class QuantizedClassifierIO(_ClassifierIO):
    """The classifier IO for quantized students: same featurize -> logits -> label contract, no torch."""

    kind = "quantized_classifier"

    def logits_batch(self, model: Any, raw_inputs: list[Any]) -> np.ndarray:
        """Featurize raw inputs and return quantized-model logits."""
        if not raw_inputs:  # empty batch: (0, K), skip the forward (reshape can't infer -1 at size 0)
            return np.empty((0, len(self.labels)), dtype=np.float32)
        return np.asarray(model.logits(self.features(raw_inputs))).reshape(len(raw_inputs), -1)

    def to_spec(self) -> dict[str, Any]:
        """Serialize the quantized classifier IO adapter."""
        fam = "text" if isinstance(self.featurizer, HashedNGram) else "record"
        return {
            "kind": self.kind,
            "featurizer_kind": fam,
            "featurizer": self.featurizer.to_spec(),
            "labels": self.labels,
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> QuantizedClassifierIO:
        """Reconstruct the quantized classifier IO adapter from a spec."""
        feat_cls = HashedNGram if spec.get("featurizer_kind", "text") == "text" else HashedRecord
        return cls(feat_cls.from_spec(spec["featurizer"]), spec["labels"])


register_adapter(QuantizedClassifierIO.kind, QuantizedClassifierIO.from_spec)

from mixle.task.artifact import register_arrays_builder  # noqa: E402  (after class defs, avoids partial-module use)

register_arrays_builder("mixle.quantized_mlp", QuantizedMLP.from_arrays)


def _torch_linears(module: Any) -> list[Any]:
    return [m for m in module.modules() if type(m).__name__ == "Linear"]


def quantize_mlp(student: TaskModel, *, bits: int = 8, clip_percentile: float | None = None) -> TaskModel:
    """Quantize a trained torch MLP student to an int8/int4, numpy-inference :class:`TaskModel`.

    Per-tensor symmetric weight quantization (``scale = max|W| / qmax`` with ``qmax`` 127 for int8, 7
    for int4); biases stay fp32 (they are a negligible byte fraction and quantizing them buys
    nothing). The returned student reuses the same featurizer and label list, reports
    ``payload="arrays"`` (int4 weights nibble-packed on disk: two per byte), and -- having no torch
    dependence at inference -- qualifies for ``torch_free`` devices. LNS needs LUT matmul kernels
    (``mixle.engines.lns``) and is left explicitly unimplemented.

    ``clip_percentile`` guards heavy-tailed weights. Plain max-scaling lets one outlier set the whole
    layer's scale: at int4 (``qmax=7``) a single weight 30x the rest quantizes everything else to 0,
    collapsing the layer. When set (e.g. ``99.9``), the scale is derived from that percentile of
    ``|W|`` instead of the max, and weights above it saturate at ``+/-qmax`` -- the bulk of the
    distribution keeps its resolution at the cost of clipping a few outliers. Default ``None`` keeps
    the exact max-scale behavior (bit-identical on well-behaved weights).
    """
    if bits not in _QMAX:
        raise NotImplementedError(
            f"bits={bits}: MLP precisions are {sorted(_QMAX)}; LNS applies to structured students via lns_classifier"
        )
    if student.payload != "torch":
        raise ValueError("quantize_mlp expects a torch MLP student (payload='torch')")
    if clip_percentile is not None and not (0.0 < clip_percentile <= 100.0):
        raise ValueError("clip_percentile must be in (0, 100]")
    linears = _torch_linears(student.model)
    if not linears:
        raise ValueError("student module has no Linear layers to quantize")

    layers: list[tuple[np.ndarray, float, np.ndarray]] = []
    for lin in linears:
        w = lin.weight.detach().cpu().numpy().astype(np.float64)
        b = (
            lin.bias.detach().cpu().numpy().astype(np.float32)
            if lin.bias is not None
            else np.zeros(w.shape[0], dtype=np.float32)
        )
        wq, scale = quantize_dequantize_array(w, bits=bits, clip_percentile=clip_percentile)
        layers.append((wq, scale, b))

    qmodel = QuantizedMLP(layers, bits=bits)
    adapter = QuantizedClassifierIO(student.adapter.featurizer, student.adapter.labels)
    meta = dict(student.meta)
    meta["quantized"] = {
        "bits": bits,
        "scheme": "per-tensor symmetric",
        "clip_percentile": clip_percentile,
        "fp32_bytes": 4 * sum(w.size for w, _s, _b in layers),
    }
    return TaskModel(
        qmodel,
        adapter,
        builder="mixle.quantized_mlp",
        config={},
        payload="arrays",
        task=student.task,
        meta=meta,
    )


# --- LNS: integer log-space inference for structured students -----------------------------------------------

_LOG_ZERO_INT = -(2**40)  # integer sentinel for log 0 (-inf); adds across factors cannot overflow int64


class LNSStructuredClassifierIO(StructuredClassifierIO):
    """The structured classifier executed in the log-number system: integers above the leaf boundary.

    A structured student's per-label score is a *sum of factor log-densities* -- in log-space that is
    products of probabilities, which is exactly what :class:`~mixle.engines.lns.LogNumberSystem` runs
    on integers: each factor's log-density is quantized once at the leaf boundary (``k = round(logp /
    step)``), then the per-label accumulation is integer ADDs, mixture components fold with the
    integer ``logadd`` LUT, the classification is an integer argmax, and the posterior is the integer
    log-softmax of :mod:`mixle.engines.lns_nn` -- no ``exp``/``log`` anywhere above the leaves (one
    ``exp`` only if you ask for linear-scale probabilities). The dequantized scores match the float
    classifier within the engine's documented bound (~``1.5 * step`` per fold), so ``step`` is a
    dial between integer-width and fidelity.

    Categorical factors are pre-quantized to integer tables at first use, so their leaves are pure
    integer lookups -- on an all-discrete schema inference touches no floats at all. Continuous
    leaves evaluate in float and quantize at the boundary, the same contract as the engine's
    ``SumProductCircuit``.
    """

    kind = "lns_structured_classifier"

    def __init__(self, field_keys: list[str] | None, label_index: int, labels: list[str], step: float = 1e-2) -> None:
        super().__init__(field_keys, label_index, labels)
        self.step = float(step)
        from mixle.engines.lns import LogNumberSystem

        self._lns = LogNumberSystem(step=self.step)

    # -- integer scoring -------------------------------------------------------------------------
    def _quantize_term(self, logp: float) -> int:
        if not np.isfinite(logp):
            return _LOG_ZERO_INT
        return int(np.rint(logp / self.step))

    # -- compiled integer tables: categorical leaves become pure lookups --------------------------
    def _compile_factor(self, factor: Any) -> tuple[dict, int] | None:
        """Pre-quantize a categorical factor to an integer table ``{key: k}`` (+ unseen default).

        Marginal ``CategoricalDistribution`` -> ``{value: k}``; a ``ConditionalDistribution`` whose
        branches are all categorical -> ``{(parent_key, value): k}``. Returns ``None`` for anything
        else (continuous leaves keep the float-then-quantize boundary path).
        """
        pmap = getattr(factor, "pmap", None)
        if isinstance(pmap, dict):  # categorical marginal
            table = {v: self._quantize_term(float(np.log(p)) if p > 0 else -np.inf) for v, p in pmap.items()}
            log_default = float(getattr(factor, "log_default_value", -np.inf))
            return table, self._quantize_term(log_default)
        dmap = getattr(factor, "dmap", None)
        if isinstance(dmap, dict) and dmap:
            table = {}
            for key, branch in dmap.items():
                branch_pmap = getattr(branch, "pmap", None)
                if not isinstance(branch_pmap, dict):
                    return None  # a non-categorical branch: leave the whole factor on the float path
                for v, p in branch_pmap.items():
                    table[(key, v)] = self._quantize_term(float(np.log(p)) if p > 0 else -np.inf)
            return table, _LOG_ZERO_INT  # unseen (parent, value) pair carries no mass
        return None

    def _compiled_tables(self, tree: Any) -> list[tuple[dict, int] | None]:
        cache = getattr(self, "_table_cache", None)
        if cache is None:
            cache = self._table_cache = {}
        key = id(tree)
        if key not in cache:
            cache[key] = [self._compile_factor(f) for f in tree.factors]
        return cache[key]

    def _tree_int_score(self, tree: Any, row: tuple) -> int:
        """Integer log-joint of one row under one dependency tree: quantized factor terms, integer adds.

        Categorical factors resolve through pre-quantized integer tables (a dict lookup of an int --
        no float log-density at all); continuous leaves evaluate in float and quantize at the
        boundary, the same contract as the engine's ``SumProductCircuit`` leaves.
        """
        tables = self._compiled_tables(tree)
        total = 0
        for i, parent in enumerate(tree.parents):
            compiled = tables[i]
            if compiled is not None:  # pure integer lookup
                table, default = compiled
                lookup = row[i] if parent is None else (tree._key(i, row[parent]), row[i])
                k = table.get(lookup, default)
            else:
                from mixle.inference.structure import _safe_log_density

                if parent is None:
                    term = _safe_log_density(tree.factors[i], row[i])
                else:
                    term = _safe_log_density(tree.factors[i], (tree._key(i, row[parent]), row[i]))
                k = self._quantize_term(term)
            if k <= _LOG_ZERO_INT:
                return _LOG_ZERO_INT
            total += k
        return total

    def int_logits_batch(self, model: Any, raw_inputs: list[Any]) -> np.ndarray:
        """Per-label INTEGER log-joint scores ``(m, K)`` -- the whole combination is integer math."""
        values = [self._values(r) for r in raw_inputs]
        out = np.full((len(values), len(self.labels)), _LOG_ZERO_INT, dtype=np.int64)
        components = getattr(model, "components", None)
        if components is not None:  # mixture: integer logadd across components with quantized log-weights
            log_w = np.log(np.clip(np.asarray(model.weights, dtype=np.float64), 1e-300, None))
            wk = self._lns.quantize(log_w)
            for k, label in enumerate(self.labels):
                for i, v in enumerate(values):
                    row = self._augment(v, label)
                    scores = np.array(
                        [self._tree_int_score(c, row) + int(wk[j]) for j, c in enumerate(components)],
                        dtype=np.int64,
                    )
                    live = scores > _LOG_ZERO_INT // 2
                    if live.any():
                        out[i, k] = int(self._lns.logsumexp(scores[live].reshape(1, -1), axis=-1)[0])
        else:
            for k, label in enumerate(self.labels):
                for i, v in enumerate(values):
                    out[i, k] = self._tree_int_score(model, self._augment(v, label))
        return out

    # -- the classifier contract on integers ------------------------------------------------------
    def logits_batch(self, model: Any, raw_inputs: list[Any]) -> np.ndarray:
        """Return floating logit values decoded from integer log-space scores."""
        z = self.int_logits_batch(model, raw_inputs).astype(np.float64) * self.step
        z[z <= _LOG_ZERO_INT // 2 * self.step] = -np.inf
        return z

    def proba_batch(self, model: Any, raw_inputs: list[Any]) -> np.ndarray:
        """Posterior via the INTEGER log-softmax (max + LUT); one exp at the very end for linear scale.

        The LUT rounds each log-probability to ~``step``, so the raw ``exp`` sums to ``1 +/- K*step/2``;
        the final float renormalization (free -- we already left integer space for the exp) removes
        that systematic drift without touching the integer pipeline.
        """
        from mixle.engines.lns_nn import log_softmax

        ints = self.int_logits_batch(model, raw_inputs)
        dead = ints <= _LOG_ZERO_INT // 2
        ints = np.where(dead.all(axis=1, keepdims=True), 0, ints)  # all-impossible row -> uniform
        p = np.exp(log_softmax(ints * self.step, self._lns, axis=-1))
        return p / p.sum(axis=1, keepdims=True)

    def predict_batch(self, model: Any, raw_inputs: list[Any]) -> list[str]:
        """Return integer-logit argmax labels for a batch of raw inputs."""
        idx = self.int_logits_batch(model, raw_inputs).argmax(axis=1)  # pure integer decision
        return [self.labels[i] for i in idx]

    def to_spec(self) -> dict[str, Any]:
        """Serialize the LNS structured-classifier adapter."""
        spec = super().to_spec()
        spec["kind"] = self.kind
        spec["step"] = self.step
        return spec

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> LNSStructuredClassifierIO:
        """Reconstruct the LNS structured-classifier adapter from a spec."""
        return cls(spec.get("field_keys"), spec["label_index"], spec["labels"], step=spec.get("step", 1e-2))


register_adapter(LNSStructuredClassifierIO.kind, LNSStructuredClassifierIO.from_spec)


def lns_classifier(student: TaskModel, *, step: float = 1e-2) -> TaskModel:
    """Re-execute a structured student in integer log-space (the LNS rung for structured students).

    The fitted model is unchanged (same factors, same JSON artifact); what changes is *how inference
    runs*: factor log-densities are quantized once at the leaf boundary, and everything above --
    per-label accumulation, mixture folding, the argmax decision, the posterior's log-softmax -- is
    integer add/max/LUT arithmetic (:class:`LNSStructuredClassifierIO`). ``step`` trades fidelity for
    integer width; the dequantized scores match the float classifier within ~``1.5 * step`` per fold.
    This is compute quantization (transcendental-free combination), not weight compression -- pair it
    with the structured student's already compact JSON payload.
    """
    if not isinstance(student.adapter, StructuredClassifierIO) or student.payload != "json":
        raise ValueError("lns_classifier expects a structured student (from distill_structured)")
    adapter = LNSStructuredClassifierIO(
        student.adapter.field_keys, student.adapter.label_index, student.adapter.labels, step=step
    )
    meta = dict(student.meta)
    meta["lns"] = {"step": float(step), "max_fold_error": 1.5 * float(step)}
    return TaskModel(student.model, adapter, payload="json", task=student.task, meta=meta)
