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

import copy
import math
from typing import Any

import numpy as np

from mixle.task.model import (
    HashedNGram,
    HashedRecord,
    ImpossibleEvidenceError,
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
    if clip_percentile is not None:
        try:
            clip_percentile = float(clip_percentile)
        except (TypeError, ValueError) as exc:
            raise ValueError("clip_percentile must be finite and in (0, 100]") from exc
        if not math.isfinite(clip_percentile) or not (0.0 < clip_percentile <= 100.0):
            raise ValueError("clip_percentile must be finite and in (0, 100]")
    qmax = _QMAX[bits]
    try:
        w = np.asarray(w, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be a finite numeric array") from exc
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be a finite numeric array")
    if clip_percentile is None:
        wmax = float(np.max(np.abs(w))) if w.size else 0.0
    else:  # scale off a high percentile so outliers saturate instead of dictating the scale
        wmax = float(np.percentile(np.abs(w), clip_percentile)) if w.size else 0.0
    scale = (wmax / qmax) or 1.0
    wq = np.clip(np.round(w / scale), -qmax, qmax).astype(np.int8)
    return wq, scale


def dequantize_symmetric(wq: np.ndarray, scale: float) -> np.ndarray:
    """Inverse of :func:`quantize_dequantize_array`: ``wq * scale`` as float64."""
    try:
        weights = np.asarray(wq, dtype=np.float64)
        scale = float(scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("quantized weights and scale must be finite numeric values") from exc
    if not np.all(np.isfinite(weights)) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("quantized weights must be finite and scale must be finite and positive")
    return weights * scale


def _pack_nibbles(w: np.ndarray) -> np.ndarray:
    """Pack an int8 array with values in [-7, 7] into two-per-byte uint8 (offset-8 nibbles)."""
    raw = np.asarray(w)
    if not np.issubdtype(raw.dtype, np.integer) or np.any(raw < -7) or np.any(raw > 7):
        raise ValueError("int4 weights must be integers in [-7, 7]")
    flat = (raw.astype(np.int16).reshape(-1) + 8).astype(np.uint8)  # [-7,7] -> [1,15]
    if flat.size % 2:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])  # pad nibble (0 = unused code)
    return (flat[0::2] << 4) | flat[1::2]


def _unpack_nibbles(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Inverse of :func:`_pack_nibbles`: uint8 pairs -> int8 values in [-7, 7] with ``shape``."""
    if not shape or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError("packed int4 shape must contain positive integers")
    raw = np.asarray(packed)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0) or np.any(raw > 255):
        raise ValueError("packed int4 weights must be a one-dimensional byte array")
    p = raw.astype(np.uint8)
    expected = (int(np.prod(shape)) + 1) // 2
    if p.size != expected:
        raise ValueError(f"packed int4 payload has {p.size} bytes, expected {expected}")
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
        normalized = []
        qmax = _QMAX[self.bits]
        previous_width = None
        for index, layer in enumerate(layers):
            if not isinstance(layer, (tuple, list)) or len(layer) != 3:
                raise ValueError("each quantized layer must be a (weights, scale, bias) triple")
            raw_w, raw_scale, raw_b = layer
            weights = np.asarray(raw_w)
            if weights.ndim != 2 or 0 in weights.shape or not np.issubdtype(weights.dtype, np.integer):
                raise ValueError("quantized layer weights must be a nonempty 2-D integer array")
            if np.any(weights < -qmax) or np.any(weights > qmax):
                raise ValueError(f"weight magnitude exceeds the int{self.bits} range [-{qmax}, {qmax}]")
            weights = weights.astype(np.int8, copy=True)
            try:
                scale = float(raw_scale)
                bias = np.asarray(raw_b, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError("quantized layer scale and bias must be numeric") from exc
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError("quantized layer scale must be finite and positive")
            if bias.ndim != 1 or bias.shape[0] != weights.shape[0] or not np.all(np.isfinite(bias)):
                raise ValueError("quantized layer bias must be finite with one value per output")
            if previous_width is not None and weights.shape[1] != previous_width:
                raise ValueError(f"quantized layer {index} input width does not match the preceding output")
            normalized.append((weights, scale, bias.copy()))
            previous_width = weights.shape[0]
        self.layers = normalized

    def logits(self, feats: np.ndarray) -> np.ndarray:
        """Compute dequantized logits for a feature matrix."""
        x = np.asarray(feats, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.layers[0][0].shape[1]:
            raise ValueError("features must be a 2-D matrix matching the quantized model input width")
        if not np.all(np.isfinite(x)):
            raise ValueError("features must contain only finite values")
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
        try:
            raw_k = np.asarray(arrays["n_layers"])
            raw_bits = np.asarray(arrays.get("bits", 8))
            if raw_k.size != 1 or raw_bits.size != 1:
                raise ValueError
            k = int(raw_k.reshape(()))
            bits = int(raw_bits.reshape(()))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("quantized payload must contain scalar n_layers and bits") from exc
        if k <= 0:
            raise ValueError("quantized payload n_layers must be positive")
        layers = []
        for i in range(k):
            try:
                w = arrays[f"w{i}"]
                scale = arrays[f"s{i}"]
                bias = arrays[f"b{i}"]
            except KeyError as exc:
                raise ValueError(f"quantized payload is missing layer {i} arrays") from exc
            if bits == 4:
                try:
                    shape = tuple(int(d) for d in np.asarray(arrays[f"shape{i}"]).reshape(-1))
                except KeyError as exc:
                    raise ValueError(f"quantized payload is missing layer {i} shape") from exc
                w = _unpack_nibbles(w, shape)
            scale_array = np.asarray(scale)
            if scale_array.size != 1:
                raise ValueError(f"quantized layer {i} scale must be scalar")
            layers.append((w, float(scale_array.reshape(())), bias))
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


def _verified_sequential_linears(module: Any) -> list[Any]:
    """Accept only the exact flat Linear/ReLU graph implemented by :class:`QuantizedMLP`."""
    import torch

    if not isinstance(module, torch.nn.Sequential):
        raise ValueError("quantize_mlp supports only a flat torch.nn.Sequential MLP")
    children = list(module._modules.values())
    if not children or len(children) % 2 == 0:
        raise ValueError("MLP graph must be Linear/ReLU pairs ending in Linear")
    if len({id(child) for child in children}) != len(children):
        raise ValueError("MLP graph cannot contain shared module instances")
    for index, child in enumerate(children):
        expected = torch.nn.Linear if index % 2 == 0 else torch.nn.ReLU
        if type(child) is not expected:
            raise ValueError("MLP graph must alternate exact Linear and ReLU modules and end in Linear")
    if list(module.modules())[1:] != children:
        raise ValueError("MLP graph must be flat with no nested or hidden submodules")
    linears = children[0::2]
    for previous, following in zip(linears, linears[1:]):
        if previous.out_features != following.in_features:
            raise ValueError("MLP Linear dimensions do not form one connected chain")
    return linears


def _prove_quantized_parity(source: Any, qmodel: QuantizedMLP) -> float:
    """Prove NumPy inference matches the same graph with dequantized Torch weights."""
    import torch

    reference = copy.deepcopy(source).cpu().eval()
    linears = list(reference._modules.values())[0::2]
    with torch.no_grad():
        for linear, (weights, scale, bias) in zip(linears, qmodel.layers, strict=True):
            linear.weight.copy_(torch.from_numpy(weights.astype(np.float32) * scale))
            if linear.bias is not None:
                linear.bias.copy_(torch.from_numpy(bias))
            elif np.any(bias != 0.0):
                raise RuntimeError("bias-free source layer acquired a nonzero quantized bias")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        probe = torch.randn((8, linears[0].in_features), generator=generator)
        expected = reference(probe).detach().cpu().numpy()
    actual = qmodel.logits(probe.numpy())
    # Scaled to float32 resolution and to the magnitude of the logits, not fixed at 1e-5. Both sides
    # compute the same graph in float32, but Torch and NumPy dispatch to different BLAS kernels that
    # reassociate a matmul differently, so agreement is bounded by unit roundoff (1.19e-7) times the
    # accumulation width times the output scale -- for logits of even modest magnitude that lands
    # right on 1e-5, which is why this passed on macOS/Accelerate and failed on Linux/OpenBLAS with
    # identical weights. What the check exists to catch is a NumPy graph that computes something
    # DIFFERENT: a transposed weight, a dropped bias, a misapplied scale. Those disagree by a
    # fraction of the output, orders of magnitude above this bound, and are still refused.
    resolution = float(np.finfo(np.float32).eps)
    magnitude = float(np.max(np.abs(expected), initial=0.0))
    if expected.shape != actual.shape or not np.allclose(
        expected,
        actual,
        rtol=64.0 * resolution,
        atol=64.0 * resolution * max(1.0, magnitude),
    ):
        raise RuntimeError("quantized NumPy graph failed dequantized Torch parity")
    return float(np.max(np.abs(expected - actual), initial=0.0))


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
    if clip_percentile is not None:
        try:
            clip_percentile = float(clip_percentile)
        except (TypeError, ValueError) as exc:
            raise ValueError("clip_percentile must be finite and in (0, 100]") from exc
        if not math.isfinite(clip_percentile) or not (0.0 < clip_percentile <= 100.0):
            raise ValueError("clip_percentile must be finite and in (0, 100]")
    linears = _verified_sequential_linears(student.model)

    layers: list[tuple[np.ndarray, float, np.ndarray]] = []
    for lin in linears:
        w = lin.weight.detach().cpu().numpy().astype(np.float64)
        b = (
            lin.bias.detach().cpu().numpy().astype(np.float32)
            if lin.bias is not None
            else np.zeros(w.shape[0], dtype=np.float32)
        )
        if not np.all(np.isfinite(w)) or not np.all(np.isfinite(b)):
            raise ValueError("source MLP weights and biases must be finite")
        wq, scale = quantize_dequantize_array(w, bits=bits, clip_percentile=clip_percentile)
        layers.append((wq, scale, b))

    qmodel = QuantizedMLP(layers, bits=bits)
    if not isinstance(student.adapter, _ClassifierIO):
        raise ValueError("quantize_mlp requires a classifier adapter")
    if len(student.adapter.labels) != linears[-1].out_features:
        raise ValueError("classifier label count must match the MLP output width")
    feature_width = getattr(student.adapter.featurizer, "dim", None)
    if feature_width is not None and int(feature_width) != linears[0].in_features:
        raise ValueError("classifier feature width must match the MLP input width")
    parity_error = _prove_quantized_parity(student.model, qmodel)
    adapter = QuantizedClassifierIO(student.adapter.featurizer, student.adapter.labels)
    meta = dict(student.meta)
    meta["quantized"] = {
        "bits": bits,
        "scheme": "per-tensor symmetric",
        "clip_percentile": clip_percentile,
        "fp32_bytes": 4 * sum(w.size for w, _s, _b in layers),
        "dequantized_graph_parity_max_abs_error": parity_error,
        "verified_architecture": "flat_linear_relu",
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

    def __init__(
        self,
        field_keys: list[str] | None,
        label_index: int,
        labels: list[str],
        step: float = 1e-2,
        *,
        field_count: int | None = None,
    ) -> None:
        super().__init__(field_keys, label_index, labels, field_count=field_count)
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

    @staticmethod
    def _factor_signature(factor: Any) -> tuple[Any, ...]:
        """Content signature for compiled categorical state, not merely object identity."""
        pmap = getattr(factor, "pmap", None)
        if isinstance(pmap, dict):
            return (
                "categorical",
                tuple(sorted(((repr(key), float(value)) for key, value in pmap.items()))),
                float(getattr(factor, "log_default_value", -np.inf)),
            )
        dmap = getattr(factor, "dmap", None)
        if isinstance(dmap, dict):
            branches = []
            for key, branch in dmap.items():
                branch_pmap = getattr(branch, "pmap", None)
                if not isinstance(branch_pmap, dict):
                    return ("uncompiled", id(factor))
                branches.append(
                    (
                        repr(key),
                        tuple(sorted((repr(value), float(probability)) for value, probability in branch_pmap.items())),
                    )
                )
            return ("conditional_categorical", tuple(sorted(branches)))
        return ("uncompiled", id(factor))

    def _compiled_tables(self, tree: Any) -> list[tuple[dict, int] | None]:
        cache = getattr(self, "_table_cache", None)
        if cache is None:
            cache = self._table_cache = {}
        key = id(tree)
        signature = (
            tuple(getattr(tree, "parents", ())),
            tuple(self._factor_signature(factor) for factor in tree.factors),
        )
        cached = cache.get(key)
        if cached is None or cached[0] != signature:
            cached = (signature, [self._compile_factor(factor) for factor in tree.factors])
            cache[key] = cached
        return cached[1]

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
        impossible = np.flatnonzero(dead.all(axis=1))
        if impossible.size:
            raise ImpossibleEvidenceError(impossible.tolist())
        p = np.exp(log_softmax(ints * self.step, self._lns, axis=-1))
        return p / p.sum(axis=1, keepdims=True)

    def predict_batch(self, model: Any, raw_inputs: list[Any]) -> list[str]:
        """Return integer-logit argmax labels for a batch of raw inputs."""
        logits = self.int_logits_batch(model, raw_inputs)
        impossible = np.flatnonzero((logits <= _LOG_ZERO_INT // 2).all(axis=1))
        if impossible.size:
            raise ImpossibleEvidenceError(impossible.tolist())
        idx = logits.argmax(axis=1)  # pure integer decision
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
        return cls(
            spec.get("field_keys"),
            spec["label_index"],
            spec["labels"],
            step=spec.get("step", 1e-2),
            field_count=spec.get("field_count"),
        )


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
        student.adapter.field_keys,
        student.adapter.label_index,
        student.adapter.labels,
        step=step,
        field_count=student.adapter.field_count,
    )
    meta = dict(student.meta)
    meta["lns"] = {"step": float(step), "max_fold_error": 1.5 * float(step)}
    return TaskModel(student.model, adapter, payload="json", task=student.task, meta=meta)
