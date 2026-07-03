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
through the artifact as an ``.npz``, and reports its true byte size. LNS (log-number-system,
transcendental-free; ``mixle.engines.lns``) is the remaining rung -- it needs LUT matmul kernels,
so it stays explicitly unimplemented here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.task.model import HashedNGram, HashedRecord, TaskModel, _ClassifierIO, register_adapter

__all__ = ["QuantizedMLP", "QuantizedClassifierIO", "quantize_mlp"]

_QMAX = {8: 127, 4: 7}  # symmetric integer range per weight precision


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
        return np.asarray(model.logits(self.features(raw_inputs))).reshape(len(raw_inputs), -1)

    def to_spec(self) -> dict[str, Any]:
        fam = "text" if isinstance(self.featurizer, HashedNGram) else "record"
        return {
            "kind": self.kind,
            "featurizer_kind": fam,
            "featurizer": self.featurizer.to_spec(),
            "labels": self.labels,
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> QuantizedClassifierIO:
        feat_cls = HashedNGram if spec.get("featurizer_kind", "text") == "text" else HashedRecord
        return cls(feat_cls.from_spec(spec["featurizer"]), spec["labels"])


register_adapter(QuantizedClassifierIO.kind, QuantizedClassifierIO.from_spec)

from mixle.task.artifact import register_arrays_builder  # noqa: E402  (after class defs, avoids partial-module use)

register_arrays_builder("mixle.quantized_mlp", QuantizedMLP.from_arrays)


def _torch_linears(module: Any) -> list[Any]:
    return [m for m in module.modules() if type(m).__name__ == "Linear"]


def quantize_mlp(student: TaskModel, *, bits: int = 8) -> TaskModel:
    """Quantize a trained torch MLP student to an int8/int4, numpy-inference :class:`TaskModel`.

    Per-tensor symmetric weight quantization (``scale = max|W| / qmax`` with ``qmax`` 127 for int8, 7
    for int4); biases stay fp32 (they are a negligible byte fraction and quantizing them buys
    nothing). The returned student reuses the same featurizer and label list, reports
    ``payload="arrays"`` (int4 weights nibble-packed on disk: two per byte), and -- having no torch
    dependence at inference -- qualifies for ``torch_free`` devices. LNS needs LUT matmul kernels
    (``mixle.engines.lns``) and is left explicitly unimplemented.
    """
    if bits not in _QMAX:
        raise NotImplementedError(f"bits={bits}: supported precisions are {sorted(_QMAX)} (LNS not wired yet)")
    if student.payload != "torch":
        raise ValueError("quantize_mlp expects a torch MLP student (payload='torch')")
    linears = _torch_linears(student.model)
    if not linears:
        raise ValueError("student module has no Linear layers to quantize")

    qmax = _QMAX[bits]
    layers: list[tuple[np.ndarray, float, np.ndarray]] = []
    for lin in linears:
        w = lin.weight.detach().cpu().numpy().astype(np.float64)
        b = (
            lin.bias.detach().cpu().numpy().astype(np.float32)
            if lin.bias is not None
            else np.zeros(w.shape[0], dtype=np.float32)
        )
        scale = float(np.max(np.abs(w)) / qmax) or 1.0
        wq = np.clip(np.round(w / scale), -qmax, qmax).astype(np.int8)
        layers.append((wq, scale, b))

    qmodel = QuantizedMLP(layers, bits=bits)
    adapter = QuantizedClassifierIO(student.adapter.featurizer, student.adapter.labels)
    meta = dict(student.meta)
    meta["quantized"] = {
        "bits": bits,
        "scheme": "per-tensor symmetric",
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
