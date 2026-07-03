"""Post-training quantization of distilled MLP students: int8 weights, numpy-only inference.

An fp32 MLP student costs ``4 bytes x params`` and needs torch at inference. Quantizing to int8
(per-tensor symmetric: ``W ~ round(W / s) * s`` with one fp32 scale per layer) cuts the weight bytes
4x -- and because the dequantized forward pass is three numpy matmuls, the quantized student needs
**no torch at all** on the device: it joins the structured students in the torch-free deployable
class, while keeping the MLP's shape. Accuracy is whatever it *measures* after quantization -- the
edge search (:func:`mixle.task.edge.distill_for_edge`) scores the quantized model's real agreement,
so the bits axis trades measured bytes against measured fidelity, never assumed ones.

``quantize_mlp(student)`` converts a trained torch student in place of retraining; the result stores
int8 arrays (``payload="arrays"``), round-trips through the artifact as an ``.npz``, and reports its
true byte size. int4 / LNS (log-number-system, transcendental-free) are the natural next rungs; they
need packing and LUT kernels respectively, so int8 is the honestly-shipped first step.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mixle.task.model import HashedNGram, HashedRecord, TaskModel, _ClassifierIO, register_adapter

__all__ = ["QuantizedMLP", "QuantizedClassifierIO", "quantize_mlp"]


class QuantizedMLP:
    """An int8-weight MLP with a pure-numpy forward pass.

    ``layers`` is ``[(W_int8 (out, in), scale fp32, bias fp32 (out,)), ...]``; the forward is
    ``x @ (W * s).T + b`` with ReLU between layers -- exactly the dequantized version of the trained
    torch stack, so its logits match torch-on-dequantized-weights to float tolerance.
    """

    def __init__(self, layers: list[tuple[np.ndarray, float, np.ndarray]]) -> None:
        if not layers:
            raise ValueError("QuantizedMLP needs at least one layer")
        self.layers = [(np.asarray(w, dtype=np.int8), float(s), np.asarray(b, dtype=np.float32)) for w, s, b in layers]

    def logits(self, feats: np.ndarray) -> np.ndarray:
        x = np.asarray(feats, dtype=np.float32)
        last = len(self.layers) - 1
        for i, (w, s, b) in enumerate(self.layers):
            x = x @ (w.astype(np.float32) * s).T + b
            if i != last:
                x = np.maximum(x, 0.0)
        return x

    def nbytes(self) -> int:
        """Deployable payload bytes: int8 weights + fp32 biases + one fp32 scale per layer."""
        return int(sum(w.nbytes + b.nbytes + 4 for w, _s, b in self.layers))

    def macs(self) -> int:
        """Per-inference multiply-accumulates (int8 x fp32 dequant multiplies count the same)."""
        return int(sum(w.shape[0] * w.shape[1] for w, _s, _b in self.layers))

    # -- artifact arrays payload --
    def to_arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {"n_layers": np.asarray(len(self.layers), dtype=np.int64)}
        for i, (w, s, b) in enumerate(self.layers):
            out[f"w{i}"] = w
            out[f"s{i}"] = np.asarray(s, dtype=np.float32)
            out[f"b{i}"] = b
        return out

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> QuantizedMLP:
        k = int(np.asarray(arrays["n_layers"]).reshape(()))
        return cls(
            [(arrays[f"w{i}"], float(np.asarray(arrays[f"s{i}"]).reshape(())), arrays[f"b{i}"]) for i in range(k)]
        )


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
    """Quantize a trained torch MLP student to an int8, numpy-inference :class:`TaskModel`.

    Per-tensor symmetric weight quantization (``scale = max|W| / 127``); biases stay fp32 (they are a
    negligible byte fraction and quantizing them buys nothing). The returned student reuses the same
    featurizer and label list, reports ``payload="arrays"``, and -- having no torch dependence at
    inference -- qualifies for ``torch_free`` devices. Only ``bits=8`` ships today; int4 needs nibble
    packing and LNS needs LUT kernels (``mixle.engines.lns``), both left explicitly unimplemented.
    """
    if bits != 8:
        raise NotImplementedError(f"bits={bits}: only int8 post-training quantization is implemented")
    if student.payload != "torch":
        raise ValueError("quantize_mlp expects a torch MLP student (payload='torch')")
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
        scale = float(np.max(np.abs(w)) / 127.0) or 1.0
        wq = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        layers.append((wq, scale, b))

    qmodel = QuantizedMLP(layers)
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
