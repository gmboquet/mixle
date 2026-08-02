"""F11: deployment of an accepted checkpoint family through the existing stack.

**Scope, read this first.** F11 is, per the roadmap, thin composition over machinery that already
exists and is already receipted:

* **J2** (:mod:`mixle.task.checkpoint_family_ladder`) builds the family itself: a headline causal LM
  plus a ladder of decreasing-size rungs, each with its own real eval report.
* **I1** (:mod:`mixle.models.unified_quantizer`) turns a torch model's real parameter tensors into
  "I-quantized artifacts" -- per-tensor auto-picked quantization with a measured bytes/error receipt.
* **J4** (:mod:`mixle.task.frontier_to_native`) builds the edge tier: a frontier-distilled, LNS-compressed,
  calibrated student served behind a :class:`~mixle.task.cascade.Cascade`, with its own cost/quality receipt.
* **Economics** (:mod:`mixle.task.economics`) supplies :class:`~mixle.task.economics.CostModel`, the unit
  costs a per-request dollar figure is built from.

This module does not re-derive compression, quantization, distillation, calibration, or cost
arithmetic; it takes J2's family, I1-quantizes every accepted rung (plus the headline) into a real
measured artifact, and reports a cost/quality **frontier** across those artifacts -- the roadmap's "end-to-end
serve receipt (cost/quality frontier plot)" acceptance criterion -- next to J4's own served-cascade
receipt for the edge tier.

**A real constraint this discovered, not glossed over: two receipted "quality" axes don't share
units, so this module does not force them onto one line.** J2's family rungs are causal LMs scored by
F10's synthetic eval suite (:mod:`mixle.models.eval_harness`) -- perplexity plus three accuracy-style
tasks. J4's edge student is a distilled classifier scored by held-out label accuracy on whatever task
it was trained for. Both are real, receipted quality numbers, but they measure different capabilities
on different tasks; averaging them into one scalar would manufacture a false equivalence the
underlying receipts do not support. So :func:`deploy_family` reports two things side by side instead:
a same-axis cost/quality **frontier across the J2 family + headline** (comparable, because every point
is the SAME eval suite on the SAME task), and J4's own :class:`~mixle.task.frontier_to_native.CascadeReceipt`
as the edge tier's cost/quality trade on ITS task, unmerged. ``ServeReceipt.summary()`` prints both.

**Why this does not build a full :class:`~mixle.task.router.Router` across the whole family.**
``Router`` requires every non-final tier to expose ``decide(x)`` (a :class:`~mixle.task.calibrate.CalibratedTaskModel`
shape returning a label or ``ESCALATE``); J2's family rungs are plain causal LMs with no calibrated
decision boundary, and building one would mean inventing an eval-suite-specific classifier wrapper
this task does not ask for. J4's own 2-tier :class:`~mixle.task.cascade.Cascade` (student, calibrated;
frontier, a callable) already IS the calibrated-routing piece this module reuses unmodified via
:func:`~mixle.task.frontier_to_native.build_served_cascade` -- what's genuinely new here is turning
J2's *causal-LM* family into comparable priced artifacts, which is a quantization/costing question,
not a routing one.

**Artifact and cost contracts.** A family point is emitted only for a rung that passed J2's quality
gate. Each point retains the exact per-parameter quantized payload, an executable model materialized
from that payload, a canonical state payload, and a SHA-256 content address. The receipt therefore
identifies something that can actually be served and independently verified, rather than summarizing
temporary arrays that have already been discarded.

Every deployable is executed on the caller's declared ``probe_inputs`` under PyTorch's FLOP profiler.
Per-request cost is priced from those measured operations relative to the headline:
``cost_per_request = cost.c_frontier * (inference_flops / headline_inference_flops)``. Storage bytes
remain separately receipted and use each source tensor's real dtype width.
"""

from __future__ import annotations

import copy
import hashlib
import struct
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.models.unified_quantizer import QuantizationBudgetError, QuantizedTensor, quantize_tensor
from mixle.task.checkpoint_family_ladder import FamilyLadderResult
from mixle.task.economics import CostModel
from mixle.task.frontier_to_native import CascadeReceipt
from mixle.utils.immutable import freeze_receipt_container

__all__ = [
    "ArtifactReceipt",
    "DenseParameter",
    "DeployableArtifact",
    "FrontierPoint",
    "QuantizedParameter",
    "ServeReceipt",
    "StateTensor",
    "quantize_family_artifacts",
    "deploy_family",
]

# The three F10 eval tasks that are accuracy-style (higher_is_better=True, bounded roughly in [0, 1]);
# held_out_perplexity is excluded (lower_is_better, unbounded) so the frontier's quality axis is a
# single consistent direction ("higher = better") without needing to invert/rescale perplexity.
_QUALITY_TASKS = ("modular_arithmetic", "parity_reasoning", "in_context_induction")


def _quality_score(eval_report: Any) -> float:
    """Mean accuracy across :data:`_QUALITY_TASKS` -- F10's real per-task scores, not a re-derived metric."""
    scores = eval_report.scores()
    vals = [scores[t] for t in _QUALITY_TASKS if t in scores]
    if not vals:
        raise ValueError(f"eval_report for {eval_report.checkpoint_id!r} has none of {_QUALITY_TASKS}")
    return float(np.mean(vals))


@dataclass(frozen=True)
class ArtifactReceipt:
    """One model's I1-quantized deployment artifact: real measured bytes/error rolled up over every
    parameter tensor. Tensors whose encoding overhead would exceed their dense source are retained
    exactly and reported as ``dense`` rather than being made larger in the name of compression."""

    name: str
    n_tensors: int
    dense_bytes: int
    quantized_bytes: int
    compression_ratio: float
    mean_reconstruction_error: float
    method_counts: dict[str, int]
    content_digest: str

    def __post_init__(self) -> None:
        # Detached and sealed: these were caller-owned containers stored by reference on a
        # frozen dataclass, so a mutation after construction rewrote evidence that had already
        # been recorded (MXR-080-1876).
        object.__setattr__(self, "method_counts", freeze_receipt_container(self.method_counts))

    def summary(self) -> str:
        methods = ", ".join(f"{m}={n}" for m, n in sorted(self.method_counts.items()))
        return (
            f"{self.name}: {self.n_tensors} tensors, {self.dense_bytes}B dense -> {self.quantized_bytes}B "
            f"quantized ({self.compression_ratio:.2f}x), mean_recon_error={self.mean_reconstruction_error:.4g} "
            f"[{methods}], sha256={self.content_digest[:12]}"
        )


@dataclass(frozen=True)
class StateTensor:
    """Canonical bytes for one tensor in the executable artifact state."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes = field(repr=False)


@dataclass(frozen=True)
class DenseParameter:
    """Exact fallback for a tensor whose quantized payload plus metadata would be larger."""

    array: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        array = np.ascontiguousarray(self.array).copy()
        array.setflags(write=False)
        object.__setattr__(self, "array", array)

    @property
    def method(self) -> str:
        return "dense"

    def reconstruct(self) -> np.ndarray:
        return self.array.copy()

    def nbytes(self) -> int:
        return self.array.nbytes


@dataclass(frozen=True)
class QuantizedParameter:
    """A named compressed or exact-dense payload retained by a deployable artifact."""

    name: str
    tensor: QuantizedTensor | DenseParameter = field(repr=False)


def _state_payload(model: Any) -> tuple[StateTensor, ...]:
    """Snapshot every unique parameter and buffer without assuming NumPy supports its dtype."""
    import torch

    tensors = [*model.named_parameters(), *model.named_buffers()]
    result: list[StateTensor] = []
    names: set[str] = set()
    for name, tensor in tensors:
        if name in names:
            raise ValueError(f"model state contains duplicate tensor name {name!r}")
        names.add(name)
        value = tensor.detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        result.append(StateTensor(name=name, dtype=str(value.dtype), shape=tuple(value.shape), data=raw))
    return tuple(sorted(result, key=lambda item: item.name))


def _content_digest(model: Any, state: tuple[StateTensor, ...]) -> str:
    digest = hashlib.sha256(b"mixle-deployable-torch-state-v1\0")
    architecture = f"{type(model).__module__}.{type(model).__qualname__}".encode()
    digest.update(struct.pack("<Q", len(architecture)))
    digest.update(architecture)
    for item in state:
        for value in (item.name.encode(), item.dtype.encode()):
            digest.update(struct.pack("<Q", len(value)))
            digest.update(value)
        digest.update(struct.pack("<Q", len(item.shape)))
        for dimension in item.shape:
            digest.update(struct.pack("<Q", dimension))
        digest.update(struct.pack("<Q", len(item.data)))
        digest.update(item.data)
    return digest.hexdigest()


@dataclass(frozen=True)
class DeployableArtifact:
    """Content-addressed executable model plus the exact payload that produced its served weights."""

    name: str
    model: Any = field(repr=False, compare=False)
    parameters: tuple[QuantizedParameter, ...] = field(repr=False)
    state: tuple[StateTensor, ...] = field(repr=False)
    receipt: ArtifactReceipt

    @property
    def content_digest(self) -> str:
        return self.receipt.content_digest

    @property
    def dense_bytes(self) -> int:
        return self.receipt.dense_bytes

    @property
    def quantized_bytes(self) -> int:
        return self.receipt.quantized_bytes

    @property
    def n_tensors(self) -> int:
        return self.receipt.n_tensors

    @property
    def compression_ratio(self) -> float:
        return self.receipt.compression_ratio

    @property
    def mean_reconstruction_error(self) -> float:
        return self.receipt.mean_reconstruction_error

    @property
    def method_counts(self) -> dict[str, int]:
        return dict(self.receipt.method_counts)

    def verify(self) -> bool:
        """Return whether the executable model still matches its immutable content address."""
        state = _state_payload(self.model)
        return state == self.state and _content_digest(self.model, state) == self.content_digest

    def serve(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the verified artifact; mutation after publication fails closed."""
        if not self.verify():
            raise RuntimeError(f"deployable artifact {self.name!r} no longer matches {self.content_digest}")
        return self.model(*args, **kwargs)

    def summary(self) -> str:
        return self.receipt.summary()


def quantize_family_artifacts(model: Any, *, name: str, bits: int = 8, seed: int = 0) -> DeployableArtifact:
    """I1: ``quantize_tensor(method="auto")`` over every real, non-empty parameter tensor of ``model``.

    Retains I1's selected payload for each parameter and uses an exact dense fallback when no encoding
    fits the literal byte budget. The per-model receipt contains real totals over the source dtypes,
    never a single assumed bits-per-parameter constant. ``seed`` is offset per tensor (``seed + i``).
    """
    import torch

    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    deployed = copy.deepcopy(model)
    deployed.eval()
    original_parameters = dict(model.named_parameters())
    deployed_parameters = dict(deployed.named_parameters())
    if original_parameters.keys() != deployed_parameters.keys():
        raise ValueError("copied model changed its named-parameter topology")

    errors: list[float] = []
    method_counts: dict[str, int] = {}
    payloads: list[QuantizedParameter] = []
    buffer_bytes = sum(buffer.detach().numel() * buffer.detach().element_size() for buffer in model.buffers())
    dense_bytes = buffer_bytes
    quantized_bytes = buffer_bytes
    for i, (parameter_name, p) in enumerate(original_parameters.items()):
        arr = p.detach().cpu().numpy()
        if arr.size == 0:
            continue
        try:
            payload: QuantizedTensor | DenseParameter = quantize_tensor(arr, method="auto", bits=bits, seed=seed + i)
        except QuantizationBudgetError:
            payload = DenseParameter(arr)
        payloads.append(QuantizedParameter(parameter_name, payload))
        dense_bytes += arr.nbytes
        if isinstance(payload, DenseParameter):
            quantized_bytes += payload.nbytes()
            errors.append(0.0)
        else:
            quantized_bytes += payload.receipt.nbytes
            errors.append(payload.receipt.reconstruction_error)
        method_counts[payload.method] = method_counts.get(payload.method, 0) + 1
        target = deployed_parameters[parameter_name]
        reconstructed = torch.as_tensor(payload.reconstruct(), dtype=target.dtype, device=target.device)
        with torch.no_grad():
            target.copy_(reconstructed)
    if not payloads:
        raise ValueError(f"model {name!r} has no non-empty parameter tensors to quantize")

    state = _state_payload(deployed)
    content_digest = _content_digest(deployed, state)
    receipt = ArtifactReceipt(
        name=name,
        n_tensors=len(payloads),
        dense_bytes=dense_bytes,
        quantized_bytes=quantized_bytes,
        compression_ratio=(dense_bytes / quantized_bytes) if quantized_bytes else float("inf"),
        mean_reconstruction_error=float(np.mean(errors)),
        method_counts=method_counts,
        content_digest=content_digest,
    )
    return DeployableArtifact(
        name=name,
        model=deployed,
        parameters=tuple(payloads),
        state=state,
        receipt=receipt,
    )


@dataclass(frozen=True)
class FrontierPoint:
    """One priced, quality-scored point on the family's cost/quality frontier."""

    name: str
    real_target: str
    artifact: DeployableArtifact
    inference_flops: int
    cost_per_request: float
    quality: float


@dataclass(frozen=True)
class ServeReceipt:
    """F11's end-to-end serve receipt: the J2-family cost/quality frontier, plus J4's own edge-tier
    served-cascade receipt reported alongside it (see the module docstring for why the two axes are
    kept separate rather than merged)."""

    points: list[FrontierPoint]
    probe_digest: str
    edge_cascade: CascadeReceipt | None = field(default=None)

    def __post_init__(self) -> None:
        # Detached and sealed: these were caller-owned containers stored by reference on a
        # frozen dataclass, so a mutation after construction rewrote evidence that had already
        # been recorded (MXR-080-1876).
        object.__setattr__(self, "points", freeze_receipt_container(self.points))

    def frontier_sorted_by_cost(self) -> list[FrontierPoint]:
        return sorted(self.points, key=lambda p: p.cost_per_request)

    def is_monotone_frontier(self, *, tol: float = 1e-9) -> bool:
        """Real, checkable claim: walking the family points cheapest-first, quality never DROPS below
        tolerance -- i.e. paying more for a bigger rung is never strictly worse on the eval suite."""
        ordered = self.frontier_sorted_by_cost()
        return all(b.quality >= a.quality - tol for a, b in zip(ordered, ordered[1:]))

    def frontier_plot(self, *, width: int = 40) -> str:
        """A deterministic, dependency-free ASCII cost/quality scatter (no matplotlib in this repo's
        dependency set) -- x = cost_per_request (log-scaled across the family's real span), y = quality.
        This is a real, reproducible rendering of the receipted numbers below, not decoration."""
        ordered = self.frontier_sorted_by_cost()
        costs = [p.cost_per_request for p in ordered]
        c_lo, c_hi = min(costs), max(costs)
        log_lo = np.log10(c_lo) if c_lo > 0 else 0.0
        log_hi = np.log10(c_hi) if c_hi > 0 else 0.0
        span = (log_hi - log_lo) or 1.0

        rows = []
        for p in ordered:
            log_c = np.log10(p.cost_per_request) if p.cost_per_request > 0 else log_lo
            col = int(round((log_c - log_lo) / span * (width - 1)))
            bar = [" "] * width
            bar[col] = "*"
            rows.append(f"  {''.join(bar)}  {p.name:12s} cost=${p.cost_per_request:.6f}  quality={p.quality:.3f}")
        header = f"  cost/quality frontier (x: log cost, ${c_lo:.6f}..${c_hi:.6f}; * = one family point)"
        return "\n".join([header, *rows])

    def summary(self) -> str:
        lines = [
            "F11 served family -- J2 family cost/quality frontier:",
            f"  probe sha256: {self.probe_digest}",
        ]
        for p in self.frontier_sorted_by_cost():
            lines.append(
                f"  {p.name:12s} ({p.real_target}): cost=${p.cost_per_request:.6f}/req quality={p.quality:.3f} "
                f"work={p.inference_flops} FLOPs | {p.artifact.summary()}"
            )
        lines.append(f"  monotone (quality non-decreasing in cost): {self.is_monotone_frontier()}")
        lines.append("")
        lines.append(self.frontier_plot())
        if self.edge_cascade is not None:
            lines.append("")
            lines.append("J4 edge-tier served-cascade receipt (own task, own quality axis):")
            lines.append(self.edge_cascade.summary())
        return "\n".join(lines)


def _probe_digest(probe_inputs: Any) -> str:
    """Content-address a nested probe of tensors and simple immutable values."""
    import torch

    digest = hashlib.sha256(b"mixle-deployment-probe-v1\0")

    def write(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            write(("tensor", str(tensor.dtype), tuple(tensor.shape), raw))
        elif isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("probe input dictionaries must have string keys")
            digest.update(b"d")
            for key in sorted(value):
                write(key)
                write(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(b"t" if isinstance(value, tuple) else b"l")
            digest.update(struct.pack("<Q", len(value)))
            for item in value:
                write(item)
        elif isinstance(value, bytes):
            digest.update(b"b")
            digest.update(struct.pack("<Q", len(value)))
            digest.update(value)
        elif isinstance(value, str):
            encoded = value.encode()
            digest.update(b"s")
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
        elif value is None or isinstance(value, (bool, int, float)):
            encoded = repr(value).encode()
            digest.update(type(value).__name__.encode() + b"\0")
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
        else:
            raise TypeError(f"unsupported probe input type: {type(value).__name__}")

    write(probe_inputs)
    return digest.hexdigest()


def _measure_inference_flops(artifact: DeployableArtifact, probe_inputs: Any) -> int:
    """Execute one probe and return profiler-observed floating-point operations."""
    import torch

    if probe_inputs is None:
        raise ValueError("probe_inputs are required to measure deployment inference work")
    activities = [torch.profiler.ProfilerActivity.CPU]
    try:
        device = next(artifact.model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def invoke() -> Any:
        if isinstance(probe_inputs, dict):
            return artifact.model(**probe_inputs)
        if isinstance(probe_inputs, tuple):
            return artifact.model(*probe_inputs)
        return artifact.model(probe_inputs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with torch.profiler.profile(activities=activities, with_flops=True) as profile:
            with torch.inference_mode():
                output = invoke()
    if not isinstance(output, torch.Tensor) or output.numel() == 0:
        raise ValueError(f"probe for artifact {artifact.name!r} must return a non-empty torch.Tensor")
    if not bool(torch.isfinite(output).all()):
        raise ValueError(f"probe for artifact {artifact.name!r} returned non-finite values")
    flops = sum(int(event.flops or 0) for event in profile.key_averages())
    if flops <= 0:
        raise ValueError(f"profiler measured no floating-point work for artifact {artifact.name!r}")
    return flops


def deploy_family(
    family: FamilyLadderResult,
    headline_model: Any,
    *,
    probe_inputs: Any,
    edge_cascade_receipt: CascadeReceipt | None = None,
    cost: CostModel | None = None,
    bits: int = 8,
    seed: int = 0,
) -> ServeReceipt:
    """Build F11's end-to-end serve receipt from a J2 :class:`FamilyLadderResult` plus its own
    ``headline_model``.

    Every accepted rung (and the headline) is I1-quantized into a real
    :class:`DeployableArtifact`
    (:func:`quantize_family_artifacts`); each artifact's per-request cost is priced off ``cost``
    (a :class:`~mixle.task.economics.CostModel`, ``CostModel(c_frontier=1.0)`` by default) scaled by
    measured inference work relative to the headline's own artifact (see module docstring); quality is
    J2's own real :class:`~mixle.models.eval_harness.EvalReport` for that rung/headline, reduced to
    :func:`_quality_score`. ``edge_cascade_receipt`` (optional) is J4's own
    :class:`~mixle.task.frontier_to_native.CascadeReceipt` for the edge tier, carried through
    unmodified and reported alongside the family frontier rather than merged into it.
    """
    cost = cost if cost is not None else CostModel(c_frontier=1.0)

    headline_artifact = quantize_family_artifacts(headline_model, name="headline", bits=bits, seed=seed)
    headline_flops = _measure_inference_flops(headline_artifact, probe_inputs)

    points = [
        FrontierPoint(
            name="headline",
            real_target="headline",
            artifact=headline_artifact,
            inference_flops=headline_flops,
            cost_per_request=cost.c_frontier,
            quality=_quality_score(family.headline_eval),
        )
    ]
    for i, rung in enumerate(rung for rung in family.rungs if rung.within_eval_budget):
        artifact = quantize_family_artifacts(rung.model, name=rung.name, bits=bits, seed=seed + 1000 * (i + 1))
        inference_flops = _measure_inference_flops(artifact, probe_inputs)
        rel_cost = cost.c_frontier * (inference_flops / headline_flops)
        points.append(
            FrontierPoint(
                name=rung.name,
                real_target=rung.real_target,
                artifact=artifact,
                inference_flops=inference_flops,
                cost_per_request=rel_cost,
                quality=_quality_score(rung.eval_report),
            )
        )

    return ServeReceipt(
        points=points,
        probe_digest=_probe_digest(probe_inputs),
        edge_cascade=edge_cascade_receipt,
    )
