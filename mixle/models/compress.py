"""One ``compress()`` front door unifying sampling KD, non-sampling (data-free), and hybrid
compression (roadmap J1).

Three method FAMILIES, one entry point
----------------------------------------
1. **``"sampling_kd"``** -- this codebase's EXISTING response/hint/attention/relational KD
   machinery (:mod:`mixle.task.distill_methods`, "already in-tree and load-bearing" per the roadmap
   context anchors). :func:`compress` wraps :func:`~mixle.task.distill_methods.response_distill`
   directly: build a fresh, smaller student architecture and train it against the real teacher on
   REAL calibration data (real forward/backward passes) -- this is the expensive-but-accurate full
   method every other method here is measured against.
2. **``"non_sampling"``** -- the data-free compression stack G1-G3 already built: G1's moment-
   propagation surrogate (:mod:`mixle.models.moment_propagation`) and G3's coarsening operator
   (:func:`mixle.models.coarsening.coarsen`, which itself calls G1's laws and is the direct
   consumer of G2's Sigma-weighted projections per that module's own docstring). No real forward
   pass over calibration data is used anywhere in this path -- only propagated Gaussian LAWS.
3. **``"hybrid"``** -- the genuinely novel piece: run ``non_sampling`` first, read off G1's own
   per-stage closure-error receipts that :func:`~mixle.models.coarsening.coarsen` already produces
   (``ScaleReceipt.surrogate_closure_error`` -- how far the Gaussian-surrogate assumption itself is
   from a real Monte Carlo check at that stage, i.e. "where is the surrogate blind"), rank stages by
   that receipt via A5's :func:`~mixle.task.acquire.acquire` (adapted here: the "pool" is compression
   STAGES rather than A5's original unlabeled data pool, and the score is the closure-error receipt
   rather than an EIG/disagreement score over a classifier's predictions -- see
   :func:`_closure_error_strategy`), and spend a REAL but TINY sampling-KD budget
   (:func:`_finetune_stages`, built directly on :func:`~mixle.task.distill_methods.kd_loss`) ONLY on
   the worst-closure-error stages' own parameters -- everywhere else keeps the free non_sampling
   result untouched.

**``method="auto"``**: mirrors roadmap I1's (:mod:`mixle.models.unified_quantizer`) exact pattern
one level up the compression stack -- run every method once, measure its REAL quality against the
teacher, and let :class:`mixle.task.bandit.UCB1` (I1's own picker, not a new one) sweep the arms
(here the arms are the three METHOD FAMILIES, "which compression method for this
layer/stage"-scale, rather than I1's four per-tensor quantization primitives) and pick the highest
real reward. Every choice -- auto or explicit -- carries a :class:`CompressionReceipt`, I1's
``QuantizationReceipt`` translated to this module's vocabulary: the real measured quality and
sample cost of the method that WAS picked, plus (for auto) the same real numbers for every method
that was considered and rejected.

Build vs. borrow
-----------------
Nothing here reimplements G1/G2/G3, A5, or the sampling-KD machinery -- :func:`compress` is
composition: :func:`~mixle.models.coarsening.coarsen` for ``non_sampling``,
:func:`~mixle.task.distill_methods.response_distill` / :func:`~mixle.task.distill_methods.kd_loss`
for ``sampling_kd``/the hybrid fine-tune step, :func:`~mixle.task.acquire.acquire` for hybrid's
stage ranking, :class:`~mixle.task.bandit.UCB1` for ``auto``.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.models.coarsening import CoarsenResult, ScaleReceipt, coarsen
from mixle.models.moment_propagation import GaussianLaw, _as_law, _to_numpy
from mixle.task.acquire import acquire, register_strategy
from mixle.task.bandit import UCB1
from mixle.task.distill_methods import DistillResult, _agreement, kd_loss, response_distill

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

__all__ = [
    "METHODS",
    "MethodCandidate",
    "CompressionReceipt",
    "CompressedModel",
    "compress",
]

METHODS: tuple[str, ...] = ("sampling_kd", "non_sampling", "hybrid")

_STAGE_INDEX_RE = re.compile(r"^(?:merged|kept)\[(\d+)\]")


# --------------------------------------------------------------------------------------------------
# receipts
# --------------------------------------------------------------------------------------------------


@dataclass
class MethodCandidate:
    """Real, measured numbers for one method family considered for one ``compress()`` call -- the
    raw material every :class:`CompressionReceipt` (chosen or rejected) is built from, mirroring
    :class:`mixle.models.unified_quantizer.MethodCandidate`."""

    method: str
    quality: float  # teacher-agreement on eval_data, higher is better; nan if unmeasured
    sample_count: int  # real calibration samples this method actually consumed
    reward: float  # what the auto-pick bandit compared


@dataclass
class CompressionReceipt:
    """Explains why ``method`` was used: its own measured numbers, plus -- for auto-pick -- the
    same real numbers for every OTHER method considered and rejected. Mirrors
    :class:`mixle.models.unified_quantizer.QuantizationReceipt`."""

    method: str
    auto: bool
    quality: float
    sample_count: int
    candidates: dict[str, MethodCandidate] = field(default_factory=dict)
    notes: str = ""

    def rejected(self) -> dict[str, MethodCandidate]:
        return {m: c for m, c in self.candidates.items() if m != self.method}


@dataclass
class CompressedModel:
    """Unified result: the compressed module plus the receipt explaining which method produced it,
    and (when the chosen/underlying path touched ``non_sampling``) the raw per-stage
    :class:`~mixle.models.coarsening.ScaleReceipt` map that path's closure-error signal came from."""

    model: Any
    method: str
    receipt: CompressionReceipt
    non_sampling_receipts: dict[str, ScaleReceipt] = field(default_factory=dict)
    hybrid_selected_stages: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------------------


def _default_input_law(model: Any) -> GaussianLaw:
    """Data-free input law for the token+pos residual stream: mean/covariance of the model's own
    token-embedding rows (the marginal distribution over which token enters the stack), the same
    "no real data, just the model's own weights" spirit :mod:`mixle.models.moment_propagation`
    documents for its input law."""
    w = _to_numpy(model.tok.weight)
    mu = w.mean(axis=0)
    covar = np.cov(w, rowvar=False) + 1e-4 * np.eye(w.shape[1])
    return _as_law(mu, covar)


def _as_long_tensor(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.long()
    return torch.as_tensor(x, dtype=torch.long)


def _quality(student: Any, teacher: Any, eval_data: Any) -> float:
    """Teacher-agreement (fraction of matching argmax next-token predictions) on ``eval_data`` --
    higher is better, ``1.0`` for a model identical to the teacher, the common scalar every method
    (and the no-compression baseline) is compared on."""
    if eval_data is None:
        return float("nan")
    x = _as_long_tensor(eval_data)
    student.eval()
    teacher.eval()
    with torch.no_grad():
        s_logits = student(x)
        t_logits = teacher(x)
    return _agreement(s_logits, t_logits)


# --------------------------------------------------------------------------------------------------
# 1. non_sampling -- G1 + G3 (coarsen), data-free
# --------------------------------------------------------------------------------------------------


def _non_sampling(
    model: Any, input_law: GaussianLaw, budget: float, trust_region: float, n_mc: int, seed: int
) -> CoarsenResult:
    # coarsen() re-uses the ORIGINAL block modules directly (kept blocks are the same nn.Module
    # instances, MergedBlock wraps block_a/block_b by reference) rather than copying them -- correct
    # for a purely data-free, inference-only compression, but hybrid's own fine-tune step below runs
    # real backprop through the coarsened model, and doing that against ALIASED parameters would
    # silently mutate the TEACHER too. Deep-copy the model first so every downstream user of the
    # coarsened result owns independent weights.
    return coarsen(
        copy.deepcopy(model), budget=budget, trust_region=trust_region, input_law=input_law, n_mc=n_mc, seed=seed
    )


# --------------------------------------------------------------------------------------------------
# 2. sampling_kd -- wraps mixle.task.distill_methods.response_distill directly
# --------------------------------------------------------------------------------------------------


def _sampling_kd(
    model: Any, calibration_data: Any, target_n_layer: int, epochs: int, lr: float, seed: int
) -> DistillResult:
    from mixle.models.transformer import build_causal_lm

    student = build_causal_lm(
        vocab=model.vocab, d_model=model.d_model, n_layer=target_n_layer, n_head=model.n_head, block=model.block
    )
    x = _as_long_tensor(calibration_data)
    return response_distill(student, model, x, epochs=epochs, lr=lr, seed=seed, baseline=False)


# --------------------------------------------------------------------------------------------------
# 3. hybrid -- non_sampling base + A5-style acquire() ranking of G1's closure-error receipts +
#    real sampling-KD micro-calibration confined to the worst stages
# --------------------------------------------------------------------------------------------------


def _closure_error_strategy(pool: Any, model: Any, **_: Any) -> np.ndarray:
    """The hybrid method's acquisition SCORE: G1's own per-stage closure-error receipt (how far the
    Gaussian-surrogate assumption is from a real Monte Carlo check at that stage). ``pool`` is a
    list of ``(stage_name, ScaleReceipt)`` pairs (not A5's original text/record pool); ``model`` is
    unused -- this is the "custom callable strategy" extension point
    :func:`mixle.task.acquire.acquire` documents, adapted from "which pool item is most worth
    labeling" to "which compression stage is most worth REAL supervision"."""
    return np.array([r.surrogate_closure_error for _, r in pool], dtype=np.float64)


register_strategy("closure_error", _closure_error_strategy)


def _stage_block_indices(stage_names: list[str]) -> list[int]:
    """Parse the ``out_idx`` (index into the coarsened model's ``blocks`` list) embedded in
    :func:`~mixle.models.coarsening.coarsen`'s own receipt-map key format (``"merged[i]<-..."`` /
    ``"kept[i]<-..."``)."""
    indices = []
    for name in stage_names:
        m = _STAGE_INDEX_RE.match(name)
        if m:
            indices.append(int(m.group(1)))
    return indices


def _set_trainable_blocks(model: Any, block_indices: list[int]) -> None:
    """Freeze every parameter except the flagged blocks' -- the "receipt-directed" part of
    receipt-directed micro-calibration: only the closure-error-flagged stages get real gradient
    updates."""
    for p in model.parameters():
        p.requires_grad_(False)
    for idx in block_indices:
        for p in model.blocks[idx].parameters():
            p.requires_grad_(True)


def _unfreeze_all(model: Any) -> None:
    for p in model.parameters():
        p.requires_grad_(True)


def _finetune_stages(student: Any, teacher: Any, x: Any, epochs: int, lr: float) -> None:
    """A small, direct sampling-KD fine-tune loop built on
    :func:`mixle.task.distill_methods.kd_loss` -- deliberately NOT
    :func:`~mixle.task.distill_methods.response_distill`, which re-initializes the student's
    parameters before training (correct for building a fresh ``sampling_kd`` student, wrong here:
    hybrid must FINE-TUNE the existing non_sampling weights, not discard them). Only parameters with
    ``requires_grad=True`` (set by :func:`_set_trainable_blocks`) receive gradient updates."""
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(x)
    trainable = [p for p in student.parameters() if p.requires_grad]
    if not trainable:
        return
    opt = torch.optim.Adam(trainable, lr=lr)
    student.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        loss = kd_loss(student(x), teacher_logits, None, temperature=4.0, alpha=1.0)
        loss.backward()
        opt.step()
    student.eval()


def _hybrid(
    model: Any,
    input_law: GaussianLaw,
    budget: float,
    trust_region: float,
    n_mc: int,
    seed: int,
    calibration_data: Any,
    sample_fraction: float,
    sample_budget: int | None,
    max_stages: int,
    epochs: int,
    lr: float,
) -> tuple[Any, CoarsenResult, dict[str, Any]]:
    result = _non_sampling(model, input_law, budget, trust_region, n_mc, seed)
    ns_model = result.model

    pool = [(name, r) for name, r in result.receipt_map.items() if np.isfinite(r.surrogate_closure_error)]
    n_calib = len(calibration_data)
    if sample_budget is not None:
        budget_n = max(1, min(int(sample_budget), n_calib))
    else:
        budget_n = max(1, min(n_calib, int(np.ceil(sample_fraction * n_calib))))

    if not pool or budget_n <= 0:
        return ns_model, result, {"selected_stages": [], "sample_count": 0}

    k_stages = max(1, min(max_stages, len(pool)))
    selected = acquire(pool, model=None, k=k_stages, strategy="closure_error")
    selected_names = [name for name, _r in selected]
    block_indices = _stage_block_indices(selected_names)
    if not block_indices:
        return ns_model, result, {"selected_stages": selected_names, "sample_count": 0}

    x = _as_long_tensor(calibration_data)[:budget_n]
    _set_trainable_blocks(ns_model, block_indices)
    _finetune_stages(ns_model, model, x, epochs=epochs, lr=lr)
    _unfreeze_all(ns_model)

    return ns_model, result, {"selected_stages": selected_names, "sample_count": int(x.shape[0])}


# --------------------------------------------------------------------------------------------------
# 4. auto -- UCB1 picks per-call among the three method families, mirroring I1's _auto_pick exactly
# --------------------------------------------------------------------------------------------------


def _auto_pick(
    model: Any,
    calibration_data: Any,
    eval_data: Any,
    input_law: GaussianLaw,
    budget: float,
    trust_region: float,
    n_mc: int,
    seed: int,
    kd_epochs: int,
    kd_lr: float,
    hybrid_sample_fraction: float,
    hybrid_sample_budget: int | None,
    hybrid_max_stages: int,
    hybrid_epochs: int,
    hybrid_lr: float,
    target_n_layer: int,
) -> tuple[str, Any, dict[str, MethodCandidate], dict[str, Any]]:
    """The D5/ConditionalJIT/I1 pattern at "which compression method for this layer/stage" scale: a
    small :class:`~mixle.task.bandit.UCB1` controller picks an arm (method family) using the REAL
    measured teacher-agreement of each method, run once and cached -- exactly
    :func:`mixle.models.unified_quantizer._auto_pick`'s cold-start-to-completion sweep, translated
    from per-tensor reconstruction error to per-model teacher-agreement."""
    payloads: dict[str, Any] = {}
    candidates: dict[str, MethodCandidate] = {}
    extra: dict[str, Any] = {}

    ns_result = _non_sampling(model, input_law, budget, trust_region, n_mc, seed)
    ns_model = ns_result.model
    q_ns = _quality(ns_model, model, eval_data)
    payloads["non_sampling"] = ns_model
    candidates["non_sampling"] = MethodCandidate("non_sampling", quality=q_ns, sample_count=0, reward=q_ns)
    extra["non_sampling_receipts"] = ns_result.receipt_map

    sk_result = _sampling_kd(model, calibration_data, target_n_layer, kd_epochs, kd_lr, seed)
    sk_model = sk_result.student
    q_sk = _quality(sk_model, model, eval_data)
    n_sk = int(len(calibration_data))
    payloads["sampling_kd"] = sk_model
    candidates["sampling_kd"] = MethodCandidate("sampling_kd", quality=q_sk, sample_count=n_sk, reward=q_sk)

    hy_model, hy_result, hy_info = _hybrid(
        model,
        input_law,
        budget,
        trust_region,
        n_mc,
        seed,
        calibration_data,
        hybrid_sample_fraction,
        hybrid_sample_budget,
        hybrid_max_stages,
        hybrid_epochs,
        hybrid_lr,
    )
    q_hy = _quality(hy_model, model, eval_data)
    payloads["hybrid"] = hy_model
    candidates["hybrid"] = MethodCandidate("hybrid", quality=q_hy, sample_count=hy_info["sample_count"], reward=q_hy)
    extra["hybrid_receipts"] = hy_result.receipt_map
    extra["hybrid_selected_stages"] = hy_info["selected_stages"]

    order = list(candidates.keys())
    bandit = UCB1(n_arms=len(order), seed=seed)
    for _ in range(len(order)):
        arm = bandit.select()
        m = order[arm]
        bandit.update(arm, candidates[m].reward)
    best_arm = int(np.argmax(bandit.means))
    best_method = order[best_arm]
    return best_method, payloads[best_method], candidates, extra


# --------------------------------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------------------------------


def compress(
    model: Any,
    method: str = "auto",
    *,
    calibration_data: Any = None,
    eval_data: Any = None,
    sample_budget: int | None = None,
    input_law: GaussianLaw | None = None,
    budget: float = 5.0,
    trust_region: float = 5.0,
    n_mc: int = 64,
    seed: int = 0,
    kd_epochs: int = 200,
    kd_lr: float = 1e-2,
    hybrid_sample_fraction: float = 0.01,
    hybrid_max_stages: int = 1,
    hybrid_epochs: int = 40,
    hybrid_lr: float = 5e-4,
    target_n_layer: int | None = None,
) -> CompressedModel:
    """The single compression entry point (roadmap J1): dispatches to one of the three method
    families, or lets a :class:`~mixle.task.bandit.UCB1` :class:`CompressionReceipt`-carrying
    picker choose (``method="auto"``, default).

    Args:
        model: a real :class:`mixle.models.transformer.CausalLM` (or a coarsened one).
        method: ``"auto"`` (default), ``"sampling_kd"``, ``"non_sampling"``, or ``"hybrid"``.
        calibration_data: integer token-context tensor/array ``(N, L)`` -- required for
            ``"sampling_kd"``, ``"hybrid"``, and ``"auto"`` (the two methods that touch real data).
        eval_data: held-out integer token-context tensor/array used to MEASURE quality
            (teacher-agreement against ``model``); defaults to ``calibration_data`` if omitted.
        sample_budget: an explicit cap (absolute count) on how many REAL calibration samples
            ``"hybrid"``/``"auto"``'s hybrid arm may use; if ``None``, ``hybrid_sample_fraction`` of
            ``len(calibration_data)`` is used instead.
        input_law: the data-free input law for G1's propagation; defaults to
            :func:`_default_input_law` (the model's own token-embedding statistics).
        budget, trust_region, n_mc: forwarded to :func:`mixle.models.coarsening.coarsen`.
        kd_epochs, kd_lr: forwarded to the fresh, from-scratch ``sampling_kd`` student's full
            training loop.
        hybrid_sample_fraction: fraction of ``calibration_data`` hybrid may use when
            ``sample_budget`` is not given (default 1%, matching the acceptance criterion).
        hybrid_max_stages: how many worst-closure-error stages hybrid fine-tunes (default 1).
        hybrid_epochs, hybrid_lr: forwarded to hybrid's own micro-calibration fine-tune loop --
            deliberately separate from ``kd_epochs``/``kd_lr``: hybrid fine-tunes ALREADY-good
            non_sampling weights on a HANDFUL of real samples (a light nudge), a completely
            different regime from training a fresh architecture from scratch on the full
            calibration set, and reusing the same hyperparameters for both badly overfits hybrid's
            tiny sample budget in practice.
        target_n_layer: depth of the fresh ``sampling_kd`` student; defaults to
            ``coarsen``'s own natural 2x depth cut (``max(1, model.n_layer // 2)``) so all three
            methods are compared at a MATCHED compression ratio.

    Returns:
        CompressedModel
    """
    if not _HAS_TORCH:
        raise RuntimeError("compress requires torch (mixle.models.transformer is torch-only).")
    if method not in ("auto",) + METHODS:
        raise ValueError(f"method must be 'auto' or one of {METHODS}, got {method!r}")

    input_law = input_law if input_law is not None else _default_input_law(model)
    if target_n_layer is None:
        target_n_layer = max(1, model.n_layer // 2)
    if eval_data is None:
        eval_data = calibration_data

    if method == "non_sampling":
        result = _non_sampling(model, input_law, budget, trust_region, n_mc, seed)
        quality = _quality(result.model, model, eval_data)
        receipt = CompressionReceipt(
            method="non_sampling",
            auto=False,
            quality=quality,
            sample_count=0,
            notes="data-free coarsen() depth-merge (G1 laws + G3 operator); no real forward pass over "
            "calibration data was used.",
        )
        return CompressedModel(
            model=result.model, method="non_sampling", receipt=receipt, non_sampling_receipts=result.receipt_map
        )

    if method == "sampling_kd":
        if calibration_data is None:
            raise ValueError("compress(method='sampling_kd') requires calibration_data")
        sk_result = _sampling_kd(model, calibration_data, target_n_layer, kd_epochs, kd_lr, seed)
        n_samples = int(len(calibration_data))
        quality = _quality(sk_result.student, model, eval_data)
        receipt = CompressionReceipt(
            method="sampling_kd",
            auto=False,
            quality=quality,
            sample_count=n_samples,
            notes=f"full response_distill KD (mixle.task.distill_methods) using all {n_samples} calibration samples.",
        )
        return CompressedModel(model=sk_result.student, method="sampling_kd", receipt=receipt)

    if method == "hybrid":
        if calibration_data is None:
            raise ValueError("compress(method='hybrid') requires calibration_data")
        hy_model, hy_result, hy_info = _hybrid(
            model,
            input_law,
            budget,
            trust_region,
            n_mc,
            seed,
            calibration_data,
            hybrid_sample_fraction,
            sample_budget,
            hybrid_max_stages,
            hybrid_epochs,
            hybrid_lr,
        )
        quality = _quality(hy_model, model, eval_data)
        receipt = CompressionReceipt(
            method="hybrid",
            auto=False,
            quality=quality,
            sample_count=hy_info["sample_count"],
            notes=f"non_sampling base + acquire()-ranked (closure_error) micro-calibration on stages "
            f"{hy_info['selected_stages']} using {hy_info['sample_count']} real samples.",
        )
        return CompressedModel(
            model=hy_model,
            method="hybrid",
            receipt=receipt,
            non_sampling_receipts=hy_result.receipt_map,
            hybrid_selected_stages=hy_info["selected_stages"],
        )

    # method == "auto"
    if calibration_data is None:
        raise ValueError("compress(method='auto') requires calibration_data (needed by the sampling_kd/hybrid arms)")
    best_method, best_model, candidates, extra = _auto_pick(
        model,
        calibration_data,
        eval_data,
        input_law,
        budget,
        trust_region,
        n_mc,
        seed,
        kd_epochs,
        kd_lr,
        hybrid_sample_fraction,
        sample_budget,
        hybrid_max_stages,
        hybrid_epochs,
        hybrid_lr,
        target_n_layer,
    )
    receipt = CompressionReceipt(
        method=best_method,
        auto=True,
        quality=candidates[best_method].quality,
        sample_count=candidates[best_method].sample_count,
        candidates=candidates,
        notes=f"UCB1 evaluated all {len(candidates)} methods once (real measured reward=teacher-agreement); "
        f"picked {best_method!r} over "
        + ", ".join(
            f"{m}(quality={c.quality:.4g}, samples={c.sample_count})" for m, c in candidates.items() if m != best_method
        ),
    )
    return CompressedModel(
        model=best_model,
        method=best_method,
        receipt=receipt,
        non_sampling_receipts=extra.get("non_sampling_receipts", {}),
        hybrid_selected_stages=extra.get("hybrid_selected_stages", []),
    )
