"""Classic knowledge-distillation families as composable, verified procedures on Torch modules.

:mod:`mixle.task.distill` distills a teacher's labels into a local student (response/label
distillation, no gradients through the teacher). This module is its representation-matching sibling: given a
trained torch teacher and an untrained torch student, transfer knowledge through the classic KD signals --

- :func:`response_distill` -- Hinton dark-knowledge: temperature-softened ``KL(teacher || student)`` mixed with
  the hard-label loss.
- :func:`multi_teacher_distill` -- distill from several teachers via averaged (or weighted) soft targets.
- :func:`hint_distill` -- FitNets feature/hint transfer: match an intermediate student feature to a teacher
  feature (a learned linear regressor bridges a dimensionality gap), read out with forward hooks.
- :func:`attention_transfer` -- Zagoruyko-Komodakis: match spatial attention maps ``sum_c F_c^2`` between a
  teacher and student layer.
- :func:`relational_distill` -- RKD: match the pairwise *distance* (and *angle*) structure of a batch in
  feature space, so the student preserves relations rather than absolute activations.
- :func:`sequence_level_distill` -- Kim-Rush: for a compact LM, train the student on teacher-generated sequences
  (the teacher's argmax continuation stands in for the reference).

Everything is deterministic given ``seed`` and needs only torch. Each returns a compact result record with the
before/after fidelity numbers the tests assert on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


def _torch() -> Any:
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("mixle.task.distill_methods requires torch.") from e
    return torch


@dataclass
class DistillResult:
    """Outcome of a distillation run: the trained student plus before/after fidelity numbers.

    ``metric`` names what ``before``/``after`` measure (e.g. ``"teacher_agreement"``, ``"soft_kl"``,
    ``"feature_gap"``). ``improved`` is the sign-aware verdict -- higher-is-better for agreement/accuracy,
    lower-is-better for a KL/gap. ``history`` holds the per-epoch training loss.
    """

    student: Any
    metric: str
    before: float
    after: float
    lower_is_better: bool
    extra: dict[str, Any] = field(default_factory=dict)
    history: list[float] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        """Whether the after metric is better than the before metric."""
        return self.after < self.before if self.lower_is_better else self.after > self.before

    @property
    def gain(self) -> float:
        """Signed improvement, always positive when the student got better."""
        return (self.before - self.after) if self.lower_is_better else (self.after - self.before)


# --------------------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------------------


def _softmax_logits(logits: Any, temperature: float) -> Any:
    torch = _torch()
    return torch.nn.functional.log_softmax(logits / temperature, dim=-1)


def _kd_kl(student_logits: Any, teacher_logits: Any, temperature: float) -> Any:
    """Temperature-softened ``KL(teacher || student)`` (Hinton), scaled by ``T^2`` so its gradient magnitude
    is comparable to the hard-loss term."""
    torch = _torch()
    t = float(temperature)
    log_p_student = _softmax_logits(student_logits, t)
    p_teacher = torch.softmax(teacher_logits / t, dim=-1)
    # KL(teacher || student) = sum p_teacher (log p_teacher - log p_student); the constant-in-student entropy
    # term is dropped, leaving the cross-entropy the optimizer actually minimizes, scaled by T^2.
    return torch.nn.functional.kl_div(log_p_student, p_teacher, reduction="batchmean") * (t * t)


def _agreement(student_logits: Any, teacher_logits: Any) -> float:
    """Fraction of rows where the student and teacher argmax agree."""
    torch = _torch()
    with torch.no_grad():
        return float((student_logits.argmax(-1) == teacher_logits.argmax(-1)).float().mean().cpu().item())


def _soft_kl(student_logits: Any, teacher_logits: Any, temperature: float = 1.0) -> float:
    torch = _torch()
    with torch.no_grad():
        return float(_kd_kl(student_logits, teacher_logits, temperature).cpu().item())


def _accuracy(logits: Any, y: Any) -> float:
    torch = _torch()
    with torch.no_grad():
        return float((logits.argmax(-1) == y).float().mean().cpu().item())


def _clone_untrained(student: Any, seed: int) -> Any:
    """A fresh copy of ``student`` with re-initialized (seed-reset) parameters -- the no-distillation baseline."""
    import copy

    torch = _torch()
    base = copy.deepcopy(student)
    torch.manual_seed(seed)
    for m in base.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    return base


# --------------------------------------------------------------------------------------------------
# response / dark-knowledge distillation (Hinton et al. 2015)
# --------------------------------------------------------------------------------------------------


def kd_loss(
    student_logits: Any,
    teacher_logits: Any,
    y: Any | None = None,
    *,
    temperature: float = 4.0,
    alpha: float = 0.9,
) -> Any:
    """The Hinton KD loss: ``alpha * T^2 * KL(teacher||student)`` + ``(1-alpha) * CE(student, y)``.

    With ``y=None`` (or ``alpha=1``) it is pure soft-target distillation. Returned as a scalar torch tensor so
    it composes into any training loop.
    """
    torch = _torch()
    soft = _kd_kl(student_logits, teacher_logits, temperature)
    if y is None or alpha >= 1.0:
        return alpha * soft
    hard = torch.nn.functional.cross_entropy(student_logits, y)
    return alpha * soft + (1.0 - alpha) * hard


def response_distill(
    student: Any,
    teacher: Any,
    x: Any,
    y: Any | None = None,
    *,
    temperature: float = 4.0,
    alpha: float = 0.9,
    epochs: int = 300,
    lr: float = 1e-2,
    seed: int = 0,
    baseline: bool = True,
) -> DistillResult:
    """Distill ``teacher``'s soft outputs into ``student`` on inputs ``x`` (labels ``y`` optional for the hard mix).

    Trains ``student`` in place with :func:`kd_loss`. When ``baseline`` is set, an identically-initialized copy is
    trained on the hard labels alone (or, if ``y`` is None, left untrained) and the two students' teacher-agreement
    is compared -- the number the test asserts the KD student wins.
    """
    torch = _torch()
    x, teacher_logits = _teacher_logits(teacher, x)

    torch.manual_seed(seed)
    student = _reset(student, seed)
    before = _agreement(_forward(student, x), teacher_logits)

    hist = _train(
        student,
        lambda: kd_loss(_forward(student, x), teacher_logits, y, temperature=temperature, alpha=alpha),
        epochs,
        lr,
    )
    after = _agreement(_forward(student, x), teacher_logits)

    extra: dict[str, Any] = {"soft_kl": _soft_kl(_forward(student, x), teacher_logits, temperature)}
    if baseline:
        base = _clone_untrained(student, seed)
        if y is not None:
            _train(base, lambda: torch.nn.functional.cross_entropy(_forward(base, x), y), epochs, lr)
        extra["baseline_agreement"] = _agreement(_forward(base, x), teacher_logits)
        extra["baseline_soft_kl"] = _soft_kl(_forward(base, x), teacher_logits, temperature)
    if y is not None:
        extra["student_accuracy"] = _accuracy(_forward(student, x), y)
    return DistillResult(student, "teacher_agreement", before, after, lower_is_better=False, extra=extra, history=hist)


# --------------------------------------------------------------------------------------------------
# multi-teacher ensemble distillation
# --------------------------------------------------------------------------------------------------


def multi_teacher_distill(
    student: Any,
    teachers: Sequence[Any],
    x: Any,
    y: Any | None = None,
    *,
    weights: Sequence[float] | None = None,
    temperature: float = 4.0,
    alpha: float = 0.9,
    epochs: int = 300,
    lr: float = 1e-2,
    seed: int = 0,
) -> DistillResult:
    """Distill an averaged (or ``weights``-weighted) ensemble of ``teachers`` into ``student``.

    The soft target is the weighted mean of the teachers' softmax probabilities (in probability space, so the
    ensemble is a genuine mixture). ``after`` is the student's agreement with the *ensemble* argmax; ``extra``
    reports the mean single-teacher agreement the ensemble student should beat.
    """
    torch = _torch()
    x = _as_tensor(x)
    logits = [_teacher_logits(t, x)[1] for t in teachers]
    w = _norm_weights(weights, len(teachers))
    ens_prob = sum(wi * torch.softmax(li, dim=-1) for wi, li in zip(w, logits))
    ens_argmax = ens_prob.argmax(-1)

    torch.manual_seed(seed)
    student = _reset(student, seed)
    before = _ensemble_agreement(_forward(student, x), ens_argmax)

    hist = _train(
        student,
        lambda: _ensemble_kd_loss(_forward(student, x), ens_prob, y, temperature, alpha),
        epochs,
        lr,
    )
    after = _ensemble_agreement(_forward(student, x), ens_argmax)

    # each teacher's own soft-KD student, to prove the ensemble beats the average single teacher
    single = []
    for li in logits:
        s = _clone_untrained(student, seed)
        _train(s, lambda s=s, li=li: kd_loss(_forward(s, x), li, y, temperature=temperature, alpha=alpha), epochs, lr)
        single.append(_ensemble_agreement(_forward(s, x), ens_argmax))
    extra = {
        "mean_single_teacher_agreement": float(sum(single) / len(single)),
        "single_teacher_agreements": single,
    }
    return DistillResult(student, "ensemble_agreement", before, after, lower_is_better=False, extra=extra, history=hist)


def _ensemble_kd_loss(student_logits: Any, ens_prob: Any, y: Any | None, temperature: float, alpha: float) -> Any:
    torch = _torch()
    t = float(temperature)
    # soften the *ensemble probabilities* to temperature T by re-normalizing p^(1/T)
    soft_target = ens_prob.clamp_min(1e-12).pow(1.0 / t)
    soft_target = soft_target / soft_target.sum(-1, keepdim=True)
    log_p_student = _softmax_logits(student_logits, t)
    soft = torch.nn.functional.kl_div(log_p_student, soft_target, reduction="batchmean") * (t * t)
    if y is None or alpha >= 1.0:
        return alpha * soft
    return alpha * soft + (1.0 - alpha) * torch.nn.functional.cross_entropy(student_logits, y)


def _ensemble_agreement(student_logits: Any, ens_argmax: Any) -> float:
    torch = _torch()
    with torch.no_grad():
        return float((student_logits.argmax(-1) == ens_argmax).float().mean().cpu().item())


# --------------------------------------------------------------------------------------------------
# feature / hint distillation (FitNets, Romero et al. 2015)
# --------------------------------------------------------------------------------------------------


class _Hook:
    """Captures the output of a named submodule on the next forward pass."""

    def __init__(self, module: Any, name: str) -> None:
        self.value: Any = None
        target = dict(module.named_modules())[name]
        self._h = target.register_forward_hook(self._grab)

    def _grab(self, _mod: Any, _inp: Any, out: Any) -> None:
        self.value = out

    def remove(self) -> None:
        self._h.remove()


def hint_distill(
    student: Any,
    teacher: Any,
    x: Any,
    *,
    student_layer: str,
    teacher_layer: str,
    epochs: int = 300,
    lr: float = 1e-2,
    seed: int = 0,
) -> DistillResult:
    """FitNets hint transfer: drive ``student``'s ``student_layer`` feature toward ``teacher``'s ``teacher_layer``.

    A learned linear regressor maps the student feature into the teacher's feature dimension when they differ; the
    loss is the MSE between the regressed student feature and the (detached) teacher feature, read out with forward
    hooks. ``before``/``after`` are the normalized feature gap -- lower is better.
    """
    torch = _torch()
    x = _as_tensor(x)
    teacher.eval()

    th = _Hook(teacher, teacher_layer)
    with torch.no_grad():
        teacher(x)
    teach_feat = _flatten_feat(th.value).detach()
    th.remove()

    torch.manual_seed(seed)
    student = _reset(student, seed)
    sh = _Hook(student, student_layer)

    student(x)
    stud_dim = _flatten_feat(sh.value).shape[-1]
    teach_dim = teach_feat.shape[-1]
    regressor = torch.nn.Identity() if stud_dim == teach_dim else torch.nn.Linear(stud_dim, teach_dim)

    def feat_gap() -> float:
        with torch.no_grad():
            student(x)
            sf = regressor(_flatten_feat(sh.value))
            return float(_norm_gap(sf, teach_feat).cpu().item())

    before = feat_gap()

    params = list(student.parameters()) + list(regressor.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    hist = []
    for _ in range(int(epochs)):
        opt.zero_grad()
        student(x)
        sf = regressor(_flatten_feat(sh.value))
        loss = torch.nn.functional.mse_loss(sf, teach_feat)
        loss.backward()
        opt.step()
        hist.append(float(loss.detach().cpu().item()))
    after = feat_gap()
    sh.remove()
    return DistillResult(
        student, "feature_gap", before, after, lower_is_better=True, extra={"regressor": regressor}, history=hist
    )


def _flatten_feat(t: Any) -> Any:
    """Flatten a feature to ``(batch, features)`` -- handles both ``(N, D)`` and conv ``(N, C, H, W)`` shapes."""
    return t.reshape(t.shape[0], -1)


def _norm_gap(a: Any, b: Any) -> Any:
    """Mean L2 distance between rows, normalized by the teacher feature scale (scale-free feature-match metric)."""
    torch = _torch()
    denom = b.norm(dim=-1).mean().clamp_min(1e-8)
    return (a - b).norm(dim=-1).mean() / denom


# --------------------------------------------------------------------------------------------------
# attention transfer (Zagoruyko & Komodakis 2017)
# --------------------------------------------------------------------------------------------------


def _attention_map(feat: Any) -> Any:
    """Spatial attention map ``sum_c |F_c|^2`` flattened and L2-normalized per sample (the AT statistic)."""
    torch = _torch()
    # feat: (N, C, H, W) -> (N, H*W); if already (N, D) treat channels as the summed axis trivially
    if feat.dim() == 4:
        a = feat.pow(2).sum(1).reshape(feat.shape[0], -1)
    else:
        a = feat.pow(2)
    return torch.nn.functional.normalize(a, p=2, dim=1)


def attention_transfer(
    student: Any,
    teacher: Any,
    x: Any,
    *,
    student_layer: str,
    teacher_layer: str,
    beta: float = 1.0,
    epochs: int = 300,
    lr: float = 1e-2,
    seed: int = 0,
) -> DistillResult:
    """Match the spatial attention map of ``student_layer`` to ``teacher_layer`` (Zagoruyko-Komodakis).

    Loss is the MSE between the two L2-normalized attention maps ``sum_c F_c^2``. ``before``/``after`` are the
    attention-map gap -- lower is better.
    """
    torch = _torch()
    x = _as_tensor(x)
    teacher.eval()

    th = _Hook(teacher, teacher_layer)
    with torch.no_grad():
        teacher(x)
    teach_att = _attention_map(th.value).detach()
    th.remove()

    torch.manual_seed(seed)
    student = _reset(student, seed)
    sh = _Hook(student, student_layer)

    def att_gap() -> float:
        with torch.no_grad():
            student(x)
            return float((_attention_map(sh.value) - teach_att).norm(dim=-1).mean().cpu().item())

    before = att_gap()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    hist = []
    for _ in range(int(epochs)):
        opt.zero_grad()
        student(x)
        loss = beta * torch.nn.functional.mse_loss(_attention_map(sh.value), teach_att)
        loss.backward()
        opt.step()
        hist.append(float(loss.detach().cpu().item()))
    after = att_gap()
    sh.remove()
    return DistillResult(student, "attention_gap", before, after, lower_is_better=True, history=hist)


# --------------------------------------------------------------------------------------------------
# relational KD (Park et al. 2019)
# --------------------------------------------------------------------------------------------------


def _pairwise_distances(feat: Any) -> Any:
    """RKD-D distance potential: pairwise L2 distances normalized by their nonzero mean."""
    torch = _torch()
    d = torch.cdist(feat, feat, p=2)
    mask = d > 0
    mu = d[mask].mean() if mask.any() else torch.ones((), device=d.device, dtype=d.dtype)
    return d / mu.clamp_min(1e-8)


def _pairwise_angles(feat: Any) -> Any:
    """RKD-A angle potential: cosine of the angle at each vertex over triplets (i, j, k)."""
    torch = _torch()
    # e_ij = normalize(feat_i - feat_j); angle_ijk = <e_ij, e_kj>
    diff = feat.unsqueeze(0) - feat.unsqueeze(1)  # (N, N, D): diff[i, j] = feat_i - feat_j... use j as anchor
    e = torch.nn.functional.normalize(diff, p=2, dim=-1)  # (N, N, D)
    # angle at anchor j between i and k: <e[i,j], e[k,j]>
    return torch.einsum("ijd,kjd->ijk", e, e)


def relational_distill(
    student: Any,
    teacher: Any,
    x: Any,
    *,
    student_layer: str,
    teacher_layer: str,
    use_angle: bool = True,
    dist_weight: float = 25.0,
    angle_weight: float = 50.0,
    epochs: int = 300,
    lr: float = 1e-2,
    seed: int = 0,
) -> DistillResult:
    """RKD: match the pairwise *distance* (and optionally *angle*) structure of a batch in feature space.

    Instead of matching absolute activations, RKD matches relations between samples -- so the student need not
    live in the teacher's coordinate frame, only preserve its geometry. ``before``/``after`` are the distance-
    structure gap -- lower is better.
    """
    torch = _torch()
    x = _as_tensor(x)
    teacher.eval()

    th = _Hook(teacher, teacher_layer)
    with torch.no_grad():
        teacher(x)
    tf = _flatten_feat(th.value).detach()
    th.remove()
    teach_dist = _pairwise_distances(tf).detach()
    teach_ang = _pairwise_angles(tf).detach() if use_angle else None

    torch.manual_seed(seed)
    student = _reset(student, seed)
    sh = _Hook(student, student_layer)

    def dist_gap() -> float:
        with torch.no_grad():
            student(x)
            return float((_pairwise_distances(_flatten_feat(sh.value)) - teach_dist).abs().mean().cpu().item())

    before = dist_gap()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    hist = []
    huber = torch.nn.functional.smooth_l1_loss
    for _ in range(int(epochs)):
        opt.zero_grad()
        student(x)
        sf = _flatten_feat(sh.value)
        loss = dist_weight * huber(_pairwise_distances(sf), teach_dist)
        if use_angle:
            loss = loss + angle_weight * huber(_pairwise_angles(sf), teach_ang)
        loss.backward()
        opt.step()
        hist.append(float(loss.detach().cpu().item()))
    after = dist_gap()
    sh.remove()
    return DistillResult(student, "distance_gap", before, after, lower_is_better=True, history=hist)


# --------------------------------------------------------------------------------------------------
# sequence-level KD (Kim & Rush 2016)
# --------------------------------------------------------------------------------------------------


def sequence_level_distill(
    student: Any,
    teacher: Callable[[Any], Any],
    prompts: Any,
    *,
    gen_length: int,
    vocab_size: int,
    epochs: int = 300,
    lr: float = 1e-2,
    seed: int = 0,
    baseline: bool = True,
) -> DistillResult:
    """Kim-Rush sequence-level KD for a compact autoregressive LM.

    ``teacher(context) -> next-token logits`` generates a hard continuation (its greedy argmax sequence) for each
    prompt; the student is then trained by teacher forcing to reproduce that teacher-generated sequence. ``x`` are
    integer ``prompts`` of shape ``(N, L0)``; ``student`` maps ``(N, L)`` token ids to ``(N, L, vocab)`` logits.
    ``before``/``after`` are the student's token match to the teacher's sequences -- higher is better.
    """
    torch = _torch()
    prompts = _as_long(prompts)

    # 1. teacher generates hard target sequences (greedy) -- the Kim-Rush "sequence-level" target.
    with torch.no_grad():
        targets = _greedy_generate(teacher, prompts, gen_length)  # (N, gen_length)

    torch.manual_seed(seed)
    student = _reset(student, seed)

    def match() -> float:
        with torch.no_grad():
            pred = _greedy_generate(_lm_step(student), prompts, gen_length)
            return float((pred == targets).float().mean().cpu().item())

    before = match()

    # 2. teacher-force the student on (prompt + target) -> next tokens.
    full = torch.cat([prompts, targets], dim=1)  # (N, L0 + gen_length)
    inp = full[:, :-1]
    tgt = full[:, 1:]
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    hist = []
    for _ in range(int(epochs)):
        opt.zero_grad()
        logits = student(inp)  # (N, L, vocab)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
        loss.backward()
        opt.step()
        hist.append(float(loss.detach().cpu().item()))
    after = match()

    extra: dict[str, Any] = {}
    if baseline:
        base = _clone_untrained(student, seed + 1)
        extra["baseline_match"] = float(
            (_greedy_generate(_lm_step(base), prompts, gen_length) == targets).float().mean().cpu().item()
        )
    return DistillResult(student, "sequence_match", before, after, lower_is_better=False, extra=extra, history=hist)


def _lm_step(student: Any) -> Callable[[Any], Any]:
    """Wrap a full-sequence LM ``(N, L) -> (N, L, V)`` as a next-token stepper ``(N, L) -> (N, V)``."""

    def step(ctx: Any) -> Any:
        return student(ctx)[:, -1, :]

    return step


def _greedy_generate(step: Callable[[Any], Any], prompts: Any, gen_length: int) -> Any:
    """Autoregressively roll out ``gen_length`` greedy tokens from ``step(context) -> next logits``."""
    torch = _torch()
    ctx = prompts
    out = []
    for _ in range(int(gen_length)):
        logits = step(ctx)
        nxt = logits.argmax(-1, keepdim=True)
        out.append(nxt)
        ctx = torch.cat([ctx, nxt], dim=1)
    return torch.cat(out, dim=1)


# --------------------------------------------------------------------------------------------------
# small shared training / tensor plumbing
# --------------------------------------------------------------------------------------------------


def _as_tensor(x: Any) -> Any:
    torch = _torch()
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, dtype=torch.float32)


def _as_long(x: Any) -> Any:
    torch = _torch()
    if isinstance(x, torch.Tensor):
        return x.long()
    return torch.as_tensor(x, dtype=torch.long)


def _forward(module: Any, x: Any) -> Any:
    return module(x)


def _teacher_logits(teacher: Any, x: Any) -> tuple[Any, Any]:
    """Evaluate a teacher (module or callable) on ``x`` under ``no_grad``; return ``(x_tensor, detached logits)``."""
    torch = _torch()
    x = _as_tensor(x)
    if hasattr(teacher, "eval"):
        teacher.eval()
    with torch.no_grad():
        logits = teacher(x)
    return x, logits.detach()


def _reset(student: Any, seed: int) -> Any:
    """Re-initialize a student's parameters under ``seed`` so a distillation run starts from a known point."""
    torch = _torch()
    torch.manual_seed(seed)
    for m in student.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    return student


def _norm_weights(weights: Sequence[float] | None, k: int) -> list[float]:
    if weights is None:
        return [1.0 / k] * k
    w = [float(x) for x in weights]
    if len(w) != k:
        raise ValueError("weights must have one entry per teacher")
    s = sum(w)
    if s <= 0:
        raise ValueError("teacher weights must sum to a positive value")
    return [x / s for x in w]


def _train(student: Any, loss_fn: Callable[[], Any], epochs: int, lr: float) -> list[float]:
    """Adam-train ``student`` for ``epochs`` steps against ``loss_fn`` (recomputed each step); return loss history."""
    torch = _torch()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    student.train()
    hist = []
    for _ in range(int(epochs)):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        hist.append(float(loss.detach().cpu().item()))
    return hist
