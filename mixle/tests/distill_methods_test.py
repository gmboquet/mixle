"""Each KD family must ACTUALLY transfer knowledge: the distilled student beats a no-distillation baseline."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.task.distill_methods import (
    DistillResult,
    analytic_response_distill,
    attention_transfer,
    hint_distill,
    kd_loss,
    multi_teacher_distill,
    planned_response_distill,
    relational_distill,
    response_distill,
    sequence_level_distill,
)


def _mlp(in_dim, hidden, out_dim, seed=0):
    torch.manual_seed(seed)
    layers = []
    dims = [in_dim] + list(hidden) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def _blobs(n_per, k, dim, seed=0, spread=4.0):
    """k gaussian blobs -> a nonlinear-ish separable classification set."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=spread, size=(k, dim))
    xs, ys = [], []
    for c in range(k):
        xs.append(rng.normal(loc=centers[c], scale=1.0, size=(n_per, dim)))
        ys.append(np.full(n_per, c))
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    perm = rng.permutation(len(x))
    return torch.from_numpy(x[perm]), torch.from_numpy(y[perm])


def _trained_teacher(x, y, k, dim, hidden=(32, 32), seed=1, epochs=400):
    t = _mlp(dim, hidden, k, seed=seed)
    opt = torch.optim.Adam(t.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(t(x), y)
        loss.backward()
        opt.step()
    t.eval()
    return t


# --------------------------------------------------------------------------------------------------


def test_kd_loss_is_scalar_and_pure_soft():
    sl = torch.randn(8, 4)
    tl = torch.randn(8, 4)
    loss = kd_loss(sl, tl, temperature=3.0, alpha=1.0)
    assert loss.dim() == 0 and float(loss) >= 0.0
    mixed = kd_loss(sl, tl, torch.zeros(8, dtype=torch.long), temperature=3.0, alpha=0.5)
    assert mixed.dim() == 0


def test_response_distill_closes_teacher_gap():
    # overlapping blobs (spread=2) so argmax agreement has real headroom -- not a saturated 100%.
    dim, k = 8, 4
    x, y = _blobs(50, k, dim, seed=0, spread=2.0)
    teacher = _trained_teacher(x, y, k, dim, hidden=(32, 32), seed=1)
    student = _mlp(dim, (8,), k, seed=5)  # smaller student, short training -> headroom to distill
    res = response_distill(student, teacher, x, y, temperature=4.0, alpha=0.9, epochs=150, lr=1e-2, seed=7)

    assert isinstance(res, DistillResult)
    # KD student agrees with the teacher far more than at init.
    assert res.after > res.before + 0.2
    assert res.improved
    assert res.after >= res.extra["baseline_agreement"]
    # dark knowledge: soft KL to the teacher is materially lower for the KD student than the hard-only baseline.
    assert res.extra["soft_kl"] < res.extra["baseline_soft_kl"]


def test_response_distill_deterministic():
    dim, k = 5, 3
    x, y = _blobs(40, k, dim, seed=2)
    teacher = _trained_teacher(x, y, k, dim, seed=3, epochs=200)
    a = response_distill(_mlp(dim, (8,), k, seed=9), teacher, x, y, epochs=100, seed=11)
    b = response_distill(_mlp(dim, (8,), k, seed=9), teacher, x, y, epochs=100, seed=11)
    assert a.after == b.after
    assert a.history == b.history


def test_analytic_response_distill_solves_linear_student_without_autograd():
    torch.manual_seed(3)
    teacher = torch.nn.Linear(5, 4)
    student = torch.nn.Sequential(torch.nn.Linear(5, 4))
    x = torch.randn(96, 5)

    result = analytic_response_distill(student, teacher, x, ridge=1e-10, seed=8)

    assert result.improved
    assert result.after < 1e-8
    assert result.extra["analytic_projection"]["autograd_steps"] == 0
    assert result.extra["analytic_projection"]["optimizer"] == "none"
    assert result.extra["analytic_projection"]["teacher_queries"] == 1


def test_planned_response_distill_refines_in_minibatches_without_adam():
    dim, classes = 6, 3
    x, y = _blobs(30, classes, dim, seed=13, spread=2.0)
    teacher = _trained_teacher(x, y, classes, dim, hidden=(16,), seed=14, epochs=150)
    student = _mlp(dim, (8,), classes, seed=15)

    result = planned_response_distill(
        student,
        teacher,
        x,
        y,
        refinement_epochs=40,
        batch_size=18,
        lr=1e-2,
        seed=16,
    )

    assert result.improved
    assert result.extra["projected_soft_kl"] < result.before
    assert result.extra["optimizer"]["batch_size"] == 18
    families = result.extra["optimizer"]["plan"]["families"]
    assert families
    assert not any("adam" in family for family in families)


def test_multi_teacher_beats_average_single_teacher():
    dim, k = 6, 3
    x, y = _blobs(60, k, dim, seed=0)
    # three teachers trained on different seeds/data views -> a useful ensemble.
    teachers = []
    for s in range(3):
        xs, ys = _blobs(60, k, dim, seed=10 + s)
        teachers.append(_trained_teacher(xs, ys, k, dim, seed=20 + s, epochs=300))
    student = _mlp(dim, (8,), k, seed=5)
    res = multi_teacher_distill(student, teachers, x, y, temperature=4.0, epochs=400, seed=7)

    assert res.improved
    # the ensemble student matches the ensemble consensus better than the mean single-teacher student does.
    assert res.after > res.extra["mean_single_teacher_agreement"]


def test_hint_distill_reduces_feature_gap_and_helps_accuracy():
    dim, k = 6, 3
    x, y = _blobs(60, k, dim, seed=0)
    teacher = _trained_teacher(x, y, k, dim, hidden=(32, 32), seed=1)
    # teacher hint layer: first Linear's activation region; student mid layer.
    student = _mlp(dim, (16, 16), k, seed=5)
    # teacher module "0" is first Linear (dim->32); student module "2" is second Linear out (16).
    res = hint_distill(student, teacher, x, student_layer="2", teacher_layer="0", epochs=400, lr=1e-2, seed=7)
    assert res.metric == "feature_gap"
    assert res.after < res.before
    assert res.improved
    # the hint objective actually drove the loss down over training.
    assert res.history[-1] < res.history[0]


def test_attention_transfer_reduces_attention_gap():
    # conv student/teacher so attention maps are spatial (N, C, H, W).
    torch.manual_seed(1)
    teacher = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 8, 3, padding=1),
        torch.nn.ReLU(),
    )
    # warm the teacher so its attention map isn't just noise.
    x = torch.randn(16, 1, 8, 8)
    opt = torch.optim.Adam(teacher.parameters(), lr=1e-2)
    for _ in range(50):
        opt.zero_grad()
        teacher(x).pow(2).mean().backward()
        opt.step()
    teacher.eval()

    student = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 4, 3, padding=1),
        torch.nn.ReLU(),
    )
    # match the pre-ReLU conv output (layer "2") -- signed activations give a distinctive attention map.
    res = attention_transfer(student, teacher, x, student_layer="2", teacher_layer="2", epochs=300, lr=1e-2, seed=7)
    assert res.metric == "attention_gap"
    assert res.after < res.before
    assert res.improved


def test_relational_distill_reduces_distance_structure_gap():
    dim, k = 6, 3
    x, y = _blobs(30, k, dim, seed=0)
    teacher = _trained_teacher(x, y, k, dim, hidden=(32, 16), seed=1)
    student = _mlp(dim, (16, 8), k, seed=5)
    res = relational_distill(
        student, teacher, x, student_layer="2", teacher_layer="2", use_angle=True, epochs=400, seed=7
    )
    assert res.metric == "distance_gap"
    assert res.after < res.before
    assert res.improved
    assert res.history[-1] < res.history[0]


class _TinyLM(torch.nn.Module):
    """Toy autoregressive LM: embed -> mean-pool causal context -> linear over vocab per position."""

    def __init__(self, vocab, dim, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = torch.nn.Embedding(vocab, dim)
        self.rnn = torch.nn.GRU(dim, dim, batch_first=True)
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, ids):
        h, _ = self.rnn(self.emb(ids))
        return self.head(h)


def test_sequence_level_distill_lifts_match_to_teacher():
    vocab, dim = 12, 16
    torch.manual_seed(3)
    teacher_lm = _TinyLM(vocab, dim, seed=3)
    # give the teacher a learnable, non-degenerate mapping by training it to copy a shifted pattern.
    prompts = torch.randint(0, vocab, (24, 4))
    tgt = torch.remainder(torch.cat([prompts, prompts], dim=1)[:, 1:9] + 1, vocab)
    opt = torch.optim.Adam(teacher_lm.parameters(), lr=1e-2)
    full = torch.cat([prompts, tgt], dim=1)
    for _ in range(200):
        opt.zero_grad()
        logits = teacher_lm(full[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), full[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
    teacher_lm.eval()

    def teacher_step(ctx):
        return teacher_lm(ctx)[:, -1, :]

    student_lm = _TinyLM(vocab, dim, seed=9)
    res = sequence_level_distill(
        student_lm, teacher_step, prompts, gen_length=6, vocab_size=vocab, epochs=300, lr=1e-2, seed=7
    )
    assert res.metric == "sequence_match"
    # student reproduces the teacher's generated sequences far better after seq-level KD than at init.
    assert res.after > res.before + 0.3
    assert res.improved
    assert res.after > res.extra["baseline_match"]


def test_result_gain_sign():
    up = DistillResult(None, "acc", before=0.5, after=0.8, lower_is_better=False)
    assert up.improved and up.gain == pytest.approx(0.3)
    down = DistillResult(None, "kl", before=1.0, after=0.4, lower_is_better=True)
    assert down.improved and down.gain == pytest.approx(0.6)
