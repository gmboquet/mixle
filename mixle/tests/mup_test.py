"""Acceptance tests for F9 (muP / hyperparameter transfer) -- see ``mixle/models/mup.py``.

The headline claim under test: tune a learning rate once, cheaply, on a small "base width" transformer,
then use :func:`mixle.models.mup.transfer_lr` to *predict* (no search) the optimal lr at a larger
"target width" -- and that prediction should land close to what an INDEPENDENT small hyperparameter
search at the target width finds on its own. A contrast test shows the standard (non-muP) parametrization
does *not* have this property: naively reusing the base width's tuned lr at a much wider target performs
measurably worse than the muP-transferred lr.

The "tuned optimum" searches use :class:`mixle.doe.optimizer.BayesianOptimizer` (mixle's own ask/tell
Bayesian-optimization machinery, ``mixle.doe``) over ``log10(lr)`` -- a real, if small, hyperparameter
search, not a hand-rolled loop, per the F5-adjacent guidance that this codebase's own DOE tools are the
right way to do "find the tuned optimum at width X".

All training here is on tiny synthetic data with tiny models (``d_model`` in 32..256, a couple hundred
parameters to a few thousand) specifically so the search loops (many training runs) are cheap -- this is
squarely muP's designed use case: verify hyperparameter transfer AT SMALL SCALE, because muP's entire
premise is that what's tuned small transfers to big.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from mixle.doe.optimizer import BayesianOptimizer  # noqa: E402
from mixle.models.mup import (  # noqa: E402
    apply_mup_init,
    classify_causal_lm_params,
    init_std_multiplier,
    lr_multiplier,
    mup_param_groups,
    output_forward_multiplier,
    transfer_lr,
)
from mixle.models.transformer import build_causal_lm  # noqa: E402

# The formula/classification/param-group unit tests below are individually marked `fast` (they run in
# the default `pytest` gate); the two full training-loop acceptance tests are marked `slow` only (they
# run many small training loops via BayesianOptimizer search and are excluded from the default fast
# gate, matching this repo's `-m fast` default -- see pyproject.toml's addopts comment).

VOCAB = 8
BLOCK = 8
BATCH = 64
N_LAYER = 2
N_HEAD = 2
BASE_WIDTH = 32

# Fixed ground-truth rule for the synthetic task -- shared by every training AND eval batch so the
# model is always being trained and scored against the SAME target function (see next-token bigram
# task below); only the input tokens are resampled per seed/step.
_TASK_PERM = np.random.RandomState(0xBEEF).permutation(VOCAB)


def _batches(seed: int, n_batches: int):
    """A trivial "bigram" next-token task: y = fixed_permutation(x[:, -1]). Easy to learn quickly,
    but from-scratch (random init), so a bad lr (too small: undertrained; too large: unstable) is
    visibly worse than a well-tuned one -- exactly the sensitivity muP's lr-transfer rule needs to
    be tested against.
    """
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_batches):
        x = rng.randint(0, VOCAB, size=(BATCH, BLOCK))
        y = _TASK_PERM[x[:, -1]]
        out.append((x.astype(np.float32), y.astype(np.int64)))
    return out


def _train_eval(d_model: int, lr: float, seed: int, *, use_mup: bool, n_steps: int) -> float:
    """Train a tiny CausalLM at ``d_model`` with a single flat Adam lr, return held-out CE loss.

    ``use_mup=True`` applies muP init (:func:`apply_mup_init`, relative to ``BASE_WIDTH``) and the muP
    output-readout multiplier at the logits; ``use_mup=False`` uses the model's standard (untouched)
    init and no readout multiplier -- the "naive" comparison. Both use the SAME flat ``lr`` for every
    parameter (this is deliberately the simple, single-hyperparameter regime a capacity ladder actually
    tunes -- not per-role optimizer groups, which is what :func:`mup_param_groups` is for and is tested
    separately in ``test_mup_param_groups_scale_lr_by_role``).
    """
    torch.manual_seed(seed)
    model = build_causal_lm(VOCAB, d_model, N_LAYER, N_HEAD, BLOCK)
    width_mult = d_model / BASE_WIDTH
    if use_mup:
        apply_mup_init(model, base_width=BASE_WIDTH, base_std=0.02)
        out_mult = output_forward_multiplier(width_mult)
    else:
        out_mult = 1.0
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for x, y in _batches(seed=100 + seed, n_batches=n_steps):
        xt, yt = torch.from_numpy(x), torch.from_numpy(y)
        loss = F.cross_entropy(model(xt) * out_mult, yt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # bound single-seed lr-instability blowups
        opt.step()
    model.eval()
    with torch.no_grad():
        losses = [
            F.cross_entropy(model(torch.from_numpy(x)) * out_mult, torch.from_numpy(y)).item()
            for x, y in _batches(seed=999, n_batches=8)  # fixed held-out batches, same task rule
        ]
    return float(np.mean(losses))


def _avg_loss(d_model: int, log10_lr: float, *, use_mup: bool, seeds: tuple[int, ...], n_steps: int) -> float:
    lr = 10.0**log10_lr
    return float(np.mean([_train_eval(d_model, lr, s, use_mup=use_mup, n_steps=n_steps) for s in seeds]))


def _bo_tune_log_lr(
    d_model: int, bounds, *, use_mup: bool, seeds: tuple[int, ...], n_steps: int, n_iter: int, seed: int
) -> tuple[float, float]:
    """Search for the best flat lr at ``d_model`` over ``bounds`` (log10-lr) using mixle's own
    ask/tell Bayesian optimizer (:class:`mixle.doe.optimizer.BayesianOptimizer`) -- a real, if small,
    hyperparameter search, standing in for "the per-rung tuned optimum" in the acceptance criterion.
    """
    opt = BayesianOptimizer(bounds, acq="ei", n_init=6, seed=seed)
    for _ in range(n_iter):
        x = opt.ask()
        y = _avg_loss(d_model, float(x[0]), use_mup=use_mup, seeds=seeds, n_steps=n_steps)
        opt.tell(x, y)
    best = opt.best
    return float(10.0 ** best.best_x[0]), float(best.best_y)


# --- 3. hand-checkable unit tests of the abc-parametrization formulas themselves -----------------


@pytest.mark.fast
def test_init_std_multiplier_matches_published_mup_formula():
    # input role: Theta(1), never rescaled.
    assert init_std_multiplier("input", width_mult=1.0) == pytest.approx(1.0)
    assert init_std_multiplier("input", width_mult=4.0) == pytest.approx(1.0)
    assert init_std_multiplier("input", width_mult=0.25) == pytest.approx(1.0)
    # hidden role: variance ~ 1/width_mult -> std ~ width_mult**-0.5.
    assert init_std_multiplier("hidden", width_mult=4.0) == pytest.approx(0.5)
    assert init_std_multiplier("hidden", width_mult=16.0) == pytest.approx(0.25)
    assert init_std_multiplier("hidden", width_mult=1.0) == pytest.approx(1.0)
    # output role: variance ~ 1/width_mult**2 -> std ~ width_mult**-1 (an extra 1/width_mult vs hidden).
    assert init_std_multiplier("output", width_mult=4.0) == pytest.approx(0.25)
    assert init_std_multiplier("output", width_mult=16.0) == pytest.approx(1.0 / 16.0)


@pytest.mark.fast
def test_lr_multiplier_matches_published_mup_formula():
    # input role: constant lr, never rescaled.
    assert lr_multiplier("input", width_mult=8.0) == pytest.approx(1.0)
    # hidden and output roles: lr ~ 1/width_mult (the headline Adam-muP lr-transfer rule).
    assert lr_multiplier("hidden", width_mult=4.0) == pytest.approx(0.25)
    assert lr_multiplier("hidden", width_mult=1.0) == pytest.approx(1.0)
    assert lr_multiplier("output", width_mult=8.0) == pytest.approx(0.125)


@pytest.mark.fast
def test_output_forward_multiplier_matches_hidden_and_output_lr_scaling():
    assert output_forward_multiplier(width_mult=1.0) == pytest.approx(1.0)
    assert output_forward_multiplier(width_mult=4.0) == pytest.approx(0.25)


@pytest.mark.fast
def test_transfer_lr_and_transfer_init_std_apply_the_role_multiplier():
    # width doubles -> hidden lr predicted to halve; base_width==target_width -> identity.
    assert transfer_lr(1e-3, base_width=32, target_width=64) == pytest.approx(5e-4)
    assert transfer_lr(1e-3, base_width=32, target_width=32) == pytest.approx(1e-3)
    assert transfer_lr(1e-3, base_width=32, target_width=32, role="input") == pytest.approx(1e-3)


@pytest.mark.fast
def test_unknown_role_raises():
    with pytest.raises(ValueError):
        init_std_multiplier("bogus", width_mult=2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        lr_multiplier("bogus", width_mult=2.0)  # type: ignore[arg-type]


# --- classification / param-group unit tests (fast, no training) ---------------------------------


@pytest.mark.fast
def test_classify_causal_lm_params_assigns_expected_roles():
    model = build_causal_lm(vocab=VOCAB, d_model=16, n_layer=2, n_head=2, block=BLOCK)
    roles = classify_causal_lm_params(model)
    assert roles["tok.weight"] == "input"
    assert roles["pos.weight"] == "input"
    assert roles["ln.weight"] == "input"
    assert roles["ln.bias"] == "input"
    assert roles["blocks.0.ln1.weight"] == "input"
    assert roles["blocks.0.attn.qkv.weight"] == "hidden"
    assert roles["blocks.0.attn.proj.weight"] == "hidden"
    assert roles["blocks.0.mlp.0.weight"] == "hidden"
    assert roles["blocks.0.mlp.2.weight"] == "hidden"
    # head.weight is tied to tok.weight (mixle/models/transformer.py: `self.head.weight = self.tok.weight`)
    # so it is the SAME nn.Parameter and torch's named_parameters() de-dupes it -- it must not appear
    # as a second, separately-classified entry.
    assert "head.weight" not in roles
    assert model.head.weight is model.tok.weight


@pytest.mark.fast
def test_mup_param_groups_scale_lr_by_role():
    model = build_causal_lm(vocab=VOCAB, d_model=64, n_layer=2, n_head=2, block=BLOCK)
    groups = mup_param_groups(model, base_width=32, lr=1e-2)  # width_mult = 64/32 = 2
    by_role = {g["mup_role"]: g for g in groups}
    assert set(by_role) == {"input", "hidden"}  # no untied "output" group in this tied-embedding model
    assert by_role["input"]["lr"] == pytest.approx(1e-2)  # unscaled
    assert by_role["hidden"]["lr"] == pytest.approx(5e-3)  # 1e-2 * 2**-1
    # every model parameter is accounted for exactly once across the groups.
    grouped = sum((g["params"] for g in groups), [])
    assert len(grouped) == len(list(model.parameters()))


@pytest.mark.fast
def test_apply_mup_init_rescales_hidden_weight_std_with_width():
    torch.manual_seed(0)
    small = build_causal_lm(vocab=VOCAB, d_model=32, n_layer=2, n_head=2, block=BLOCK)
    apply_mup_init(small, base_width=32, base_std=0.02)
    torch.manual_seed(0)
    big = build_causal_lm(vocab=VOCAB, d_model=128, n_layer=2, n_head=2, block=BLOCK)
    apply_mup_init(big, base_width=32, base_std=0.02)

    small_std = small.blocks[0].attn.proj.weight.detach().std().item()
    big_std = big.blocks[0].attn.proj.weight.detach().std().item()
    # width_mult = 128/32 = 4 -> hidden std multiplier = 4**-0.5 = 0.5, so big's std should be
    # ~half of small's (measured on a few thousand samples, generous tolerance for sampling noise).
    assert big_std == pytest.approx(0.5 * small_std, rel=0.25)

    # LayerNorm affine params keep their identity init (Theta(1), width-independent) at every width.
    assert torch.allclose(big.blocks[0].ln1.weight, torch.ones_like(big.blocks[0].ln1.weight))
    assert torch.allclose(big.blocks[0].ln1.bias, torch.zeros_like(big.blocks[0].ln1.bias))


# --- 1. the core acceptance criterion: transferred lr vs. independently-tuned optimum ------------


@pytest.mark.slow
def test_mup_lr_transfer_matches_independently_tuned_optimum():
    """Tune once at BASE_WIDTH=32, transfer to widths 64 and 128 (mixle's rungs i-ii capacity-ladder
    proxy), and check the muP-predicted lr lands close to what an independent search finds at each
    target width -- reporting the actual measured ratio, not just asserting "close enough".
    """
    seeds = (1, 2, 3)
    n_steps = 60

    base_lr, base_loss = _bo_tune_log_lr(
        BASE_WIDTH, [(-4.0, -1.0)], use_mup=True, seeds=seeds, n_steps=n_steps, n_iter=12, seed=1
    )
    assert base_loss < 1.0  # sanity: the base-width search actually found something that learns the task

    ratios = []
    for target_width, bo_seed in ((64, 2), (128, 3)):
        predicted = transfer_lr(base_lr, BASE_WIDTH, target_width)
        # search a window around the prediction so the independent search isn't handed the answer,
        # but also isn't forced to rediscover the whole [1e-4, 1e-1] range from scratch every time.
        window = [(np.log10(predicted) - 1.5, np.log10(predicted) + 1.5)]
        tuned_lr, _tuned_loss = _bo_tune_log_lr(
            target_width, window, use_mup=True, seeds=seeds, n_steps=n_steps, n_iter=12, seed=bo_seed
        )
        ratio = predicted / tuned_lr
        ratios.append(ratio)
        print(
            f"[muP transfer] base_width={BASE_WIDTH} base_lr={base_lr:.4g} -> "
            f"target_width={target_width} predicted_lr={predicted:.4g} tuned_lr={tuned_lr:.4g} "
            f"ratio={ratio:.3f} ({abs(1 - ratio) * 100:.1f}% off)"
        )
        # muP is an asymptotic (large-width) guarantee; at these deliberately tiny, fast-to-train
        # widths (32 -> 64/128) the transferred lr is expected to land within a factor of ~3x of an
        # independent search's optimum -- loose relative to production-scale muP papers (which report
        # single-digit-percent transfer at 100x+ width) precisely because we're validating the
        # mechanism cheaply, not because the rule is expected to be looser at scale.
        assert 1.0 / 3.0 <= ratio <= 3.0, (
            f"transferred/tuned lr ratio {ratio:.3f} outside [1/3, 3] at width {target_width}"
        )

    print(f"[muP transfer] mean |ratio - 1| = {np.mean([abs(1 - r) for r in ratios]) * 100:.1f}%")


# --- 2. sanity contrast: WITHOUT muP, naively reusing the base lr does not transfer --------------


@pytest.mark.slow
def test_without_mup_naive_lr_reuse_is_worse_than_mup_transfer():
    """At a much wider target (16x the base width), reusing the base width's tuned lr verbatim with
    STANDARD (non-muP) parametrization should measurably underperform using the muP-transferred lr --
    this is the actual point of muP, demonstrated as an explicit contrast rather than tested in isolation.
    """
    seeds = (1, 2, 3)
    n_steps = 60
    target_width = 8 * BASE_WIDTH  # 256 -- large enough for standard parametrization to visibly break

    base_lr, _ = _bo_tune_log_lr(
        BASE_WIDTH, [(-4.0, -1.0)], use_mup=True, seeds=seeds, n_steps=n_steps, n_iter=12, seed=1
    )
    predicted = transfer_lr(base_lr, BASE_WIDTH, target_width)

    loss_transferred = _avg_loss(target_width, np.log10(predicted), use_mup=True, seeds=seeds, n_steps=n_steps)
    loss_naive = _avg_loss(target_width, np.log10(base_lr), use_mup=False, seeds=seeds, n_steps=n_steps)

    print(
        f"[muP contrast] target_width={target_width} base_lr={base_lr:.4g} predicted_lr={predicted:.4g}\n"
        f"  loss @ muP-transferred lr           = {loss_transferred:.4f}\n"
        f"  loss @ naive reuse (no muP, no rescale) = {loss_naive:.4f}"
    )
    # entropy floor for VOCAB=8 uniform predictions is ln(8) ~= 2.079; the naive run should be at or
    # above it (i.e. it has learned nothing, or worse, destabilized), while the transferred run should
    # have made real progress below it.
    assert loss_transferred < np.log(VOCAB) * 0.75
    assert loss_naive > loss_transferred * 2.0
