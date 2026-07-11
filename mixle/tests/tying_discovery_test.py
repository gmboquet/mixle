"""Tests for H4 tying discovery: profile similarity, candidate proposal, and apply-with-receipts.

These are the acceptance tests for H4: "discovered ties on a trained small LM give stated param reduction
within a loss budget." Test 2 trains a tiny causal LM briefly, plants a known tying opportunity (one head's Q
weight copied -- with noise -- into another head's slot), and confirms discovery surfaces exactly that pair.
Test 3 applies the discovered tie and asserts both a bounded output-parity delta and a strictly non-zero
parameter reduction, printing the actual measured numbers.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.tying_discovery import (
    apply_tie,
    profile_distance,
    propose_ties,
    tensor_profile,
)
from mixle.models.transformer import build_causal_lm


def test_profile_of_permuted_tensor_is_identical():
    """Sorting removes the arrangement: a permutation of a tensor has an IDENTICAL profile."""
    torch.manual_seed(0)
    base = torch.randn(64)
    perm = base[torch.randperm(64)]
    p_base = tensor_profile(base, n_quantiles=128)
    p_perm = tensor_profile(perm, n_quantiles=128)
    # Same multiset of values -> same sorted curve -> same profile, up to float roundoff.
    assert profile_distance(p_base, p_perm) < 1e-6


def test_profile_of_permuted_tensor_of_different_shape_is_still_identical():
    """Profiles are shape-agnostic: reshaping (which is a permutation of a flat view) doesn't move the profile."""
    torch.manual_seed(1)
    base = torch.randn(8, 8)
    reshaped = base.flatten()[torch.randperm(64)].reshape(4, 16)
    p_base = tensor_profile(base, n_quantiles=64)
    p_reshaped = tensor_profile(reshaped, n_quantiles=64)
    assert profile_distance(p_base, p_reshaped) < 1e-6


def test_profile_distance_is_large_for_clearly_different_distributions():
    """N(0, 1) vs N(10, 5): very different scale/mean -> large profile distance."""
    torch.manual_seed(2)
    a = torch.randn(2000) * 1.0 + 0.0
    b = torch.randn(2000) * 5.0 + 10.0
    p_a = tensor_profile(a, n_quantiles=256)
    p_b = tensor_profile(b, n_quantiles=256)
    distance = profile_distance(p_a, p_b)
    # These distributions barely overlap; the gap between quantile functions should be on the order of the
    # mean shift (10) minus some slack for the differing spreads -- comfortably bigger than any noise-level
    # distance we'd see between near-duplicate tensors (see the "close" case below, which is < 1).
    assert distance > 5.0


def test_profile_distance_is_small_for_near_duplicate_tensors():
    """A tensor plus a small amount of noise should have a small, comfortably-separated profile distance."""
    torch.manual_seed(3)
    base = torch.randn(500)
    noisy = base + torch.randn(500) * 0.01
    p_base = tensor_profile(base, n_quantiles=256)
    p_noisy = tensor_profile(noisy, n_quantiles=256)
    distance = profile_distance(p_base, p_noisy)
    assert distance < 0.1


def _synthetic_batch(vocab=16, block=8, n=32, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randint(0, vocab, size=(n, block)).astype(float)
    y = rng.randint(0, vocab, size=n).astype(int)
    return torch.as_tensor(x), torch.as_tensor(y, dtype=torch.long)


def _train_briefly(model, x, y, steps=20, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    return float(loss.item())


def _head_slice(module, layer, head, part, d_model, n_head):
    """Return the (d_head, d_model) weight sub-block of a Block's fused qkv.weight for one head's Q/K/V.

    ``CausalAttention.qkv`` is a single ``nn.Linear(d_model, 3 * d_model)``; its weight (shape
    ``(3*d_model, d_model)``) reshapes as ``(3, n_head, d_head, d_model)`` in the same order the forward pass
    uses (``reshape(b, t, 3, h, d//h)``). Used only for the discovery-side test (profile comparison of
    per-head slices) -- applying a tie needs a whole standalone ``nn.Parameter``, see
    ``_whole_tensor_candidates`` below for the tensors actually tied.
    """
    d_head = d_model // n_head
    part_idx = {"q": 0, "k": 1, "v": 2}[part]
    qkv_weight = module.blocks[layer].attn.qkv.weight  # (3*d_model, d_model)
    reshaped = qkv_weight.reshape(3, n_head, d_head, d_model)
    return reshaped[part_idx, head]  # (d_head, d_model) view into qkv.weight


def _build_and_train():
    torch.manual_seed(42)
    vocab, d_model, n_layer, n_head, block = 16, 32, 2, 4, 8
    model = build_causal_lm(vocab, d_model, n_layer, n_head, block)
    x, y = _synthetic_batch(vocab=vocab, block=block)
    loss = _train_briefly(model, x, y)
    return model, x, y, loss, dict(vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block)


def _plant_tying_opportunity(model, cfg, noise_std=1e-3):
    """Deliberately copy layer0/head0's Q weight into layer1/head2's Q slot, with a bit of noise.

    Small noise simulates the realistic "very similar but not identical" case tying discovery is meant to
    catch (an exact copy would be a degenerate, uninterestingly-easy case).
    """
    d_model, n_head = cfg["d_model"], cfg["n_head"]
    d_head = d_model // n_head
    with torch.no_grad():
        src = _head_slice(model, layer=0, head=0, part="q", d_model=d_model, n_head=n_head).clone()
        qkv1 = model.blocks[1].attn.qkv.weight  # (3*d_model, d_model)
        reshaped = qkv1.reshape(3, n_head, d_head, d_model)
        reshaped[0, 2] = src + torch.randn(d_head, d_model) * noise_std
        model.blocks[1].attn.qkv.weight.copy_(reshaped.reshape(3 * d_model, d_model))


def _collect_head_profiles_input(model, cfg):
    d_model, n_head, n_layer = cfg["d_model"], cfg["n_head"], cfg["n_layer"]
    named = {}
    for layer in range(n_layer):
        for head in range(n_head):
            for part in ("q", "k", "v"):
                name = "layer%d.head%d.%s" % (layer, head, part)
                named[name] = _head_slice(model, layer, head, part, d_model, n_head)
    return named


def test_tying_discovery_surfaces_the_planted_pair():
    model, x, y, loss, cfg = _build_and_train()
    assert np.isfinite(loss)

    _plant_tying_opportunity(model, cfg)

    named = _collect_head_profiles_input(model, cfg)
    candidates = propose_ties(named, n_quantiles=128)
    assert len(candidates) > 0

    planted = {"layer0.head0.q", "layer1.head2.q"}
    top = candidates[0]
    top_pair = {top.name_a, top.name_b}
    # The planted pair should be the single closest (or at worst clearly separated within the top few) --
    # check it is #1, and that its distance is well below the next-closest UNRELATED pair's distance.
    assert top_pair == planted, "expected planted pair on top, got %r (distance=%.4g)" % (top_pair, top.distance)

    unrelated_distances = [c.distance for c in candidates if {c.name_a, c.name_b} != planted]
    assert unrelated_distances, "need at least one unrelated pair to compare separation against"
    assert top.distance < min(unrelated_distances) / 2.0, (
        "planted pair (%.4g) should be clearly separated from the closest unrelated pair (%.4g)"
        % (top.distance, min(unrelated_distances))
    )


def _plant_whole_tensor_tying_opportunity(model, noise_std=1e-3):
    """Deliberately copy block0's whole MLP input-projection weight into block1's, plus small noise.

    Unlike the fused ``qkv.weight`` (whose per-head Q/K/V slices are views, not standalone
    ``nn.Parameter``s), ``mlp[0].weight`` is a real, independently addressable ``nn.Parameter`` per block --
    so it is directly tie-able via :func:`apply_tie`, which needs an attribute path it can ``setattr`` a
    shared ``nn.Parameter`` onto.
    """
    with torch.no_grad():
        src = model.blocks[0].mlp[0].weight.detach().clone()
        model.blocks[1].mlp[0].weight.copy_(src + torch.randn_like(src) * noise_std)


def test_apply_tie_gives_bounded_parity_and_real_param_reduction():
    model, x, y, loss, cfg = _build_and_train()
    _plant_whole_tensor_tying_opportunity(model)

    named = {"blocks.%d.mlp[0].weight" % layer: model.blocks[layer].mlp[0].weight for layer in range(cfg["n_layer"])}
    candidates = propose_ties(named, n_quantiles=128)
    assert len(candidates) > 0
    top = candidates[0]
    assert {top.name_a, top.name_b} == {"blocks.0.mlp[0].weight", "blocks.1.mlp[0].weight"}

    receipt = apply_tie(
        model,
        "blocks.0.mlp.0.weight",
        "blocks.1.mlp.0.weight",
        inputs=x,
        strategy="average",
    )

    print("H4 parity receipt: max_abs_diff=%.6g relative_l2=%.6g" % (receipt.max_abs_diff, receipt.relative_l2))
    print(
        "H4 param reduction: %d -> %d params (-%d, -%.2f%%)"
        % (
            receipt.params_before,
            receipt.params_after,
            receipt.params_reduced,
            100.0 * receipt.params_reduced_fraction,
        )
    )

    # Tolerance: block1's mlp[0].weight was planted as block0's mlp[0].weight plus noise_std=1e-3 Gaussian
    # noise (see _plant_whole_tensor_tying_opportunity), so "average" replaces each of the two tensors with
    # something within ~noise_std/2 of where it started -- a small, bounded perturbation, not zero (weight
    # tying is NOT assumed output-preserving; this is the measured receipt). 0.25 relative-L2 is a generous
    # multiple of what that small a weight perturbation, propagated through two un-normalized-after residual
    # blocks plus an output head, is expected to produce -- loose enough to not be flaky over the model's
    # random init/training seed, tight enough to demonstrate the tie was genuinely near-function-preserving
    # (as the profile-similarity threshold that nominated it promised) rather than an arbitrary large edit.
    assert np.isfinite(receipt.max_abs_diff)
    assert np.isfinite(receipt.relative_l2)
    assert receipt.relative_l2 < 0.25

    # Real, strictly non-zero parameter reduction: two (d_model, 4*d_model) tensors collapse to one.
    assert receipt.params_reduced > 0
    assert receipt.params_after < receipt.params_before
