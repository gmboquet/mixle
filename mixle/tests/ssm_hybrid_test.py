"""E5 part 2 acceptance receipts for the hybrid block (local attention + selective-scan SSM + E2 far
field), see notes/designs/E5.md.

Three receipts:
1. ``HybridBlock`` satisfies the ``ContextMechanism`` protocol, trains via ``train_tbptt`` without error,
   and ``detach()`` actually cuts the TBPTT backward graph for all three sub-mechanisms' state.
2. ``report()``'s routing-mass receipt is a real reading of the forward pass's own softmax weights:
   ``local + far_field + ssm`` routing mass sums to 1.0 (float tolerance), and is explicitly not described
   as fractional output attribution.
3. A matched-parameter comparison of the hybrid, local-only, and SSM-only mechanisms under E7's corrected
   dependency-only training protocol. The old full-sequence protocol was dominated by identity targets and
   produced obsolete superiority numbers; this receipt records the corrected measurements without
   asserting that the current experimental hybrid has graduated.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.context_spine import ContextMechanism, SlidingWindowSpine, train_tbptt  # noqa: E402
from mixle.experimental.long_context_eval import _train_and_probe, copy_suite, needle_suite  # noqa: E402
from mixle.experimental.selective_scan import SelectiveScan  # noqa: E402
from mixle.experimental.ssm_hybrid import HybridBlock  # noqa: E402

# torch / experimental / slow markers come from mixle/tests/conftest.py's FILE_MARKERS table.


def _n_params(m) -> int:
    return int(sum(p.numel() for p in m.parameters()))


def _chunks(x, y, chunk_size: int) -> list:
    return [(x[:, i : i + chunk_size], y[:, i : i + chunk_size]) for i in range(0, x.shape[1], chunk_size)]


def _build(seed: int, cls, **kwargs):
    torch.manual_seed(seed)
    return cls(**kwargs)


# -------------------------------------------------------------------------------------------------------
# 1. ContextMechanism protocol conformance + TBPTT training + detach cuts the backward graph.
# -------------------------------------------------------------------------------------------------------


def test_hybrid_block_is_context_mechanism_and_trains():
    vocab = 10
    m = _build(
        0, HybridBlock, vocab=vocab, d_model=16, n_layer=1, n_head=2, window=5, d_state=4, ssm_expand=1, max_clusters=2
    )
    assert isinstance(m, ContextMechanism)

    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    rng = np.random.RandomState(1)
    x = torch.as_tensor(rng.randint(0, vocab, size=(1, 20)), dtype=torch.long)
    y = torch.as_tensor(rng.randint(0, vocab, size=(1, 20)), dtype=torch.long)
    state = m.init_state(1)
    chunks = _chunks(x, y, 5)
    receipt = train_tbptt(m, state, chunks, opt, detach_horizon=2)
    assert len(receipt["losses"]) == len(chunks)
    assert all(math.isfinite(loss_v) for loss_v in receipt["losses"])


def test_detach_cuts_the_backward_graph_for_all_three_branches():
    vocab = 10
    m = _build(
        0, HybridBlock, vocab=vocab, d_model=16, n_layer=1, n_head=2, window=5, d_state=4, ssm_expand=1, max_clusters=2
    )
    rng = np.random.RandomState(2)
    x = torch.as_tensor(rng.randint(0, vocab, size=(1, 16)), dtype=torch.long)
    y = torch.as_tensor(rng.randint(0, vocab, size=(1, 16)), dtype=torch.long)

    state = m.init_state(1)
    state, _ = m.step(state, (x[:, :8], y[:, :8]))
    state = m.detach(state)

    for k in state.near.cache_k:
        assert k is None or not k.requires_grad
    for v in state.near.cache_v:
        assert v is None or not v.requires_grad
    for bank in state.banks:
        assert not bank.mu_k.requires_grad
        assert not bank.sigma_vk.requires_grad
    for h in state.ssm_h:
        assert h is None or not h.requires_grad

    _, loss = m.step(state, (x[:, 8:], y[:, 8:]))
    loss.backward()
    assert m.qkv[0].weight.grad is not None
    assert torch.isfinite(m.qkv[0].weight.grad).all()


# -------------------------------------------------------------------------------------------------------
# 2. Routing receipt: real softmax-mass reading, sums to 1, non-negative.
# -------------------------------------------------------------------------------------------------------


def test_routing_receipt_sums_to_one_and_does_not_claim_output_attribution():
    vocab = 10
    m = _build(
        0, HybridBlock, vocab=vocab, d_model=16, n_layer=1, n_head=2, window=5, d_state=4, ssm_expand=1, max_clusters=2
    )
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    rng = np.random.RandomState(3)
    x = torch.as_tensor(rng.randint(0, vocab, size=(1, 20)), dtype=torch.long)
    y = torch.as_tensor(rng.randint(0, vocab, size=(1, 20)), dtype=torch.long)
    state = m.init_state(1)
    chunks = _chunks(x, y, 5)
    train_tbptt(m, state, chunks, opt, detach_horizon=len(chunks))

    report = m.report()
    print(f"[E5 part-2 receipt] routing report after a real trained step: {report}")
    assert set(report) == {
        "local_routing_mass",
        "far_field_routing_mass",
        "ssm_routing_mass",
    }
    total = sum(report.values())
    assert math.isclose(total, 1.0, abs_tol=1e-5), f"routing masses should sum to 1.0, got {total}"
    for name, share in report.items():
        assert share >= -1e-8, f"{name} share is negative: {share}"
        assert math.isfinite(share)
    # not a degenerate all-mass-on-one-branch receipt for THIS random init/training -- real evidence the
    # gate/joint-softmax are doing something, not fabricated to look balanced.
    assert max(report.values()) < 1.0 - 1e-6


def test_far_bank_advances_at_each_query_window_crossing_and_is_chunk_invariant():
    vocab = 10
    model = _build(
        4,
        HybridBlock,
        vocab=vocab,
        d_model=8,
        n_layer=1,
        n_head=2,
        window=2,
        d_state=2,
        ssm_expand=1,
        max_clusters=1,
    )
    x = torch.tensor([[1, 2, 3, 4, 5, 6]])
    y = torch.tensor([[2, 3, 4, 5, 6, 7]])

    full_state, full_loss = model.step(model.init_state(1), (x, y))
    full_bank = full_state.banks[0]
    full_receipt = model.last_receipts[0]

    chunked_state = model.init_state(1)
    chunk_losses = []
    for start in range(0, x.shape[1], 2):
        chunked_state, chunk_loss = model.step(
            chunked_state,
            (x[:, start : start + 2], y[:, start : start + 2]),
        )
        chunk_losses.append(chunk_loss)
    chunked_bank = chunked_state.banks[0]

    assert full_receipt["query_aligned"] is True
    assert full_receipt["ingestion_steps"] == x.shape[1] - model.window
    assert full_receipt["accounting"]["dual_visible_position_count"] == 0
    assert full_receipt["accounting"]["conserved"] is True
    torch.testing.assert_close(full_bank.count, chunked_bank.count)
    torch.testing.assert_close(full_bank.mu_k, chunked_bank.mu_k)
    torch.testing.assert_close(full_bank.mu_v, chunked_bank.mu_v)
    torch.testing.assert_close(full_bank.sigma_kk, chunked_bank.sigma_kk)
    torch.testing.assert_close(full_bank.sigma_vk, chunked_bank.sigma_vk)
    torch.testing.assert_close(full_state.near.cache_k[0], chunked_state.near.cache_k[0])
    torch.testing.assert_close(full_state.ssm_h[0], chunked_state.ssm_h[0])
    torch.testing.assert_close(full_loss, torch.stack(chunk_losses).mean())


def test_log_density_is_additive_sequence_log_likelihood():
    vocab = 5
    model = _build(
        5,
        HybridBlock,
        vocab=vocab,
        d_model=8,
        n_layer=1,
        n_head=2,
        window=2,
        d_state=2,
        ssm_expand=1,
        max_clusters=1,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    x = torch.tensor([[0, 1, 2], [0, 1, 2]])
    y = torch.tensor([[1, 2, 3], [1, 2, 3]])
    score = model.log_density(x, y)
    torch.testing.assert_close(score, torch.full((2,), -3.0 * math.log(vocab)))


@pytest.mark.parametrize(
    ("x", "y", "error"),
    [
        (torch.ones(2, dtype=torch.long), torch.ones(2, dtype=torch.long), ValueError),
        (torch.ones((1, 0), dtype=torch.long), torch.ones((1, 0), dtype=torch.long), ValueError),
        (torch.ones((1, 2), dtype=torch.float32), torch.ones((1, 2), dtype=torch.float32), TypeError),
    ],
)
def test_log_density_rejects_invalid_events(x, y, error):
    model = HybridBlock(
        5,
        d_model=8,
        n_layer=1,
        n_head=2,
        window=2,
        d_state=2,
        ssm_expand=1,
        max_clusters=1,
    )
    with pytest.raises(error):
        model.log_density(x, y)


# -------------------------------------------------------------------------------------------------------
# 3. Matched-params E7 comparison: hybrid vs. local-only vs. SSM-only ablations, local + global tasks.
# -------------------------------------------------------------------------------------------------------


def test_hybrid_ablations_are_measured_at_matched_params():
    vocab = 8
    window = 5

    hybrid = _build(
        0,
        HybridBlock,
        vocab=vocab,
        d_model=16,
        n_layer=1,
        n_head=2,
        window=window,
        d_state=4,
        ssm_expand=1,
        max_clusters=2,
    )
    local_only = _build(1, SlidingWindowSpine, vocab=vocab, d_model=18, n_layer=1, n_head=1, window=window)
    ssm_only = _build(2, SelectiveScan, vocab=vocab, d_model=18, d_state=6, n_layer=1, expand=1)

    params = {"hybrid": _n_params(hybrid), "local_only": _n_params(local_only), "ssm_only": _n_params(ssm_only)}
    spread = (max(params.values()) - min(params.values())) / min(params.values())
    print(f"[E5 part-2 receipt] matched-param configuration: {params} (spread={spread:.3%})")
    assert spread < 0.05, f"ablations are not matched within 5% of total parameters: {params}"

    results: dict[str, dict] = {}
    for name, mechanism in [("hybrid", hybrid), ("local_only", local_only), ("ssm_only", ssm_only)]:
        opt = torch.optim.Adam(mechanism.parameters(), lr=2e-2)
        rng = np.random.RandomState(0)
        local_task = _train_and_probe(
            mechanism,
            opt,
            copy_suite,
            distance=3,
            vocab=vocab,
            chunk_size=2,
            n_train_steps=400,
            n_eval_trials=40,
            rng=rng,
        )
        global_task = _train_and_probe(
            mechanism,
            opt,
            needle_suite,
            distance=8,
            vocab=vocab,
            chunk_size=4,
            n_train_steps=400,
            n_eval_trials=40,
            rng=rng,
        )
        results[name] = {
            "local_success": local_task["loss_threshold_success_rate"],
            "global_success": global_task["loss_threshold_success_rate"],
        }
        print(
            f"[E5 part-2 receipt] {name}: local(copy@3) success={local_task['loss_threshold_success_rate']:.3f} "
            f"loss={local_task['mean_probe_loss']:.3f} | "
            f"global(needle@8) success={global_task['loss_threshold_success_rate']:.3f} "
            f"loss={global_task['mean_probe_loss']:.3f} (chance={global_task['chance_loss']:.3f})"
        )

    for receipt in results.values():
        assert 0.0 <= receipt["local_success"] <= 1.0
        assert 0.0 <= receipt["global_success"] <= 1.0
