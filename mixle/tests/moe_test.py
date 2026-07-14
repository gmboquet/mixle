"""Acceptance tests for roadmap H2 (mixle/models/moe.py): MoE block + dense->MoE upcycling.

Three receipts, each with a real measured number in the assertion message:

1. matched-active-FLOPs loss win vs dense at small scale (Switch-Transformer-style top_k=1 routing
   keeps active compute/token equal to the dense MLP's, while total capacity scales with n_experts).
2. balance receipts (mixle.models.moe.expert_collapse_receipt) flag a deliberately-induced "merged"
   collapse and do NOT false-positive on a well-balanced run; a deliberately unstable routing schedule
   also trips "shattered" without tripping "merged".
3. upcycling receipt: the freshly-upcycled MoE's output is measurably close to (not identical to) the
   source dense block's output.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.moe import (
    MoEBlock,
    SharedResidualMoEMLP,
    expert_collapse_receipt,
    factorize_dense_mlp_to_moe,
    upcycle_dense_to_moe,
    upcycle_dense_to_shared_residual_moe,
)
from mixle.models.transformer import Block, CausalLM


def _regime_maps(vocab, n_regimes, seed=7):
    """``n_regimes`` fixed random permutations of the vocabulary -- a multi-regime "next token = f(last
    token)" ground truth. Which permutation applies is decided by the FIRST token's residue mod
    ``n_regimes``, so the same last token maps to a different next token depending on regime: a single
    dense FFN width must represent all regimes' mappings superimposed, while an MoE with >= n_regimes
    experts can let routing specialize one expert per regime -- exactly the capacity argument the
    published MoE-vs-dense-at-matched-FLOPs result rests on, reproduced here at toy scale.
    """
    rng = np.random.RandomState(seed)
    return [rng.permutation(vocab) for _ in range(n_regimes)]


def _synthetic_batch(vocab=16, block=12, n=32, seed=0, n_regimes=4):
    rng = np.random.RandomState(seed)
    x = rng.randint(0, vocab, size=(n, block))
    maps = _regime_maps(vocab, n_regimes)
    regime = x[:, 0] % n_regimes
    last = x[:, -1]
    y = np.array([maps[regime[i]][last[i]] for i in range(n)], dtype=np.int64)
    return torch.as_tensor(x.astype(float)), torch.as_tensor(y)


def _build_dense_lm(vocab, d_model, n_layer, n_head, block, seed):
    torch.manual_seed(seed)
    return CausalLM(vocab, d_model, n_layer, n_head, block)


def _build_moe_lm(vocab, d_model, n_layer, n_head, block, n_experts, top_k, seed):
    torch.manual_seed(seed)
    lm = CausalLM(vocab, d_model, n_layer, n_head, block)
    # top_k=1, expert_hidden=4*d_model (the dense MLP's own hidden width) matches active FLOPs/token to
    # the dense MLP exactly: one expert forward pass per token, same shape as the dense path.
    lm.blocks = torch.nn.ModuleList(
        [MoEBlock(d_model, n_head, n_experts, top_k=top_k, expert_hidden=4 * d_model) for _ in range(n_layer)]
    )
    return lm


def _train(lm, steps, vocab, block, *, n_regimes=4, batch_size=32, lr=3e-3, seed=0, moe=False):
    opt = torch.optim.Adam(lm.parameters(), lr=lr)
    losses = []
    routing_history = [] if moe else None
    for step in range(steps):
        x, y = _synthetic_batch(vocab=vocab, block=block, n=batch_size, seed=seed * 1000 + step, n_regimes=n_regimes)
        logits = lm(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        if moe:
            aux = sum(blk.aux_loss for blk in lm.blocks)
            (loss + aux).backward()
            routing_history.append(lm.blocks[0].routing_weights.numpy())
        else:
            loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.detach()))
    return losses, routing_history


class MatchedFlopsLossWinTest:
    def test_moe_beats_dense_at_matched_active_flops(self):
        # vocab=48 with a hidden width of only 4*d_model=64 gives the dense FFN just enough room to
        # squeeze by, not enough to comfortably superimpose n_regimes=6 independent permutation tables;
        # the MoE has the SAME active hidden width per token (top_k=1, expert_hidden=4*d_model) but
        # n_experts=n_regimes total experts to dedicate one per regime -- more total capacity at equal
        # active FLOPs/token, the textbook matched-FLOPs MoE-vs-dense setup.
        vocab, d_model, n_layer, n_head, block = 48, 16, 2, 4, 12
        n_experts, top_k, n_regimes, steps, batch_size = 6, 1, 6, 250, 48

        dense = _build_dense_lm(vocab, d_model, n_layer, n_head, block, seed=0)
        moe = _build_moe_lm(vocab, d_model, n_layer, n_head, block, n_experts, top_k, seed=0)

        dense_losses, _ = _train(
            dense, steps, vocab, block, n_regimes=n_regimes, batch_size=batch_size, seed=1, moe=False
        )
        moe_losses, _ = _train(moe, steps, vocab, block, n_regimes=n_regimes, batch_size=batch_size, seed=1, moe=True)

        dense_final = float(np.mean(dense_losses[-15:]))
        moe_final = float(np.mean(moe_losses[-15:]))

        print(f"[matched-FLOPs] dense final loss={dense_final:.4f}  moe final loss={moe_final:.4f}")
        assert moe_final < dense_final, (
            f"expected MoE (final loss {moe_final:.4f}) to beat dense (final loss {dense_final:.4f}) "
            "at matched active FLOPs/token"
        )


class ExpertCollapseReceiptTest:
    def _forced_merged_history(self, n_experts=4, rounds=6, n_tokens=64, seed=0):
        """Deliberately collapsed routing: nearly all mass on expert 0 every round (load-balance loss
        disabled + a gate strongly favoring one expert), no round-to-round variation."""
        rng = np.random.RandomState(seed)
        history = []
        for _ in range(rounds):
            probs = np.full((n_tokens, n_experts), 1.0e-3)
            probs[:, 0] = 1.0 - 1.0e-3 * (n_experts - 1)
            probs += rng.uniform(0, 1.0e-4, size=probs.shape)
            probs /= probs.sum(axis=1, keepdims=True)
            history.append(probs)
        return history

    def _balanced_history(self, n_experts=4, rounds=6, n_tokens=64, seed=0):
        """Well-behaved routing: near-uniform utilization, stable round to round (small Dirichlet noise
        around a uniform base, same base every round)."""
        rng = np.random.RandomState(seed)
        history = []
        base = np.full(n_experts, 1.0 / n_experts)
        for _ in range(rounds):
            noise = rng.normal(0, 0.02, size=(n_tokens, n_experts))
            probs = np.clip(base[None, :] + noise, 1.0e-3, None)
            probs /= probs.sum(axis=1, keepdims=True)
            history.append(probs)
        return history

    def _shattered_history(self, n_experts=4, rounds=6, n_tokens=64, seed=0):
        """Deliberately unstable routing: average utilization looks balanced (each expert gets a turn),
        but each individual round dumps essentially all mass on a DIFFERENT expert, so no expert ever
        sees a consistent token distribution -- the failure mode averaging alone would miss."""
        history = []
        for r in range(rounds):
            favored = r % n_experts
            probs = np.full((n_tokens, n_experts), 1.0e-3)
            probs[:, favored] = 1.0 - 1.0e-3 * (n_experts - 1)
            probs /= probs.sum(axis=1, keepdims=True)
            history.append(probs)
        return history

    def test_forced_merged_collapse_is_flagged(self):
        history = self._forced_merged_history()
        receipt = expert_collapse_receipt(history)
        print(
            f"[collapse] forced-merged effective_experts={receipt['effective_experts']:.2f} of {receipt['n_experts']}"
        )
        assert receipt["merged"] is True, receipt["diagnosis"]

    def test_balanced_run_does_not_false_positive(self):
        history = self._balanced_history()
        receipt = expert_collapse_receipt(history)
        print(
            f"[collapse] balanced effective_experts={receipt['effective_experts']:.2f} "
            f"instability={receipt['instability']:.3f}"
        )
        assert receipt["merged"] is False, receipt["diagnosis"]
        assert receipt["shattered"] is False, receipt["diagnosis"]

    def test_forced_shattered_instability_is_flagged_without_merged(self):
        history = self._shattered_history()
        receipt = expert_collapse_receipt(history)
        print(
            f"[collapse] shattered instability={receipt['instability']:.3f} "
            f"effective_experts={receipt['effective_experts']:.2f}"
        )
        assert receipt["shattered"] is True, receipt["diagnosis"]
        # the pooled utilization across all rounds is still ~uniform (each expert had its turn), so this
        # scenario should NOT also read as "merged" -- averaging alone hides it, instability catches it.
        assert receipt["merged"] is False, receipt["diagnosis"]


class UpcyclingReceiptTest:
    def _trained_looking_dense(self, d_model=32, n_head=4, seed=0):
        torch.manual_seed(seed)
        dense = Block(d_model, n_head)
        # give the dense block real (trained-looking) weights rather than fresh init
        with torch.no_grad():
            for p in dense.parameters():
                p.add_(torch.randn_like(p) * 0.02)
        return dense

    def test_upcycled_output_is_close_to_dense(self):
        dense = self._trained_looking_dense()
        moe_block, receipt = upcycle_dense_to_moe(dense, n_experts=4, top_k=1, seed=0, noise_std=0.01)

        print(f"[upcycle] relative_output_diff={receipt['relative_output_diff']:.4f}")
        assert receipt["relative_output_diff"] > 0.0, "upcycled output should not be bit-identical to dense"
        assert receipt["relative_output_diff"] < 0.2, (
            f"upcycled output diverged too far from dense (relative diff {receipt['relative_output_diff']:.4f})"
        )

    def test_upcycling_is_closer_than_a_fresh_random_moe(self):
        """The whole point of upcycling is starting close to the dense function instead of from random
        init -- confirm the upcycled block actually beats a freshly-initialized (non-upcycled) MoE block
        of the same shape on the same probe."""
        d_model, n_head = 32, 4
        dense = self._trained_looking_dense(d_model=d_model, n_head=n_head)
        moe_block, receipt = upcycle_dense_to_moe(dense, n_experts=4, top_k=1, seed=0, noise_std=0.01)

        torch.manual_seed(1)
        fresh = MoEBlock(d_model, n_head, 4, top_k=1, expert_hidden=4 * d_model)
        probe = torch.randn(1, 64, d_model, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            dense.eval()
            fresh.eval()
            dense_out = dense(probe)
            fresh_out = fresh(probe)
            fresh_gap = float(
                torch.linalg.norm(fresh_out - dense_out) / torch.linalg.norm(dense_out).clamp_min(1.0e-12)
            )

        print(f"[upcycle] upcycled_diff={receipt['relative_output_diff']:.4f}  fresh_random_diff={fresh_gap:.4f}")
        assert receipt["relative_output_diff"] < fresh_gap

    def test_larger_noise_diverges_more(self):
        dense = self._trained_looking_dense()

        _, small_noise_receipt = upcycle_dense_to_moe(dense, n_experts=4, top_k=1, seed=0, noise_std=0.001)
        _, large_noise_receipt = upcycle_dense_to_moe(dense, n_experts=4, top_k=1, seed=0, noise_std=0.2)

        assert small_noise_receipt["relative_output_diff"] < large_noise_receipt["relative_output_diff"]


class SharedResidualMoETest:
    def test_dense_factorization_is_function_preserving_at_matched_active_width(self):
        torch.manual_seed(19)
        dense = Block(24, 4).mlp
        factored, receipt = factorize_dense_mlp_to_moe(
            dense,
            n_experts=4,
            common_fraction=0.5,
            top_k=1,
        )
        probe = torch.randn(3, 11, 24)
        with torch.no_grad():
            expected = dense(probe)
            actual = factored(probe)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert receipt["relative_output_diff"] < 1e-5
        assert factored.active_hidden == dense[0].out_features

    def test_only_routed_residual_experts_execute(self):
        module = SharedResidualMoEMLP(
            8,
            4,
            shared_hidden=16,
            residual_hidden=16,
            top_k=1,
        )
        calls = [0, 0, 0, 0]
        handles = []
        for idx, expert in enumerate(module.residual_experts):
            handles.append(
                expert.register_forward_hook(lambda _m, _i, _o, idx=idx: calls.__setitem__(idx, calls[idx] + 1))
            )
        module(torch.ones(32, 8))
        for handle in handles:
            handle.remove()
        assert sum(calls) == 1

    def test_disjoint_penalty_prefers_peaked_routing(self):
        module = SharedResidualMoEMLP(
            4,
            3,
            shared_hidden=4,
            residual_hidden=4,
            top_k=1,
        )
        x = torch.ones(16, 4)
        with torch.no_grad():
            module.gate.weight.zero_()
            module(x)
            uniform = float(module.last_disjoint_loss)
            module.gate.weight[0].fill_(4.0)
            module.gate.weight[1:].fill_(-4.0)
            module(x)
            peaked = float(module.last_disjoint_loss)
        assert peaked < uniform * 0.01

    def test_top1_task_loss_reaches_the_gate(self):
        dense = Block(12, 3).mlp
        module, _ = factorize_dense_mlp_to_moe(dense, n_experts=3, top_k=1)
        loss = module(torch.randn(10, 12)).square().mean()
        loss.backward()
        assert module.gate.weight.grad is not None
        assert torch.linalg.vector_norm(module.gate.weight.grad) > 0.0

    def test_block_upcycle_preserves_the_full_block(self):
        torch.manual_seed(23)
        dense = Block(24, 4)
        moe, receipt = upcycle_dense_to_shared_residual_moe(
            dense,
            n_experts=4,
            common_fraction=0.5,
            top_k=1,
        )
        probe = torch.randn(2, 13, 24)
        with torch.no_grad():
            expected = dense(probe)
            actual = moe(probe)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert receipt["active_hidden"] == dense.mlp[0].out_features
