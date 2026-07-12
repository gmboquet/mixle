"""E10 quantized-key cell attention: the collapse identity on the real code path, streaming invariance,
protocol conformance, and the honesty receipts (occupancy, drops, eval-mode loss purity).

The headline test is exactness: a far-field cell store built by ``_fold_evictions`` must produce -- through
``_far_bank`` and the spine's joint softmax algebra -- the IDENTICAL output to an unmerged store holding
every evicted token in its own count-1 slot. That is the theorem the mechanism rests on (falsified
end-to-end 2026-07-12, ``experiments/group_attention/RESULTS.md``): merging same-cell tokens loses nothing
because attention weights are cell-constant.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.context_spine import train_tbptt  # noqa: E402
from mixle.experimental.quantized_key_attention import (  # noqa: E402
    ProductQuantizer,
    QuantizedKeyAttentionSpine,
    _far_bank,
    _fold_evictions,
    _windowed_logits,
)


def _store_dicts(spine, state):
    """Per (layer, batch, head): {flat cell id: count} -- chunking-invariant content snapshot."""
    radix = spine.codes_per_block ** torch.arange(spine.n_blocks, dtype=torch.long)
    out = []
    for layer in range(spine.n_layer):
        codes, counts = state.codes[layer], state.counts[layer]
        flat = (codes * radix).sum(-1)
        layer_dicts = []
        for bi in range(counts.shape[0]):
            for hi in range(counts.shape[1]):
                occ = counts[bi, hi] > 0
                layer_dicts.append(
                    {int(f): int(c) for f, c in zip(flat[bi, hi][occ].tolist(), counts[bi, hi][occ].tolist())}
                )
        out.append(layer_dicts)
    return out


def _stream(spine, state, x, chunk_size):
    with torch.no_grad():
        for start in range(0, x.shape[1], chunk_size):
            xc = x[:, start : start + chunk_size]
            state, _ = spine.step(state, (xc, xc))
    return state


class CollapseExactnessTest:
    def test_merged_cell_store_equals_per_token_store_through_the_joint_softmax(self):
        """Cell aggregation is an identity, not an approximation: fold 200 tokens into a 64-slot store,
        and compare against a 200-slot store with one count-1 slot per token -- outputs must agree to
        float64 precision through the exact functions the spine calls."""
        torch.manual_seed(0)
        n_head, head_dim, n_tokens = 2, 8, 200
        pq = ProductQuantizer(head_dim, n_blocks=2, codes_per_block=4).double()

        keys = torch.randn(1, n_tokens, n_head, head_dim, dtype=torch.float64)
        values = torch.randn(1, n_tokens, n_head, head_dim, dtype=torch.float64)
        evicted_codes = pq.encode(keys)

        capacity = 64  # 4^2 = 16 possible cells per head, so 64 slots can never overflow
        merged = _fold_evictions(
            torch.zeros(1, n_head, capacity, 2, dtype=torch.long),
            torch.zeros(1, n_head, capacity, dtype=torch.long),
            torch.zeros(1, n_head, capacity, head_dim, dtype=torch.float64),
            evicted_codes,
            values,
            codes_per_block=4,
        )
        assert merged[3] == 0, "the collapse claim is only tested on a drop-free store"
        assert int((merged[1] > 0).sum(-1).max()) < n_tokens, "tokens must actually have merged"

        reference = (
            evicted_codes.permute(0, 2, 1, 3),  # (1, n_head, n_tokens, n_blocks): one slot per token
            torch.ones(1, n_head, n_tokens, dtype=torch.long),
            values.permute(0, 2, 1, 3),
        )

        q_raw = torch.randn(1, 3, n_head, head_dim, dtype=torch.float64)
        k_raw = torch.randn(1, 3, n_head, head_dim, dtype=torch.float64)
        v_raw = torch.randn(1, 3, n_head, head_dim, dtype=torch.float64)
        near_logits, gap_logits, values, *_ = _windowed_logits(
            q_raw, k_raw, v_raw, None, None, window=8, head_dim=head_dim, pos=n_tokens, pq=pq
        )
        assert bool(torch.isneginf(gap_logits).all()), "a 3-token cache-less chunk has no gap tokens"

        outs = []
        for codes, counts, vsum in (merged[:3], reference):
            far_logits, vbar = _far_bank(codes, counts, vsum, q_raw, pq)
            joint = torch.cat([far_logits, gap_logits, near_logits], dim=-1).softmax(dim=-1)
            n_cells = far_logits.shape[-1]
            n_unfolded = gap_logits.shape[-1]
            outs.append(
                joint[..., :n_cells] @ vbar
                + joint[..., n_cells : n_cells + n_unfolded] @ values
                + joint[..., n_cells + n_unfolded :] @ values
            )
        assert torch.allclose(outs[0], outs[1], atol=1e-12), f"max err {(outs[0] - outs[1]).abs().max()}"

    def test_gap_bank_equals_folding_those_tokens_into_cells(self):
        """The blind-spot fix's own identity: tokens past the window horizon attended as count-1
        quantized gap tokens must produce the same distribution as first folding them into the far
        store -- so visibility cannot depend on chunk-boundary timing."""
        torch.manual_seed(8)
        n_head, head_dim, window = 2, 8, 4
        pq = ProductQuantizer(head_dim, n_blocks=2, codes_per_block=4).double()
        cache_k = torch.randn(1, 12, n_head, head_dim, dtype=torch.float64)  # 12 > window: 8-token gap
        cache_v = torch.randn(1, 12, n_head, head_dim, dtype=torch.float64)
        q_raw = torch.randn(1, 2, n_head, head_dim, dtype=torch.float64)
        k_raw = torch.randn(1, 2, n_head, head_dim, dtype=torch.float64)
        v_raw = torch.randn(1, 2, n_head, head_dim, dtype=torch.float64)
        empty = (
            torch.zeros(1, n_head, 32, 2, dtype=torch.long),
            torch.zeros(1, n_head, 32, dtype=torch.long),
            torch.zeros(1, n_head, 32, head_dim, dtype=torch.float64),
        )

        def joint_out(cache_k, cache_v, k_chunk, v_chunk, store, pos):
            near, gap, values, *_ = _windowed_logits(
                q_raw, k_chunk, v_chunk, cache_k, cache_v, window=window, head_dim=head_dim, pos=pos, pq=pq
            )
            far, vbar = _far_bank(*store, q_raw, pq)
            joint = torch.cat([far, gap, near], dim=-1).softmax(dim=-1)
            n_cells, n_unfolded = far.shape[-1], gap.shape[-1]
            return (
                joint[..., :n_cells] @ vbar
                + joint[..., n_cells : n_cells + n_unfolded] @ values
                + joint[..., n_cells + n_unfolded :] @ values
            )

        # Arm 1: 12 cache tokens unfolded -- queries reach the old ones through the gap bank.
        out_gap = joint_out(cache_k, cache_v, k_raw, v_raw, empty, pos=12)
        # Arm 2: tokens at least `window` behind the FIRST query (j <= pos - window) are quantized for
        # every query in the chunk, so folding them first must change nothing.
        n_fold = 12 - window + 1
        folded = _fold_evictions(*empty, pq.encode(cache_k[:, :n_fold]), cache_v[:, :n_fold], codes_per_block=4)
        assert folded[3] == 0
        out_folded = joint_out(cache_k[:, n_fold:], cache_v[:, n_fold:], k_raw, v_raw, folded[:3], pos=12)
        assert torch.allclose(out_gap, out_folded, atol=1e-12), f"max err {(out_gap - out_folded).abs().max()}"

    def test_fold_preserves_integer_counts_and_value_sum_gradients(self):
        torch.manual_seed(1)
        pq = ProductQuantizer(8, n_blocks=2, codes_per_block=4)
        keys = torch.randn(1, 40, 2, 8)
        values = torch.randn(1, 40, 2, 8, requires_grad=True)
        codes, counts, vsum, dropped = _fold_evictions(
            torch.zeros(1, 2, 32, 2, dtype=torch.long),
            torch.zeros(1, 2, 32, dtype=torch.long),
            torch.zeros(1, 2, 32, 8),
            pq.encode(keys),
            values,
            codes_per_block=4,
        )
        assert dropped == 0
        assert counts.dtype == torch.long and int(counts.sum()) == 40 * 2  # exact integer bookkeeping
        assert vsum.requires_grad, "value sums must carry the chunk's graph until detach()"
        vsum.sum().backward()
        assert values.grad is not None and torch.isfinite(values.grad).all()


class StreamingInvarianceTest:
    def test_chunk_size_does_not_change_far_store_content_or_probe_loss(self):
        """After the same 96-token prefix, chunk sizes 4/8/16 (all <= window) must leave identical
        integer cell contents and probe losses equal to float-addition-order tolerance."""
        torch.manual_seed(2)
        spine = QuantizedKeyAttentionSpine(11, d_model=16, n_layer=2, n_head=2, window=16, max_cells=64)
        x = torch.randint(0, 11, (1, 96))
        probe = (x[:, -1:], x[:, -1:])

        losses, stores = [], []
        for chunk_size in (4, 8, 16):
            state = _stream(spine, spine.init_state(1), x, chunk_size)
            with torch.no_grad():
                _, loss = spine.step(state, probe)
            losses.append(float(loss))
            stores.append(_store_dicts(spine, state))
        assert stores[0] == stores[1] == stores[2], "integer cell content must be chunking-invariant"
        assert max(losses) - min(losses) < 1e-4, f"probe losses diverged across chunkings: {losses}"

    def test_far_field_engages_beyond_the_window(self):
        torch.manual_seed(3)
        spine = QuantizedKeyAttentionSpine(7, d_model=16, n_layer=1, n_head=2, window=8, max_cells=64)
        state = _stream(spine, spine.init_state(1), torch.randint(0, 7, (1, 64)), 8)
        receipt = spine.occupancy_receipt(state)
        assert receipt["occupied_cells_per_layer"][0] > 0
        assert receipt["dropped_tokens"] == 0
        assert receipt["possible_cells"] == 16**2


class ProtocolAndTrainingTest:
    def test_tbptt_training_reaches_the_codebooks(self):
        torch.manual_seed(4)
        spine = QuantizedKeyAttentionSpine(9, d_model=16, n_layer=1, n_head=2, window=8, max_cells=32)
        opt = torch.optim.Adam(spine.parameters(), lr=1e-3)
        x = torch.randint(0, 9, (1, 40))
        chunks = [(x[:, i : i + 8], x[:, i : i + 8]) for i in range(0, 40, 8)]
        before = spine.pq[0].codebooks.detach().clone()
        result = train_tbptt(spine, spine.init_state(1), chunks, opt, detach_horizon=2)
        assert all(np.isfinite(loss) for loss in result["losses"])
        assert not torch.equal(before, spine.pq[0].codebooks.detach()), "codebooks must train"

    def test_detach_cuts_the_graph(self):
        torch.manual_seed(5)
        spine = QuantizedKeyAttentionSpine(7, d_model=16, n_layer=1, n_head=2, window=8, max_cells=32)
        state = spine.init_state(1)
        x = torch.randint(0, 7, (1, 24))
        for start in (0, 8, 16):  # long enough that evictions fold grad-carrying values into vsum
            state, _ = spine.step(state, (x[:, start : start + 8], x[:, start : start + 8]))
        assert state.vsum[0].requires_grad
        detached = spine.detach(state)
        assert not detached.vsum[0].requires_grad
        assert all(not c.requires_grad for c in detached.cache_k if c is not None)
        assert detached.pos == state.pos and detached.drops == state.drops

    def test_eval_probe_loss_is_pure_cross_entropy(self):
        """Under no_grad the commitment term must NOT contaminate the loss the E7 referee thresholds."""
        torch.manual_seed(6)
        spine = QuantizedKeyAttentionSpine(7, d_model=16, n_layer=1, n_head=2, window=8, max_cells=32)
        torch.manual_seed(6)
        no_commit = QuantizedKeyAttentionSpine(
            7, d_model=16, n_layer=1, n_head=2, window=8, max_cells=32, commit_weight=0.0
        )
        chunk = (torch.randint(0, 7, (1, 8)), torch.randint(0, 7, (1, 8)))
        with torch.no_grad():
            _, eval_loss = spine.step(spine.init_state(1), chunk)
            _, eval_loss_nc = no_commit.step(no_commit.init_state(1), chunk)
        _, train_loss = spine.step(spine.init_state(1), chunk)
        assert float(eval_loss) == pytest.approx(float(eval_loss_nc)), "no_grad loss must be commit-free"
        assert float(train_loss.detach()) > float(eval_loss), "grad-enabled loss must include the commitment term"

    def test_overflow_drops_are_counted_never_silent(self):
        torch.manual_seed(7)
        spine = QuantizedKeyAttentionSpine(13, d_model=16, n_layer=1, n_head=2, window=4, max_cells=1)
        state = _stream(spine, spine.init_state(1), torch.randint(0, 13, (1, 64)), 4)
        assert spine.occupancy_receipt(state)["dropped_tokens"] > 0
