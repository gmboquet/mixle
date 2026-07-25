"""E4 acceptance receipts for the hierarchical summary tree (see notes/designs/E4.md).

Four receipts:
1. needle retrieval beyond the exact/local window -- E7's ``needle_suite`` at ``distance > window``,
   ``SummaryTreeSpine`` averaged over several seeds vs. ``SlidingWindowSpine`` baseline, matched
   architecture, same seeds.
2. ablation -- ``aux_weight > 0`` vs ``aux_weight == 0``, same needle evaluation, same seeds: the
   auxiliary loss version scores materially higher.
3. positions stable under re-chunking -- identical token stream through two different chunk-size
   schedules produces identical tree topology (paths, node ids, finalization order) for every node.
4. stop-gradient horizon (receipted) -- detachment after ``H`` subsequent evictions, within the caller's
   detach cadence, plus the "load-bearing, not just bookkept" consequence check: a loss built purely from
   an archived (detached) node's summary must not touch ``compressor``'s gradient.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.context_spine import SlidingWindowSpine  # noqa: E402
from mixle.experimental.long_context_eval import _train_and_probe, needle_suite  # noqa: E402
from mixle.experimental.summary_tree import SummaryTreeSpine, digits_of, lca_depth  # noqa: E402

# torch / experimental / slow markers come from mixle/tests/conftest.py's FILE_MARKERS table.

VOCAB = 6
WINDOW = 4
FANOUT = 2
DISTANCE = 10  # well beyond WINDOW -- SlidingWindowSpine cannot see it by construction
CHUNK_SIZE = 4
SEEDS = (1, 2, 3, 4, 5)


def _run_tree(seed: int, *, aux_weight: float) -> dict:
    torch.manual_seed(seed)
    m = SummaryTreeSpine(
        VOCAB,
        d_model=24,
        n_layer=2,
        n_head=4,
        window=WINDOW,
        fanout=FANOUT,
        detach_horizon_nodes=4,
        aux_weight=aux_weight,
    )
    opt = torch.optim.Adam(m.parameters(), lr=1.5e-2)
    rng = np.random.RandomState(seed + 100)
    return _train_and_probe(
        m,
        opt,
        needle_suite,
        distance=DISTANCE,
        vocab=VOCAB,
        chunk_size=CHUNK_SIZE,
        n_train_steps=150,
        n_eval_trials=50,
        rng=rng,
    )


def _run_baseline(seed: int) -> dict:
    torch.manual_seed(seed)
    m = SlidingWindowSpine(VOCAB, d_model=24, n_layer=2, n_head=4, window=WINDOW)
    opt = torch.optim.Adam(m.parameters(), lr=1.5e-2)
    rng = np.random.RandomState(seed + 100)
    return _train_and_probe(
        m,
        opt,
        needle_suite,
        distance=DISTANCE,
        vocab=VOCAB,
        chunk_size=CHUNK_SIZE,
        n_train_steps=150,
        n_eval_trials=50,
        rng=rng,
    )


def test_needle_retrieval_beyond_window_beats_baseline():
    tree_acc = [_run_tree(s, aux_weight=0.2)["loss_threshold_success_rate"] for s in SEEDS]
    base_acc = [_run_baseline(s)["loss_threshold_success_rate"] for s in SEEDS]
    tree_mean, base_mean = float(np.mean(tree_acc)), float(np.mean(base_acc))

    print(
        f"[E4 receipt 1] needle accuracy at distance={DISTANCE} > window={WINDOW}: "
        f"SummaryTreeSpine mean={tree_mean:.3f} {tree_acc}  SlidingWindowSpine mean={base_mean:.3f} {base_acc}"
    )
    # SlidingWindowSpine is at-chance by construction past its window -- this is a real, measured
    # (not asserted-by-construction) margin, not just "different": the tree materially outperforms.
    assert tree_mean > base_mean + 0.1, (
        f"tree needle accuracy ({tree_mean:.3f}) is not materially above baseline ({base_mean:.3f})"
    )
    assert base_mean < 0.1, f"baseline unexpectedly solved distance > window: {base_mean:.3f}"


def test_ablation_auxiliary_loss_improves_summary_usefulness():
    on = [_run_tree(s, aux_weight=0.2)["loss_threshold_success_rate"] for s in SEEDS]
    off = [_run_tree(s, aux_weight=0.0)["loss_threshold_success_rate"] for s in SEEDS]
    on_mean, off_mean = float(np.mean(on)), float(np.mean(off))

    print(
        f"[E4 receipt 2] needle accuracy: aux_weight=0.2 mean={on_mean:.3f} {on}  aux_weight=0.0 mean={off_mean:.3f} {off}"
    )
    assert on_mean > off_mean, (
        f"auxiliary loss did not improve summary usefulness: on={on_mean:.3f} vs off={off_mean:.3f}"
    )


def test_positions_stable_under_rechunking():
    """Same token stream, two different chunk-size schedules covering the same total length --
    identical tree topology (path/level/g for every finalized node), by construction of the
    mixed-radix carry propagation processing evicted tokens one at a time regardless of arrival batching."""
    torch.manual_seed(0)
    length = 64
    x = torch.randint(0, VOCAB, (1, length))
    y = torch.randint(0, VOCAB, (1, length))

    def _topology(chunk_size: int) -> list[tuple[int, int, tuple[int, ...]]]:
        torch.manual_seed(0)
        m = SummaryTreeSpine(VOCAB, d_model=16, n_layer=1, n_head=2, window=WINDOW, fanout=FANOUT, aux_weight=0.0)
        state = m.init_state(1)
        with torch.no_grad():
            for i in range(0, length, chunk_size):
                state, _ = m.step(state, (x[:, i : i + chunk_size], y[:, i : i + chunk_size]))
        nodes = []
        for lvl in state.live:
            for n in lvl:
                nodes.append((n.level, n.g, n.path))
        return sorted(nodes)

    topo_32 = _topology(chunk_size=4)
    topo_8 = _topology(chunk_size=8)
    topo_16 = _topology(chunk_size=16)

    print(
        f"[E4 receipt 3] node count chunk_size=4: {len(topo_32)}  chunk_size=8: {len(topo_8)}  chunk_size=16: {len(topo_16)}"
    )
    assert len(topo_32) > 0, "test degenerate: no nodes finalized at all"
    assert topo_32 == topo_8 == topo_16, "tree topology (level, g, path) differs across chunk-size schedules"


def test_frontier_is_bounded_non_overlapping_and_incomplete_leaves_are_visible():
    torch.manual_seed(11)
    model = SummaryTreeSpine(
        VOCAB,
        d_model=16,
        n_layer=1,
        n_head=2,
        window=2,
        fanout=2,
        aux_weight=0.0,
    )
    state = model.init_state(1)
    x = torch.randint(0, VOCAB, (1, 18))
    y = torch.randint(0, VOCAB, (1, 18))

    state, _ = model.step(state, (x[:, :3], y[:, :3]))
    assert state.evicted_count == 1
    assert state.receipt["pending_leaves"] == 1
    assert len(model._far_field_nodes(state)) == 1

    state, _ = model.step(state, (x[:, 3:], y[:, 3:]))
    receipt = state.receipt
    assert receipt["evicted_tokens_per_stream"] == 16
    assert receipt["represented_leaf_mass"] == 16
    assert receipt["non_overlapping"] is True
    assert receipt["conserved"] is True
    assert receipt["stored_frontier_nodes"] <= receipt["storage_bound"]
    assert receipt["frontier_nodes"] < state.evicted_count


def test_outputs_and_frontier_are_invariant_to_chunk_schedule():
    torch.manual_seed(12)
    model = SummaryTreeSpine(
        VOCAB,
        d_model=16,
        n_layer=1,
        n_head=2,
        window=3,
        fanout=2,
        aux_weight=0.0,
    )
    x = torch.randint(0, VOCAB, (1, 12))
    y = torch.randint(0, VOCAB, (1, 12))

    with torch.no_grad():
        full_state, full_loss = model.step(model.init_state(1), (x, y))
        chunked_state = model.init_state(1)
        chunked_losses = []
        for start in range(0, x.shape[1], 3):
            chunked_state, chunk_loss = model.step(
                chunked_state,
                (x[:, start : start + 3], y[:, start : start + 3]),
            )
            chunked_losses.append(chunk_loss)

    torch.testing.assert_close(full_loss, torch.stack(chunked_losses).mean())
    assert full_state.receipt == chunked_state.receipt
    full_nodes = sorted(model._stored_frontier_nodes(full_state), key=lambda node: (node.level, node.g))
    chunked_nodes = sorted(
        model._stored_frontier_nodes(chunked_state),
        key=lambda node: (node.level, node.g),
    )
    assert [(node.level, node.g, node.represented_leaves) for node in full_nodes] == [
        (node.level, node.g, node.represented_leaves) for node in chunked_nodes
    ]
    for full_node, chunked_node in zip(full_nodes, chunked_nodes, strict=True):
        torch.testing.assert_close(full_node.histogram, chunked_node.histogram)
        torch.testing.assert_close(full_node.summary[0], chunked_node.summary[0])


def test_digits_of_and_lca_depth_basic_properties():
    """Sanity-checks the integer building blocks the re-chunking receipt and the far-field bias
    depend on, independent of any trained model."""
    assert digits_of(0, 4) == (0,)
    assert digits_of(5, 2) == (1, 0, 1)  # 5 = 1*1 + 0*2 + 1*4
    assert digits_of(13, 4) == (1, 3)  # 13 = 1*1 + 3*4

    # a node at level 1 with g=0 covers leaf positions [0, fanout); a query at position 0 is INSIDE
    # that range, so lca_depth should be 0 (no climbing needed -- the query's own leaf ancestor at
    # level 1 already equals the node's g).
    assert lca_depth(0, node_level=1, node_g=0, fanout=4) == 0
    assert lca_depth(3, node_level=1, node_g=0, fanout=4) == 0
    # a query far outside that node's range must climb at least one level.
    assert lca_depth(100, node_level=1, node_g=0, fanout=4) > 0
    # lca_depth is monotonically non-decreasing as the query moves further from the node's range.
    near = lca_depth(10, node_level=1, node_g=0, fanout=2)
    far = lca_depth(10_000, node_level=1, node_g=0, fanout=2)
    assert far >= near


def test_stop_gradient_horizon_exact_and_load_bearing():
    """The horizon receipt: (a) every archived frontier node is detached after at least ``H`` later
    evictions, with at most one caller chunk of cadence overshoot, and (b) the horizon is load-bearing:
    a loss built purely from an archived summary cannot touch ``compressor`` gradients."""
    torch.manual_seed(0)
    H = 2
    m = SummaryTreeSpine(
        VOCAB, d_model=16, n_layer=1, n_head=2, window=2, fanout=2, detach_horizon_nodes=H, aux_weight=0.1
    )
    state = m.init_state(1)
    length = 30
    x = torch.randint(0, VOCAB, (1, length))
    y = torch.randint(0, VOCAB, (1, length))
    chunks = [(x[:, i : i + 2], y[:, i : i + 2]) for i in range(0, length, 2)]

    total_loss = None
    for chunk in chunks:
        state, loss = m.step(state, chunk)
        total_loss = loss if total_loss is None else total_loss + loss
        state = m.detach(state)  # called every step -- exact per-node accounting requires this cadence
    total_loss.backward()

    n_checked = 0
    for level_pending in state.archived:
        for node in level_pending:
            diff = node.detached_at_evicted_count - node.finalized_step
            assert H <= diff < H + 2, (
                f"level {node.level} node g={node.g}: horizon accounting exceeds detach cadence (diff={diff}, H={H})"
            )
            assert node.summary[0].requires_grad is False and node.summary[0].grad_fn is None
            n_checked += 1
    print(f"[E4 receipt 4a] {n_checked} archived frontier nodes with receipted horizon accounting (H={H})")
    assert n_checked > 0, "test degenerate: no nodes were archived"

    archived_node = next(node for level_nodes in state.archived for node in level_nodes)
    m.zero_grad()
    probe_loss = m.predict_head(archived_node.summary[0]).sum()
    probe_loss.backward()
    compressor_touched = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.compressor.parameters())
    predict_head_touched = m.predict_head.weight.grad is not None
    print(
        f"[E4 receipt 4b] isolated backward from an archived summary: predict_head touched={predict_head_touched}, compressor touched={compressor_touched}"
    )
    assert predict_head_touched, "sanity check failed: the isolated backward pass did not run at all"
    assert not compressor_touched, "stop-gradient horizon is not load-bearing: gradient leaked into compressor"
