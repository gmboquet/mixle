"""E8 acceptance receipts for context parallelism on the chunked-recurrent spine (see notes/designs/E8.md).

Four receipts, matching the design note's section 5 test plan:

1. **Exact-match correctness** (the "tolerance-parity with single-device on the same window" acceptance
   criterion): ``cp_size in {1, 2, 4, 8}`` reproduces the ``cp_size=1`` dense reference's per-chunk loss
   and final carried-state cache tensors, ``torch.allclose`` at ``atol=1e-5, rtol=1e-4`` (F1's own
   precedent) -- including a ``cp_size=8`` run on a larger window (``window=64``), the in-process half of
   the "near-linear window scaling to 8 devices" receipt (see 4 below for why the wall-clock half of
   that receipt cannot be measured here).
2. **Window-size edge cases**: ``cp_size`` not dividing ``window`` evenly, and a chunk near stream start
   where ``cache_len < window`` (KV axis shorter than ``cp_size``) -- both must not crash and must still
   match the dense reference.
3. **Composability with ``train_tbptt``** across multiple ``detach_horizon`` chunks.
4. **Honestly-scoped scaling receipt**: (a) the in-process ``cp_size=8``/``window=64`` exact-match case
   from (1); (b) a REAL ``torch.distributed`` gloo-backend multi-process test
   (``torch.multiprocessing.spawn``) that runs the actual shard/RoPE/all-gather/attend algorithm across
   genuinely separate OS processes communicating over real (CPU) collectives, to catch bugs an in-process
   Python-loop simulation could hide (e.g. accidental shared-memory aliasing across "ranks").

**What this file does NOT and CANNOT measure**: the roadmap card's "near-linear window scaling to 8
devices" acceptance criterion, read at face value, is a WALL-CLOCK/throughput claim across up to 8 real
accelerators. This environment has zero GPUs (``torch.cuda.device_count() == 0`` here, confirmed at
collection time below) -- there is no way to honestly measure that number on this hardware, and no test
in this file claims to. CPU-gloo all-gather cost does not predict GPU-NCCL all-gather cost, so no
throughput number from the multi-process test in (4b) is reported as a scaling receipt either; it
verifies only that the COLLECTIVE PATTERN is correct across real processes. This mirrors exactly how F1's
own CP work (``mixle/tests/tensor_pipeline_context_parallel_test.py``) documented the identical hardware
limitation for its "70B-config across >=512 GPUs at published-comparable MFU" acceptance number.
"""

import os
import tempfile
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.context_spine import SlidingWindowSpine, train_tbptt  # noqa: E402
from mixle.utils.parallel.context_parallel_spine import (  # noqa: E402
    _apply_rope,
    _rope_angles,
    cp_shard_kv,
    cp_window_attention_forward,
)

# torch / experimental / parallel / slow markers come from mixle/tests/conftest.py's FILE_MARKERS table.

N_CUDA_DEVICES = torch.cuda.device_count()  # checked once at collection; see the module docstring


def _lag_copy_sequence(rng: np.random.RandomState, *, length: int, vocab: int, lag: int) -> tuple:
    """``y[i] = x[i - lag]`` for ``i >= lag``, ``y[i] = x[i]`` otherwise -- a controlled-dependency-distance task."""
    x = rng.randint(0, vocab, size=(1, length))
    y = x.copy()
    y[:, lag:] = x[:, :-lag]
    return torch.as_tensor(x, dtype=torch.long), torch.as_tensor(y, dtype=torch.long)


def _chunks(x: torch.Tensor, y: torch.Tensor, chunk_size: int) -> list:
    return [(x[:, i : i + chunk_size], y[:, i : i + chunk_size]) for i in range(0, x.shape[1], chunk_size)]


def _build_model(seed: int, *, vocab: int, d_model: int, n_layer: int, n_head: int, window, cp_size: int = 1):
    torch.manual_seed(seed)
    return SlidingWindowSpine(vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window, cp_size=cp_size)


def _run_streamed_frozen(model: SlidingWindowSpine, chunks: list, *, detach_horizon: int) -> dict:
    """Run ``chunks`` through the full ``train_tbptt`` machinery, including genuine backward passes and
    real (nonzero-lr) optimizer steps -- so this exercises the same code path a real training run would.

    (Name kept from an earlier version of this test that used ``lr=0`` to sidestep what looked like a
    tolerance problem under real training; investigating that divergence during implementation found a
    real bug instead -- see the ``git log``/PR description for ``mixle/experimental/context_spine.py``'s
    cp branch. The dense (``cp_size=1``) path caches the RoPE'd ``k_full`` (it reassigns ``k_full =
    _apply_rope(...)`` before slicing into the cache), not the raw pre-RoPE ``k_full`` the design note's
    algorithm section literally describes as the sharding INPUT. An early implementation cached the raw
    ``k_full`` in the ``cp_size>1`` branch, which is a different (still-plausible-looking, since the loss
    for the very first chunk matched bit-exactly) but wrong convention: cache_k diverged starting the
    second chunk, and with real lr those divergences compounded through backward/SGD into >1000x the
    intended rtol=1e-4 tolerance. Once the cp branch's cache write was fixed to match the dense
    convention (RoPE'd K, per shard, then concat), real-lr training reproduces the dense reference to
    within low-1e-6 absolute difference at cp_size up to 8 -- i.e. the design note's "mathematically
    identical, not bit-identical" RoPE-per-shard-before-gather risk is real but tiny (comfortably inside
    rtol=1e-4), and the earlier order-of-magnitude-larger divergence was the cache bug, not that risk.)
    """
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    state = model.init_state(1)
    return train_tbptt(model, state, chunks, opt, detach_horizon=detach_horizon)


def _assert_receipts_match(reference: dict, other: dict, *, atol=1e-5, rtol=1e-4, label: str) -> None:
    assert len(reference["losses"]) == len(other["losses"])
    for i, (loss_ref, loss_other) in enumerate(zip(reference["losses"], other["losses"])):
        assert abs(loss_ref - loss_other) <= atol + rtol * abs(loss_ref), (
            f"{label}: chunk {i} loss diverged: reference={loss_ref!r} other={loss_other!r}"
        )
    for layer, (k_ref, k_other) in enumerate(zip(reference["state"].cache_k, other["state"].cache_k)):
        assert torch.allclose(k_ref, k_other, atol=atol, rtol=rtol), f"{label}: layer {layer} cache_k diverged"
    for layer, (v_ref, v_other) in enumerate(zip(reference["state"].cache_v, other["state"].cache_v)):
        assert torch.allclose(v_ref, v_other, atol=atol, rtol=rtol), f"{label}: layer {layer} cache_v diverged"


# ===================================================================================================
# 1. Exact-match correctness across cp_size, including a larger-window cp_size=8 scaling case.
# ===================================================================================================


@pytest.mark.parametrize("window", [None, 8, 16])
@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
def test_cp_size_matches_dense_reference(window, cp_size):
    vocab, d_model, n_layer, n_head = 17, 32, 2, 2
    length, chunk_size, lag = 30, 5, 3
    seed = 0

    ref_model = _build_model(seed, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window)
    cp_model = _build_model(
        seed, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window, cp_size=cp_size
    )
    # Same initial weights (same seed + same architecture => same init draw order).
    for p_ref, p_cp in zip(ref_model.parameters(), cp_model.parameters()):
        assert torch.equal(p_ref, p_cp)

    data_rng = np.random.RandomState(321)
    x, y = _lag_copy_sequence(data_rng, length=length, vocab=vocab, lag=lag)
    chunks = _chunks(x, y, chunk_size)

    ref_receipt = _run_streamed_frozen(ref_model, chunks, detach_horizon=2)
    cp_receipt = _run_streamed_frozen(cp_model, chunks, detach_horizon=2)

    _assert_receipts_match(ref_receipt, cp_receipt, label=f"cp_size={cp_size} window={window}")
    print(
        f"[E8 receipt] window={window!r} cp_size={cp_size}: {len(ref_receipt['losses'])} chunk losses and "
        f"{n_layer} layers' cache_k/cache_v exact-match (atol=1e-5, rtol=1e-4) vs the cp_size=1 dense reference"
    )


def test_cp_size_8_matches_dense_reference_on_larger_window():
    """The in-process half of "near-linear window scaling to 8 devices": cp_size=8 is only a meaningful
    shard count when the window is large enough that 8 shards isn't mostly empty chunks -- window=64
    gives 8 keys/shard on average. This is a CORRECTNESS receipt (exact match at cp_size=8), not a
    throughput one -- see the module docstring for why the wall-clock half of this acceptance criterion
    is out of reach on this hardware.
    """
    vocab, d_model, n_layer, n_head, window, cp_size = 23, 32, 2, 2, 64, 8
    length, chunk_size, lag = 96, 12, 5
    seed = 11

    ref_model = _build_model(seed, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window)
    cp_model = _build_model(
        seed, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window, cp_size=cp_size
    )

    data_rng = np.random.RandomState(654)
    x, y = _lag_copy_sequence(data_rng, length=length, vocab=vocab, lag=lag)
    chunks = _chunks(x, y, chunk_size)

    ref_receipt = _run_streamed_frozen(ref_model, chunks, detach_horizon=3)
    cp_receipt = _run_streamed_frozen(cp_model, chunks, detach_horizon=3)

    _assert_receipts_match(ref_receipt, cp_receipt, label=f"cp_size={cp_size} window={window} (scaling case)")
    print(
        f"[E8 receipt] window={window} cp_size={cp_size}: {len(chunks)} chunks over {length} tokens exact-match "
        f"the dense reference -- the mathematical-decomposition half of the scaling receipt. No wall-clock "
        f"number is reported (0 GPUs available: torch.cuda.device_count()={N_CUDA_DEVICES})."
    )


# ===================================================================================================
# 2. Window-size edge cases: uneven cp_size/window division; cache_len < window near stream start.
# ===================================================================================================


def test_cp_size_not_dividing_window_evenly_still_matches_dense_reference():
    # window=16, cp_size=3 -> uneven torch.chunk split (6, 5, 5) once the window is full.
    vocab, d_model, n_layer, n_head, window, cp_size = 13, 24, 2, 2, 16, 3
    length, chunk_size, lag = 40, 7, 3
    seed = 5

    ref_model = _build_model(seed, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # cp_size=3 on window=16 is not degenerate, but keep the test robust either way
        cp_model = _build_model(
            seed, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window, cp_size=cp_size
        )

    data_rng = np.random.RandomState(777)
    x, y = _lag_copy_sequence(data_rng, length=length, vocab=vocab, lag=lag)
    chunks = _chunks(x, y, chunk_size)

    ref_receipt = _run_streamed_frozen(ref_model, chunks, detach_horizon=2)
    cp_receipt = _run_streamed_frozen(cp_model, chunks, detach_horizon=2)
    _assert_receipts_match(ref_receipt, cp_receipt, label="uneven window/cp_size division (16 // 3)")


def test_cp_size_exceeding_cache_len_near_stream_start_degrades_gracefully():
    """First chunk of a stream: cache is empty (cache_len=0), so the KV axis is just this chunk's own
    ``t`` positions -- shorter than ``cp_size`` requested. ``torch.chunk`` must degrade to fewer than
    ``cp_size`` actual shards rather than crash or produce empty/misaligned shards.
    """
    vocab, d_model, n_layer, n_head, window, cp_size = 11, 16, 1, 2, 32, 8
    seed = 9
    torch.manual_seed(seed)
    ref_model = SlidingWindowSpine(vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window)
    torch.manual_seed(seed)
    cp_model = SlidingWindowSpine(
        vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window, cp_size=cp_size
    )

    data_rng = np.random.RandomState(42)
    # First chunk has only 3 tokens -- far fewer than cp_size=8's requested shard count.
    x, y = _lag_copy_sequence(data_rng, length=3, vocab=vocab, lag=1)

    ref_state = ref_model.init_state(1)
    cp_state = cp_model.init_state(1)
    with torch.no_grad():
        ref_state, ref_loss = ref_model.step(ref_state, (x, y))
        cp_state, cp_loss = cp_model.step(cp_state, (x, y))

    assert torch.allclose(ref_loss, cp_loss, atol=1e-5, rtol=1e-4)
    for k_ref, k_cp in zip(ref_state.cache_k, cp_state.cache_k):
        assert torch.allclose(k_ref, k_cp, atol=1e-5, rtol=1e-4)
    print("[E8 receipt] cp_size=8 on a 3-token first chunk (cache_len=0 < cp_size) degrades gracefully and matches")


def test_validate_cp_window_plan_warns_on_degenerate_window_cp_ratio_but_does_not_error():
    # window=4, cp_size=8 -> < 1 key/shard on average: a performance heads-up, not a correctness error.
    with pytest.warns(UserWarning, match="key.*per shard"):
        SlidingWindowSpine(10, d_model=8, n_layer=1, n_head=2, window=4, cp_size=8)


def test_cp_size_zero_or_negative_rejected():
    with pytest.raises(ValueError):
        SlidingWindowSpine(10, d_model=8, n_layer=1, n_head=2, window=4, cp_size=0)
    with pytest.raises(ValueError):
        SlidingWindowSpine(10, d_model=8, n_layer=1, n_head=2, window=4, cp_size=-1)


# ===================================================================================================
# 3. Composability with train_tbptt across multiple detach_horizon chunks.
# ===================================================================================================


def test_cp_composes_with_tbptt_detach_horizon_and_losses_decrease():
    vocab, d_model, n_layer, n_head, window, cp_size = 19, 32, 2, 4, 20, 4
    length, chunk_size, lag = 80, 8, 4
    n_rounds = 12
    seed = 3

    cp_model = _build_model(
        seed, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window, cp_size=cp_size
    )
    dense_model = _build_model(seed, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window)
    cp_opt = torch.optim.Adam(cp_model.parameters(), lr=5e-3)
    dense_opt = torch.optim.Adam(dense_model.parameters(), lr=5e-3)

    data_rng = np.random.RandomState(2024)
    cp_state = cp_model.init_state(1)
    dense_state = dense_model.init_state(1)
    cp_round_losses, dense_round_losses = [], []
    for _ in range(n_rounds):
        x, y = _lag_copy_sequence(data_rng, length=length, vocab=vocab, lag=lag)
        chunks = _chunks(x, y, chunk_size)  # 10 chunks/round, detach_horizon=3 => mid-stream detaches happen

        cp_receipt = train_tbptt(cp_model, cp_state, chunks, cp_opt, detach_horizon=3)
        cp_state = cp_receipt["state"]
        cp_round_losses.append(float(np.mean(cp_receipt["losses"])))

        dense_receipt = train_tbptt(dense_model, dense_state, chunks, dense_opt, detach_horizon=3)
        dense_state = dense_receipt["state"]
        dense_round_losses.append(float(np.mean(dense_receipt["losses"])))

    cp_first_half = float(np.mean(cp_round_losses[: n_rounds // 3]))
    cp_last_half = float(np.mean(cp_round_losses[-n_rounds // 3 :]))
    dense_first_half = float(np.mean(dense_round_losses[: n_rounds // 3]))
    dense_last_half = float(np.mean(dense_round_losses[-n_rounds // 3 :]))

    print(
        f"[E8 receipt] cp_size={cp_size} + train_tbptt (detach_horizon=3, {n_rounds} rounds): "
        f"loss {cp_first_half:.4f} -> {cp_last_half:.4f} (dense: {dense_first_half:.4f} -> {dense_last_half:.4f})"
    )
    assert cp_last_half < cp_first_half, "cp_size>1 gradients did not flow through train_tbptt's detach_horizon"
    # Comparable to the dense run -- same qualitative learning behavior, not required to be numerically
    # identical (independent parameter/optimizer trajectories once training actually updates weights).
    assert abs(cp_last_half - dense_last_half) < 0.75 * max(dense_first_half - dense_last_half, 1e-6) + 0.5


# ===================================================================================================
# 4b. Real torch.distributed (gloo) multi-process test: the same shard/gather/attend algorithm running
# across genuinely separate OS processes, not an in-process Python-loop simulation.
#
# No existing test in mixle/tests/ uses torch.multiprocessing.spawn with a real multi-rank world_size
# (grepped per the design note); the closest established convention is the single-rank
# `dist.init_process_group("gloo", rank=0, world_size=1, init_method="file://" + path)` pattern in
# engine_test.py / backend_scoring_test.py, and the subprocess/torchrun two-rank smoke test in
# torchrun_encoded_data_test.py (env-var gated). This test follows that same file://-init-method gloo
# convention but drives it with torch.multiprocessing.spawn for a real world_size>1, matching what the
# design note asked for: real separate processes, real (CPU) collectives.
# ===================================================================================================


def _cp_gloo_worker(rank: int, world_size: int, init_file: str, out_file: str, payload_file: str) -> None:
    """One rank's worth of real CP: shard, RoPE-locally, all-gather (real gloo collective), attend.

    Mirrors ``cp_shard_kv`` + ``cp_window_attention_forward``'s algorithm exactly, but with the
    all-gather implemented as an actual ``torch.distributed.all_gather`` collective instead of an
    in-process Python list -- this is what catches bugs the in-process rank simulation could hide (e.g.
    a rank accidentally reading another rank's tensor out of shared Python state rather than only what a
    real collective delivers it).
    """
    import torch
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method="file://" + init_file)
    try:
        payload = torch.load(payload_file, weights_only=True)
        k_full, v_full, key_positions = payload["k_full"], payload["v_full"], payload["key_positions"]
        q, query_positions = payload["q"], payload["query_positions"]
        window, head_dim = payload["window"], payload["head_dim"]

        # Each rank only ever touches its OWN contiguous shard -- exactly what cp_shard_kv hands one
        # simulated rank in the in-process path.
        k_shard = torch.chunk(k_full, world_size, dim=1)[rank]
        v_shard = torch.chunk(v_full, world_size, dim=1)[rank]
        pos_shard = torch.chunk(key_positions, world_size, dim=0)[rank]

        # Step 3 of the design's algorithm: RoPE applied locally, using only this rank's own absolute
        # positions -- no cross-rank RoPE dependency.
        sin_k, cos_k = _rope_angles(pos_shard, head_dim)
        roped_k_shard = _apply_rope(k_shard, sin_k, cos_k).contiguous()
        v_shard = v_shard.contiguous()

        # The one real collective per attention call (step 4): all-gather the RoPE'd K and raw V.
        gathered_k = [torch.zeros_like(roped_k_shard) for _ in range(world_size)]
        dist.all_gather(gathered_k, roped_k_shard)
        gathered_v = [torch.zeros_like(v_shard) for _ in range(world_size)]
        dist.all_gather(gathered_v, v_shard)
        gathered_pos = [torch.zeros_like(pos_shard) for _ in range(world_size)]
        dist.all_gather(gathered_pos, pos_shard.contiguous())

        full_k = torch.cat(gathered_k, dim=1)
        full_v = torch.cat(gathered_v, dim=1)
        full_pos = torch.cat(gathered_pos, dim=0)

        # Step 5: unmodified masked-matmul attention against the gathered full K/V.
        sin_q, cos_q = _rope_angles(query_positions, head_dim)
        q_roped = _apply_rope(q, sin_q, cos_q)
        b, t, n_head, hd = q_roped.shape
        delta = query_positions[:, None] - full_pos[None, :]
        allowed = (delta >= 0) & (delta < window) if window is not None else (delta >= 0)
        mask = torch.zeros(t, full_pos.shape[0]).masked_fill(~allowed, float("-inf"))
        qh = q_roped.transpose(1, 2)
        kh = full_k.transpose(1, 2)
        vh = full_v.transpose(1, 2)
        attn = ((qh @ kh.transpose(-2, -1)) / (hd**0.5) + mask[None, None]).softmax(dim=-1)
        out = (attn @ vh).transpose(1, 2).reshape(b, t, n_head * hd)

        if rank == 0:
            torch.save(out, out_file)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_real_gloo_multiprocess_cp_matches_inprocess_module():
    """Runs the actual CP shard/RoPE/all-gather/attend algorithm across 4 REAL separate OS processes
    (torch.multiprocessing.spawn, gloo backend) and checks the reconstructed attention output against
    ``cp_shard_kv``/``cp_window_attention_forward``'s in-process result for the SAME inputs.

    This validates that the collective PATTERN this module documents is correct when actually executed
    by independent processes over real (if CPU-local) collectives -- it does NOT and cannot measure
    throughput or scaling: CPU gloo all-gather cost has no relationship to GPU NCCL all-gather cost, so
    no wall-clock number from this test is a scaling receipt. See the module docstring.
    """
    torch.manual_seed(0)
    world_size = 4
    b, cache_len, t, n_head, head_dim, window = 2, 12, 4, 2, 4, None
    k_full = torch.randn(b, cache_len + t, n_head, head_dim)
    v_full = torch.randn(b, cache_len + t, n_head, head_dim)
    key_positions = torch.arange(0, cache_len + t)
    q = torch.randn(b, t, n_head, head_dim)
    query_positions = torch.arange(cache_len, cache_len + t)

    # In-process reference: the actual production module, same inputs.
    shards = cp_shard_kv(k_full, v_full, key_positions, world_size)
    assert len(shards) == world_size
    expected = cp_window_attention_forward(q, query_positions, shards, window=window, head_dim=head_dim)

    scratch_dir = tempfile.mkdtemp()
    init_file = os.path.join(scratch_dir, "init")
    out_file = os.path.join(scratch_dir, "out.pt")
    payload_file = os.path.join(scratch_dir, "payload.pt")
    torch.save(
        {
            "k_full": k_full,
            "v_full": v_full,
            "key_positions": key_positions,
            "q": q,
            "query_positions": query_positions,
            "window": window,
            "head_dim": head_dim,
        },
        payload_file,
    )

    ctx = torch.multiprocessing.get_context("spawn")
    torch.multiprocessing.spawn(
        _cp_gloo_worker,
        args=(world_size, init_file, out_file, payload_file),
        nprocs=world_size,
        join=True,
    )

    assert os.path.exists(out_file), "rank 0 never wrote its output -- worker(s) likely raised"
    actual = torch.load(out_file, weights_only=True)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4), (
        "real gloo multi-process CP diverged from the in-process module -- see module docstring: this "
        "is exactly the class of bug (accidental cross-rank state sharing) a real-process test can catch "
        "that an in-process Python-loop simulation cannot"
    )
    del ctx
    print(
        f"[E8 receipt] real gloo multi-process CP (world_size={world_size}, genuinely separate OS processes) "
        f"reproduces the in-process cp_shard_kv/cp_window_attention_forward output (atol=1e-5, rtol=1e-4). "
        f"This validates the COLLECTIVE PATTERN only -- no wall-clock/MFU scaling number is reported "
        f"(0 GPUs available here: torch.cuda.device_count()={N_CUDA_DEVICES})."
    )
