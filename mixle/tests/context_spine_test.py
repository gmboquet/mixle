"""E1 acceptance receipts for the chunked-recurrent training spine (see notes/designs/E1.md).

Three receipts, each asserting a stated threshold from the roadmap card:
1. windowed multi-chunk training matches the full-attention-equivalent single-chunk configuration
   within 2% when dependencies are within the window (same seeds).
2. wall-clock is linear in total streamed length (2x length => <=2.3x time).
3. carried state is bitwise-deterministic given a seed.
"""

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.context_spine import SlidingWindowSpine, train_tbptt  # noqa: E402

# torch / experimental / slow markers come from mixle/tests/conftest.py's FILE_MARKERS table.


def _lag_copy_sequence(rng: np.random.RandomState, *, length: int, vocab: int, lag: int) -> tuple:
    """``y[i] = x[i - lag]`` for ``i >= lag``, ``y[i] = x[i]`` otherwise -- a controlled-dependency-distance task."""
    x = rng.randint(0, vocab, size=(1, length))
    y = x.copy()
    y[:, lag:] = x[:, :-lag]
    return torch.as_tensor(x, dtype=torch.long), torch.as_tensor(y, dtype=torch.long)


def _chunks(x: torch.Tensor, y: torch.Tensor, chunk_size: int) -> list:
    return [(x[:, i : i + chunk_size], y[:, i : i + chunk_size]) for i in range(0, x.shape[1], chunk_size)]


def _build_model(seed: int, *, vocab: int, d_model: int, n_layer: int, n_head: int, window) -> SlidingWindowSpine:
    torch.manual_seed(seed)
    return SlidingWindowSpine(vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window)


def test_windowed_matches_full_attention_within_2_percent():
    vocab, d_model, n_layer, n_head = 12, 16, 1, 2
    length, chunk_size, lag = 24, 8, 5  # lag < chunk_size (within-chunk) AND lag can straddle a chunk boundary
    n_steps = 25
    seed = 0

    full_model = _build_model(seed, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=length)
    win_model = _build_model(
        seed, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=chunk_size + lag
    )
    full_opt = torch.optim.Adam(full_model.parameters(), lr=1e-2)
    win_opt = torch.optim.Adam(win_model.parameters(), lr=1e-2)

    data_rng = np.random.RandomState(123)
    full_losses, win_losses = [], []
    for _ in range(n_steps):
        x, y = _lag_copy_sequence(data_rng, length=length, vocab=vocab, lag=lag)

        full_state = full_model.init_state(1)
        full_receipt = train_tbptt(full_model, full_state, [(x, y)], full_opt, detach_horizon=1)
        full_losses.append(full_receipt["losses"][0])

        win_state = win_model.init_state(1)
        chunks = _chunks(x, y, chunk_size)
        win_receipt = train_tbptt(win_model, win_state, chunks, win_opt, detach_horizon=len(chunks))
        win_losses.append(float(np.mean(win_receipt["losses"])))

    full_mean = float(np.mean(full_losses))
    win_mean = float(np.mean(win_losses))
    rel_diff = abs(full_mean - win_mean) / full_mean

    print(
        f"[E1 receipt] full-attention mean loss={full_mean:.6f}  windowed mean loss={win_mean:.6f}  "
        f"relative diff={rel_diff:.6f}"
    )
    assert rel_diff <= 0.02, f"windowed training diverged from full-attention baseline: rel_diff={rel_diff:.4f}"


def test_wallclock_linear_in_total_length():
    # Sized (chunk_size, d_model, repeat count) so each timed run is tens of milliseconds -- large enough that
    # perf_counter/scheduler noise (which dominated at the original ~3ms scale and made this test flaky) is a
    # small fraction of the signal.
    vocab, d_model, n_layer, n_head, window, chunk_size = 12, 96, 2, 4, 32, 48
    n_chunks_short, n_chunks_long = 40, 80  # 2x the chunk count => 2x the total streamed length

    model = _build_model(0, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window)
    rng = np.random.RandomState(7)

    def _time_n_chunks(n_chunks: int) -> float:
        state = model.init_state(1)
        chunks = []
        for _ in range(n_chunks):
            x, y = _lag_copy_sequence(rng, length=chunk_size, vocab=vocab, lag=3)
            chunks.append((x, y))
        start = time.perf_counter()
        with torch.no_grad():
            for chunk in chunks:
                state, _ = model.step(state, chunk)
        return time.perf_counter() - start

    for _ in range(3):  # warm-up: exclude first-call overhead (lazy CUDA/MKL init, allocator warmup) from timing
        _time_n_chunks(n_chunks_short)

    # Median of many trials (not min-of-few): min-of-few is biased low by the fastest of several noisy runs and
    # occasionally lets a single fast `short_time` push the ratio above threshold; the median is far more stable
    # under this repo's parallel test runner (pytest-xdist), where sibling workers contend for CPU.
    short_time = float(np.median([_time_n_chunks(n_chunks_short) for _ in range(9)]))
    long_time = float(np.median([_time_n_chunks(n_chunks_long) for _ in range(9)]))
    ratio = long_time / short_time

    print(
        f"[E1 receipt] {n_chunks_short} chunks: {short_time:.4f}s  {n_chunks_long} chunks: {long_time:.4f}s  "
        f"ratio={ratio:.3f} (threshold <= 2.3)"
    )
    assert ratio <= 2.3, f"wall-clock did not scale linearly with length: ratio={ratio:.3f}"


def test_carried_state_bitwise_deterministic():
    vocab, d_model, n_layer, n_head, window, chunk_size = 12, 16, 1, 2, 12, 6
    seed = 42

    def _run():
        model = _build_model(seed, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        rng = np.random.RandomState(99)
        x, y = _lag_copy_sequence(rng, length=chunk_size * 3, vocab=vocab, lag=4)
        chunks = _chunks(x, y, chunk_size)
        state = model.init_state(1)
        receipt = train_tbptt(model, state, chunks, opt, detach_horizon=1)
        return receipt

    receipt_a = _run()
    receipt_b = _run()

    assert receipt_a["losses"] == receipt_b["losses"]
    for k_a, k_b in zip(receipt_a["state"].cache_k, receipt_b["state"].cache_k):
        assert torch.equal(k_a, k_b)
    for v_a, v_b in zip(receipt_a["state"].cache_v, receipt_b["state"].cache_v):
        assert torch.equal(v_a, v_b)
    print(f"[E1 receipt] determinism: {len(receipt_a['losses'])} chunk losses bitwise-identical across seeded reruns")
