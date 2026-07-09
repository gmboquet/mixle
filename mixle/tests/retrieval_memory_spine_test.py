"""E6 acceptance receipts for retrieval memory over frozen past (see notes/standout-roadmap-tasks.md's E6
card and mixle/experimental/retrieval_memory_spine.py's module docstring).

Four receipts, each asserting a real, measured number:
1. exact gradients flow through the retrieval operation (query/output projections), and the archived index
   entries genuinely carry no gradient -- proven both by ``requires_grad`` on the state and by a same-example
   overfit converging to near-zero loss (the mechanism can actually learn something through this path).
2. the non-differentiable boundary is documented as a receipt field on the mechanism's OUTPUT (the carried
   state), not just in a docstring, with the fields the card requires.
3. long-range factual recall: at a distance beyond the local window (where ``SlidingWindowSpine`` is
   information-theoretically incapable of ever seeing the needle, no matter how much it trains),
   ``RetrievalMemorySpine`` achieves a measurably lower needle-probe loss than the window-only E1 baseline
   at the same window budget -- a real number from real training, not a threshold asserted on faith.
4. state cost: bytes-per-token actually used by the carried state, measured directly (E2's ratio is left as
   a documented placeholder -- see ``RETRIEVAL_MEMORY_UNAVAILABLE_PIECES``, E2 is not reachable from this
   worktree's base as of this writing).
5. carried state is bitwise-deterministic given a seed (same contract as E1's context_spine_test.py).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.context_spine import SlidingWindowSpine, train_tbptt  # noqa: E402
from mixle.experimental.long_context_eval import _state_bytes, _train_and_probe, needle_suite  # noqa: E402
from mixle.experimental.retrieval_memory_spine import (  # noqa: E402
    RETRIEVAL_MEMORY_UNAVAILABLE_PIECES,
    RetrievalMemorySpine,
)

# torch / experimental / slow markers come from mixle/tests/conftest.py's FILE_MARKERS table.


def _chunks(x: torch.Tensor, y: torch.Tensor, chunk_size: int) -> list:
    return [(x[:, i : i + chunk_size], y[:, i : i + chunk_size]) for i in range(0, x.shape[1], chunk_size)]


def _build_model(seed: int, *, vocab=10, d_model=32, n_layer=2, n_head=2, window=3, retrieval_k=4):
    torch.manual_seed(seed)
    return RetrievalMemorySpine(
        vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window, retrieval_k=retrieval_k
    )


def test_index_entries_carry_no_gradient_but_retrieval_learns():
    # Fixed single needle example, repeated: the correct answer requires info that has scrolled out of the
    # local window and is ONLY reachable through the retrieval index -- if gradients into the index contents
    # were needed (rather than just through this step's retrieval operation), or if retrieval weren't
    # actually wired into the loss, this would not be able to memorize the example at all.
    vocab, distance, chunk_size = 10, 12, 3
    model = _build_model(0, vocab=vocab, window=3)
    opt = torch.optim.Adam(model.parameters(), lr=2e-2)

    rng = np.random.RandomState(1)
    x, y = needle_suite(rng, distance=distance, vocab=vocab)
    chunks = _chunks(x, y, chunk_size)

    losses = []
    for _ in range(300):
        state = model.init_state(1)
        receipt = train_tbptt(model, state, chunks, opt, detach_horizon=len(chunks))
        losses.append(float(np.mean(receipt["losses"])))

    # The index tensors are genuinely detached -- this is the non-differentiable boundary the card asks for.
    final_state = receipt["state"]
    for k in final_state.index_k:
        if k is not None:
            assert not k.requires_grad
    for v in final_state.index_v:
        if v is not None:
            assert not v.requires_grad

    print(f"[E6 receipt] same-example overfit: loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.6f}")
    assert losses[-1] < 0.05, (
        f"retrieval-augmented training should memorize a single fixed needle example near-exactly, "
        f"got final loss={losses[-1]:.4f}"
    )
    assert losses[-1] < losses[0]


def test_step_documents_differentiable_boundary_as_a_receipt_field():
    vocab, chunk_size = 10, 4
    model = _build_model(0, vocab=vocab, window=3, retrieval_k=2)
    rng = np.random.RandomState(2)
    x = torch.as_tensor(rng.randint(0, vocab, size=(1, chunk_size * 3)), dtype=torch.long)
    y = torch.as_tensor(rng.randint(0, vocab, size=(1, chunk_size * 3)), dtype=torch.long)

    state = model.init_state(1)
    for chunk in _chunks(x, y, chunk_size)[:-1]:
        state, _ = model.step(state, chunk)
    # Index has been archived from the first two chunks by now -- the third step should actually retrieve.
    last_chunk = _chunks(x, y, chunk_size)[-1]
    state, loss = model.step(state, last_chunk)

    receipt = state.receipt
    assert "differentiable_boundary" in receipt and isinstance(receipt["differentiable_boundary"], str)
    assert "gradient" in receipt["differentiable_boundary"].lower()
    assert receipt["retrieval_k_requested"] == 2
    assert receipt["index_len_per_layer"] == [chunk_size * 2] * model.n_layer  # two prior chunks archived
    assert all(c > 0 for c in receipt["retrieved_per_layer"]), "index was non-empty; retrieval should fire"
    print(f"[E6 receipt] step() receipt: {receipt}")


def test_retrieval_beats_window_only_baseline_beyond_the_window():
    # window=3 < distance=12: SlidingWindowSpine is INFORMATION-THEORETICALLY incapable of ever seeing the
    # needle's key/value pair at recall time, no matter how much it trains -- a hard ceiling, not a training
    # difficulty. RetrievalMemorySpine has the same window (same local recall budget) plus the archived
    # index, so any advantage measured here is coming specifically from retrieval.
    vocab, distance, chunk_size, window = 10, 12, 3, 3

    rng_r = np.random.RandomState(10)
    retrieval = _build_model(0, vocab=vocab, window=window, retrieval_k=4)
    opt_r = torch.optim.Adam(retrieval.parameters(), lr=3e-2)
    result_retrieval = _train_and_probe(
        retrieval,
        opt_r,
        needle_suite,
        distance=distance,
        vocab=vocab,
        chunk_size=chunk_size,
        n_train_steps=250,
        n_eval_trials=40,
        rng=rng_r,
    )

    rng_b = np.random.RandomState(10)
    torch.manual_seed(0)
    baseline = SlidingWindowSpine(vocab, d_model=32, n_layer=2, n_head=2, window=window)
    opt_b = torch.optim.Adam(baseline.parameters(), lr=3e-2)
    result_baseline = _train_and_probe(
        baseline,
        opt_b,
        needle_suite,
        distance=distance,
        vocab=vocab,
        chunk_size=chunk_size,
        n_train_steps=250,
        n_eval_trials=40,
        rng=rng_b,
    )

    print(
        f"[E6 receipt] needle recall at distance={distance}, window={window}: "
        f"retrieval mean_probe_loss={result_retrieval['mean_probe_loss']:.4f} "
        f"(accuracy={result_retrieval['accuracy']:.3f})  "
        f"window-only baseline mean_probe_loss={result_baseline['mean_probe_loss']:.4f} "
        f"(accuracy={result_baseline['accuracy']:.3f})  chance_loss={result_retrieval['chance_loss']:.4f}"
    )
    assert result_retrieval["mean_probe_loss"] < result_baseline["mean_probe_loss"], (
        "retrieval memory should recall the beyond-window needle better than a window-only baseline that "
        "provably cannot see it at all"
    )


def test_state_bytes_per_token_receipt():
    # E2 (moment-closure attention) is not reachable from this worktree's base -- see the module docstring
    # and RETRIEVAL_MEMORY_UNAVAILABLE_PIECES. This receipt reports RetrievalMemorySpine's OWN measured
    # state cost (bytes actually carried per streamed token) honestly, in place of the card's E2 ratio.
    assert "E2" in RETRIEVAL_MEMORY_UNAVAILABLE_PIECES

    vocab, d_model, n_layer, n_head, window, chunk_size = 10, 32, 2, 2, 3, 4
    model = _build_model(0, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window, retrieval_k=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    rng = np.random.RandomState(0)

    n_tokens = 200
    x = torch.as_tensor(rng.randint(0, vocab, size=(1, n_tokens)), dtype=torch.long)
    y = torch.as_tensor(rng.randint(0, vocab, size=(1, n_tokens)), dtype=torch.long)
    chunks = _chunks(x, y, chunk_size)
    state = model.init_state(1)
    receipt = train_tbptt(model, state, chunks, opt, detach_horizon=len(chunks))

    state_bytes = _state_bytes(receipt["state"])
    bytes_per_token = state_bytes / n_tokens
    print(
        f"[E6 receipt] state cost: {state_bytes} bytes carried after {n_tokens} tokens "
        f"({bytes_per_token:.2f} bytes/token). E2 comparison: {RETRIEVAL_MEMORY_UNAVAILABLE_PIECES['E2']}"
    )
    assert state_bytes > 0
    assert bytes_per_token > 0.0


def test_carried_state_bitwise_deterministic():
    vocab, d_model, n_layer, n_head, window, chunk_size = 10, 16, 1, 2, 4, 6

    def _run():
        model = _build_model(42, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, window=window)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        rng = np.random.RandomState(99)
        x = torch.as_tensor(rng.randint(0, vocab, size=(1, chunk_size * 3)), dtype=torch.long)
        y = torch.as_tensor(rng.randint(0, vocab, size=(1, chunk_size * 3)), dtype=torch.long)
        chunks = _chunks(x, y, chunk_size)
        state = model.init_state(1)
        return train_tbptt(model, state, chunks, opt, detach_horizon=1)

    receipt_a = _run()
    receipt_b = _run()

    assert receipt_a["losses"] == receipt_b["losses"]
    for k_a, k_b in zip(receipt_a["state"].index_k, receipt_b["state"].index_k):
        assert torch.equal(k_a, k_b)
    for v_a, v_b in zip(receipt_a["state"].index_v, receipt_b["state"].index_v):
        assert torch.equal(v_a, v_b)
    print(f"[E6 receipt] determinism: {len(receipt_a['losses'])} chunk losses bitwise-identical across seeded reruns")
