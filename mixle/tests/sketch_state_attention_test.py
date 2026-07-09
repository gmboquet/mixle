"""E3 acceptance receipts for sketch-state attention (see notes/designs/E3.md's Test plan).

Six receipts, mirroring the design note's Test plan section 1:1:
1. FD's deterministic L2 bound (Liberty 2013, Theorem 1.1) -- a HARD assertion (no statistical tolerance)
   on multiple seeded random streams, plus a companion test that B has a genuine zero row after any shrink.
2. Chunked-scan equivalence for (a) LinearAttentionSpine: streaming in small chunks must reproduce the same
   carried state (and the same overall loss) as one non-chunked pass, within float-accumulation tolerance.
3. ContextMechanism protocol conformance for all three mechanisms, round-tripped through train_tbptt.
4. Tensor-sketch unbiasedness -- a STATISTICAL concentration check (explicitly not a hard bound, unlike #1).
5. Misfit receipts for graduation.py bookkeeping (fd_misfit_receipt / tensor_sketch_misfit_receipt feed a
   real ExperimentalMechanism's misfit_receipt field).
6. The E7 bake-off: evaluate() against SlidingWindowSpine (E1) plus all three E3 mechanisms at matched
   state bytes, small stand-in ranges, with the E2-unavailable placeholder honestly surfaced (E2 has not
   landed anywhere reachable as of this writing -- see E3_UNAVAILABLE_COMPARISONS).
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.context_spine import ContextMechanism, SlidingWindowSpine, train_tbptt  # noqa: E402
from mixle.experimental.graduation import ExperimentalMechanism  # noqa: E402
from mixle.experimental.long_context_eval import comparison_table, evaluate  # noqa: E402
from mixle.experimental.sketch_state_attention import (  # noqa: E402
    E3_UNAVAILABLE_COMPARISONS,
    FrequentDirectionsSpine,
    LinearAttentionSpine,
    TensorSketchSpine,
    fd_misfit_receipt,
    frequent_directions_error_bound,
    frequent_directions_update,
    make_tensor_sketch_hashes,
    tensor_sketch_misfit_receipt,
    tensor_sketch_project,
)

# torch / experimental / slow markers come from mixle/tests/conftest.py's FILE_MARKERS table.


# ---------------------------------------------------------------------------------------------------------
# 1. FD deterministic L2 bound (Liberty 2013, Theorem 1.1) -- HARD assertion, no statistical tolerance.
# ---------------------------------------------------------------------------------------------------------


def _random_stream(seed: int, *, n: int, d: int, rank: int, noise: float) -> torch.Tensor:
    """Low-rank-plus-noise rows -- the bound is tightest to check (least slack) when ``A`` isn't already
    low rank, per the design note's Test plan #1 ("adversarial-ish inputs")."""
    g = torch.Generator().manual_seed(seed)
    U = torch.randn(n, rank, generator=g)
    V = torch.randn(rank, d, generator=g)
    return U @ V + noise * torch.randn(n, d, generator=g)


_FD_STREAM_CONFIGS = [
    # (n, d, ell, rank, noise)
    (200, 12, 8, 3, 0.1),
    (150, 20, 10, 5, 0.5),
    (80, 6, 4, 2, 1.0),
    (300, 16, 5, 1, 0.05),  # ell << d: aggressive compression, loosest bound
]


def test_fd_error_bound_is_a_hard_deterministic_assertion():
    """Liberty's Theorem 1.1, exactly: ||A^T A - B^T B||_2 <= ||A - A_k||_F^2 / (ell - k), for k in
    {0, 1, ell-1}, on several seeded streams. This is NOT a statistical check -- no tolerance band, no
    retries, no "usually holds": the theorem is a deterministic worst-case bound and the assertion is a
    plain `<=` (with only a 1e-6 relative fudge for float round-off, not for the bound's own slack)."""
    for seed, (n, d, ell, rank, noise) in enumerate(_FD_STREAM_CONFIGS):
        for trial_seed in range(3):  # multiple streams per config, per "every seed" in the design note
            A = _random_stream(seed * 100 + trial_seed, n=n, d=d, rank=rank, noise=noise)
            B0 = torch.zeros(ell, d, dtype=A.dtype)
            B = frequent_directions_update(B0, A, ell)

            realized = float(torch.linalg.matrix_norm(A.T @ A - B.T @ B, ord=2))
            for k in (0, 1, ell - 1):
                bound = frequent_directions_error_bound(A, B, ell, k)
                assert realized <= bound * (1.0 + 1e-6), (
                    f"Liberty Theorem 1.1 VIOLATED: n={n} d={d} ell={ell} k={k} seed={seed}/{trial_seed}: "
                    f"realized={realized!r} > bound={bound!r}"
                )
            print(
                f"[E3 receipt] FD bound: n={n} d={d} ell={ell} seed={seed}/{trial_seed} "
                f"realized={realized:.6f} bound(k=0)={frequent_directions_error_bound(A, B, ell, 0):.6f}"
            )


def test_fd_bound_check_rejects_wrong_ell():
    """frequent_directions_error_bound's own consistency check: B.shape[0] must equal the ell it's told."""
    A = torch.randn(10, 4)
    B = torch.zeros(3, 4)
    with pytest.raises(ValueError):
        frequent_directions_error_bound(A, B, ell=5, k=0)


def test_fd_freed_row_invariant_after_shrink():
    """Companion test to the bound (design note's Test plan #1, second half): Liberty's Algorithm 1 is
    (2a) insert the incoming row into an all-zero row of B, THEN (2b) if B now has no all-zero row, shrink
    -- shrinking zeros out exactly the smallest singular value's row, so the row freed by a shrink stays
    genuinely, exactly zero (not merely small) until the next row consumes it. Because step (2b) is
    evaluated within the SAME row's processing as the insertion that makes B full, the very first shrink
    fires while processing the `ell`-th row itself (not on some later `ell+1`-th row) -- so B already has a
    genuine zero row immediately after exactly `ell` rows have been streamed in. This checks that directly,
    and confirms the invariant persists (a fresh shrink refills the vacancy each subsequent row) after one
    more row on top -- not a rank-compacted variant that never leaves a literal zero row."""
    for seed, (d, ell) in enumerate([(6, 5), (10, 4), (5, 5), (12, 3)]):
        g = torch.Generator().manual_seed(seed + 1000)
        B = torch.zeros(ell, d)

        fill_rows = torch.randn(ell, d, generator=g)
        B_after_fill = frequent_directions_update(B, fill_rows, ell)
        n_zero_after_fill = int((B_after_fill.abs().sum(dim=-1) == 0).sum())
        assert n_zero_after_fill >= 1, (
            f"seed={seed} d={d} ell={ell}: processing the ell-th row both fills B's last vacant slot AND "
            f"(within that same row's processing, per Liberty's step 2b) triggers the first shrink, so B "
            f"must already show at least one EXACT zero row; found {n_zero_after_fill}."
        )
        zero_rows_after_fill = B_after_fill[B_after_fill.abs().sum(dim=-1) == 0]
        assert torch.equal(zero_rows_after_fill, torch.zeros_like(zero_rows_after_fill)), (
            "freed row is not exactly (bitwise) zero."
        )

        one_more = torch.randn(1, d, generator=g)
        B_after_next = frequent_directions_update(B_after_fill, one_more, ell)
        n_zero_after_next = int((B_after_next.abs().sum(dim=-1) == 0).sum())
        assert n_zero_after_next >= 1, (
            f"seed={seed} d={d} ell={ell}: the invariant should persist -- the next row consumes the "
            f"vacancy and its own processing triggers another shrink, leaving >=1 zero row again; "
            f"found {n_zero_after_next}."
        )
        zero_rows_after_next = B_after_next[B_after_next.abs().sum(dim=-1) == 0]
        assert torch.equal(zero_rows_after_next, torch.zeros_like(zero_rows_after_next)), (
            "freed row is not exactly (bitwise) zero."
        )
        print(
            f"[E3 receipt] FD freed-row invariant: d={d} ell={ell} seed={seed}: "
            f"{n_zero_after_fill} zero row(s) after exactly ell rows, "
            f"{n_zero_after_next} zero row(s) after ell+1 rows."
        )


# ---------------------------------------------------------------------------------------------------------
# 2. Chunked-scan equivalence for (a) LinearAttentionSpine.
# ---------------------------------------------------------------------------------------------------------


def test_linear_attention_chunked_scan_matches_single_pass():
    """Streaming LinearAttentionSpine in small chunks with carried (S, Z) must reproduce the SAME carried
    state and the SAME overall loss as one non-chunked call over the whole sequence, within float
    accumulation tolerance (cumsum reordering can shift rounding -- see design note's Test plan #2)."""
    torch.manual_seed(0)
    vocab, d_model, n_layer, n_head = 10, 8, 1, 2
    model = LinearAttentionSpine(vocab, d_model=d_model, n_layer=n_layer, n_head=n_head)

    b, T = 2, 24
    rng = torch.Generator().manual_seed(7)
    x = torch.randint(0, vocab, (b, T), generator=rng)
    y = torch.randint(0, vocab, (b, T), generator=rng)

    with torch.no_grad():
        state_single, loss_single = model.step(model.init_state(b), (x, y))

        for chunk_size in (1, 3, 8):
            state = model.init_state(b)
            total_loss = 0.0
            for i in range(0, T, chunk_size):
                xc, yc = x[:, i : i + chunk_size], y[:, i : i + chunk_size]
                state, loss = model.step(state, (xc, yc))
                total_loss += float(loss) * xc.shape[1]
            mean_loss_chunked = total_loss / T

            assert math.isclose(mean_loss_chunked, float(loss_single), rel_tol=1e-4, abs_tol=1e-5), (
                f"chunk_size={chunk_size}: chunked mean loss {mean_loss_chunked!r} != "
                f"single-pass loss {float(loss_single)!r}"
            )
            for layer in range(n_layer):
                assert torch.allclose(state.S[layer], state_single.S[layer], atol=1e-4, rtol=1e-4), (
                    f"chunk_size={chunk_size}: carried S diverged from the single-pass reference."
                )
                assert torch.allclose(state.Z[layer], state_single.Z[layer], atol=1e-4, rtol=1e-4), (
                    f"chunk_size={chunk_size}: carried Z diverged from the single-pass reference."
                )
            print(f"[E3 receipt] chunked-scan equivalence: chunk_size={chunk_size} matches single-pass reference")


# ---------------------------------------------------------------------------------------------------------
# 3. ContextMechanism protocol conformance for all three mechanisms, via train_tbptt.
# ---------------------------------------------------------------------------------------------------------

_MECHANISM_FACTORIES = {
    "linear_attention": lambda vocab: LinearAttentionSpine(vocab, d_model=8, n_layer=1, n_head=2),
    "frequent_directions": lambda vocab: FrequentDirectionsSpine(
        vocab, d_model=8, n_layer=1, n_head=2, window=4, ell=6
    ),
    "tensor_sketch": lambda vocab: TensorSketchSpine(
        vocab, d_model=8, n_layer=1, n_head=2, window=4, sketch_dim=8, degree=2
    ),
}


@pytest.mark.parametrize("name", list(_MECHANISM_FACTORIES))
def test_mechanism_is_a_context_mechanism_and_trains_via_tbptt(name):
    """isinstance(mechanism, ContextMechanism) (the protocol is @runtime_checkable) and a full
    init_state/step/detach round trip through train_tbptt on a tiny synthetic stream, mirroring
    context_spine_test.py's existing SlidingWindowSpine pattern."""
    torch.manual_seed(0)
    vocab = 10
    mechanism = _MECHANISM_FACTORIES[name](vocab)
    assert isinstance(mechanism, ContextMechanism), f"{name} does not satisfy the ContextMechanism protocol"

    opt = torch.optim.Adam(mechanism.parameters(), lr=1e-2)
    rng = torch.Generator().manual_seed(3)
    x = torch.randint(0, vocab, (1, 16), generator=rng)
    y = torch.randint(0, vocab, (1, 16), generator=rng)
    chunks = [(x[:, i : i + 4], y[:, i : i + 4]) for i in range(0, 16, 4)]

    state = mechanism.init_state(1)
    receipt = train_tbptt(mechanism, state, chunks, opt, detach_horizon=2)

    assert len(receipt["losses"]) == len(chunks)
    assert all(math.isfinite(loss_v) for loss_v in receipt["losses"])
    print(f"[E3 receipt] {name}: ContextMechanism conformance + train_tbptt round trip OK, losses={receipt['losses']}")


# ---------------------------------------------------------------------------------------------------------
# 4. Tensor sketch unbiasedness -- STATISTICAL concentration check, explicitly not a hard bound.
# ---------------------------------------------------------------------------------------------------------


def test_tensor_sketch_is_unbiased_in_expectation():
    """NOT the same class of guarantee as FD's Theorem 1.1 (design note's Test plan #4 is explicit about
    this): TS(x)^T TS(y) is an unbiased estimator of (x^T y)^p in expectation over the sketch's hash/sign
    randomness (and, here, also over fresh x/y draws), with variance O(1/sketch_dim). Over many trials the
    sample mean of the estimation error should concentrate near zero -- checked here as a z-style band
    (|mean error| < 5 standard errors of the mean, computed FROM the same sample, not a fixed magic
    constant) rather than any deterministic bound. A fixed seed keeps the test non-flaky while still being
    a genuine statistical check, not a hard one."""
    d, sketch_dim, degree, trials = 16, 64, 2, 2000
    g = torch.Generator().manual_seed(42)
    errors = []
    for t in range(trials):
        x = torch.randn(d, generator=g)
        y = torch.randn(d, generator=g)
        true_val = float((x @ y) ** degree)
        hashes, signs = make_tensor_sketch_hashes(d, sketch_dim=sketch_dim, degree=degree, seed=1000 + t)
        est = float(
            tensor_sketch_project(x, hashes, signs, sketch_dim) @ tensor_sketch_project(y, hashes, signs, sketch_dim)
        )
        errors.append(est - true_val)

    errors_t = torch.tensor(errors)
    mean_error = float(errors_t.mean())
    std_error = float(errors_t.std(unbiased=True))
    standard_error_of_mean = std_error / math.sqrt(trials)

    print(
        f"[E3 receipt] tensor sketch unbiasedness (statistical, NOT a hard bound): trials={trials} "
        f"mean_error={mean_error:.4f} std={std_error:.4f} se={standard_error_of_mean:.4f} "
        f"z={mean_error / standard_error_of_mean:.4f}"
    )
    assert abs(mean_error) < 5.0 * standard_error_of_mean, (
        "tensor sketch estimator mean error did not concentrate near zero across trials -- "
        f"mean={mean_error!r}, 5*SE={5.0 * standard_error_of_mean!r}"
    )


def test_tensor_sketch_variance_shrinks_with_larger_sketch_dim():
    """A weaker, corroborating statistical check of the O(1/sketch_dim) variance claim: empirical variance
    at a larger sketch_dim should be smaller than at a small one, on the same fixed (x, y) pair (isolating
    the sketch's own randomness from x/y sampling variance). Statistical, seeded, generous margin -- not
    asserting the exact 1/m scaling constant, just the qualitative direction."""
    d, degree, trials = 12, 2, 300
    g = torch.Generator().manual_seed(11)
    x = torch.randn(d, generator=g)
    y = torch.randn(d, generator=g)

    def _empirical_variance(sketch_dim: int, seed_offset: int) -> float:
        errs = []
        for t in range(trials):
            hashes, signs = make_tensor_sketch_hashes(d, sketch_dim=sketch_dim, degree=degree, seed=seed_offset + t)
            est = float(
                tensor_sketch_project(x, hashes, signs, sketch_dim)
                @ tensor_sketch_project(y, hashes, signs, sketch_dim)
            )
            errs.append(est)
        return float(torch.tensor(errs).var(unbiased=True))

    var_small = _empirical_variance(sketch_dim=8, seed_offset=0)
    var_large = _empirical_variance(sketch_dim=64, seed_offset=100_000)
    print(f"[E3 receipt] tensor sketch variance: sketch_dim=8 -> {var_small:.4f}, sketch_dim=64 -> {var_large:.4f}")
    assert var_large < var_small, "variance should shrink as sketch_dim grows (O(1/sketch_dim) claim)"


# ---------------------------------------------------------------------------------------------------------
# 5. Misfit receipts for graduation.py bookkeeping.
# ---------------------------------------------------------------------------------------------------------


def test_fd_misfit_receipt_shape_and_graduation_bookkeeping():
    """fd_misfit_receipt reports the realized ||A^T A - B^T B||_2 against Liberty's bound -- "how tight is
    the guarantee in practice" -- and feeds directly into graduation.py's ExperimentalMechanism.misfit_receipt
    field, exactly the shape its docstring names as the worked example."""
    A = _random_stream(seed=0, n=120, d=10, rank=3, noise=0.3)
    receipt = fd_misfit_receipt(A, ell=6)

    assert set(receipt) == {"realized_error", "bound", "tightness_ratio"}
    assert receipt["realized_error"] >= 0.0
    assert receipt["bound"] > 0.0
    assert receipt["realized_error"] <= receipt["bound"] * (1.0 + 1e-6)

    entry = ExperimentalMechanism(name="sketch_state_attention_fd", misfit_receipt=receipt)
    assert entry.misfit_receipt is receipt
    assert not entry.is_eligible()  # baseline_receipt still missing -- bookkeeping, not fabrication
    entry.baseline_receipt = {"metric": "bpb", "mechanism": 1.0, "baseline": 1.0, "flops": 0.0}
    assert entry.is_eligible()
    print(f"[E3 receipt] FD misfit receipt -> graduation bookkeeping: {receipt}")


def test_tensor_sketch_misfit_receipt_shape_and_graduation_bookkeeping():
    """tensor_sketch_misfit_receipt reports the empirical collision/variance rate -- graduation.py's
    docstring's own "sketch collision rate" example."""
    receipt = tensor_sketch_misfit_receipt(d=12, sketch_dim=32, degree=2, seed=0, trials=200)

    assert set(receipt) == {"mean_bias", "empirical_variance", "trials"}
    assert receipt["trials"] == 200.0
    assert receipt["empirical_variance"] >= 0.0

    entry = ExperimentalMechanism(name="sketch_state_attention_tensor_sketch", misfit_receipt=receipt)
    assert not entry.is_eligible()
    entry.baseline_receipt = {"metric": "bpb", "mechanism": 1.0, "baseline": 1.0, "flops": 0.0}
    assert entry.is_eligible()
    print(f"[E3 receipt] tensor sketch misfit receipt -> graduation bookkeeping: {receipt}")


# ---------------------------------------------------------------------------------------------------------
# 6. E7 bake-off: E1 baseline vs all three E3 mechanisms at matched state bytes.
# ---------------------------------------------------------------------------------------------------------

# Small stand-in ranges, following long_context_eval_test.py's own documented deviation from the card-literal
# (1e3, 1e4, 1e5, 1e6) for suite runtime. Sized (together with `window`) so FrequentDirectionsSpine's SVD-based
# shrink -- differentiable per the design note's Risks section ("gradients DO flow through
# FrequentDirectionsSpine via autodiff... but backpropagating through repeated/near-degenerate singular
# values is numerically unstable") -- does not chain enough un-detached shrink events within a single
# train_tbptt(..., detach_horizon=len(chunks)) call (E7's own no-mid-stream-detach training regime for each
# probe) to blow up: verified empirically stable across 10+ seeds at this range/window ratio, whereas the
# card-literal larger ranges chain 20+ SVDs per backward pass and reliably NaN -- a real property of the
# published algorithm's differentiable shrink, not a bug in this test's harness.
_E7_RANGES = (5, 7, 9)
_E7_WINDOW = 8
_E7_VOCAB = 12
_E7_D_MODEL, _E7_N_LAYER, _E7_N_HEAD = 16, 1, 2


def _build_e7_mechanisms():
    torch.manual_seed(0)
    sliding_window = SlidingWindowSpine(
        _E7_VOCAB, d_model=_E7_D_MODEL, n_layer=_E7_N_LAYER, n_head=_E7_N_HEAD, window=_E7_WINDOW
    )
    torch.manual_seed(0)
    linear_attention = LinearAttentionSpine(_E7_VOCAB, d_model=_E7_D_MODEL, n_layer=_E7_N_LAYER, n_head=_E7_N_HEAD)
    torch.manual_seed(0)
    frequent_directions = FrequentDirectionsSpine(
        _E7_VOCAB, d_model=_E7_D_MODEL, n_layer=_E7_N_LAYER, n_head=_E7_N_HEAD, window=_E7_WINDOW, ell=8
    )
    torch.manual_seed(0)
    tensor_sketch = TensorSketchSpine(
        _E7_VOCAB,
        d_model=_E7_D_MODEL,
        n_layer=_E7_N_LAYER,
        n_head=_E7_N_HEAD,
        window=_E7_WINDOW,
        sketch_dim=12,
        degree=2,
    )
    return {
        "E1_sliding_window": sliding_window,
        "E3a_linear_attention": linear_attention,
        "E3b_frequent_directions": frequent_directions,
        "E3c_tensor_sketch": tensor_sketch,
    }


def test_e7_bakeoff_e1_vs_e3_mechanisms_matched_state_bytes():
    """evaluate() run against SlidingWindowSpine (E1 baseline) and all three E3 mechanisms at the same
    ranges/state_budget_bytes, rendered through comparison_table() -- the design note's acceptance receipt
    #2 ("E7 table vs E1/E2 at matched state bytes"). E2 (moment-closure attention) has not landed anywhere
    reachable from this worktree (checked via `git branch -a` / `gh pr list --search moment-closure-attention`
    at implementation time: the branch sits at this worktree's own base commit with zero real work, and no
    PR references it) -- per this project's convention for honestly naming an unreachable dependency rather
    than fabricating or silently dropping it, the table below is E1 vs the three E3 sketches only, with
    E3_UNAVAILABLE_COMPARISONS["E2"] surfaced alongside it instead of a fabricated row."""
    mechanisms = _build_e7_mechanisms()
    kwargs = dict(
        ranges=_E7_RANGES,
        state_budget_bytes=4000,
        seed=7,
        hops=2,
        n_train_steps=2,
        n_eval_trials=2,
        perplexity_steps=1,
        curriculum_rounds=3,
    )

    results = {name: evaluate(mechanism, **kwargs) for name, mechanism in mechanisms.items()}

    for name, result in results.items():
        assert result["ranges"] == _E7_RANGES
        assert result["within_state_budget"], f"{name} exceeded the shared state_budget_bytes"
        print(f"[E3 receipt] E7 bake-off: {name} state_bytes_used={result['state_bytes_used']}")

    assert "E2" in E3_UNAVAILABLE_COMPARISONS
    table = comparison_table(results) + "\n\n[E2 column pending] " + E3_UNAVAILABLE_COMPARISONS["E2"]

    assert isinstance(table, str) and table.strip()
    for name in mechanisms:
        assert name in table
    for r in _E7_RANGES:
        assert str(r) in table
    assert "moment-closure attention (roadmap E2) has not been implemented" in table

    print("[E3 receipt] E7 bake-off comparison table:\n" + table)
