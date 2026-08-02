"""Focused streaming, buffer, normalization, and theorem contracts for sketch attention."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.sketch_state_attention import (  # noqa: E402
    FrequentDirectionsSpine,
    TensorSketchSpine,
    _tensor_sketch_far_scan,
    frequent_directions_error_bound,
    frequent_directions_update,
)

pytestmark = pytest.mark.experimental


def _run_chunks(model, x, chunk_size):
    state = model.init_state(x.shape[0])
    total_nll = 0.0
    with torch.no_grad():
        for start in range(0, x.shape[1], chunk_size):
            chunk = x[:, start : start + chunk_size]
            state, loss = model.step(state, (chunk, chunk))
            total_nll += float(loss) * chunk.shape[1]
    return state, total_nll / x.shape[1]


@pytest.mark.parametrize(
    "factory,state_fields",
    [
        (
            lambda: FrequentDirectionsSpine(
                11,
                d_model=8,
                n_layer=1,
                n_head=2,
                window=3,
                ell=6,
            ),
            ("B", "Z"),
        ),
        (
            lambda: TensorSketchSpine(
                11,
                d_model=8,
                n_layer=1,
                n_head=2,
                window=3,
                sketch_dim=64,
                degree=2,
                seed=4,
            ),
            ("C", "Z"),
        ),
    ],
)
def test_far_state_and_loss_are_chunk_boundary_invariant(factory, state_fields):
    torch.manual_seed(0)
    model = factory()
    tokens = torch.arange(9).remainder(model.vocab)[None, :]
    reference_state, reference_loss = _run_chunks(model, tokens, tokens.shape[1])

    for chunk_size in (1, 2, 3):
        state, loss = _run_chunks(model, tokens, chunk_size)
        assert loss == pytest.approx(reference_loss, abs=1e-6)
        for field in state_fields:
            for actual, expected in zip(getattr(state, field), getattr(reference_state, field)):
                assert torch.allclose(actual, expected, atol=1e-6)
        assert state.receipt["chunk_boundary_invariant_update"] is True


def test_tensor_sketch_hashes_and_signs_are_registered_buffers():
    model = TensorSketchSpine(
        7,
        d_model=8,
        n_layer=2,
        n_head=2,
        window=2,
        sketch_dim=16,
        degree=2,
    )
    buffers = dict(model.named_buffers())
    expected_names = {
        f"_tensor_{kind}_{layer}_{degree}" for kind in ("hash", "sign") for layer in range(2) for degree in range(2)
    }
    assert expected_names <= buffers.keys()
    assert expected_names <= model.state_dict().keys()

    model.double()
    assert all(hash_values.dtype == torch.long for layer in model._hashes for hash_values in layer)
    assert all(sign_values.dtype == torch.float64 for layer in model._signs for sign_values in layer)


def test_tensor_sketch_far_branch_is_normalized_by_kernel_mass():
    hashes = [torch.arange(2, dtype=torch.long)]
    signs = [torch.ones(2)]
    q = torch.tensor([[[[1.0, 2.0]], [[1.0, 2.0]]]])
    evicted_k = torch.tensor([[[[0.5, 1.0]], [[1.5, 0.5]]]])
    evicted_v = torch.tensor([[[[3.0]], [[3.0]]]])
    C = torch.zeros(1, 1, 2, 1)
    Z = torch.zeros(1, 1, 2)
    Z_ts = torch.zeros(1, 1, 2)

    # Degree 1 (one hash/sign pair): the exact Z is already the right degree, so this path is
    # unchanged by the MXR-080-1853 repair -- which is the point of keeping it exact there.
    _, _, _, output, minimum = _tensor_sketch_far_scan(
        q,
        C,
        Z,
        Z_ts,
        evicted_k,
        evicted_v,
        hashes,
        signs,
        cache_len=1,
        window=1,
        sketch_dim=2,
        head_dim=1,
        far_count_before=0,
    )

    assert torch.allclose(output, torch.full_like(output, 3.0))
    assert minimum is not None and minimum > 0


def test_tensor_sketch_nonpositive_normalizer_fails_closed():
    hashes = [torch.arange(2, dtype=torch.long)]
    signs = [torch.ones(2)]
    q = torch.ones(1, 1, 1, 2)
    C = torch.zeros(1, 1, 2, 1)
    Z = -torch.ones(1, 1, 2)
    with pytest.raises(ValueError, match="strictly positive"):
        _tensor_sketch_far_scan(
            q,
            C,
            Z,
            torch.zeros(1, 1, 2),
            None,
            None,
            hashes,
            signs,
            cache_len=0,
            window=1,
            sketch_dim=2,
            head_dim=1,
            far_count_before=1,
        )


@pytest.mark.parametrize(
    "ell,k",
    [
        (0, 0),
        (3, -1),
        (3, 3),
        (5, 0),
    ],
)
def test_fd_theorem_rejects_invalid_ell_and_k_domains(ell, k):
    A = torch.randn(8, 4)
    B = torch.zeros(max(ell, 1), 4)
    with pytest.raises(ValueError):
        frequent_directions_error_bound(A, B, ell, k)


def test_fd_update_rejects_shape_dtype_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="ell"):
        frequent_directions_update(torch.zeros(5, 4), torch.zeros(1, 4), ell=5)
    with pytest.raises(ValueError, match="feature dimension"):
        frequent_directions_update(torch.zeros(3, 4), torch.zeros(1, 5), ell=3)
    rows = torch.zeros(1, 4)
    rows[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        frequent_directions_update(torch.zeros(3, 4), rows, ell=3)
