"""Execution, memory, and covariance contracts for transformer moment propagation."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.moment_propagation import (  # noqa: E402
    GaussianLaw,
    _gelu_scalar_moments,
    _project_correlation_fixed_diagonal,
    gelu_law,
    iter_moments,
    propagate_moments,
)
from mixle.models.transformer import build_causal_lm  # noqa: E402


def _law(dim: int) -> GaussianLaw:
    return GaussianLaw(mu=np.linspace(-0.2, 0.2, dim), covar=np.eye(dim) * 0.4 + 0.02)


def _model() -> object:
    torch.manual_seed(0)
    return build_causal_lm(vocab=6, d_model=4, n_layer=1, n_head=1, block=8).double()


@pytest.mark.parametrize(
    "options",
    [
        {"n_mc": 1},
        {"n_mc": 2.5},
        {"seq_len": 0},
        {"seed": -1},
        {"probe_batch_size": 0},
    ],
)
def test_moment_propagation_rejects_invalid_probe_controls(options: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        propagate_moments(_model(), _law(4), **options)


def test_iterator_is_lazy_batched_and_uses_target_dtype_eval_mode() -> None:
    model = _model()
    model.train()
    model.blocks[0].mlp.eval()
    original_modes = {module: module.training for module in model.modules()}
    observations: list[tuple[bool, torch.dtype, torch.device, int]] = []

    def record(module: object, args: tuple[torch.Tensor, ...]) -> None:
        value = args[0]
        observations.append((module.training, value.dtype, value.device, value.shape[0]))

    handles = [
        model.blocks[0].register_forward_pre_hook(record),
        model.ln.register_forward_pre_hook(record),
        model.head.register_forward_pre_hook(record),
    ]
    try:
        iterator = iter_moments(
            model,
            _law(4),
            n_mc=5,
            seq_len=3,
            seed=4,
            probe_batch_size=2,
        )
        assert observations == []
        first = next(iterator)
        remaining = list(iterator)
    finally:
        for handle in handles:
            handle.remove()

    assert first.name == "block[0]"
    assert [receipt.name for receipt in remaining] == ["ln_f", "head"]
    assert observations
    assert all(training is False for training, _, _, _ in observations)
    assert all(dtype == torch.float64 for _, dtype, _, _ in observations)
    assert all(device == next(model.parameters()).device for _, _, device, _ in observations)
    assert all(batch_size <= 2 for _, _, _, batch_size in observations)
    assert all(module.training == original_modes[module] for module in model.modules())


def test_materializer_explicitly_retains_every_iterator_receipt() -> None:
    model = _model()
    streamed = list(iter_moments(model, _law(4), n_mc=4, seq_len=3, seed=7, probe_batch_size=2))
    materialized = propagate_moments(model, _law(4), n_mc=4, seq_len=3, seed=7, probe_batch_size=2)

    assert len(materialized) == len(model.blocks) + 2
    assert [receipt.name for receipt in materialized] == [receipt.name for receipt in streamed]
    for left, right in zip(materialized, streamed, strict=True):
        np.testing.assert_allclose(left.law.mu, right.law.mu)
        np.testing.assert_allclose(left.law.covar, right.law.covar)
        assert left.closure_error == pytest.approx(right.closure_error)


def test_fixed_marginal_projection_repairs_indefinite_correlation_with_receipt() -> None:
    candidate = np.asarray(
        [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ]
    )
    corrected, receipt = _project_correlation_fixed_diagonal(candidate, np.ones(3))

    assert receipt.min_eigenvalue_before < 0.0
    assert receipt.min_eigenvalue_after > 0.0
    assert receipt.correction_frobenius > 0.0
    assert receipt.marginal_variance_max_error == 0.0
    np.testing.assert_array_equal(np.diag(corrected), np.ones(3))
    assert np.linalg.eigvalsh(corrected)[0] > 0.0


def test_gelu_preserves_exact_scalar_variances_and_returns_correction_receipt() -> None:
    mean = np.asarray([-2.0, 0.3, 1.5])
    covariance = np.asarray(
        [
            [2.0, 0.8, -0.2],
            [0.8, 1.0, 0.6],
            [-0.2, 0.6, 1.5],
        ]
    )
    law = GaussianLaw(mu=mean, covar=covariance)
    expected_mean, expected_variance, _ = _gelu_scalar_moments(mean, np.diag(covariance))
    output, _, receipt = gelu_law(law, return_receipt=True)

    np.testing.assert_allclose(output.mu, expected_mean, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(np.diag(output.covar), expected_variance, rtol=0.0, atol=1.0e-12)
    assert np.linalg.eigvalsh(output.covar)[0] > 0.0
    assert receipt.marginal_variance_max_error == 0.0
    assert receipt.min_eigenvalue_after > 0.0


def test_block_receipt_exposes_gelu_covariance_correction() -> None:
    first = next(iter_moments(_model(), _law(4), n_mc=4, seq_len=3, probe_batch_size=2))
    correction = first.diagnostics["gelu_covariance"]
    assert correction.correction_frobenius >= 0.0
    assert correction.marginal_variance_max_error == 0.0
