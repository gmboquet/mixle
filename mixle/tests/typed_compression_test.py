"""Contract-gated compression and persistent error-feedback tests."""

import json

import numpy as np
import pytest

from mixle.experimental.typed_runtime import (
    CompressionMethod,
    ErrorFeedbackCompressor,
    MergeLaw,
    ObjectiveKind,
    UpdateContract,
    UpdateKind,
    compile_update_graph,
)
from mixle.stats import GaussianDistribution, GaussianEstimator

pytestmark = [pytest.mark.experimental, pytest.mark.fast]


def _approximate_contract():
    return UpdateContract(
        objective_kind=ObjectiveKind.MLE,
        update_kind=UpdateKind.PRECONDITIONED,
        merge_law=MergeLaw.LOW_RANK,
        exact=False,
        declared_by="test",
    )


def test_low_rank_payload_reduces_bytes_and_receipts_realized_error():
    rng = np.random.default_rng(3)
    delta = rng.normal(size=(64, 64))
    compressor = ErrorFeedbackCompressor(default_rank=2, exact_threshold_bytes=0)
    payload = compressor.compress("weight", delta, _approximate_contract())

    assert payload.method is CompressionMethod.LOW_RANK
    assert payload.receipt.payload_bytes < payload.receipt.input_bytes
    assert payload.receipt.compression_ratio > 5.0
    reconstruction = payload.reconstruct()
    residual = compressor.residual("weight")
    np.testing.assert_allclose(delta, reconstruction + residual, rtol=1.0e-12, atol=1.0e-12)
    assert payload.receipt.realized_l2_error == pytest.approx(np.linalg.norm(residual))
    json.dumps(payload.as_dict(), allow_nan=False)


def test_exact_statistical_contract_stays_dense_even_when_rank_is_requested():
    contract = compile_update_graph(GaussianDistribution(0.0, 1.0), GaussianEstimator()).node("n0000").contract
    delta = np.arange(4_096, dtype=np.float64).reshape(64, 64)
    payload = ErrorFeedbackCompressor(exact_threshold_bytes=0).compress("stats", delta, contract, rank=1)
    assert payload.method is CompressionMethod.DENSE
    assert payload.receipt.exact
    np.testing.assert_array_equal(payload.reconstruct(), delta)


def test_error_feedback_reduces_cumulative_bias_against_memoryless_low_rank():
    rng = np.random.default_rng(7)
    delta = rng.normal(size=(32, 32))
    contract = _approximate_contract()
    rounds = 12

    with_feedback = ErrorFeedbackCompressor(default_rank=1, exact_threshold_bytes=0)
    feedback_sum = np.zeros_like(delta)
    for _ in range(rounds):
        feedback_sum += with_feedback.compress("weight", delta, contract).reconstruct()

    memoryless_sum = np.zeros_like(delta)
    for _ in range(rounds):
        memoryless = ErrorFeedbackCompressor(default_rank=1, exact_threshold_bytes=0)
        memoryless_sum += memoryless.compress("weight", delta, contract).reconstruct()

    target = rounds * delta
    feedback_error = np.linalg.norm(target - feedback_sum)
    memoryless_error = np.linalg.norm(target - memoryless_sum)
    assert feedback_error < memoryless_error


def test_error_feedback_checkpoint_produces_same_next_payload_and_residual():
    rng = np.random.default_rng(11)
    first = rng.normal(size=(24, 24))
    second = rng.normal(size=(24, 24))
    contract = _approximate_contract()
    original = ErrorFeedbackCompressor(default_rank=2, exact_threshold_bytes=0)
    original.compress("weight", first, contract)

    restored = ErrorFeedbackCompressor()
    restored.load_state_dict(original.state_dict())
    expected = original.compress("weight", second, contract)
    actual = restored.compress("weight", second, contract)

    assert actual.payload_hash == expected.payload_hash
    np.testing.assert_array_equal(restored.residual("weight"), original.residual("weight"))


def test_vector_uses_topk_and_small_approximate_statistics_remain_dense():
    vector = np.linspace(-10.0, 10.0, 1_000)
    contract = _approximate_contract()
    sparse = ErrorFeedbackCompressor(exact_threshold_bytes=0, default_topk_fraction=0.05).compress(
        "vector", vector, contract
    )
    assert sparse.method is CompressionMethod.TOPK
    assert sparse.receipt.rank_or_nnz == 50

    small = ErrorFeedbackCompressor(exact_threshold_bytes=10_000).compress("small", vector[:10], contract)
    assert small.method is CompressionMethod.DENSE
