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

pytestmark = [pytest.mark.experimental]


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
    payload = compressor.compress(
        "weight",
        delta,
        _approximate_contract(),
        allow_approximation=True,
        maximum_relative_l2_error=1.0,
    )

    assert payload.method is CompressionMethod.LOW_RANK
    assert payload.receipt.payload_bytes < payload.receipt.input_bytes
    assert payload.receipt.compression_ratio > 5.0
    reconstruction = payload.reconstruct()
    assert compressor.residual("weight") is None
    acknowledgement = compressor.acknowledge(payload, applied=True)
    assert acknowledgement.applied
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
        payload = with_feedback.compress(
            "weight",
            delta,
            contract,
            allow_approximation=True,
            maximum_relative_l2_error=1.0,
        )
        feedback_sum += payload.reconstruct()
        with_feedback.acknowledge(payload, applied=True)

    memoryless_sum = np.zeros_like(delta)
    for _ in range(rounds):
        memoryless = ErrorFeedbackCompressor(default_rank=1, exact_threshold_bytes=0)
        payload = memoryless.compress(
            "weight",
            delta,
            contract,
            allow_approximation=True,
            maximum_relative_l2_error=1.0,
        )
        memoryless_sum += payload.reconstruct()
        memoryless.acknowledge(payload, applied=True)

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
    first_payload = original.compress(
        "weight",
        first,
        contract,
        allow_approximation=True,
        maximum_relative_l2_error=1.0,
    )
    original.acknowledge(first_payload, applied=True)

    restored = ErrorFeedbackCompressor()
    restored.load_state_dict(original.state_dict())
    expected = original.compress(
        "weight",
        second,
        contract,
        allow_approximation=True,
        maximum_relative_l2_error=1.0,
    )
    actual = restored.compress(
        "weight",
        second,
        contract,
        allow_approximation=True,
        maximum_relative_l2_error=1.0,
    )

    assert actual.payload_hash == expected.payload_hash
    original.acknowledge(expected, applied=True)
    restored.acknowledge(actual, applied=True)
    np.testing.assert_array_equal(restored.residual("weight"), original.residual("weight"))


def test_vector_uses_topk_and_small_approximate_statistics_remain_dense():
    vector = np.linspace(-10.0, 10.0, 1_000)
    contract = _approximate_contract()
    compressor = ErrorFeedbackCompressor(exact_threshold_bytes=0, default_topk_fraction=0.05)
    sparse = compressor.compress(
        "vector",
        vector,
        contract,
        allow_approximation=True,
        maximum_relative_l2_error=1.0,
    )
    assert sparse.method is CompressionMethod.TOPK
    assert sparse.receipt.rank_or_nnz == 50
    compressor.acknowledge(sparse, applied=True)

    small = ErrorFeedbackCompressor(exact_threshold_bytes=10_000).compress(
        "small",
        vector[:10],
        contract,
        allow_approximation=True,
        maximum_relative_l2_error=1.0,
    )
    assert small.method is CompressionMethod.DENSE


def test_lossy_transport_requires_separate_authorization_and_error_budget():
    rng = np.random.default_rng(19)
    delta = rng.normal(size=(32, 32))
    compressor = ErrorFeedbackCompressor(exact_threshold_bytes=0)
    dense = compressor.compress("weight", delta, _approximate_contract(), rank=1)
    assert dense.method is CompressionMethod.DENSE
    assert dense.receipt.exact
    compressor.acknowledge(dense, applied=True)

    exact_low_rank_contract = UpdateContract(
        objective_kind=ObjectiveKind.MLE,
        update_kind=UpdateKind.EXACT_CLOSED_FORM,
        merge_law=MergeLaw.LOW_RANK,
        exact=True,
        declared_by="test-exact-low-rank-merge",
    )
    with pytest.raises(ValueError, match="exact update contract"):
        compressor.compress(
            "exact",
            delta,
            exact_low_rank_contract,
            allow_approximation=True,
            maximum_relative_l2_error=1.0,
        )

    budgeted = compressor.compress(
        "budgeted",
        delta,
        _approximate_contract(),
        allow_approximation=True,
        maximum_relative_l2_error=0.0,
    )
    assert budgeted.method is CompressionMethod.DENSE
    assert budgeted.receipt.exact


def test_rejected_delivery_does_not_advance_committed_residual():
    delta = np.arange(1.0, 257.0).reshape(16, 16)
    compressor = ErrorFeedbackCompressor(default_rank=1, exact_threshold_bytes=0)
    first = compressor.compress(
        "weight",
        delta,
        _approximate_contract(),
        allow_approximation=True,
        maximum_relative_l2_error=1.0,
    )
    assert compressor.residual("weight") is None
    with pytest.raises(RuntimeError, match="unacknowledged"):
        compressor.compress("weight", delta, _approximate_contract())
    rejected = compressor.acknowledge(first, applied=False)
    assert not rejected.applied
    assert compressor.residual("weight") is None

    retry = compressor.compress(
        "weight",
        delta,
        _approximate_contract(),
        allow_approximation=True,
        maximum_relative_l2_error=1.0,
    )
    assert retry.payload_hash == first.payload_hash
    compressor.acknowledge(retry, applied=True)
    assert compressor.residual("weight") is not None


def test_integer_payload_never_uses_lossy_factor_casts():
    delta = np.arange(64, dtype=np.int64).reshape(8, 8)
    compressor = ErrorFeedbackCompressor(exact_threshold_bytes=0)
    payload = compressor.compress(
        "integer",
        delta,
        _approximate_contract(),
        allow_approximation=True,
        maximum_relative_l2_error=1.0,
        rank=1,
    )
    assert payload.method is CompressionMethod.DENSE
    assert payload.receipt.exact
    np.testing.assert_array_equal(payload.reconstruct(), delta)
    assert payload.reconstruct().dtype == delta.dtype
