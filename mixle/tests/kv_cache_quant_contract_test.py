"""Fast correctness contracts for clustered KV-cache quantization."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.kv_cache_quant import (
    AffineQuantized,
    dequantize_kv_cache,
    quantize_cluster_outliers,
    quantize_kv_cache,
    verify_cluster_token_conservation,
)

pytestmark = pytest.mark.experimental


def _payload(flat_k, flat_v):
    return {
        0: {
            "member_indices": torch.tensor([0, 2, 4], device=flat_k.device),
            "indices": torch.tensor([2], device=flat_k.device),
            "k": flat_k[[2]],
            "v": flat_v[[2]],
        },
        1: {
            "member_indices": torch.tensor([1, 3, 5], device=flat_k.device),
            "indices": torch.tensor([5], device=flat_k.device),
            "k": flat_k[[5]],
            "v": flat_v[[5]],
        },
    }


def test_cluster_tails_contain_only_non_outlier_members_and_conserve_counts():
    flat_k = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2)
    flat_v = flat_k + 100
    encoded = quantize_cluster_outliers(_payload(flat_k, flat_v), flat_k, flat_v)

    np.testing.assert_array_equal(encoded[0].tail_indices, [0, 4])
    np.testing.assert_array_equal(encoded[1].tail_indices, [1, 3])
    receipt = verify_cluster_token_conservation(encoded, 6)
    assert receipt.conserved
    assert (receipt.input_count, receipt.outlier_count, receipt.tail_count) == (6, 2, 4)


def test_birth_receipt_exposes_a_partition_that_round_trips_through_compressor():
    from mixle.experimental.moment_closure_attention import ClusterBank, birth_and_merge

    means = torch.tensor([[[-5.0, -5.0], [5.0, 5.0]]])
    bank = ClusterBank(
        count=torch.full((1, 2), 10.0),
        mu_k=means,
        mu_v=means.clone(),
        sigma_kk=torch.ones(1, 2, 2),
        sigma_vk=torch.zeros(1, 2, 2, 2),
        n_clusters=2,
        max_clusters=2,
    )
    k = torch.tensor([[[[-5.1, -4.9]], [[-4.8, -5.0]], [[5.0, 5.2]], [[4.9, 4.8]]]])
    v = k.clone()
    _, receipt = birth_and_merge(
        bank,
        k,
        v,
        birth_threshold=-1e9,
        merge_threshold=-1.0,
        outlier_top_k=1,
    )
    flat_k = k.reshape(4, 1, 2)
    flat_v = v.reshape(4, 1, 2)
    encoded = quantize_cluster_outliers(receipt["per_cluster_outlier_tokens"], flat_k, flat_v)
    assert verify_cluster_token_conservation(encoded, 4).conserved


def test_overlapping_memberships_are_rejected():
    flat_k = torch.randn(6, 2, 2)
    flat_v = torch.randn(6, 2, 2)
    payload = _payload(flat_k, flat_v)
    payload[1]["member_indices"] = torch.tensor([0, 1, 3, 5])
    with pytest.raises(ValueError, match="exact partition"):
        quantize_cluster_outliers(payload, flat_k, flat_v)


def test_outlier_must_belong_to_its_cluster():
    flat_k = torch.randn(6, 2, 2)
    flat_v = torch.randn(6, 2, 2)
    payload = _payload(flat_k, flat_v)
    payload[0]["indices"] = torch.tensor([1])
    payload[0]["k"] = flat_k[[1]]
    payload[0]["v"] = flat_v[[1]]
    with pytest.raises(ValueError, match="must be cluster members"):
        quantize_cluster_outliers(payload, flat_k, flat_v)


@pytest.mark.parametrize("bad", [torch.tensor([]), torch.tensor([float("nan")]), torch.tensor([float("inf")])])
def test_quantizer_rejects_empty_and_nonfinite_input(bad):
    with pytest.raises(ValueError):
        quantize_kv_cache(bad)


def test_fp8_scales_values_outside_native_range_without_overflow():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build has no fp8 dtype")
    source = torch.tensor([-100_000.0, 0.0, 100_000.0])
    encoded = quantize_kv_cache(source, mode="fp8")
    reconstructed = dequantize_kv_cache(encoded)
    assert encoded.scale > 1.0
    assert torch.isfinite(reconstructed).all()
    torch.testing.assert_close(reconstructed[[0, 2]], source[[0, 2]], rtol=0.05, atol=0)


def test_dequantizer_rejects_nonfinite_scale():
    with pytest.raises(ValueError, match="positive and finite"):
        dequantize_kv_cache(AffineQuantized(torch.ones(1, dtype=torch.int8), float("inf"), "int8"))


def test_cuda_membership_indexing_stays_device_local_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    flat_k = torch.randn(6, 2, 2, device="cuda")
    flat_v = torch.randn(6, 2, 2, device="cuda")
    encoded = quantize_cluster_outliers(_payload(flat_k, flat_v), flat_k, flat_v)
    assert encoded[0].outlier_k.codes.device.type == "cuda"
