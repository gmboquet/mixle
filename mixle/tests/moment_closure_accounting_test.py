"""One-observation/one-representation contracts for moment-closure attention."""

import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.moment_closure_attention import (
    ClusterBank,
    MomentClosureAttention,
    _empty_cluster_bank,
    birth_and_merge,
    ingest_cluster_batch,
)

pytestmark = pytest.mark.experimental


def test_birth_ingestion_counts_each_observation_once():
    bank = _empty_cluster_bank(2, 4, 4, device="cpu", dtype=torch.float32)
    k = torch.randn(1, 3, 2, 4)
    v = torch.randn(1, 3, 2, 4)
    updated, receipt = ingest_cluster_batch(bank, k, v, birth_threshold=-2.0)
    torch.testing.assert_close(updated.count[:, : updated.n_clusters].sum(dim=1), torch.full((2,), 3.0))
    assert receipt["birth_incorporated_batch"] is True
    assert receipt["soft_update_applied"] is False
    assert receipt["count_conserved"] is True


def test_far_bank_contains_only_tokens_evicted_from_near_window():
    torch.manual_seed(2)
    model = MomentClosureAttention(
        9,
        d_model=8,
        n_layer=1,
        n_head=2,
        window=2,
        max_clusters=4,
        birth_threshold=-2.0,
    )
    state = model.init_state(1)

    x1 = torch.tensor([[1, 2, 3]])
    state, _ = model.step(state, (x1, x1))
    assert state.near.cache_k[0].shape[1] == 2
    torch.testing.assert_close(state.banks[0].count[:, : state.banks[0].n_clusters].sum(dim=1), torch.ones(2))
    accounting1 = model.last_receipts[0]["accounting"]
    assert accounting1["near_tokens_per_stream"] == 2
    assert accounting1["expected_far_tokens_per_head"] == 1
    assert accounting1["overlap_tokens"] == 0
    assert accounting1["conserved"] is True

    x2 = torch.tensor([[4, 5]])
    state, _ = model.step(state, (x2, x2))
    assert state.near.cache_k[0].shape[1] == 2
    torch.testing.assert_close(
        state.banks[0].count[:, : state.banks[0].n_clusters].sum(dim=1),
        torch.full((2,), 3.0),
    )
    accounting2 = model.last_receipts[0]["accounting"]
    assert accounting2["expected_far_tokens_per_head"] == 3
    assert accounting2["positions_seen_per_stream"] == 5
    assert accounting2["conserved"] is True


def test_non_evicted_tokens_do_not_enter_far_bank():
    model = MomentClosureAttention(9, d_model=8, n_layer=1, n_head=2, window=4)
    state = model.init_state(1)
    x = torch.tensor([[1, 2, 3]])
    state, _ = model.step(state, (x, x))
    assert state.banks[0].n_clusters == 0
    assert model.last_receipts[0]["expected_ingested_tokens_per_head"] == 0


def test_merge_uses_multivariate_geometry_not_flat_coordinate_histograms():
    # The two means contain the same coordinate values in a different order. Flattening their values loses
    # that distinction, while their variance-normalized vector distance is large.
    bank = ClusterBank(
        count=torch.tensor([[100.0, 100.0]]),
        mu_k=torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
        mu_v=torch.zeros(1, 2, 2),
        sigma_kk=torch.full((1, 2, 2), 0.01),
        sigma_vk=torch.zeros(1, 2, 2, 2),
        n_clusters=2,
        max_clusters=2,
    )
    k = torch.tensor([[[[0.0, 1.0]]]])
    v = torch.zeros_like(k)
    updated, receipt = birth_and_merge(
        bank,
        k,
        v,
        birth_threshold=-1e9,
        merge_threshold=3.0,
        outlier_top_k=0,
    )
    assert updated.n_clusters == 2
    assert receipt["merged"] == []


def test_invalid_batch_change_is_rejected_before_accounting_can_drift():
    model = MomentClosureAttention(9, d_model=8, n_layer=1, n_head=2, window=2)
    state = model.init_state(1)
    x = torch.ones((2, 1), dtype=torch.long)
    with pytest.raises(ValueError, match="batch size"):
        model.step(state, (x, x))


def test_preexisting_overlap_is_rejected_before_more_tokens_are_processed():
    model = MomentClosureAttention(9, d_model=8, n_layer=1, n_head=2, window=2)
    state = model.init_state(1)
    x = torch.tensor([[1, 2, 3]])
    state, _ = model.step(state, (x, x))
    state.banks[0].count[:, 0] += 1.0
    next_token = torch.tensor([[4]])
    with pytest.raises(RuntimeError, match="one-token/one-representation"):
        model.step(state, (next_token, next_token))


def test_hybrid_consumer_uses_the_same_eviction_only_ingestion_contract():
    from mixle.experimental.ssm_hybrid import HybridBlock

    model = HybridBlock(
        9,
        d_model=8,
        n_layer=1,
        n_head=2,
        window=2,
        d_state=2,
        ssm_expand=1,
        max_clusters=2,
    )
    state = model.init_state(1)
    x = torch.tensor([[1, 2, 3]])
    state, _ = model.step(state, (x, x))
    torch.testing.assert_close(state.banks[0].count[:, : state.banks[0].n_clusters].sum(dim=1), torch.ones(2))
    assert model.last_receipts[0]["accounting"]["conserved"] is True
