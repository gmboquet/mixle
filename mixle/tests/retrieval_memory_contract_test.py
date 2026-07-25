"""Focused position, rotation, and layout contracts for retrieval memory."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.context_spine import _apply_rope, _rope_angles  # noqa: E402
from mixle.experimental.retrieval_memory_spine import (  # noqa: E402
    RetrievalMemorySpine,
    _merge_attention_heads,
)

pytestmark = pytest.mark.experimental


def _model(**overrides):
    config = {
        "vocab": 11,
        "d_model": 8,
        "n_layer": 1,
        "n_head": 2,
        "window": 2,
        "retrieval_k": 2,
    }
    config.update(overrides)
    return RetrievalMemorySpine(**config)


def test_cache_and_index_partition_stream_positions_without_duplicates():
    torch.manual_seed(0)
    model = _model()
    state = model.init_state(1)
    tokens = torch.arange(8).remainder(model.vocab)[None, :]
    with torch.no_grad():
        for start in range(0, 8, 2):
            chunk = tokens[:, start : start + 2]
            state, _ = model.step(state, (chunk, chunk))

    assert state.index_pos[0].tolist() == list(range(6))
    assert state.cache_k[0].shape[1] == model.window
    assert state.receipt["cache_index_position_overlap_per_layer"] == [0]
    assert state.receipt["dual_visible_position_count_per_layer"] == [0]
    assert state.receipt["index_len_per_layer"][0] + state.cache_k[0].shape[1] == state.pos


def test_cached_raw_key_is_rotated_exactly_once_when_archived():
    torch.manual_seed(1)
    model = _model()
    first = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        embedded = model.tok(first)
        normalized = model.ln1[0](embedded)
        raw_keys = model.qkv[0](normalized).reshape(1, 3, 3, model.n_head, model.head_dim)[:, :, 1]

        state, _ = model.step(model.init_state(1), (first, first))
        second = torch.tensor([[4]])
        state, _ = model.step(state, (second, second))

    sin, cos = _rope_angles(torch.tensor([1]), model.head_dim)
    expected_once = _apply_rope(raw_keys[:, 1:2], sin, cos)
    assert state.index_pos[0].tolist() == [0, 1]
    assert torch.allclose(state.index_k[0][:, 1:2], expected_once)


def test_head_merge_preserves_time_then_head_layout():
    per_head = torch.tensor(
        [
            [
                [[10.0], [11.0], [12.0]],
                [[20.0], [21.0], [22.0]],
            ]
        ]
    )
    merged = _merge_attention_heads(per_head)
    assert merged.tolist() == [[[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]]


def test_long_chunk_never_makes_one_position_local_and_retrieved_for_same_query():
    torch.manual_seed(2)
    model = _model(window=3, retrieval_k=4)
    tokens = torch.arange(9).remainder(model.vocab)[None, :]
    with torch.no_grad():
        state, _ = model.step(model.init_state(1), (tokens, tokens))
    assert state.receipt["archived_this_step_per_layer"] == [6]
    assert state.receipt["dual_visible_position_count_per_layer"] == [0]
    assert state.index_pos[0].tolist() == list(range(6))


def test_index_cap_keeps_latest_evicted_positions_and_no_cache_overlap():
    torch.manual_seed(3)
    model = _model(max_index_tokens=3)
    state = model.init_state(1)
    tokens = torch.arange(9).remainder(model.vocab)[None, :]
    with torch.no_grad():
        for start in range(0, 9, 3):
            chunk = tokens[:, start : start + 3]
            state, _ = model.step(state, (chunk, chunk))
    assert state.index_pos[0].tolist() == [4, 5, 6]
    assert state.receipt["cache_index_position_overlap_per_layer"] == [0]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"window": 0}, "window"),
        ({"retrieval_k": True}, "retrieval_k"),
        ({"max_index_tokens": 0}, "max_index_tokens"),
        ({"d_model": 7, "n_head": 2}, "divisible"),
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _model(**kwargs)


def test_step_rejects_empty_or_mismatched_chunks():
    model = _model()
    state = model.init_state(1)
    with pytest.raises(ValueError, match="non-empty"):
        model.step(state, (torch.empty(1, 0, dtype=torch.long), torch.empty(1, 0, dtype=torch.long)))
    with pytest.raises(ValueError, match="batch_size"):
        model.step(
            state,
            (
                torch.ones(2, 1, dtype=torch.long),
                torch.ones(2, 1, dtype=torch.long),
            ),
        )
