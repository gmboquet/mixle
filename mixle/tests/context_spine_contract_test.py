"""Cache and RoPE invariants for the streaming context spine."""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.context_parallel_spine import (  # noqa: E402
    _rope_angles as cp_rope_angles,
)
from mixle.experimental.context_parallel_spine import (
    cp_shard_kv,
    cp_window_attention_forward,
    validate_cp_window_plan,
)
from mixle.experimental.context_spine import (  # noqa: E402
    SlidingWindowSpine,
    _rope_angles,
)


def _chunk() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([[1, 2, 3]], dtype=torch.long)
    y = torch.tensor([[2, 3, 4]], dtype=torch.long)
    return x, y


def test_cache_stores_raw_keys_and_explicit_absolute_positions() -> None:
    torch.manual_seed(4)
    model = SlidingWindowSpine(8, d_model=8, n_layer=1, n_head=2, window=4)
    x, y = _chunk()
    state = model.init_state(1)

    with torch.no_grad():
        hidden = model.tok(x)
        normalized = model.ln1[0](hidden)
        raw_k = model.qkv[0](normalized).reshape(1, 3, 3, 2, 4)[:, :, 1]
        state, _ = model.step(state, (x, y))

    torch.testing.assert_close(state.cache_k[0], raw_k)
    assert torch.equal(state.cache_positions[0], torch.arange(3))
    assert state.pos == 3

    with torch.no_grad():
        state, _ = model.step(state, (x[:, :2], y[:, :2]))
    assert torch.equal(state.cache_positions[0], torch.arange(1, 5))
    assert state.cache_k[0].shape == (1, 4, 2, 4)


def test_streamed_full_window_matches_one_shot_loss() -> None:
    torch.manual_seed(9)
    model = SlidingWindowSpine(11, d_model=8, n_layer=2, n_head=2, window=None)
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    y = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        _, full_loss = model.step(model.init_state(1), (x, y))
        state, first_loss = model.step(model.init_state(1), (x[:, :2], y[:, :2]))
        _, second_loss = model.step(state, (x[:, 2:], y[:, 2:]))
    torch.testing.assert_close((first_loss + second_loss) / 2.0, full_loss, atol=1e-6, rtol=1e-6)


def test_state_and_model_layouts_fail_closed() -> None:
    with pytest.raises(ValueError):
        SlidingWindowSpine(8, d_model=7, n_head=2)
    with pytest.raises(ValueError):
        SlidingWindowSpine(8, d_model=6, n_head=2)  # odd head dimension is invalid for RoPE
    with pytest.raises(ValueError):
        SlidingWindowSpine(8, window=0)

    model = SlidingWindowSpine(8, d_model=8, n_layer=1, n_head=2)
    state = model.init_state(1)
    with pytest.raises(ValueError):
        model.step(replace(state, n_head=3), _chunk())
    with pytest.raises(ValueError):
        model.step(replace(state, pos=-1), _chunk())
    with pytest.raises(ValueError):
        model.step(state, (_chunk()[0].float(), _chunk()[1]))


def test_rope_angles_stay_on_the_positions_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positions = torch.arange(3, device=device)
    for helper in (_rope_angles, cp_rope_angles):
        sin, cos = helper(positions, 4)
        assert sin.device == device
        assert cos.device == device


def test_context_parallel_validates_window_and_layout() -> None:
    with pytest.raises(ValueError):
        validate_cp_window_plan(2, 0)
    with pytest.raises(ValueError):
        validate_cp_window_plan(0, 4)

    k = torch.zeros(1, 3, 2, 4)
    v = torch.zeros_like(k)
    positions = torch.arange(3)
    shards = cp_shard_kv(k, v, positions, 2)
    q = torch.zeros(1, 1, 2, 4)
    with pytest.raises(ValueError):
        cp_window_attention_forward(q, torch.tensor([3]), shards, window=0, head_dim=4)
    with pytest.raises(ValueError):
        cp_shard_kv(k, v[:, :2], positions, 2)
