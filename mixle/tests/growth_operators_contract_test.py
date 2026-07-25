"""Device, layout, and state contracts for function-preserving growth."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.growth_operators import (  # noqa: E402
    insert_block,
    net2net_widen,
    verify_output_parity,
    widen_block,
)
from mixle.models.transformer import Block, build_causal_lm  # noqa: E402


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_net2net_allocates_replacements_on_source_device_and_dtype() -> None:
    device = _device()
    linear_in = torch.nn.Linear(3, 4, device=device, dtype=torch.float64)
    linear_out = torch.nn.Linear(4, 2, device=device, dtype=torch.float64)

    widened_in, widened_out, _ = net2net_widen(linear_in, linear_out, 7)
    assert widened_in.weight.device == device
    assert widened_out.weight.device == device
    assert widened_in.weight.dtype == torch.float64
    assert widened_out.weight.dtype == torch.float64

    x = torch.randn(5, 3, device=device, dtype=torch.float64)
    torch.testing.assert_close(
        widened_out(torch.nn.functional.gelu(widened_in(x))),
        linear_out(torch.nn.functional.gelu(linear_in(x))),
    )


def test_block_growth_preserves_device_dtype_and_parity_batch_contract() -> None:
    device = _device()
    block = Block(8, 2).to(device=device, dtype=torch.float64)
    widened, receipt = widen_block(block, 16, seed=2)
    assert next(widened.parameters()).device == device
    assert next(widened.parameters()).dtype == torch.float64
    assert receipt.parity is not None and receipt.parity.within_tolerance


def test_parity_verification_restores_every_shared_module_mode() -> None:
    model = build_causal_lm(vocab=9, d_model=8, n_layer=2, n_head=2, block=4)
    model.eval()
    model.blocks[0].train()
    before = {id(module): module.training for module in model.modules()}

    receipt = verify_output_parity(model, model, torch.randint(0, 9, (2, 4)))
    after = {id(module): module.training for module in model.modules()}
    assert receipt.within_tolerance
    assert after == before

    grown, receipt = insert_block(model, 1)
    assert receipt.parity is not None and receipt.parity.within_tolerance
    assert {id(module): module.training for module in model.modules()} == before
    assert grown.training is model.training
    assert grown.blocks[0].training is True


def test_parity_requires_finite_aligned_tensor_outputs() -> None:
    class TupleOutput(torch.nn.Module):
        def forward(self, x):
            return x, x

    with pytest.raises(TypeError, match="tensor-valued"):
        verify_output_parity(TupleOutput(), TupleOutput(), torch.ones(2, 2))
    with pytest.raises(ValueError, match="tolerance"):
        verify_output_parity(torch.nn.Identity(), torch.nn.Identity(), torch.ones(1), tolerance=-1.0)


def test_growth_width_and_layer_contracts_fail_closed() -> None:
    with pytest.raises(ValueError):
        net2net_widen(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1), new_width=3.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        widen_block(Block(8, 2), new_d_model=0)
    model = build_causal_lm(vocab=9, d_model=8, n_layer=1, n_head=2, block=4)
    with pytest.raises(ValueError):
        insert_block(model, position=True)  # type: ignore[arg-type]
