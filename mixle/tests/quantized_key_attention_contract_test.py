"""Focused ownership and approximation contracts for quantized-key attention."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.quantized_key_attention import (  # noqa: E402
    ProductQuantizer,
    QuantizedKeyAttentionSpine,
    SlidingCellWindow,
    quantized_softmax_weights,
)

pytestmark = pytest.mark.experimental


def test_sliding_window_owns_values_used_for_later_eviction():
    window = SlidingCellWindow(window=1, value_dim=2)
    caller_value = np.array([1.0, 2.0])
    window.push(3, caller_value)
    caller_value[:] = 100.0

    evicted = window.push(4, [5.0, 6.0])
    assert evicted is not None
    np.testing.assert_array_equal(evicted[1], [1.0, 2.0])
    counts, sums = window.totals()
    assert counts == {4: 1}
    np.testing.assert_array_equal(sums[4], [5.0, 6.0])


def test_ema_codes_reconstruct_the_returned_quantized_keys_after_update():
    torch.manual_seed(0)
    quantizer = ProductQuantizer(
        8,
        n_blocks=2,
        codes_per_block=4,
        codebook_update="ema",
        ema_decay=0.5,
    )
    quantizer.train()
    keys = torch.randn(32, 8)

    quantized, codes, _ = quantizer(keys)

    assert torch.equal(quantized, quantizer.reconstruct(codes))


def test_quantized_softmax_receipt_covers_clipped_tail_error():
    logits = torch.tensor([[0.0, -30.0, -float("inf")]], dtype=torch.float64)
    weights, receipt = quantized_softmax_weights(logits, bits=12, span=24.0, return_receipt=True)
    approximate = weights / weights.sum(dim=-1, keepdim=True)
    exact = logits.softmax(dim=-1)
    finite_positive = exact > 0
    relative_error = torch.abs(approximate[finite_positive] / exact[finite_positive] - 1.0)

    assert receipt.clipped_logits == 1
    assert receipt.max_logit_error > 5.0
    assert float(relative_error.max()) <= receipt.max_relative_probability_error_bound


def test_in_span_receipt_reduces_to_the_grid_error_scope():
    logits = torch.tensor([[0.0, -1.0, -2.0]], dtype=torch.float64)
    _, receipt = quantized_softmax_weights(logits, bits=8, span=4.0, return_receipt=True)
    assert receipt.clipped_logits == 0
    assert receipt.max_logit_error <= receipt.grid_step / 2 + 1e-12
    assert receipt.max_relative_probability_error_bound <= math.expm1(receipt.grid_step + 1e-12)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"bits": True, "span": 24.0}, "bits"),
        ({"bits": 0, "span": 24.0}, "bits"),
        ({"bits": 25, "span": 24.0}, "bits"),
        ({"bits": 8, "span": 0.0}, "span"),
        ({"bits": 8, "span": float("nan")}, "span"),
    ],
)
def test_quantized_softmax_rejects_invalid_grid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        quantized_softmax_weights(torch.tensor([[0.0]]), **kwargs)


def test_quantized_softmax_rejects_a_dtype_that_underflows_its_table():
    with pytest.raises(ValueError, match="cannot represent"):
        quantized_softmax_weights(torch.tensor([[0.0, -1.0]], dtype=torch.float16), bits=8, span=24.0)


def test_spine_carries_the_data_dependent_quantization_receipt():
    torch.manual_seed(1)
    spine = QuantizedKeyAttentionSpine(
        7,
        d_model=8,
        n_layer=1,
        n_head=1,
        window=2,
        n_blocks=2,
        codes_per_block=4,
        max_cells=8,
        lse_bits=8,
        lse_span=8.0,
    )
    tokens = torch.randint(0, 7, (1, 3))
    with torch.no_grad():
        state, _ = spine.step(spine.init_state(1), (tokens, tokens))
    receipt = state.lse_receipts[0]
    assert receipt is not None
    assert receipt.finite_logits > 0
    assert spine.occupancy_receipt(state)["quantized_softmax"][0]["bits"] == 8


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_blocks": 0}, "n_blocks"),
        ({"codes_per_block": 0}, "codes_per_block"),
        ({"ema_decay": 1.0}, "ema_decay"),
        ({"ema_eps": 0.0}, "ema_eps"),
    ],
)
def test_product_quantizer_rejects_invalid_configuration(kwargs, match):
    config = {"n_blocks": 2, "codes_per_block": 4}
    config.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ProductQuantizer(8, **config)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"lse_bits": True}, "lse_bits"),
        ({"lse_bits": 0}, "lse_bits"),
        ({"lse_bits": 25}, "lse_bits"),
        ({"lse_span": 0.0}, "lse_span"),
        ({"d_model": 7, "n_head": 2}, "divisible"),
    ],
)
def test_spine_rejects_invalid_configuration(kwargs, match):
    config = {"d_model": 8, "n_layer": 1, "n_head": 1}
    config.update(kwargs)
    with pytest.raises(ValueError, match=match):
        QuantizedKeyAttentionSpine(7, **config)
