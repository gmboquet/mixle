"""Regression: StreamingTransformerAccumulator.seq_update dropped the per-token weight.

Its in-place train step used an unweighted CrossEntropyLoss mean, so a mixture responsibility / streaming
decay / sample weight was ignored -- every token trained the module equally. It now applies a weighted mean,
which is bit-identical to the old path when the weights are uniform (so pure streaming is unchanged).
"""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.streaming_transformer_leaf import StreamingTransformerAccumulatorFactory
from mixle.models.transformer import build_causal_lm


def _batch(V=12, block=8, n=6, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randint(0, V, size=(n, block)).astype(float)
    y = rng.randint(0, V, size=n).astype(int)
    return (x, y)


def _acc(module):
    return StreamingTransformerAccumulatorFactory(module, 3e-3, "cpu").make()


def _params(module):
    return torch.cat([p.detach().flatten() for p in module.parameters()])


def test_uniform_weights_are_bit_identical_to_none():
    torch.manual_seed(0)
    base = build_causal_lm(12, 32, 2, 2, 8)
    enc = _batch()
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    acc_a, acc_b = _acc(a), _acc(b)
    acc_a.seq_update(enc, None, None)
    acc_b.seq_update(enc, np.ones(len(enc[1])), None)
    assert acc_a.last_loss == acc_b.last_loss
    assert torch.equal(_params(a), _params(b))  # pure-streaming path unchanged


def test_weights_steer_the_gradient():
    torch.manual_seed(0)
    base = build_causal_lm(12, 32, 2, 2, 8)
    enc = _batch()
    n = len(enc[1])
    w_first, w_last = np.zeros(n), np.zeros(n)
    w_first[0] = 1.0
    w_last[-1] = 1.0
    c, d = copy.deepcopy(base), copy.deepcopy(base)
    _acc(c).seq_update(enc, w_first, None)
    _acc(d).seq_update(enc, w_last, None)
    assert not torch.equal(_params(c), _params(d))  # focusing on different tokens gives different updates
