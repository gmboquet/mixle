"""Regression: StreamingTransformerAccumulator.seq_update dropped the per-token weight.

Its in-place train step used an unweighted CrossEntropyLoss mean, so a mixture responsibility / streaming
decay / sample weight was ignored -- every token trained the module equally. It now applies a weighted mean,
which is bit-identical to the old path when the weights are uniform (so pure streaming is unchanged).
"""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.streaming_transformer_leaf import (
    StreamingTransformerAccumulatorFactory,
    StreamingTransformerEstimator,
    stream_fit,
)
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
    state_a, state_b = acc_a.value(), acc_b.value()
    for gradient_a, gradient_b in zip(state_a.gradient_sums, state_b.gradient_sums):
        np.testing.assert_array_equal(gradient_a, gradient_b)
    StreamingTransformerEstimator(a, lr=3e-3).estimate(None, state_a)
    StreamingTransformerEstimator(b, lr=3e-3).estimate(None, state_b)
    assert torch.equal(_params(a), _params(b))


def test_weights_steer_the_gradient():
    torch.manual_seed(0)
    base = build_causal_lm(12, 32, 2, 2, 8)
    enc = _batch()
    n = len(enc[1])
    w_first, w_last = np.zeros(n), np.zeros(n)
    w_first[0] = 1.0
    w_last[-1] = 1.0
    c, d = copy.deepcopy(base), copy.deepcopy(base)
    acc_c, acc_d = _acc(c), _acc(d)
    acc_c.seq_update(enc, w_first, None)
    acc_d.seq_update(enc, w_last, None)
    StreamingTransformerEstimator(c, lr=3e-3).estimate(None, acc_c.value())
    StreamingTransformerEstimator(d, lr=3e-3).estimate(None, acc_d.value())
    assert not torch.equal(_params(c), _params(d))  # focusing on different tokens gives different updates


def test_partitioned_gradient_combine_matches_single_worker_update():
    torch.manual_seed(4)
    base = build_causal_lm(12, 16, 1, 2, 8).double()
    enc = _batch(n=8, seed=9)
    weights = np.linspace(0.25, 2.0, len(enc[1]))

    single_module = copy.deepcopy(base)
    single_estimator = StreamingTransformerEstimator(single_module, lr=2e-3)
    single_accumulator = single_estimator.accumulator_factory().make()
    single_accumulator.seq_update(enc, weights, None)
    single_estimator.estimate(None, single_accumulator.value())

    root_module = copy.deepcopy(base)
    root_estimator = StreamingTransformerEstimator(root_module, lr=2e-3)
    root_accumulator = root_estimator.accumulator_factory().make()
    for rows in (slice(0, 3), slice(3, None)):
        worker_module = copy.deepcopy(base)
        worker = StreamingTransformerAccumulatorFactory(worker_module, 2e-3, "cpu").make()
        worker.seq_update((enc[0][rows], enc[1][rows]), weights[rows], None)
        root_accumulator.combine(worker.value())
    root_estimator.estimate(None, root_accumulator.value())

    # Split GEMMs and one full-batch GEMM may sum products in a different order; in the model's
    # declared float64 precision the aggregated update agrees to numerical roundoff.
    torch.testing.assert_close(_params(root_module), _params(single_module), rtol=1e-9, atol=1e-10)


def test_optimizer_state_and_controls_survive_continuation():
    torch.manual_seed(7)
    base = build_causal_lm(12, 16, 1, 2, 8)
    batches = [_batch(n=5, seed=11), _batch(n=5, seed=12)]

    direct_module = copy.deepcopy(base)
    direct_estimator = StreamingTransformerEstimator(direct_module, lr=7e-4)
    for batch in batches:
        accumulator = direct_estimator.accumulator_factory().make()
        accumulator.seq_update(batch, None, None)
        direct_leaf = direct_estimator.estimate(None, accumulator.value())

    continued_module = copy.deepcopy(base)
    first_estimator = StreamingTransformerEstimator(continued_module, lr=7e-4)
    first_accumulator = first_estimator.accumulator_factory().make()
    first_accumulator.seq_update(batches[0], None, None)
    continued_leaf = first_estimator.estimate(None, first_accumulator.value())
    resumed_estimator = continued_leaf.estimator()
    second_accumulator = resumed_estimator.accumulator_factory().make()
    second_accumulator.seq_update(batches[1], None, None)
    continued_leaf = resumed_estimator.estimate(None, second_accumulator.value())

    assert continued_leaf.lr == 7e-4
    assert continued_leaf.optimizer_state is not None
    torch.testing.assert_close(_params(continued_module), _params(direct_module), rtol=0.0, atol=0.0)
    assert direct_leaf.optimizer_state["param_groups"][0]["lr"] == 7e-4

    from mixle.models.streaming_transformer_leaf import StreamingTransformer
    from mixle.utils.serialization import trusted_deserialization

    payload = continued_leaf.to_dict()
    with trusted_deserialization():
        restored = StreamingTransformer.from_dict(payload)
    assert restored.lr == 7e-4
    assert restored.optimizer_state is not None
    assert restored.optimizer_state["param_groups"][0]["lr"] == 7e-4


def test_streaming_contracts_and_weighted_telemetry_fail_closed():
    module = build_causal_lm(12, 16, 1, 2, 8)
    enc = _batch(n=3)
    before = _params(module).clone()
    streamed, telemetry = stream_fit(module, [enc, enc], lr=1.0e-3, report_every=1)
    assert not torch.equal(_params(streamed.module), before)
    assert telemetry[1] == 2 * len(enc[1])
    assert streamed.optimizer_state is not None

    module = build_causal_lm(12, 16, 1, 2, 8)
    accumulator = _acc(module)
    for bad_enc, bad_weights in (
        ((np.zeros((0, 8)), np.zeros(0)), None),
        ((enc[0], np.array([-1, 1, 2])), None),
        ((enc[0], np.array([12, 1, 2])), None),
        ((enc[0], enc[1]), np.array([1.0, -1.0, 1.0])),
        ((enc[0], enc[1]), np.array([1.0, np.inf, 1.0])),
        ((enc[0], enc[1]), np.zeros(3)),
    ):
        with pytest.raises(ValueError):
            accumulator.seq_update(bad_enc, bad_weights, None)

    weighted = _acc(copy.deepcopy(module))
    weights = np.array([0.0, 1.0, 3.0])
    weighted.seq_update(enc, weights, None)
    state = weighted.value()
    assert state.effective_weight == 4.0
    assert state.rows == 3
    assert state.loss_sum == pytest.approx(state.mean_loss * 4.0)

    with pytest.raises(ValueError, match="report_every"):
        stream_fit(copy.deepcopy(module), [enc], report_every=0)
    with pytest.raises(ValueError, match="no training batches"):
        stream_fit(copy.deepcopy(module), [], report_every=1)

    root = _acc(copy.deepcopy(module))
    changed_module = copy.deepcopy(module)
    with torch.no_grad():
        next(changed_module.parameters()).add_(0.01)
    changed = _acc(changed_module)
    changed.seq_update(enc, None, None)
    with pytest.raises(ValueError, match="different model revisions"):
        root.combine(changed.value())
