"""Fast methodological contracts for the long-context referee."""

import copy
import math

import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.context_spine import SlidingWindowSpine
from mixle.experimental.long_context_eval import (
    _train_and_probe,
    comparison_table,
    copy_suite,
    evaluate,
)

pytestmark = pytest.mark.experimental


class RecordingMechanism(torch.nn.Module):
    vocab = 5

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.calls = []

    def init_state(self, batch_size, *, device="cpu"):
        return 0

    def step(self, state, chunk):
        x, _ = chunk
        self.calls.append((torch.is_grad_enabled(), x.shape[1]))
        return state + x.shape[1], self.weight.square() + math.log(self.vocab)

    def detach(self, state):
        return state


def _model(seed=0, d_model=8):
    torch.manual_seed(seed)
    return SlidingWindowSpine(7, d_model=d_model, n_layer=1, n_head=2, window=4)


def _quick_eval(model, **overrides):
    kwargs = {
        "ranges": (2,),
        "state_budget_bytes": 100_000,
        "seed": 4,
        "n_train_steps": 1,
        "n_eval_trials": 1,
        "perplexity_steps": 1,
        "curriculum_rounds": 0,
        "compute_budget_flops": 1_000_000,
    }
    kwargs.update(overrides)
    return evaluate(model, **kwargs)


def test_controlled_probe_training_backpropagates_only_through_dependency_position():
    model = RecordingMechanism()
    model.eval()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    result = _train_and_probe(
        model,
        opt,
        copy_suite,
        distance=4,
        vocab=model.vocab,
        chunk_size=2,
        n_train_steps=1,
        n_eval_trials=1,
        rng=__import__("numpy").random.RandomState(2),
    )
    assert [length for grad_enabled, length in model.calls if grad_enabled] == [1]
    assert "accuracy" not in result
    assert result["metric"] == "loss_threshold_success_rate"
    assert model.training is False


def test_evaluate_does_not_mutate_source_model_and_cells_are_order_independent():
    source = _model(seed=8)
    before = copy.deepcopy(source.state_dict())
    forward = evaluate(
        source,
        ranges=(2, 4),
        state_budget_bytes=100_000,
        seed=9,
        n_train_steps=1,
        n_eval_trials=1,
        perplexity_steps=1,
        curriculum_rounds=0,
    )
    reverse = evaluate(
        source,
        ranges=(4, 2),
        state_budget_bytes=100_000,
        seed=9,
        n_train_steps=1,
        n_eval_trials=1,
        perplexity_steps=1,
        curriculum_rounds=0,
    )
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, before[name])
    for distance in (2, 4):
        assert forward["suites"][distance] == reverse["suites"][distance]


def test_shared_training_budget_is_enforced_and_recorded():
    result = _quick_eval(_model(), suite_training_budget_flops=1.0)
    assert result["training_flops_used"] <= result["suite_training_budget_flops"]
    assert all(
        row["training_steps"] == 0
        for suite in result["suites"].values()
        for row in (suite["needle"], suite["copy"], suite["multi_hop"], suite["perplexity"])
    )


def test_comparison_rejects_unmatched_compute_and_exceeded_state():
    small = _quick_eval(_model(d_model=8))
    large = _quick_eval(_model(d_model=16))
    with pytest.raises(ValueError, match="training FLOPs differ"):
        comparison_table({"small": small, "large": large})

    over = _quick_eval(_model(), state_budget_bytes=0)
    with pytest.raises(ValueError, match="exceeded"):
        comparison_table(over)


def test_fractional_ranges_are_not_silently_truncated():
    with pytest.raises(ValueError, match="exact integer"):
        _quick_eval(_model(), ranges=(2.5,))
