"""Focused contracts for declarative optimization programs."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.program import (  # noqa: E402
    Program,
    ReplayBuffer,
    _mgda_weights,
    alternate,
    bilevel,
    constrain,
    fisher_diagonal,
    fit,
    lora,
    maximize,
    minimize,
    pareto,
    replay,
    weighted,
)

pytestmark = pytest.mark.experimental


def test_maximized_primal_uses_sign_correct_constraint_without_mutating_move():
    theta = torch.tensor(0.0, requires_grad=True)
    objective = lambda: -((theta - 5.0) ** 2)
    move = maximize(objective, over=[theta])

    program = fit(
        move,
        constraints=[constrain(lambda: theta, 2.0, "<=")],
        steps=1_000,
        lr=0.03,
    )

    assert theta.item() == pytest.approx(2.0, abs=0.08)
    assert move.objective is objective
    assert program.constraint_state is not None
    assert program.constraint_state.primal_move_index == 0
    assert program.constraint_state.multiplier_values()[0] >= 0

    # Reusing the same move must not stack an earlier augmented closure.
    fit(move, constraints=[constrain(lambda: theta, 2.0, "<=")], steps=500, lr=0.03)
    assert theta.item() == pytest.approx(2.0, abs=0.1)
    assert move.objective is objective


def test_constraints_select_first_trainable_scalar_move_and_reject_absence():
    class CounterMove:
        def __init__(self):
            self.steps = 0

        def _step(self):
            self.steps += 1

    counter = CounterMove()
    theta = torch.tensor(0.0, requires_grad=True)
    program = fit(
        alternate(counter, minimize(lambda: (theta - 3.0) ** 2, [theta])),
        constraints=[constrain(lambda: theta, 1.0)],
        steps=600,
        lr=0.03,
    )
    assert counter.steps == 600
    assert theta.item() == pytest.approx(1.0, abs=0.1)
    assert program.constraint_state.primal_move_index == 1

    with pytest.raises(ValueError, match="gradient Move"):
        fit(Program([CounterMove()]), constraints=[constrain(lambda: 0.0)], steps=1)


def test_lora_and_mgda_auxiliary_state_inherit_dtype_and_device():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2, dtype=torch.float64))
    dtype = model[0].weight.dtype
    device = model[0].weight.device
    params = lora(model, rank=2)
    assert all(param.dtype == dtype for param in params)
    assert all(param.device == device for param in params)

    grads = [
        [torch.tensor([1.0, 0.0], dtype=torch.float64)],
        [torch.tensor([0.0, 1.0], dtype=torch.float64)],
    ]
    weights = _mgda_weights(grads, torch)
    assert weights.dtype == grads[0][0].dtype
    assert weights.device == grads[0][0].device
    assert weights.sum().item() == pytest.approx(1.0)


def test_fisher_allows_parameters_disconnected_from_a_batch():
    class PartlyUsed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Parameter(torch.tensor(1.0))
            self.unused = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, x):
            return self.used * x

    model = PartlyUsed()
    diagonal = fisher_diagonal(model, [torch.ones(4, 1)], kind="regression")
    assert diagonal[0].item() > 0
    assert diagonal[1].item() == 0


def test_empty_or_non_differentiable_program_inputs_fail_clearly():
    theta = torch.tensor(0.0, requires_grad=True)
    with pytest.raises(ValueError, match="at least one"):
        weighted([], over=[theta])
    with pytest.raises(ValueError, match="at least one objective"):
        pareto([], over=[theta])
    with pytest.raises(ValueError, match="empty buffer"):
        replay(lambda chunk: chunk.sum(), ReplayBuffer())()
    with pytest.raises(ValueError, match="at least one move"):
        fit(Program([]), steps=1)
    with pytest.raises(TypeError, match="differentiable scalar tensor"):
        fit(minimize(lambda: 0.0, [theta]), steps=1)


def test_empty_meta_task_collection_fails_clearly():
    model = torch.nn.Linear(1, 1)
    loss = lambda forward, batch: forward(batch).sum()
    move = bilevel(model, loss, loss, lambda: [])
    with pytest.raises(ValueError, match="at least one"):
        fit(move, steps=1)


def test_fisher_validates_kind_and_nonempty_batches():
    model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="kind"):
        fisher_diagonal(model, [torch.ones(1, 1)], kind="other")
    with pytest.raises(ValueError, match="at least one"):
        fisher_diagonal(model, [], kind="regression")
