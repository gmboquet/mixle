"""Focused likelihood and state-boundary contracts for selective scan."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from mixle.experimental.selective_scan import (  # noqa: E402
    SelectiveScan,
    SelectiveScanState,
    _scan_layer,
)

pytestmark = pytest.mark.experimental


def _model(**overrides):
    config = {"vocab": 5, "d_model": 4, "d_state": 3, "n_layer": 1, "expand": 1}
    config.update(overrides)
    return SelectiveScan(**config)


def test_log_density_is_token_sum_not_mean_loss():
    torch.manual_seed(0)
    model = _model()
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        _, mean_nll = model.step(model.init_state(1), (x, y))
        log_density = model.log_density(x, y)
    assert log_density.shape == (1,)
    assert log_density[0] == pytest.approx(float(-mean_nll * x.shape[1]))


def test_uniform_model_likelihood_is_additive_in_sequence_length():
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    short = torch.tensor([[0, 1, 2]])
    long = torch.tensor([[0, 1, 2, 3, 4, 0]])
    with torch.no_grad():
        short_ll = float(model.log_density(short, short)[0])
        long_ll = float(model.log_density(long, long)[0])
        duplicated_rows = model.log_density(short.repeat(2, 1), short.repeat(2, 1))
    assert short_ll == pytest.approx(-3 * math.log(model.vocab))
    assert long_ll == pytest.approx(-6 * math.log(model.vocab))
    assert float(duplicated_rows.sum()) == pytest.approx(2 * short_ll)


def test_initialized_state_is_complete_and_shape_checked():
    model = _model()
    state = model.init_state(2)
    assert state.batch_size == 2
    assert state.h[0].shape == (2, model.d_inner, model.d_state)
    assert torch.count_nonzero(state.h[0]) == 0

    bad = SelectiveScanState(h=[torch.zeros(2, 1, 1)], batch_size=2)
    tokens = torch.ones(2, 1, dtype=torch.long)
    with pytest.raises(ValueError, match=r"state\.h"):
        model.step(bad, (tokens, tokens))


def test_step_rejects_empty_mismatched_or_out_of_domain_chunks():
    model = _model()
    with pytest.raises(ValueError, match="non-empty"):
        model.step(
            model.init_state(1),
            (torch.empty(1, 0, dtype=torch.long), torch.empty(1, 0, dtype=torch.long)),
        )
    with pytest.raises(ValueError, match="batch_size"):
        tokens = torch.ones(2, 1, dtype=torch.long)
        model.step(model.init_state(1), (tokens, tokens))
    with pytest.raises(ValueError, match="token IDs"):
        model.step(model.init_state(1), (torch.tensor([[5]]), torch.tensor([[0]])))


def test_non_finite_dynamics_fail_before_scan_results_escape():
    model = _model()
    tokens = torch.ones(1, 1, dtype=torch.long)
    with torch.no_grad():
        model.A_log[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="A_log"):
        model.step(model.init_state(1), (tokens, tokens))


def test_scan_layer_rejects_empty_or_wrong_control_dimensions():
    model = _model()
    empty = torch.empty(1, 0, model.d_inner)
    with pytest.raises(ValueError, match="non-empty"):
        _scan_layer(
            empty,
            model.A_log[0],
            model.W_delta[0],
            model.W_B[0],
            model.W_C[0],
            model.D[0],
            None,
        )

    wrong_a = torch.zeros(model.d_inner + 1, model.d_state)
    u = torch.zeros(1, 1, model.d_inner)
    with pytest.raises(ValueError, match="A_log"):
        _scan_layer(
            u,
            wrong_a,
            model.W_delta[0],
            model.W_B[0],
            model.W_C[0],
            model.D[0],
            None,
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"vocab": 0}, "vocab"),
        ({"d_model": True}, "d_model"),
        ({"d_state": 0}, "d_state"),
        ({"n_layer": 0}, "n_layer"),
        ({"expand": 0}, "expand"),
    ],
)
def test_constructor_rejects_invalid_dimensions(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _model(**kwargs)


def test_log_density_rejects_empty_sequences():
    model = _model()
    empty = torch.empty(1, 0, dtype=torch.long)
    with pytest.raises(ValueError, match="non-empty"):
        model.log_density(empty, empty)
