"""Adversarial contracts for quantized hidden Markov models."""

from types import SimpleNamespace

import numpy as np
import pytest

import mixle.stats.latent.quantized_hidden_markov_model as quantized_hmm
from mixle.stats.latent.quantized_hidden_markov_model import (
    QuantizedHiddenMarkovEstimator,
    QuantizedHiddenMarkovModelDistribution,
    QuantizedHMMOptimizationError,
)


def _distribution(**overrides):
    arguments = {
        "theta": 0.5,
        "levels": ["a", "b"],
        "transition_exponents": [[0, 1], [1, 0]],
        "emission_exponents": [[0, 2], [2, 0]],
        "initial_exponents": [0, 1],
        "use_numba": False,
    }
    arguments.update(overrides)
    return QuantizedHiddenMarkovModelDistribution(**arguments)


def _statistics():
    return (
        2,
        np.array([1.0, 1.0]),
        np.array([3.0, 3.0]),
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        ({"b": 1.0, "a": 2.0}, {"b": 2.0, "a": 1.0}),
        None,
    )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("transition_exponents", [[0.9, 1], [1, 0]]),
        ("emission_exponents", [[0, 1], [0.7, 0]]),
        ("initial_exponents", [0.4, 1]),
        ("transition_exponents", [[False, 1], [1, 0]]),
    ],
)
def test_exponents_require_exact_non_boolean_integers(parameter, value):
    with pytest.raises(TypeError, match="exact non-boolean integers"):
        _distribution(**{parameter: value})


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("transition_exponents", [[-2, 1], [1, 0]]),
        ("emission_exponents", [[0, -2], [2, 0]]),
        ("initial_exponents", [-2, 1]),
    ],
)
def test_only_minus_one_is_a_structural_zero(parameter, value):
    with pytest.raises(ValueError, match="structural-zero sentinel"):
        _distribution(**{parameter: value})


def test_exponents_obey_cap_and_are_owned_by_distribution():
    transition = np.array([[0, 1], [1, 0]], dtype=np.int64)
    emission = np.array([[0, 2], [2, 0]], dtype=np.int64)
    initial = np.array([0, 1], dtype=np.int64)
    model = _distribution(
        transition_exponents=transition,
        emission_exponents=emission,
        initial_exponents=initial,
        k_max=2,
    )

    transition[0, 0] = 2
    emission[0, 0] = 2
    initial[0] = 2

    assert model.transition_exponents[0, 0] == 0
    assert model.emission_exponents[0, 0] == 0
    assert model.initial_exponents[0] == 0
    with pytest.raises(ValueError, match="cannot exceed k_max"):
        _distribution(transition_exponents=[[0, 3], [1, 0]], k_max=2)


@pytest.mark.parametrize(
    "levels",
    [
        [],
        ["a", "a"],
        [1, True],
        [np.nan, "a"],
        [["unhashable"], "a"],
    ],
)
def test_levels_are_nonempty_unique_stable_values(levels):
    with pytest.raises((TypeError, ValueError)):
        _distribution(levels=levels)


@pytest.mark.parametrize(
    "arguments",
    [
        {"num_states": 2.0},
        {"num_states": True},
        {"pseudo_count": -1.0},
        {"pseudo_count": np.nan},
        {"k_max": 2.5},
        {"fixed_theta": 1.0},
        {"max_quant_its": 1},
        {"max_quant_its": True},
        {"split_nats": 0.0},
        {"split_nats": np.inf},
        {"split_collapsed": 1},
        {"use_numba": 1},
    ],
)
def test_estimator_controls_fail_closed(arguments):
    with pytest.raises((TypeError, ValueError)):
        QuantizedHiddenMarkovEstimator(**({"num_states": 2} | arguments))


@pytest.mark.parametrize(
    "statistics",
    [
        (2, [1.0], [3.0, 3.0], [[1.0, 1.0], [1.0, 1.0]], ({"a": 3.0}, {"b": 3.0}), None),
        (2, [1.0, 1.0], [3.0, np.nan], [[1.0, 1.0], [1.0, 1.0]], ({"a": 3.0}, {"b": 3.0}), None),
        (2, [1.0, 1.0], [3.0, 3.0], [[1.0, -1.0], [1.0, 1.0]], ({"a": 3.0}, {"b": 3.0}), None),
        (2, [1.0, 1.0], [3.0, 3.0], [[1.0, 1.0], [1.0, 1.0]], ({"a": 2.0}, {"b": 3.0}), None),
        (2, [1.0, 1.0], [4.0, 3.0], [[1.0, 1.0], [1.0, 1.0]], ({"a": 4.0}, {"b": 3.0}), None),
    ],
)
def test_estimator_rejects_malformed_sufficient_statistics(statistics):
    with pytest.raises((TypeError, ValueError)):
        QuantizedHiddenMarkovEstimator(2, fixed_theta=0.5).estimate(2, statistics)


def test_fixed_theta_fit_carries_a_complete_receipt_and_stable_levels():
    model = QuantizedHiddenMarkovEstimator(
        2,
        levels=["z"],
        fixed_theta=0.5,
        k_max=8,
        use_numba=False,
    ).estimate(2, _statistics())

    receipt = model.fit_diagnostics
    assert model.levels == ["z", "a", "b"]
    assert receipt is not None
    assert receipt.converged
    assert receipt.fixed_theta
    assert receipt.theta == 0.5
    assert np.isfinite(receipt.objective)
    assert receipt.selected_start == 0
    assert receipt.selected_start_converged
    assert receipt.iterations_per_start == (1,)
    assert receipt.termination_reasons == ("fixed_theta",)


def test_free_theta_fit_identifies_the_selected_restart():
    model = QuantizedHiddenMarkovEstimator(2, k_max=8, max_quant_its=8, use_numba=False).estimate(2, _statistics())

    receipt = model.fit_diagnostics
    assert receipt is not None
    assert 0 <= receipt.selected_start < 4
    assert receipt.selected_start_converged == receipt.converged_starts[receipt.selected_start]
    assert receipt.converged == receipt.selected_start_converged
    assert len(receipt.iterations_per_start) == 4
    assert len(receipt.converged_starts) == 4
    assert len(receipt.termination_reasons) == 4
    assert set(receipt.termination_reasons) <= {"fixed_point", "cycle", "iteration_budget"}
    assert receipt.scalar_optimizer_evaluations > 0
    assert np.isfinite(receipt.objective)


def test_scalar_optimizer_failure_raises_with_a_failure_receipt(monkeypatch):
    monkeypatch.setattr(
        quantized_hmm,
        "minimize_scalar",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            x=0.5,
            fun=1.0,
            message="injected failure",
        ),
    )

    with pytest.raises(QuantizedHMMOptimizationError, match="injected failure") as raised:
        QuantizedHiddenMarkovEstimator(2, k_max=8, use_numba=False).estimate(2, _statistics())

    receipt = raised.value.diagnostics
    assert not receipt.converged
    assert receipt.termination_reasons[-1] == "scalar_optimizer_failure"
    assert receipt.selected_start == -1
    assert not receipt.selected_start_converged
