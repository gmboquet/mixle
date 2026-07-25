"""Fast split-integrity and shape contracts for the model-economy experiment."""

import math

import pytest

from mixle.experimental.model_economy import run_economy

pytestmark = pytest.mark.experimental


def test_selection_and_final_test_are_separate_declared_splits():
    report = run_economy(n_train=20, n_selection=11, n_test=13, seed=4)
    assert report.n_selection == 11
    assert report.n_test == 13
    assert report.design_distribution == "iid_standard_normal_scaled"
    assert report.selection_data_digest != report.test_data_digest
    assert report.selection_trade_mse <= report.selection_isolation_mse
    assert all(
        math.isfinite(value)
        for value in (
            report.isolation_mse,
            report.trade_mse,
            report.oracle_mse,
            report.selection_isolation_mse,
            report.selection_trade_mse,
        )
    )


@pytest.mark.parametrize("n_train,n_features", [(1, 10), (3, 20), (20, 3)])
def test_rectangular_designs_do_not_depend_on_qr_shape(n_train, n_features):
    if n_features < 6:
        cols_a, cols_b = (0,), (1,)
    else:
        cols_a, cols_b = (0, 1), (4, 5)
    report = run_economy(
        n_train=n_train,
        n_features=n_features,
        cols_a=cols_a,
        cols_b=cols_b,
        n_selection=8,
        n_test=9,
        seed=2,
    )
    assert math.isfinite(report.trade_mse)
    assert math.isfinite(report.oracle_mse)


def test_default_selection_size_is_independent_test_size_not_the_same_rows():
    report = run_economy(n_train=10, n_test=7, seed=1)
    assert report.n_selection == report.n_test == 7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_features": 0},
        {"n_train": 0},
        {"n_selection": 0},
        {"n_test": 0},
        {"noise": float("nan")},
        {"cols_a": (0, 0)},
        {"cols_a": (0,), "cols_b": (0,)},
        {"seed": 1.5},
    ],
)
def test_invalid_experiment_controls_fail_closed(kwargs):
    with pytest.raises((TypeError, ValueError)):
        run_economy(**kwargs)
