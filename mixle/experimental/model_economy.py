"""P14 (experimental, speculative) -- model economies: trading verified components beats isolation.

Two agents hold complementary training data. A buyer may adopt coefficients offered by the other agent,
but component selection is performed on a buyer-local validation split and every reported MSE is measured
later on an untouched final test split. The full-data-sharing oracle fits the exact union of the two
agents' training rows; it does not receive a separately generated, easier design.

Every train/selection/test matrix is sampled from the declared ``iid_standard_normal_scaled`` design
distribution before the non-domain columns of training rows are masked. This construction works for
underdetermined and overdetermined row/feature configurations and keeps isolation, trade, and oracle
candidates on one data-generating process.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

__all__ = ["EconomyReport", "run_economy"]

_DESIGN_DISTRIBUTION = "iid_standard_normal_scaled"


def _exact_positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive exact integer.")
    return int(value)


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _as_regression_data(x: Any, y: Any, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float)
    if design.ndim != 2 or design.shape[0] == 0 or design.shape[1] == 0:
        raise ValueError(f"{name} design must be a non-empty two-dimensional matrix.")
    if target.shape != (design.shape[0],):
        raise ValueError(f"{name} target must have shape {(design.shape[0],)}.")
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(target)):
        raise ValueError(f"{name} data must be finite.")
    return design, target


def _fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design, target = _as_regression_data(x, y, name="fit")
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    if coef.shape != (design.shape[1],) or not np.all(np.isfinite(coef)):
        raise RuntimeError("least-squares fit returned invalid coefficients.")
    return coef


def _mse(coef: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    design, target = _as_regression_data(x, y, name="mse")
    coefficient = np.asarray(coef, dtype=float)
    if coefficient.shape != (design.shape[1],) or not np.all(np.isfinite(coefficient)):
        raise ValueError(f"coef must be a finite vector of shape {(design.shape[1],)}.")
    result = float(np.mean((target - design @ coefficient) ** 2))
    if not math.isfinite(result):
        raise RuntimeError("MSE computation returned a non-finite value.")
    return result


@dataclass(frozen=True)
class EconomyReport:
    """Final-test report plus the separate selection receipt."""

    isolation_mse: float
    trade_mse: float
    oracle_mse: float
    oracle_gain: float
    trade_gain: float
    recovered_fraction: float
    adopted: int
    selection_isolation_mse: float
    selection_trade_mse: float
    n_selection: int
    n_test: int
    selection_data_digest: str
    test_data_digest: str
    design_distribution: str = _DESIGN_DISTRIBUTION


def _draw_design(rng: np.random.Generator, n_rows: int, n_features: int) -> np.ndarray:
    return rng.standard_normal((n_rows, n_features)) / math.sqrt(n_features)


def _data_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(repr(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _agent(
    rng: np.random.Generator,
    n_features: int,
    domain_cols: tuple[int, ...],
    coef_true: np.ndarray,
    n_train: int,
    n_selection: int,
    n_test: int,
    noise: float = 0.1,
) -> tuple[np.ndarray, ...]:
    """Generate independent train, selection, and final-test splits from one declared design law."""
    x_train = _draw_design(rng, n_train, n_features)
    mask = np.ones(n_features, dtype=bool)
    mask[list(domain_cols)] = False
    x_train[:, mask] = 0.0
    y_train = x_train @ coef_true + noise * rng.standard_normal(n_train)

    x_selection = _draw_design(rng, n_selection, n_features)
    y_selection = x_selection @ coef_true + noise * rng.standard_normal(n_selection)
    x_test = _draw_design(rng, n_test, n_features)
    y_test = x_test @ coef_true + noise * rng.standard_normal(n_test)
    return x_train, y_train, x_selection, y_selection, x_test, y_test


def _verify_and_adopt(
    buyer_coef: np.ndarray,
    seller_coef: np.ndarray,
    x_selection: np.ndarray,
    y_selection: np.ndarray,
    *,
    tol: float = 1e-9,
) -> tuple[np.ndarray, int]:
    """Select offered components using validation data that is never reused for final reporting."""
    design, target = _as_regression_data(x_selection, y_selection, name="selection")
    buyer = np.asarray(buyer_coef, dtype=float)
    seller = np.asarray(seller_coef, dtype=float)
    expected = (design.shape[1],)
    if buyer.shape != expected or seller.shape != expected:
        raise ValueError(f"buyer_coef and seller_coef must have shape {expected}.")
    if not np.all(np.isfinite(buyer)) or not np.all(np.isfinite(seller)):
        raise ValueError("buyer_coef and seller_coef must be finite.")
    tol = _finite_nonnegative(tol, "tol")

    coef = buyer.copy()
    adopted = 0
    current_mse = _mse(coef, design, target)
    for j, offered in enumerate(seller):
        if abs(offered) < 1e-6:
            continue
        trial = coef.copy()
        trial[j] = offered
        trial_mse = _mse(trial, design, target)
        if trial_mse < current_mse - tol:
            coef = trial
            current_mse = trial_mse
            adopted += 1
    return coef, adopted


def _domain_columns(value: Any, *, name: str, n_features: int) -> tuple[int, ...]:
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of exact integer column indices.") from exc
    if not values:
        raise ValueError(f"{name} must contain at least one column.")
    if any(isinstance(col, (bool, np.bool_)) or not isinstance(col, Integral) for col in values):
        raise TypeError(f"{name} must contain exact integer column indices.")
    columns = tuple(int(col) for col in values)
    if len(set(columns)) != len(columns) or any(col < 0 or col >= n_features for col in columns):
        raise ValueError(f"{name} must contain unique indices in [0, {n_features}).")
    return columns


def run_economy(
    *,
    n_features: int = 10,
    cols_a: Any = (0, 1, 2),
    cols_b: Any = (5, 6, 7),
    n_train: int = 400,
    n_selection: int | None = None,
    n_test: int = 400,
    noise: float = 0.1,
    seed: int = 0,
) -> EconomyReport:
    """Measure isolation, validation-selected component trade, and exact training-data sharing."""
    n_features = _exact_positive_int(n_features, "n_features")
    n_train = _exact_positive_int(n_train, "n_train")
    n_test = _exact_positive_int(n_test, "n_test")
    n_selection = n_test if n_selection is None else _exact_positive_int(n_selection, "n_selection")
    noise = _finite_nonnegative(noise, "noise")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError("seed must be a non-negative exact integer.")
    seed = int(seed)
    cols_a = _domain_columns(cols_a, name="cols_a", n_features=n_features)
    cols_b = _domain_columns(cols_b, name="cols_b", n_features=n_features)
    if set(cols_a) & set(cols_b):
        raise ValueError("cols_a and cols_b must be disjoint complementary domains.")

    seed_sequences = np.random.SeedSequence(seed).spawn(3)
    truth_rng, rng_a, rng_b = (np.random.default_rng(sequence) for sequence in seed_sequences)
    coef_true = np.zeros(n_features)
    for column in (*cols_a, *cols_b):
        coef_true[column] = truth_rng.uniform(1.5, 2.5) * truth_rng.choice((-1.0, 1.0))

    xa_tr, ya_tr, xa_sel, ya_sel, xa_te, ya_te = _agent(
        rng_a,
        n_features,
        cols_a,
        coef_true,
        n_train,
        n_selection,
        n_test,
        noise,
    )
    xb_tr, yb_tr, xb_sel, yb_sel, xb_te, yb_te = _agent(
        rng_b,
        n_features,
        cols_b,
        coef_true,
        n_train,
        n_selection,
        n_test,
        noise,
    )

    coef_a = _fit(xa_tr, ya_tr)
    coef_b = _fit(xb_tr, yb_tr)

    selection_isolation = 0.5 * (_mse(coef_a, xa_sel, ya_sel) + _mse(coef_b, xb_sel, yb_sel))
    coef_a_traded, ad_a = _verify_and_adopt(coef_a, coef_b, xa_sel, ya_sel)
    coef_b_traded, ad_b = _verify_and_adopt(coef_b, coef_a, xb_sel, yb_sel)
    selection_trade = 0.5 * (_mse(coef_a_traded, xa_sel, ya_sel) + _mse(coef_b_traded, xb_sel, yb_sel))

    # Final metrics use only the untouched final-test splits.
    isolation = 0.5 * (_mse(coef_a, xa_te, ya_te) + _mse(coef_b, xb_te, yb_te))
    trade = 0.5 * (_mse(coef_a_traded, xa_te, ya_te) + _mse(coef_b_traded, xb_te, yb_te))

    # The oracle receives exactly what "full data sharing" promises: both original training sets.
    oracle_coef = _fit(np.vstack((xa_tr, xb_tr)), np.concatenate((ya_tr, yb_tr)))
    oracle = 0.5 * (_mse(oracle_coef, xa_te, ya_te) + _mse(oracle_coef, xb_te, yb_te))

    oracle_gain = isolation - oracle
    trade_gain = isolation - trade
    recovered_fraction = trade_gain / oracle_gain if oracle_gain > 1e-12 else 0.0
    return EconomyReport(
        isolation_mse=isolation,
        trade_mse=trade,
        oracle_mse=oracle,
        oracle_gain=oracle_gain,
        trade_gain=trade_gain,
        recovered_fraction=recovered_fraction,
        adopted=ad_a + ad_b,
        selection_isolation_mse=selection_isolation,
        selection_trade_mse=selection_trade,
        n_selection=n_selection,
        n_test=n_test,
        selection_data_digest=_data_digest(xa_sel, ya_sel, xb_sel, yb_sel),
        test_data_digest=_data_digest(xa_te, ya_te, xb_te, yb_te),
    )
