"""P14 (experimental, speculative) -- model economies: trading verified components beats isolation.

Two agents hold complementary data: agent A's training data only exercises its domain's components,
agent B's only its own. Neither can fit the shared task family alone. In a **market** the agents
trade fitted components (columns + coefficients) rather than data, and -- crucially -- the *buyer
verifies* each offered component's gain on its own held-out set, so it never has to trust the
seller's claim (P5's certificates make a component's quality checkable by the buyer). The receipt
is the price signal: a component is adopted iff it measurably reduces the buyer's held-out error.

The card's first experiment, made exact: measure joint held-out gain under (a) isolation,
(b) verified component trade, (c) full data sharing (the oracle). Trade must recover >= 50% of the
oracle's gain over isolation while exchanging models only, never data.

Exploratory ``mixle.experimental`` code (P14 card).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coef


def _mse(coef: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y - x @ coef) ** 2))


@dataclass
class EconomyReport:
    isolation_mse: float
    trade_mse: float
    oracle_mse: float
    oracle_gain: float  # isolation - oracle
    trade_gain: float  # isolation - trade
    recovered_fraction: float  # trade_gain / oracle_gain
    adopted: int  # components the buyers verified and adopted


def _agent(rng, n_features, domain_cols, coef_true, n_train, n_test, noise=0.1):
    """One agent: training data exercises only its domain columns; the held-out task uses all columns."""
    q, _ = np.linalg.qr(rng.standard_normal((n_train + n_test, n_features)))
    x_full = q
    # Training rows: zero out non-domain columns so the agent can only estimate its own domain.
    x_train = x_full[:n_train].copy()
    mask = np.ones(n_features, dtype=bool)
    mask[list(domain_cols)] = False
    x_train[:, mask] = 0.0
    y_train = x_train @ coef_true + noise * rng.standard_normal(n_train)
    # Held-out task from the shared family: all columns active.
    x_test = x_full[n_train:]
    y_test = x_test @ coef_true + noise * rng.standard_normal(n_test)
    return x_train, y_train, x_test, y_test


def _verify_and_adopt(buyer_coef, seller_coef, x_test, y_test, *, tol=1e-9):
    """Buyer adopts each seller coefficient only if it reduces the buyer's OWN held-out error."""
    coef = buyer_coef.copy()
    adopted = 0
    for j in range(len(seller_coef)):
        if abs(seller_coef[j]) < 1e-6:
            continue  # seller has nothing meaningful on this column
        trial = coef.copy()
        trial[j] = seller_coef[j]
        if _mse(trial, x_test, y_test) < _mse(coef, x_test, y_test) - tol:
            coef = trial
            adopted += 1
    return coef, adopted


def run_economy(
    *, n_features: int = 10, cols_a=(0, 1, 2), cols_b=(5, 6, 7), n_train: int = 400, n_test: int = 400, seed: int = 0
) -> EconomyReport:
    """Run the two-agent market and measure isolation vs verified trade vs the data-sharing oracle."""
    rng = np.random.default_rng(seed)
    coef_true = np.zeros(n_features)
    for c in (*cols_a, *cols_b):
        coef_true[c] = rng.uniform(1.5, 2.5) * rng.choice([-1.0, 1.0])

    xa_tr, ya_tr, xa_te, ya_te = _agent(rng, n_features, cols_a, coef_true, n_train, n_test)
    xb_tr, yb_tr, xb_te, yb_te = _agent(rng, n_features, cols_b, coef_true, n_train, n_test)

    coef_a = _fit(xa_tr, ya_tr)  # A recovers its domain (cols_a), ~0 elsewhere
    coef_b = _fit(xb_tr, yb_tr)

    # (a) isolation: each agent uses its own model on its held-out task.
    iso = 0.5 * (_mse(coef_a, xa_te, ya_te) + _mse(coef_b, xb_te, yb_te))

    # (b) trade: each agent buys the other's verified-useful components (models only, never data).
    coef_a_traded, ad_a = _verify_and_adopt(coef_a, coef_b, xa_te, ya_te)
    coef_b_traded, ad_b = _verify_and_adopt(coef_b, coef_a, xb_te, yb_te)
    trade = 0.5 * (_mse(coef_a_traded, xa_te, ya_te) + _mse(coef_b_traded, xb_te, yb_te))

    # (c) oracle: full data sharing -- each agent fits on all-columns-active data.
    xa_full, ya_full, _, _ = _agent(rng, n_features, range(n_features), coef_true, n_train, 1)
    xb_full, yb_full, _, _ = _agent(rng, n_features, range(n_features), coef_true, n_train, 1)
    oracle = 0.5 * (_mse(_fit(xa_full, ya_full), xa_te, ya_te) + _mse(_fit(xb_full, yb_full), xb_te, yb_te))

    oracle_gain = iso - oracle
    trade_gain = iso - trade
    return EconomyReport(
        isolation_mse=iso,
        trade_mse=trade,
        oracle_mse=oracle,
        oracle_gain=oracle_gain,
        trade_gain=trade_gain,
        recovered_fraction=trade_gain / oracle_gain if oracle_gain > 1e-12 else 0.0,
        adopted=ad_a + ad_b,
    )
