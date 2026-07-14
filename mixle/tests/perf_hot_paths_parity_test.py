"""Exactness-parity tests for the hot-path performance changes (ledger E-1, E-2/G-3, E-3, I-2).

Every optimization here must be behavior-preserving: MVN scoring via the precomputed
inverse Cholesky factor must match the cho_solve reference to float precision (E-1), the
shared objective loop must evaluate ONE forward per Adam iteration while recording the
same history (E-2/G-3) and must recover from a NaN initial objective (I-2), and the
seq_encode chunk slicing must produce identical encodings for list and ndarray inputs
(E-3).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg as sla

from mixle.stats.compute.sequence import seq_encode
from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


# --------------------------------------------------------------------- E-1: MVN gemm parity
def test_mvn_seq_log_density_matches_cho_solve_reference() -> None:
    rng = np.random.RandomState(seed=3)
    for d in (2, 5, 16):
        a = rng.normal(size=(d, d))
        covar = a @ a.T + d * np.eye(d)
        mu = rng.normal(size=d)
        dist = MultivariateGaussianDistribution(mu, covar)
        x = rng.normal(size=(200, d))
        enc = dist.dist_to_encoder().seq_encode([row for row in x])
        got = np.asarray(dist.seq_log_density(enc), dtype=float)

        chol = sla.cho_factor(covar)
        diff = mu - x
        soln = sla.cho_solve(chol, diff.T).T
        const = -0.5 * (d * np.log(2.0 * np.pi) + np.linalg.slogdet(covar)[1])
        ref = const - 0.5 * (diff * soln).sum(axis=1)
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


# --------------------------------------------------------------------- E-2/G-3: one forward per iteration
def _torch_engine():
    torch = pytest.importorskip("torch")
    from mixle.engines import TorchEngine

    return torch, TorchEngine(dtype=torch.float64)


def test_objective_loop_runs_one_forward_per_adam_iteration() -> None:
    _torch, engine = _torch_engine()
    from mixle.inference.objectives import ObjectiveParameter, fit_parameter_objective

    calls = {"n": 0}

    def objective(p, enc, eng):
        calls["n"] += 1
        return -((p["loc"] - 3.0) ** 2)

    fit_parameter_objective([ObjectiveParameter("loc", 0.0)], objective, engine=engine, max_its=10, lr=0.1, tol=0.0)
    # one forward per step + the initial and final evaluations; the historical loop paid two
    # per step (>= 21 calls for 10 iterations)
    assert calls["n"] <= 12, calls["n"]


def test_nan_initial_objective_does_not_freeze_best_tracking() -> None:
    torch, engine = _torch_engine()
    from mixle.inference.objectives import ObjectiveParameter, fit_parameter_objective

    state = {"first": True}

    def objective(p, enc, eng):
        # NaN only at the very first (recorded, pre-step) evaluation: the L-BFGS path records a
        # standalone evaluation before each step, so the NaN lands in history[0] without ever
        # reaching a backward pass -- exactly the audit I-2 scenario
        if state["first"]:
            state["first"] = False
            return p["loc"] * torch.tensor(float("nan"), dtype=torch.float64)
        return -((p["loc"] - 3.0) ** 2)

    params, _value = fit_parameter_objective(
        [ObjectiveParameter("loc", 0.0)],
        objective,
        engine=engine,
        optimizer="lbfgs",
        max_its=30,
        lr=0.5,
        tol=0.0,
    )
    # under the pre-fix comparison a NaN incumbent never lost, so restore_best returned the
    # INITIAL parameters (loc = 0) no matter how far the objective later improved (audit I-2)
    assert abs(float(params["loc"]) - 3.0) < 1.0, float(params["loc"])


# --------------------------------------------------------------------- E-3: encode slicing parity
def _flatten_encoding(enc_chunks: list) -> np.ndarray:
    vals = []
    for _sz, enc in enc_chunks:
        vals.append(np.sort(np.asarray(enc, dtype=float).ravel()))
    return np.sort(np.concatenate(vals))


def test_seq_encode_slicing_matches_for_lists_and_arrays() -> None:
    rng = np.random.RandomState(seed=9)
    values = rng.normal(size=500)
    model = GaussianDistribution(mu=0.0, sigma2=1.0)
    for num_chunks in (1, 3):
        as_list = seq_encode(list(values), model=model, num_chunks=num_chunks)
        as_array = seq_encode(values, model=model, num_chunks=num_chunks)
        assert len(as_list) == len(as_array) == num_chunks
        assert sum(sz for sz, _ in as_list) == 500
        assert sum(sz for sz, _ in as_array) == 500
        np.testing.assert_allclose(_flatten_encoding(as_list), np.sort(values))
        np.testing.assert_allclose(_flatten_encoding(as_array), np.sort(values))


def test_seq_encode_chunks_partition_the_data_exactly() -> None:
    model = GaussianDistribution(mu=0.0, sigma2=1.0)
    data = [float(i) for i in range(11)]
    chunks = seq_encode(data, model=model, num_chunks=4)
    seen = np.sort(np.concatenate([np.asarray(enc, dtype=float).ravel() for _sz, enc in chunks]))
    np.testing.assert_allclose(seen, np.asarray(data))
