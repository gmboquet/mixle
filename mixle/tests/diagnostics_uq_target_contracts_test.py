"""Diagnostics and UQ must fail closed on non-finite or unsupported claims."""

import numpy as np
import pytest

from mixle.inference import ess, mcmc_summary, nuts, rhat, split_rhat, uq
from mixle.inference.target import _nuts_jax, _pool_chains


def _normal_target(theta):
    return -0.5 * float(theta @ theta), -theta


def test_ess_does_not_pool_away_between_chain_location_failure():
    chains = np.array([np.full(100, -5.0), np.full(100, 5.0)])[:, :, None]
    assert ess(chains)[0] < 5.0
    assert np.isinf(rhat(chains)[0])


@pytest.mark.parametrize(
    "call",
    [
        lambda: rhat(np.array([[0.0, np.nan], [0.0, 1.0]])),
        lambda: ess(np.array([0.0, np.inf])),
        lambda: split_rhat(np.ones((1, 10))),
        lambda: split_rhat(np.ones((2, 3))),
        lambda: mcmc_summary(np.array([[[0.0]], [[np.nan]]])),
    ],
)
def test_chain_diagnostics_reject_nonfinite_or_insufficient_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_sampler_results_use_rank_normalized_split_diagnostics():
    rng = np.random.RandomState(3)
    arrays = [rng.normal(size=(100, 2)) for _ in range(4)]
    result = _pool_chains(arrays, 2, 4, 123, 0.1, backend="test")
    assert result.extra["diagnostics"] == "rank_normalized_split"
    assert result.extra["target_evals_observed"]
    assert np.all(np.isfinite(result.rhat))


def test_sampler_pool_rejects_nonfinite_or_unequal_backend_outputs():
    with pytest.raises(RuntimeError, match="unequal"):
        _pool_chains([np.ones((4, 1)), np.ones((3, 1))], 1, 2, 2, 0.1, backend="test")
    with pytest.raises(RuntimeError, match="non-finite"):
        _pool_chains(
            [np.ones((4, 1)), np.array([[1.0], [1.0], [1.0], [np.nan]])],
            1,
            2,
            2,
            0.1,
            backend="test",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_samples": 0},
        {"warmup": -1},
        {"chains": 0},
        {"thin": 0},
        {"target_accept": 1.0},
        {"parallel": "unknown"},
        {"init": np.array([np.nan])},
    ],
)
def test_nuts_rejects_invalid_public_controls(kwargs):
    base = {"dim": 1, "num_samples": 4, "warmup": 0, "chains": 1}
    base.update(kwargs)
    with pytest.raises(ValueError):
        nuts(_normal_target, **base)


def test_jax_backend_rejects_controls_it_cannot_honor_before_importing_jax():
    with pytest.raises(NotImplementedError, match="mass"):
        _nuts_jax(lambda value: value, dim=1, num_samples=2, warmup=0, mass=np.array([1.0]))
    with pytest.raises(NotImplementedError, match="thin"):
        _nuts_jax(lambda value: value, dim=1, num_samples=2, warmup=0, thin=2)


def test_uncalibrated_semantic_uq_cannot_issue_a_confidence_decision():
    result = uq(lambda prompt: str(prompt))
    with pytest.raises(ValueError, match="calibrated"):
        result.confident("question")


def test_uq_rejects_nonfinite_predictions_and_calibration_values():
    with pytest.raises(ValueError, match="finite"):
        uq(lambda value: np.array([np.nan]), data=([1.0], [1.0]))
    with pytest.raises(ValueError, match="finite"):
        uq(lambda value: np.array([value]), data=([1.0], [np.inf]))


def test_torch_uq_preserves_nested_module_modes():
    torch = pytest.importorskip("torch")
    module = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Dropout(), torch.nn.Linear(2, 1))
    module.train()
    module[1].eval()
    before = [part.training for part in module.modules()]

    result = uq(module, data=([np.array([0.0]), np.array([1.0])], [0.0, 1.0]))
    result.interval(np.array([0.5]))

    assert [part.training for part in module.modules()] == before
