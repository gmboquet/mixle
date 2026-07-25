"""Adverse lifecycle, numerical, and serialization contracts for energy models."""

from __future__ import annotations

import pickle
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.energy import (  # noqa: E402
    EnergyModel,
    EnergyModelAccumulator,
    EnergyModelEstimator,
    build_convex_energy_net,
    build_energy_net,
)


class _RecordingEnergy(torch.nn.Module):
    def __init__(self, dim: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.linear = torch.nn.Linear(dim, 1)
        self.log_norm = torch.nn.Parameter(torch.zeros(()))
        self.forward_modes: list[bool] = []

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_modes.append(self.training)
        return self.linear(x).squeeze(-1)


class _BadEnergy(torch.nn.Module):
    def __init__(self, value: float, *, column: bool = False) -> None:
        super().__init__()
        self.dim = 1
        self.log_norm = torch.nn.Parameter(torch.zeros(()))
        self.value = value
        self.column = column

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        shape = (x.shape[0], 1) if self.column else (x.shape[0],)
        return torch.full(shape, self.value, dtype=x.dtype, device=x.device)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("m_steps", 0),
        ("m_steps", 1.5),
        ("lr", 0.0),
        ("lr", np.nan),
        ("noise_ratio", 0),
        ("noise_ratio", 1.5),
        ("langevin_steps", 0),
        ("langevin_step", -0.1),
        ("langevin_step", np.inf),
    ],
)
def test_energy_model_rejects_invalid_controls(name: str, value: object) -> None:
    options = {name: value}
    with pytest.raises((TypeError, ValueError), match=name):
        EnergyModel(build_energy_net(1), **options)


def test_energy_estimator_rejects_invalid_controls_too() -> None:
    with pytest.raises(ValueError, match="noise_ratio"):
        EnergyModelEstimator(build_energy_net(1), noise_ratio=0)


@pytest.mark.parametrize(
    ("x", "weights", "message"),
    [
        (np.ones((3, 2)), np.ones(2), "shape"),
        (np.ones((3, 2)), np.asarray([1.0, -1.0, 1.0]), "non-negative"),
        (np.ones((3, 2)), np.asarray([1.0, np.nan, 1.0]), "finite"),
        (np.ones((3, 2)), np.zeros(3), "positive finite total"),
        (np.asarray([[1.0, 1.0], [np.inf, 1.0]]), np.ones(2), "finite"),
        (np.ones((3, 1)), np.ones(3), "exactly 2 features"),
    ],
)
def test_energy_estimator_rejects_invalid_aligned_data(
    x: np.ndarray,
    weights: np.ndarray,
    message: str,
) -> None:
    estimator = EnergyModelEstimator(build_energy_net(2), m_steps=1)
    with pytest.raises(ValueError, match=message):
        estimator.estimate(None, (x, weights))


def test_energy_accumulator_rejects_invalid_responsibilities() -> None:
    accumulator = EnergyModelAccumulator()
    with pytest.raises(ValueError, match="non-negative"):
        accumulator.update([1.0], -1.0, None)
    with pytest.raises(ValueError, match="shape"):
        accumulator.seq_update(np.ones((2, 1)), np.ones(3), None)


def test_energy_scoring_sampling_and_fitting_restore_module_modes() -> None:
    module = _RecordingEnergy()
    model = EnergyModel(module, m_steps=1, noise_ratio=1, langevin_steps=1)

    module.train()
    model.seq_log_density(np.ones((2, 2)))
    assert module.forward_modes[-1] is False
    assert module.training is True

    model.sampler(0).sample(2)
    assert module.forward_modes[-1] is False
    assert module.training is True

    estimator = model.estimator()
    module.eval()
    estimator.estimate(None, (np.ones((3, 2)), np.ones(3)))
    assert module.forward_modes[-1] is True
    assert module.training is False


def test_energy_estimator_reuses_adam_continuation_state() -> None:
    estimator = EnergyModelEstimator(_RecordingEnergy(1), m_steps=1, noise_ratio=1)
    stats = (np.asarray([[-1.0], [0.0], [1.0]]), np.ones(3))
    estimator.estimate(None, stats)
    optimizer = estimator._optimizer
    assert optimizer is not None
    first_steps = [float(state["step"]) for state in optimizer.state.values()]

    estimator.estimate(None, stats)
    assert estimator._optimizer is optimizer
    second_steps = [float(state["step"]) for state in optimizer.state.values()]
    assert all(after > before for before, after in zip(first_steps, second_steps, strict=True))


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_energy_sampler_rejects_invalid_sizes(size: object) -> None:
    sampler = EnergyModel(build_energy_net(1), langevin_steps=1).sampler(0)
    with pytest.raises((TypeError, ValueError), match="size"):
        sampler.sample(size)


@pytest.mark.parametrize(
    "module",
    [
        _BadEnergy(np.nan),
        _BadEnergy(np.inf),
        _BadEnergy(0.0, column=True),
    ],
)
def test_energy_wrapper_rejects_nonfinite_or_wrong_shape_outputs(module: torch.nn.Module) -> None:
    model = EnergyModel(module, langevin_steps=1)
    with pytest.raises(ValueError, match="energy"):
        model.seq_log_density(np.ones((2, 1)))


def test_log_norm_uses_negative_log_partition_convention() -> None:
    module = _BadEnergy(0.0)
    with torch.no_grad():
        module.log_norm.fill_(-0.5 * np.log(2.0 * np.pi))
    score = EnergyModel(module).log_density([0.0])
    assert score == pytest.approx(-0.5 * np.log(2.0 * np.pi))


def test_convex_energy_full_module_round_trips_in_fresh_process() -> None:
    module = build_convex_energy_net(2, hidden=4, layers=3)
    payload = pickle.dumps(module, protocol=5)
    script = """
import pickle
import sys
import torch

module = pickle.loads(sys.stdin.buffer.read())
output = module.energy(torch.ones(3, 2))
assert output.shape == (3,)
assert torch.isfinite(output).all()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=payload,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
