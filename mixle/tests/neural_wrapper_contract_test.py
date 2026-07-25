"""Contract tests for neural objective wrappers and structured-network builders."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.neural import (  # noqa: E402
    CategoricalClassificationNeuralNetwork,
    GaussianRegressionNeuralNetwork,
    PoissonRegressionNeuralNetwork,
    make_deep_set,
    make_monotonic_mlp,
)


class _ModeRecordingLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features)
        self.forward_modes: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_modes.append(self.training)
        return super().forward(x)


class _NonFiniteModule(torch.nn.Module):
    def __init__(self, out_features: int, value: float) -> None:
        super().__init__()
        self.out_features = out_features
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (x.shape[0], self.out_features),
            self.value,
            dtype=x.dtype,
            device=x.device,
        )


@pytest.mark.parametrize("noise", [True, 0.0, -1.0, np.nan, np.inf])
def test_gaussian_wrapper_rejects_invalid_noise(noise: object) -> None:
    with pytest.raises((TypeError, ValueError), match="noise"):
        GaussianRegressionNeuralNetwork(torch.nn.Linear(1, 1), noise=noise)


def test_gaussian_wrapper_requires_exact_finite_geometry() -> None:
    model = GaussianRegressionNeuralNetwork(torch.nn.Linear(2, 2))
    x = np.ones((3, 2))

    with pytest.raises(ValueError, match="exactly match"):
        model.log_likelihood(x, np.ones((3, 1)))
    with pytest.raises(ValueError, match="row counts"):
        model.log_likelihood(x, np.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        model.log_likelihood(x, np.asarray([[1.0, 1.0], [np.nan, 1.0], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        model.predict(np.asarray([[1.0, np.inf]]))


def test_neural_wrappers_restore_modes_and_train_in_training_mode() -> None:
    module = _ModeRecordingLinear(1, 1)
    model = GaussianRegressionNeuralNetwork(module)
    x = np.asarray([[-1.0], [0.0], [1.0]])
    y = 2.0 * x

    module.train()
    model.log_likelihood(x, y)
    assert module.forward_modes[-1] is False
    assert module.training is True

    module.eval()
    model.fit(x, y, max_its=1, tol=0.0)
    assert module.forward_modes[-1] is True
    assert module.training is False


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0, 1]),
        np.asarray([[0], [1], [0]]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0, -1, 0]),
        np.asarray([0, 2, 0]),
    ],
)
def test_categorical_wrapper_rejects_invalid_labels(labels: np.ndarray) -> None:
    model = CategoricalClassificationNeuralNetwork(torch.nn.Linear(1, 2))
    with pytest.raises(ValueError, match="labels"):
        model.log_likelihood(np.ones((3, 1)), labels)


def test_categorical_wrapper_rejects_nonfinite_logits() -> None:
    model = CategoricalClassificationNeuralNetwork(_NonFiniteModule(2, np.nan))
    with pytest.raises(ValueError, match="finite"):
        model.log_likelihood(np.ones((2, 1)), np.asarray([0, 1]))


@pytest.mark.parametrize(
    "counts",
    [
        np.asarray([0.0, 1.5, 2.0]),
        np.asarray([0.0, -1.0, 2.0]),
        np.asarray([0.0, np.nan, 2.0]),
        np.asarray([0.0, np.inf, 2.0]),
    ],
)
def test_poisson_wrapper_rejects_invalid_counts(counts: np.ndarray) -> None:
    model = PoissonRegressionNeuralNetwork(torch.nn.Linear(1, 1))
    with pytest.raises(ValueError, match="counts"):
        model.log_likelihood(np.ones((3, 1)), counts)


def test_poisson_wrapper_rejects_nonfinite_log_rates_and_rates() -> None:
    invalid = PoissonRegressionNeuralNetwork(_NonFiniteModule(1, np.inf))
    with pytest.raises(ValueError, match="finite"):
        invalid.log_likelihood(np.ones((2, 1)), np.asarray([0, 1]))

    overflowing = PoissonRegressionNeuralNetwork(_NonFiniteModule(1, 1_000.0))
    with pytest.raises(ValueError, match="non-finite"):
        overflowing.predict_rate(np.ones((2, 1)))


@pytest.mark.parametrize("increasing", [True, False])
def test_monotonic_builder_is_pickle_stable(increasing: bool) -> None:
    torch.manual_seed(3)
    module = make_monotonic_mlp(2, [4], 1, increasing=increasing)
    restored = pickle.loads(pickle.dumps(module))
    x = torch.randn(5, 2)
    torch.testing.assert_close(restored(x), module(x))


def test_deep_set_is_pickle_stable_and_validates_set_geometry() -> None:
    torch.manual_seed(5)
    module = make_deep_set(2, [4], 3, [4], 1)
    restored = pickle.loads(pickle.dumps(module))
    x = torch.randn(3, 5, 2)
    torch.testing.assert_close(restored(x), module(x))

    with pytest.raises(ValueError, match="non-empty"):
        module(torch.empty(3, 0, 2))
    with pytest.raises(ValueError, match="element_dim"):
        module(torch.ones(3, 5, 1))
    with pytest.raises(ValueError, match="finite"):
        module(torch.full((3, 5, 2), np.nan))


@pytest.mark.parametrize("builder", [make_monotonic_mlp, make_deep_set])
def test_structured_builders_reject_fractional_dimensions(builder: object) -> None:
    if builder is make_monotonic_mlp:
        args = (1.5, [4], 1)
    else:
        args = (1.5, [4], 3, [4], 1)
    with pytest.raises(TypeError, match="integer"):
        builder(*args)
