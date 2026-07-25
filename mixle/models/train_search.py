"""Design-of-experiments helpers for language-model training recipes.

Training a language model is an expensive-objective, low-fidelity-proxy
setting: a short run over fewer steps or a data subset is a noisy estimate of
the full run's loss. :func:`tune_training` wraps
``mixle.doe.multi_fidelity_minimize`` so the search uses low-budget runs to
locate promising recipes and reserves full-budget runs to refine them.

The objective is a caller-supplied training callback
``train(recipe, budget) -> held-out loss`` where ``budget in (0, 1]`` is the
fraction of full training. :func:`lm_train_fn` provides a callback for
:class:`~mixle.models.language_model.LM`, and
:func:`extrapolate_learning_curve` predicts full-budget loss from a partial
run's curve for early stopping.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a {'positive ' if positive else ''}finite scalar")
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a {'positive ' if positive else ''}finite scalar")
    try:
        result = float(array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a {'positive ' if positive else ''}finite scalar") from exc
    if not np.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be a {'positive ' if positive else ''}finite scalar")
    return result


def _choice_sequence(values: Sequence[int], name: str) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"{name} must be a non-empty sequence of positive integers")
    choices = tuple(_positive_int(value, f"{name}[{index}]") for index, value in enumerate(values))
    if len(set(choices)) != len(choices):
        raise ValueError(f"{name} must not contain duplicate choices")
    return choices


@dataclass
class TrainingSpace:
    """The tunable axes of an LM training recipe and how a unit-cube point decodes into concrete knobs."""

    d_model_choices: Sequence[int] = (64, 128, 256, 512)
    n_layer_range: tuple[int, int] = (2, 12)
    log10_lr_range: tuple[float, float] = (-4.0, -2.0)
    batch_choices: Sequence[int] = (16, 32, 64, 128)

    def __post_init__(self) -> None:
        self.d_model_choices = _choice_sequence(self.d_model_choices, "d_model_choices")
        self.batch_choices = _choice_sequence(self.batch_choices, "batch_choices")
        if not isinstance(self.n_layer_range, tuple) or len(self.n_layer_range) != 2:
            raise ValueError("n_layer_range must be a two-item tuple of positive integers")
        layer_low = _positive_int(self.n_layer_range[0], "n_layer_range[0]")
        layer_high = _positive_int(self.n_layer_range[1], "n_layer_range[1]")
        if layer_low > layer_high:
            raise ValueError("n_layer_range lower bound must not exceed its upper bound")
        self.n_layer_range = (layer_low, layer_high)
        if not isinstance(self.log10_lr_range, tuple) or len(self.log10_lr_range) != 2:
            raise ValueError("log10_lr_range must be a two-item tuple of finite scalars")
        lr_low = _finite_float(self.log10_lr_range[0], "log10_lr_range[0]")
        lr_high = _finite_float(self.log10_lr_range[1], "log10_lr_range[1]")
        if lr_low > lr_high:
            raise ValueError("log10_lr_range lower bound must not exceed its upper bound")
        self.log10_lr_range = (lr_low, lr_high)

    def dims(self) -> int:
        """Return the dimensionality of the unit-cube recipe search space."""
        return 4

    def bounds(self) -> list[tuple[float, float]]:
        """Return unit-cube bounds for the DOE optimizer."""
        return [(0.0, 1.0)] * self.dims()

    def decode(self, point: np.ndarray) -> dict[str, Any]:
        """Decode a unit-cube point into concrete LM training hyperparameters."""
        try:
            p = np.asarray(point, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("point must be a finite one-dimensional unit-cube vector of length 4") from exc
        if p.shape != (self.dims(),):
            raise ValueError(f"point must have shape ({self.dims()},), got {p.shape}")
        if not np.all(np.isfinite(p)):
            raise ValueError("point must contain only finite coordinates")
        if np.any((p < 0.0) | (p > 1.0)):
            raise ValueError("point coordinates must lie in the closed unit interval [0, 1]")
        return {
            "d_model": int(
                self.d_model_choices[min(len(self.d_model_choices) - 1, int(p[0] * len(self.d_model_choices)))]
            ),
            "n_layer": int(round(self.n_layer_range[0] + p[1] * (self.n_layer_range[1] - self.n_layer_range[0]))),
            "lr": float(10.0 ** (self.log10_lr_range[0] + p[2] * (self.log10_lr_range[1] - self.log10_lr_range[0]))),
            "batch_size": int(
                self.batch_choices[min(len(self.batch_choices) - 1, int(p[3] * len(self.batch_choices)))]
            ),
        }


@dataclass
class TrainingSearchResult:
    """The outcome of a multi-fidelity training search: the best recipe, its full-budget loss, and the history."""

    recipe: dict[str, Any]
    loss: float
    history: Any = field(default=None)


def tune_training(
    train: Callable[[dict[str, Any], float], float],
    space: TrainingSpace | None = None,
    *,
    fidelities: tuple[float, ...] = (0.25, 1.0),
    costs: tuple[float, ...] | None = None,
    max_cost: float = 20.0,
    n_init: int | None = None,
    seed: int = 0,
) -> TrainingSearchResult:
    """Run multi-fidelity BO over a training recipe.

    ``train(recipe, budget)`` returns held-out loss, where lower is better.
    ``fidelities`` are the training-budget fractions the search may run at.
    Returns the recipe with the best full-budget loss and the full BO history.
    """
    from mixle.doe import multi_fidelity_minimize

    if not callable(train):
        raise TypeError("train must be callable")
    if space is None:
        space = TrainingSpace()
    if not isinstance(space, TrainingSpace):
        raise TypeError("space must be a TrainingSpace")

    def objective(x: np.ndarray, s: float) -> float:
        value = _finite_float(train(space.decode(x), float(s)), "training objective")
        return value

    result = multi_fidelity_minimize(
        objective, space.bounds(), fidelities=fidelities, costs=costs, max_cost=max_cost, n_init=n_init, seed=seed
    )
    if not result["target_evaluated"]:
        # `x`/`y` are None here (MXR-080-0181): the budget ran out -- or the surrogate fit failed --
        # before a single target-fidelity evaluation was affordable. Surface that plainly instead of
        # letting `np.asarray(None, dtype=np.float64)` silently become `array(nan)` and fail later with
        # an unrelated-looking error out of `space.decode` (e.g. "invalid index to scalar variable").
        target = float(max(fidelities))
        raise ValueError(
            f"tune_training: budget too tight to ever reach the target fidelity {target} -- "
            f"max_cost={max_cost}, fidelities={fidelities!r}, costs={costs!r}, n_init={n_init!r}. "
            f"The multi-fidelity search stopped ({result['stopped_reason']!r}) without affording a "
            f"single evaluation at the target fidelity, so no best recipe/loss is available. Raise "
            f"max_cost, reduce n_init, or lower the target fidelity's cost."
        )
    best_x = np.asarray(result["x"], dtype=np.float64)  # best target-fidelity point
    loss = _finite_float(result["y"], "best target-fidelity loss")
    return TrainingSearchResult(recipe=space.decode(best_x), loss=loss, history=result)


def lm_train_fn(
    token_ids: Sequence[int],
    val_ids: Sequence[int],
    *,
    vocab: int,
    block: int = 64,
    max_epochs: int = 3,
    device: str = "cpu",
) -> Callable[[dict[str, Any], float], float]:
    """Return a training callback ``(recipe, budget) -> held-out nats/token`` for ``LM``.

    ``budget in (0, 1]`` scales the number of epochs. A larger pretraining loop
    can use the same convention to scale steps or token subsets.
    """
    from mixle.models.language_model import LM

    vocab = _positive_int(vocab, "vocab")
    block = _positive_int(block, "block")
    max_epochs = _positive_int(max_epochs, "max_epochs")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")

    def validated_tokens(values: Sequence[int], name: str) -> list[int]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
            raise ValueError(f"{name} must be a non-empty sequence of token ids")
        result = []
        for index, value in enumerate(values):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name}[{index}] must be an integer token id")
            token = int(value)
            if not 0 <= token < vocab:
                raise ValueError(f"{name}[{index}]={token} is outside [0, {vocab})")
            result.append(token)
        return result

    train_ids = validated_tokens(token_ids, "token_ids")
    validation_ids = validated_tokens(val_ids, "val_ids")

    def train(recipe: dict[str, Any], budget: float) -> float:
        if not isinstance(recipe, Mapping):
            raise TypeError("recipe must be a mapping")
        budget = _finite_float(budget, "budget", positive=True)
        if budget > 1.0:
            raise ValueError("budget must be no greater than 1")
        epochs = max(1, int(round(max_epochs * budget)))
        d_model = _positive_int(recipe.get("d_model", 128), "recipe d_model")
        n_layer = _positive_int(recipe.get("n_layer", 4), "recipe n_layer")
        batch_size = _positive_int(recipe.get("batch_size", 32), "recipe batch_size")
        learning_rate = _finite_float(recipe.get("lr", 3e-3), "recipe lr", positive=True)
        lm = LM(
            vocab=vocab,
            d_model=d_model,
            n_layer=n_layer,
            block=block,
            device=device,
        )
        lm.fit(
            train_ids,
            epochs=epochs,
            batch_size=batch_size,
            lr=learning_rate,
        )
        loss = _finite_float(lm.nll(validation_ids), "validation loss")
        return loss

    return train


def extrapolate_learning_curve(steps: Sequence[float], losses: Sequence[float], *, at: float) -> float:
    """Predict the loss at budget/step ``at`` from a partial run's ``(steps, losses)`` via a power-law fit.

    Fits ``loss(t) = a + b * t^(-c)`` and evaluates it at ``at`` so a partial
    run can estimate the full-budget loss for early stopping. Invalid data, an
    underdetermined curve, or a failed/non-finite fit raises explicitly.
    """
    try:
        t = np.asarray(steps, dtype=np.float64)
        y = np.asarray(losses, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("steps and losses must be finite one-dimensional arrays") from exc
    if t.ndim != 1 or y.ndim != 1 or len(t) == 0 or len(t) != len(y):
        raise ValueError("steps and losses must be non-empty aligned one-dimensional arrays")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("steps and losses must contain only finite values")
    if np.any(t <= 0.0) or np.any(y <= 0.0):
        raise ValueError("steps and losses must be strictly positive")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("steps must be strictly increasing")
    at = _finite_float(at, "at", positive=True)
    if at <= t[-1]:
        raise ValueError("at must be greater than the last observed step")
    if len(t) < 3:
        raise ValueError("at least three observations are required to extrapolate a learning curve")
    try:
        from scipy.optimize import curve_fit

        def curve(tt: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            return a + b * np.power(tt, -c)

        a0 = float(min(y) * 0.9)
        popt, _ = curve_fit(
            curve,
            t,
            y,
            p0=[a0, float(max(y) - a0), 0.5],
            maxfev=5000,
            bounds=([-np.inf, 0.0, 1e-3], [np.inf, np.inf, 5.0]),
        )
        if not np.all(np.isfinite(popt)):
            raise RuntimeError("learning-curve fit returned non-finite parameters")
        prediction = float(curve(np.asarray([at], dtype=np.float64), *popt)[0])
        if not np.isfinite(prediction) or prediction <= 0.0:
            raise RuntimeError(f"learning-curve fit returned invalid prediction {prediction!r}")
        return prediction
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        raise RuntimeError("learning-curve extrapolation failed; no prediction is available") from exc
