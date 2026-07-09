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

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrainingSpace:
    """The tunable axes of an LM training recipe and how a unit-cube point decodes into concrete knobs."""

    d_model_choices: Sequence[int] = (64, 128, 256, 512)
    n_layer_range: tuple[int, int] = (2, 12)
    log10_lr_range: tuple[float, float] = (-4.0, -2.0)
    batch_choices: Sequence[int] = (16, 32, 64, 128)

    def dims(self) -> int:
        """Return the dimensionality of the unit-cube recipe search space."""
        return 4

    def bounds(self) -> list[tuple[float, float]]:
        """Return unit-cube bounds for the DOE optimizer."""
        return [(0.0, 1.0)] * self.dims()

    def decode(self, point: np.ndarray) -> dict[str, Any]:
        """Decode a unit-cube point into concrete LM training hyperparameters."""
        p = np.clip(np.asarray(point, dtype=np.float64), 0.0, 1.0)
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

    space = space or TrainingSpace()

    def objective(x: np.ndarray, s: float) -> float:
        return float(train(space.decode(x), float(s)))

    result = multi_fidelity_minimize(
        objective, space.bounds(), fidelities=fidelities, costs=costs, max_cost=max_cost, n_init=n_init, seed=seed
    )
    best_x = np.asarray(result["x"], dtype=np.float64)  # best target-fidelity point
    return TrainingSearchResult(recipe=space.decode(best_x), loss=float(result["y"]), history=result)


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

    def train(recipe: dict[str, Any], budget: float) -> float:
        epochs = max(1, int(round(max_epochs * float(budget))))
        lm = LM(
            vocab=vocab,
            d_model=int(recipe.get("d_model", 128)),
            n_layer=int(recipe.get("n_layer", 4)),
            block=block,
            device=device,
        )
        lm.fit(
            list(token_ids),
            epochs=epochs,
            batch_size=int(recipe.get("batch_size", 32)),
            lr=float(recipe.get("lr", 3e-3)),
        )
        return float(lm.nll(list(val_ids)))

    return train


def extrapolate_learning_curve(steps: Sequence[float], losses: Sequence[float], *, at: float) -> float:
    """Predict the loss at budget/step ``at`` from a partial run's ``(steps, losses)`` via a power-law fit.

    Fits ``loss(t) = a + b * t^(-c)`` and evaluates it at ``at`` so a partial
    run can estimate the full-budget loss for early stopping. Falls back to the
    last observed loss if the fit fails.
    """
    t = np.asarray(steps, dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)
    if len(t) < 3:
        return float(y[-1])
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
        return float(curve(np.asarray([at], dtype=np.float64), *popt)[0])
    except Exception:  # noqa: BLE001 - degrade to the last observation on any fit failure
        return float(y[-1])
