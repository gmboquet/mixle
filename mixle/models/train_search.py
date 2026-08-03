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

from mixle.utils.immutable import detach_receipt_container


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    """Exact non-negative integer, Booleans refused -- the shape a seed must have.

    ``int(seed)`` truncates: ``seed=2.9`` and ``seed=2`` name the same random stream while reading
    as different declarations, and ``seed=True`` becomes ``1``. A seed is an identifier, not a
    magnitude, so silently rounding one is silently running a different experiment.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an exact non-negative integer, got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be an exact non-negative integer, got {value!r}")
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
    """The outcome of a multi-fidelity training search: the best recipe, its full-budget loss, and the history.

    The result OWNS what it reports (MXR-080-1890). It used to store the caller's ``recipe`` mapping
    and ``history`` container by reference and validate nothing, so::

        recipe = {"d_model": 64}
        res = TrainingSearchResult(recipe=recipe, loss=1.0, history=trace)
        recipe["d_model"] = 999          # the recorded winning recipe silently became a different one

    rewrote a decision that had already been recorded, and ``loss`` accepted ``nan`` or the string
    ``"not a number"`` as a search outcome. ``__post_init__`` now copies both containers with
    :func:`~mixle.utils.immutable.detach_receipt_container` and checks that the fields are the kind
    of thing the search actually produces.

    ``seed`` records the search's common-random-numbers seed, so the reported loss is attributable
    to a reproducible run rather than to one unlabelled draw.

    Deliberately NOT checked, and why:

    * **The numeric contents of ``history``.** ``detach_receipt_container`` copies containers but
      passes elements through by identity, so the ``X``/``Y`` blocks a search accumulates stay the
      same arrays rather than being duplicated -- they are the optimizer's own freshly built output,
      nobody else holds them, and copying every evaluation block would add a real cost to a result
      that is often only glanced at. A holder can still edit those arrays in place; this defends
      against the *caller* who kept a reference, which is the aliasing defect that was reproduced.
    * **That ``recipe`` decodes to ``history["x"]``.** Inside :func:`tune_training` both come from
      the same optimizer result, so the check would be tautological; for a directly constructed
      result there is no space to decode against.
    """

    recipe: dict[str, Any]
    loss: float
    history: Any = field(default=None)
    seed: int | None = field(default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.recipe, Mapping):
            raise TypeError(
                f"recipe must be a mapping of hyperparameter name -> value, got {type(self.recipe).__name__}"
            )
        if not all(isinstance(key, str) for key in self.recipe):
            raise ValueError("recipe keys must be hyperparameter name strings")
        # dict(...) rather than the stronger freezer: `recipe` is handed straight to a training
        # callback that may type-test or mutate its own copy, and a mappingproxy is not picklable.
        self.recipe = dict(detach_receipt_container(self.recipe))
        self.loss = _finite_float(self.loss, "loss")
        self.history = detach_receipt_container(self.history)
        if self.seed is not None:
            self.seed = _nonnegative_int(self.seed, "seed")


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

    ``seed`` is recorded on the returned result (MXR-080-1890) so the reported loss is attributable
    to a reproducible search rather than to one unlabelled draw. Whether the *objective* is itself
    reproducible is the callback's contract, not this function's: :func:`lm_train_fn` supplies one
    that pins its own randomness, but an arbitrary caller-supplied ``train`` is opaque here and is
    deliberately not second-guessed.
    """
    from mixle.doe import multi_fidelity_minimize

    if not callable(train):
        raise TypeError("train must be callable")
    seed = _nonnegative_int(seed, "seed")
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
    # `seed` is recorded, not just consumed: the reported loss is only reproducible against the draw
    # that produced it, and a result that does not carry its seed cannot be re-run (MXR-080-1890).
    # `TrainingSearchResult.__post_init__` detaches `history` from the optimizer's own dict.
    return TrainingSearchResult(recipe=space.decode(best_x), loss=loss, history=result, seed=seed)


@dataclass(frozen=True)
class LMFidelity:
    """The work one declared budget actually buys, counted in TRAINING TOKENS (MXR-080-1890).

    Fidelity used to be ``max(1, round(max_epochs * budget))`` whole epochs, which makes the declared
    budget and the executed work different quantities related by a coarse many-to-one map. Measured
    over the library's own default ``max_epochs=3``, budgets ``0.05, 0.1, 0.2, 0.25, 0.3, 0.4`` all
    executed exactly ONE epoch, and at ``max_epochs=1`` *every* budget from ``0.05`` to ``1.0``
    executed one epoch -- so a "cheap screening run" and a "full run" were byte-identical work and
    the multi-fidelity premise did not hold at all.

    Counting in tokens restores resolution: the corpus is truncated to a prefix as well as repeated
    for whole epochs, so ``total_tokens`` separates budgets that whole epochs could not. ``epochs``
    and ``tokens_per_epoch`` are what is handed to :meth:`~mixle.models.language_model.LM.fit`;
    ``total_tokens = epochs * tokens_per_epoch`` is the comparable work figure. ``seed`` is the
    common-random-numbers seed the run used.
    """

    budget: float
    epochs: int
    tokens_per_epoch: int
    total_tokens: int
    seed: int


def _plan_lm_fidelity(budget: float, *, max_epochs: int, n_tokens: int, min_tokens: int, seed: int) -> LMFidelity:
    """Resolve ``budget`` into concrete, token-denominated work.

    ``budget=1.0`` reproduces the previous full-fidelity run exactly (``max_epochs`` epochs over the
    whole corpus); below that, the deficit is taken out of the token prefix rather than being
    rounded away. ``min_tokens`` is the floor a run must keep to remain trainable at all -- the
    streaming LM objective needs ``block + 1`` tokens to form one window and raises
    "token_source yielded no training batches" below that (measured), so a very small budget over a
    short corpus clamps up to that floor instead of producing an un-runnable plan.
    """
    full_tokens = max_epochs * n_tokens
    target = int(round(budget * full_tokens))
    epochs = max(1, min(max_epochs, -(-target // n_tokens)))  # ceil-div, at least one epoch
    per_epoch = min(n_tokens, max(min_tokens, int(round(target / epochs))))
    return LMFidelity(
        budget=float(budget),
        epochs=int(epochs),
        tokens_per_epoch=int(per_epoch),
        total_tokens=int(epochs * per_epoch),
        seed=int(seed),
    )


class LMTrainFn:
    """A ``(recipe, budget) -> held-out nats/token`` callback for ``LM`` that records what it ran.

    Built by :func:`lm_train_fn`. Two properties the bare closure did not have (MXR-080-1890):

    * **Fidelity is inspectable before it is paid for.** :meth:`plan` resolves a budget into the
      concrete :class:`LMFidelity` it will execute, so a caller can see whether two declared budgets
      are actually different runs without training anything.
    * **Every executed run is recorded.** :attr:`trials` returns the fidelity (work AND seed) of each
      completed call, so a finished search can be audited for budgets that collapsed onto the same
      work rather than the collapse being invisible.
    """

    def __init__(
        self,
        train_ids: list[int],
        validation_ids: list[int],
        *,
        vocab: int,
        block: int,
        max_epochs: int,
        device: str,
        seed: int,
    ) -> None:
        self._train_ids = train_ids
        self._validation_ids = validation_ids
        self._vocab = vocab
        self._block = block
        self._max_epochs = max_epochs
        self._device = device
        self._seed = seed
        # A prefix shorter than block+1 forms no training window at all; never plan below it.
        self._min_tokens = min(len(train_ids), block + 1)
        self._trials: list[LMFidelity] = []

    @property
    def seed(self) -> int:
        """The common-random-numbers seed every run uses (see :meth:`__call__`)."""
        return self._seed

    @property
    def trials(self) -> tuple[LMFidelity, ...]:
        """The fidelity of every completed run, oldest first, as an immutable snapshot."""
        return tuple(self._trials)

    def plan(self, budget: float) -> LMFidelity:
        """Resolve ``budget in (0, 1]`` into the concrete work it will execute, without running it."""
        value = _finite_float(budget, "budget", positive=True)
        if value > 1.0:
            raise ValueError("budget must be no greater than 1")
        return _plan_lm_fidelity(
            value,
            max_epochs=self._max_epochs,
            n_tokens=len(self._train_ids),
            min_tokens=self._min_tokens,
            seed=self._seed,
        )

    def __call__(self, recipe: dict[str, Any], budget: float) -> float:
        """Train one recipe at ``budget`` and return its held-out nats/token.

        Every run is driven from the SAME seed -- common random numbers. Two recipes are then
        compared over one shared draw of weight initialization and batch order, so their difference
        is the recipe's, not the draw's. Without it, repeating an identical (recipe, budget) call
        returned 6.46 then 7.65 nats/token in a measured back-to-back pair: a run-to-run spread far
        wider than the differences the search is trying to rank, which made the ranking noise.

        The torch global RNG is forked and restored, so seeding a search does not silently reseed
        the caller's other torch randomness.
        """
        import torch

        from mixle.models.language_model import LM

        if not isinstance(recipe, Mapping):
            raise TypeError("recipe must be a mapping")
        fidelity = self.plan(budget)
        d_model = _positive_int(recipe.get("d_model", 128), "recipe d_model")
        n_layer = _positive_int(recipe.get("n_layer", 4), "recipe n_layer")
        batch_size = _positive_int(recipe.get("batch_size", 32), "recipe batch_size")
        learning_rate = _finite_float(recipe.get("lr", 3e-3), "recipe lr", positive=True)
        device = torch.device(self._device)
        fork_devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(fidelity.seed)  # LM.__init__ draws its weights from the global RNG
            if device.type == "cuda":
                torch.cuda.manual_seed_all(fidelity.seed)
            lm = LM(
                vocab=self._vocab,
                d_model=d_model,
                n_layer=n_layer,
                block=self._block,
                device=self._device,
            )
            lm.fit(
                self._train_ids[: fidelity.tokens_per_epoch],
                epochs=fidelity.epochs,
                batch_size=batch_size,
                lr=learning_rate,
                seed=fidelity.seed,  # LM.fit's own shuffle stream, pinned to the same draw
            )
            loss = _finite_float(lm.nll(self._validation_ids), "validation loss")
        self._trials.append(fidelity)
        return loss


def lm_train_fn(
    token_ids: Sequence[int],
    val_ids: Sequence[int],
    *,
    vocab: int,
    block: int = 64,
    max_epochs: int = 3,
    device: str = "cpu",
    seed: int = 0,
    fidelities: Sequence[float] | None = None,
) -> LMTrainFn:
    """Return a training callback ``(recipe, budget) -> held-out nats/token`` for ``LM``.

    ``budget in (0, 1]`` is a fraction of the FULL run's training tokens
    (``max_epochs * len(token_ids)``), not a fraction of the epoch count. ``budget=1.0`` is
    unchanged -- ``max_epochs`` epochs over the whole corpus -- while a smaller budget shortens the
    corpus prefix as well as the epoch count, so distinct budgets buy distinct work. See
    :class:`LMFidelity` for the measurement that motivated this (MXR-080-1890).

    ``seed`` is the common-random-numbers seed shared by every run, which is what makes two recipes
    comparable; see :meth:`LMTrainFn.__call__`.

    ``fidelities`` is optional and off by default. Pass the exact budget ladder the search will use
    and it is checked, before any training happens, for budgets that still resolve to identical work
    -- the failure this finding is about. Token resolution runs out on a short corpus, so this can
    legitimately happen; it is reported rather than assumed away.

    Raises:
        ValueError: If ``fidelities`` is supplied and two of its budgets execute the same work.
    """
    vocab = _positive_int(vocab, "vocab")
    block = _positive_int(block, "block")
    max_epochs = _positive_int(max_epochs, "max_epochs")
    seed = _nonnegative_int(seed, "seed")
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
    train = LMTrainFn(
        train_ids,
        validated_tokens(val_ids, "val_ids"),
        vocab=vocab,
        block=block,
        max_epochs=max_epochs,
        device=device,
        seed=seed,
    )
    if fidelities is not None:
        if not isinstance(fidelities, Sequence) or isinstance(fidelities, (str, bytes)) or not fidelities:
            raise ValueError("fidelities must be a non-empty sequence of budgets in (0, 1]")
        by_work: dict[int, list[float]] = {}
        for budget in fidelities:
            plan = train.plan(budget)
            by_work.setdefault(plan.total_tokens, []).append(plan.budget)
        collapsed = {tokens: budgets for tokens, budgets in by_work.items() if len(budgets) > 1}
        if collapsed:
            raise ValueError(
                f"these declared fidelities execute identical work, so comparing recipes across them "
                f"compares nothing: {collapsed!r} (total training tokens -> the budgets that share "
                f"them). The corpus is {len(train_ids)} tokens over max_epochs={max_epochs}, "
                f"which cannot resolve budgets that close together -- raise max_epochs, lengthen the "
                f"corpus, or space the fidelities further apart."
            )
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
