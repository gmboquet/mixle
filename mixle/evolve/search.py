"""``auto_select`` and ``search``: model selection and config search from data.

* :func:`auto_select` elevates the existing automatic engine
  (:func:`mixle.utils.automatic.get_estimator`) into the evolve contract and, when the criterion is a
  proper-score :class:`~mixle.evolve.objective.Objective`, adds the held-out champion/challenger gate
  on top of the in-sample BIC pick so the returned model wins *out of sample*, not merely on BIC.

* :func:`search` searches a typed :class:`~mixle.evolve.space.Space` for the config that builds the best
  model under a held-out :class:`Objective`, with three interchangeable backends:

    * ``method='bo'``          -- encode the space as a numeric box and drive :func:`mixle.doe.minimize`.
    * ``method='evolutionary'``-- a (mu + lambda) loop over :meth:`Space.sample` / :meth:`Space.neighbors`.
    * ``method='bandit'``      -- delegate the *which-operator* decision to the
      :class:`~mixle.evolve.population.OperatorBandit` via a :class:`~mixle.evolve.population.Population`.

  ``build_fn(config) -> fitted model`` is caller-supplied, so ``search`` is family-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.evolve.improve import ImprovementResult, _split
from mixle.evolve.objective import Objective
from mixle.evolve.operators import ImprovementOperator, Refit
from mixle.evolve.space import Space
from mixle.evolve.verify import challenger_beats_champion


def _fit_auto(rows: list[Any], *, max_its: int) -> Any:
    """BIC/auto family inference + EM fit (the in-sample automatic pick)."""
    from mixle.inference.estimation import optimize
    from mixle.utils.automatic import get_estimator

    estimator = get_estimator(rows)
    return optimize(rows, estimator, max_its=max_its, out=None)


def auto_select(
    data: Sequence[Any],
    *,
    space: Any | None = None,
    criterion: str | Objective = "bic",
    verify: bool = True,
    holdout: float = 0.25,
    seed: int = 0,
    max_its: int = 20,
) -> ImprovementResult:
    """Infer and fit a model from raw ``data``, optionally gated by a held-out proper score.

    Args:
        data: the raw dataset.
        space: reserved for typed search-space selection; must currently be ``None`` for ``auto_select``.
        criterion: ``'bic'`` (delegate to the automatic in-sample pick) or a proper-score
            :class:`~mixle.evolve.objective.Objective` (add the held-out verify gate on top of BIC).
        verify: when ``criterion`` is an :class:`Objective`, whether to run the held-out gate (the BIC
            pick fitted on the train split is the *champion*; the BIC pick refitted on all data is the
            *challenger*, promoted only if it wins out of sample).
        holdout: held-out fraction for the proper-score gate.
        seed: RNG seed for the split and sampled objectives.
        max_its: EM iterations for the fits.

    Returns:
        An :class:`~mixle.evolve.improve.ImprovementResult`. For ``criterion='bic'`` it carries the
        fitted automatic model with ``verified=False`` (no out-of-sample test was requested). For an
        :class:`Objective` criterion with ``verify=True`` it carries the gate verdict and
        ``verified`` reflects whether the full-data model beats the train-only model out of sample.
    """
    if space is not None:
        raise NotImplementedError("auto_select: a typed search 'space' is a Phase-2 feature; pass space=None.")

    rows = list(data)

    if isinstance(criterion, str):
        if criterion != "bic":
            raise ValueError(
                f"string criterion must be 'bic' (got {criterion!r}); pass a proper-score Objective for "
                "out-of-sample selection."
            )
        model = _fit_auto(rows, max_its=max_its)
        return ImprovementResult(
            model,
            False,
            "auto_select[bic]",
            0.0,
            None,
            {"criterion": "bic", "family": type(model).__name__},
            None,
        )

    # proper-score Objective: BIC pick + held-out gate.
    objective: Objective = criterion
    if not verify:
        model = _fit_auto(rows, max_its=max_its)
        return ImprovementResult(
            model,
            False,
            "auto_select[%s]" % objective.name,
            0.0,
            None,
            {"criterion": objective.name, "verify": False, "family": type(model).__name__},
            None,
        )

    train, val = _split(rows, holdout, seed)
    champion = _fit_auto(train, max_its=max_its)
    # the challenger is the same automatic family warm-fitted on the full data (more data, same shape).
    challenger = Refit(max_its=max_its).propose(champion, rows, ctx={"parent_hash": None}).model

    verdict = challenger_beats_champion(
        champion,
        challenger,
        val,
        objective=objective,
        seed=seed,
    )
    if verdict.promote:
        return ImprovementResult(
            challenger,
            True,
            "auto_select[%s]" % objective.name,
            verdict.delta,
            verdict,
            {"criterion": objective.name, "family": type(challenger).__name__},
            None,
        )
    # the full-data fit did not beat the train-only fit out of sample -> keep the more-evidenced full fit
    # but report it as unverified (no out-of-sample improvement over the train-only model).
    full = _fit_auto(rows, max_its=max_its)
    return ImprovementResult(
        full,
        False,
        "auto_select[%s]" % objective.name,
        verdict.delta,
        verdict,
        {"criterion": objective.name, "family": type(full).__name__, "verified_gate": False},
        None,
    )


@dataclass(frozen=True)
class SearchResult:
    """The outcome of a :func:`search` (or :meth:`Population.run`) run."""

    best_config: dict[str, Any]
    best_model: Any
    best_score: float  # in the objective's native orientation (lower- or higher-is-better)
    history: list[dict[str, Any]] = field(default_factory=list)


def _held_out_score(
    config: dict[str, Any],
    build_fn: Callable[[dict[str, Any]], Any],
    train: list[Any],
    val: list[Any],
    objective: Objective,
) -> tuple[float, Any]:
    """Build a model from ``config`` on ``train`` and score it on ``val``; smaller is always better.

    Returns ``(canonical_score, model)`` where ``canonical_score`` is normalized to lower-is-better
    (the BO/evolutionary loops minimize it). A build/score failure is a large finite penalty, not an
    exception, so one bad config cannot abort the whole search.
    """
    try:
        model = build_fn(config)
        s = float(objective.scalar(model, val))
        canonical = s if objective.lower_is_better else -s
        if not np.isfinite(canonical):
            return 1.0e18, None
        return canonical, model
    except Exception:  # noqa: BLE001
        return 1.0e18, None


def search(
    space: Space,
    data: Sequence[Any],
    *,
    objective: Objective,
    build_fn: Callable[[dict[str, Any]], Any],
    method: str = "bo",
    n_iter: int = 25,
    holdout: float = 0.25,
    seed: int = 0,
    **method_kwargs: Any,
) -> SearchResult:
    """Search ``space`` for the config whose ``build_fn`` model scores best on a held-out split.

    Args:
        space: the typed :class:`~mixle.evolve.space.Space` to search.
        data: the raw dataset (split once into train/val here for the inner objective).
        objective: the held-out :class:`~mixle.evolve.objective.Objective` (lower-is-better aware).
        build_fn: caller-supplied ``config -> fitted model`` (the search is family-agnostic).
        method: ``'bo'`` (Bayesian optimization over the numeric box), ``'evolutionary'``
            (a (mu + lambda) loop over ``sample`` / ``neighbors``), or ``'bandit'`` (delegate the
            operator policy to an :class:`~mixle.evolve.population.OperatorBandit`).
        n_iter: search budget (BO acquisition steps / evolutionary generations / bandit generations).
        holdout: held-out fraction for the inner objective.
        seed: RNG seed.
        method_kwargs: backend-specific knobs (e.g. ``mu`` / ``lam`` for the evolutionary loop,
            ``operators`` / ``size`` for the bandit population).

    Returns:
        A :class:`SearchResult` with ``best_config`` / ``best_model`` / ``best_score`` (native
        orientation) / ``history``.
    """
    rows = list(data)
    train, val = _split(rows, holdout, seed)

    def native(canonical: float) -> float:
        return canonical if objective.lower_is_better else -canonical

    if method == "bo":
        result = _search_bo(space, train, val, objective, build_fn, n_iter=n_iter, seed=seed, **method_kwargs)
    elif method == "evolutionary":
        result = _search_evolutionary(space, train, val, objective, build_fn, n_iter=n_iter, seed=seed, **method_kwargs)
    elif method == "bandit":
        return _search_bandit(space, rows, objective, build_fn, n_iter=n_iter, seed=seed, **method_kwargs)
    else:
        raise ValueError(f"method must be 'bo' | 'evolutionary' | 'bandit' (got {method!r}).")

    best_config, best_model, best_canonical, history = result
    return SearchResult(best_config, best_model, native(best_canonical), history)


def _search_bo(
    space: Space,
    train: list[Any],
    val: list[Any],
    objective: Objective,
    build_fn: Callable[[dict[str, Any]], Any],
    *,
    n_iter: int,
    seed: int,
    n_init: int | None = None,
) -> tuple[dict[str, Any], Any, float, list[dict[str, Any]]]:
    """Drive :func:`mixle.doe.minimize` over the space's numeric box (categoricals as integer indices)."""
    from mixle.doe import minimize

    bounds = space.to_bounds()
    history: list[dict[str, Any]] = []
    # cache the best model encountered (minimize only returns x/y, not the built object).
    best: dict[str, Any] = {"score": np.inf, "config": None, "model": None}

    def numeric_objective(x: np.ndarray) -> float:
        config = space.decode(x)
        canonical, model = _held_out_score(config, build_fn, train, val, objective)
        history.append({"config": config, "score": float(canonical)})
        if canonical < best["score"]:
            best.update({"score": float(canonical), "config": config, "model": model})
        return float(canonical)

    n_init = max(2 * space.ndim + 1, 3) if n_init is None else int(n_init)
    n_init = min(n_init, max(1, n_iter))
    n_acq = max(0, int(n_iter) - n_init)
    minimize(numeric_objective, bounds, n_init=n_init, n_iter=n_acq, seed=seed, maximize=False)
    return best["config"], best["model"], float(best["score"]), history


def _search_evolutionary(
    space: Space,
    train: list[Any],
    val: list[Any],
    objective: Objective,
    build_fn: Callable[[dict[str, Any]], Any],
    *,
    n_iter: int,
    seed: int,
    mu: int = 4,
    lam: int = 8,
) -> tuple[dict[str, Any], Any, float, list[dict[str, Any]]]:
    """A (mu + lambda) evolutionary loop over ``Space.sample`` / ``Space.neighbors``.

    Maintains ``mu`` parents; each generation spawns ``lam`` offspring (a random neighbor of a random
    parent), evaluates them, and keeps the best ``mu`` of (parents + offspring). Categoricals are handled
    natively (no numeric rounding), so this is the backend for spaces BO encodes lossily.
    """
    rng = np.random.RandomState(seed)
    history: list[dict[str, Any]] = []

    def evaluate(config: dict[str, Any]) -> tuple[float, Any]:
        canonical, model = _held_out_score(config, build_fn, train, val, objective)
        history.append({"config": config, "score": float(canonical)})
        return canonical, model

    # initial parents: random samples.
    population: list[tuple[float, dict[str, Any], Any]] = []
    for _ in range(mu):
        cfg = space.sample(rng)
        score, model = evaluate(cfg)
        population.append((score, cfg, model))
    population.sort(key=lambda t: t[0])

    for _ in range(int(n_iter)):
        offspring: list[tuple[float, dict[str, Any], Any]] = []
        for _ in range(lam):
            parent = population[int(rng.randint(0, len(population)))][1]
            nbrs = space.neighbors(parent)
            child = nbrs[int(rng.randint(0, len(nbrs)))] if nbrs else space.sample(rng)
            score, model = evaluate(child)
            offspring.append((score, child, model))
        population = sorted(population + offspring, key=lambda t: t[0])[:mu]

    best_score, best_config, best_model = population[0]
    return best_config, best_model, float(best_score), history


def _search_bandit(
    space: Space,
    rows: list[Any],
    objective: Objective,
    build_fn: Callable[[dict[str, Any]], Any],
    *,
    n_iter: int,
    seed: int,
    operators: Sequence[ImprovementOperator] | None = None,
    size: int = 8,
    n_seeds: int = 3,
) -> SearchResult:
    """Delegate to an :class:`~mixle.evolve.population.OperatorBandit` via a :class:`Population`.

    The "space" here is *which operator to apply*, not a parameter box: ``build_fn`` instantiates a few
    seed structures (from random configs), and the bandit-driven :class:`Population` evolves them,
    learning which operators pay off. The returned :class:`SearchResult` carries the population champion.
    """
    from mixle.evolve.population import OperatorBandit, Population

    rng = np.random.RandomState(seed)
    seeds = []
    for _ in range(max(1, n_seeds)):
        cfg = space.sample(rng)
        try:
            seeds.append(build_fn(cfg))
        except Exception:  # noqa: BLE001
            continue
    if not seeds:
        raise ValueError("search(method='bandit'): build_fn produced no valid seed models from the space.")

    from mixle.evolve.operators import default_operators

    ops = list(operators) if operators is not None else default_operators()
    bandit = OperatorBandit(ops, seed=seed)
    pop = Population(seeds, objective=objective, operators=ops, bandit=bandit, size=size, seed=seed)
    return pop.run(rows, generations=n_iter)


__all__ = ["auto_select", "search", "SearchResult"]
