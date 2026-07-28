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

  ``build_fn(config, train_data) -> fitted model`` is caller-supplied and given ONLY the training
  split (never ``val``), so ``search`` is family-agnostic.
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
            pick fitted on the train split is the *champion*; the *challenger* is the same automatic
            family refit on that SAME train split, so the comparison on the held-out data is genuinely
            held out. Promotion deploys a fresh refit of the verified winner on all the data -- never
            the object that was itself scored -- so a promoted model still benefits from the held-out
            split's extra evidence).
        holdout: held-out fraction for the proper-score gate.
        seed: RNG seed for the split and sampled objectives.
        max_its: EM iterations for the fits.

    Returns:
        An :class:`~mixle.evolve.improve.ImprovementResult`. For ``criterion='bic'`` it carries the
        fitted automatic model with ``verified=False`` (no out-of-sample test was requested). For an
        :class:`Objective` criterion with ``verify=True`` it carries the gate verdict and ``verified``
        reflects whether the train-only challenger beat the train-only champion out of sample.
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
    # the challenger for the held-out comparison must be fit on the SAME split as the champion (train
    # only): refitting on data that also contains `val` would let the challenger see the very
    # observations it is later "tested" against, which is not a held-out comparison at all.
    challenger = Refit(max_its=max_its).propose(champion, train, ctx={"parent_hash": None}).model

    verdict = challenger_beats_champion(
        champion,
        challenger,
        val,
        objective=objective,
        seed=seed,
    )
    if verdict.promote:
        # the held-out decision is now settled on a fair, train-only comparison. Only NOW is it safe
        # to fold `val`'s extra evidence into the model actually deployed: a fresh refit of the
        # VERIFIED winner on all the data, never the object that was itself scored as "held out".
        deployed = Refit(max_its=max_its).propose(challenger, rows, ctx={"parent_hash": None}).model
        return ImprovementResult(
            deployed,
            True,
            "auto_select[%s]" % objective.name,
            verdict.delta,
            verdict,
            {"criterion": objective.name, "family": type(deployed).__name__},
            None,
        )
    # the train-only challenger did not beat the train-only champion out of sample -> keep the
    # more-evidenced full fit but report it as unverified (no out-of-sample improvement was shown).
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
    """The outcome of a :func:`search` (or :meth:`Population.run`) run.

    ``search_failed`` / ``n_evaluations`` / ``n_successes`` distinguish "we searched and found a real
    (if mediocre) model" from "every configuration failed and best_model is a hollow placeholder" --
    without them, a totally failed search (``best_model=None``, ``best_score`` at the internal penalty
    sentinel) is shaped identically to a real result. Every backend populates them, including the
    ``bandit`` one via :meth:`mixle.evolve.population.Population.run` -- it used to leave them at their
    defaults, so a run that evaluated seeds and generations still reported ``n_evaluations=0`` and
    ``search_failed=False``, which reads as a successful search that did no work.
    """

    best_config: dict[str, Any]
    best_model: Any
    best_score: float  # in the objective's native orientation (lower- or higher-is-better)
    history: list[dict[str, Any]] = field(default_factory=list)
    search_failed: bool = False  # True iff NO configuration was ever successfully built/scored
    n_evaluations: int = 0  # total build/score attempts (successes + failures)
    n_successes: int = 0  # of those, how many actually succeeded


def _positive_int(value: Any, name: str) -> int:
    """Validate a search control as an exact positive integer.

    ``n_iter`` is a work budget and ``mu``/``lam`` are population sizes: a zero, negative, or
    fractional value is not a count of anything, and a budget of zero buys no result at all (BO
    cannot fit a surrogate from no observations; the evolutionary loop cannot seed a population).
    Rejected outright rather than silently normalized into "do some work anyway", which is what
    ``n_iter=0`` used to do in both backends.
    """
    if isinstance(value, bool):
        raise ValueError(f"search: {name} must be an exact positive integer, got {value!r}")
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"search: {name} must be an exact positive integer, got {value!r}") from None
    if count != value or count < 1:
        raise ValueError(f"search: {name} must be an exact positive integer, got {value!r}")
    return count


def _held_out_score(
    config: dict[str, Any],
    build_fn: Callable[[dict[str, Any], list[Any]], Any],
    train: list[Any],
    val: list[Any],
    objective: Objective,
) -> tuple[float, Any, str | None]:
    """Build a model from ``config`` on ``train`` (and ONLY ``train``) and score it on ``val``.

    Returns ``(canonical_score, model, error)``. ``canonical_score`` is normalized to lower-is-better
    (the BO/evolutionary loops minimize it). A build/score failure is a large finite penalty, not an
    exception, so one bad config cannot abort the whole search -- but it is never silent: ``model`` is
    ``None`` and ``error`` carries a short, human-readable reason (the per-attempt receipt), so a
    caller can always tell a real success (``error is None``) from a config that merely scored at the
    penalty sentinel by coincidence.
    """
    try:
        model = build_fn(config, train)
        s = float(objective.scalar(model, val))
        canonical = s if objective.lower_is_better else -s
        if not np.isfinite(canonical):
            return 1.0e18, None, f"non-finite score: {s!r}"
        return canonical, model, None
    except Exception as exc:  # noqa: BLE001
        return 1.0e18, None, f"{type(exc).__name__}: {exc}"


def search(
    space: Space,
    data: Sequence[Any],
    *,
    objective: Objective,
    build_fn: Callable[[dict[str, Any], list[Any]], Any],
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
        build_fn: caller-supplied ``(config, train_data) -> fitted model`` (the search is
            family-agnostic). ``train_data`` is ALWAYS the train split ``search`` computed -- never
            ``val`` -- so a config can actually be fit to real data without risking a held-out leak.
        method: ``'bo'`` (Bayesian optimization over the numeric box), ``'evolutionary'``
            (a (mu + lambda) loop over ``sample`` / ``neighbors``), or ``'bandit'`` (delegate the
            operator policy to an :class:`~mixle.evolve.population.OperatorBandit`).
        n_iter: the TOTAL evaluation budget (build/score attempts), consistently for ``bo`` and
            ``evolutionary`` -- each reserves every evaluation against it before spending, so neither
            can exceed it (the evolutionary backend used to treat it as a generation count and spend
            ``mu + lam * n_iter``). Both need at least one evaluation to produce any result, so
            ``n_iter<=0`` is rejected with a clear error instead of silently spending an evaluation
            anyway. (``bandit`` interprets ``n_iter`` as a generation count, unchanged -- it delegates
            to :class:`~mixle.evolve.population.Population`, whose per-generation work depends on the
            operator policy rather than on a fixed offspring count.)
        holdout: held-out fraction for the inner objective.
        seed: RNG seed.
        method_kwargs: backend-specific knobs (e.g. ``mu`` / ``lam`` for the evolutionary loop,
            ``operators`` / ``size`` for the bandit population).

    Returns:
        A :class:`SearchResult` with ``best_config`` / ``best_model`` / ``best_score`` (native
        orientation) / ``history``, plus ``search_failed`` / ``n_evaluations`` / ``n_successes`` for
        ``bo`` / ``evolutionary`` (see :class:`SearchResult`).
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
        return _search_bandit(space, train, val, objective, build_fn, n_iter=n_iter, seed=seed, **method_kwargs)
    else:
        raise ValueError(f"method must be 'bo' | 'evolutionary' | 'bandit' (got {method!r}).")

    best_config, best_model, best_canonical, history = result
    n_evaluations = len(history)
    n_successes = sum(1 for row in history if not row.get("failed", False))
    return SearchResult(
        best_config,
        best_model,
        native(best_canonical),
        history,
        search_failed=(n_successes == 0),
        n_evaluations=n_evaluations,
        n_successes=n_successes,
    )


def _search_bo(
    space: Space,
    train: list[Any],
    val: list[Any],
    objective: Objective,
    build_fn: Callable[[dict[str, Any], list[Any]], Any],
    *,
    n_iter: int,
    seed: int,
    n_init: int | None = None,
) -> tuple[dict[str, Any], Any, float, list[dict[str, Any]]]:
    """Drive :func:`mixle.doe.minimize` over the space's numeric box (categoricals as integer indices).

    ``n_iter`` is the TOTAL evaluation budget (initial design + acquisition steps): for ``n_iter >= 1``
    exactly ``n_iter`` configs are built/scored. BO cannot fit a surrogate -- let alone propose an
    acquisition step -- from zero observed points, so ``n_iter <= 0`` is rejected outright rather than
    silently spending an evaluation anyway (the previous behavior for ``n_iter=0``).
    """
    from mixle.doe import minimize

    n_iter = _positive_int(n_iter, "n_iter")

    bounds = space.to_bounds()
    history: list[dict[str, Any]] = []
    # cache the best model encountered (minimize only returns x/y, not the built object).
    best: dict[str, Any] = {"score": np.inf, "config": None, "model": None}

    def numeric_objective(x: np.ndarray) -> float:
        config = space.decode(x)
        canonical, model, error = _held_out_score(config, build_fn, train, val, objective)
        history.append({"config": config, "score": float(canonical), "failed": error is not None, "error": error})
        if canonical < best["score"]:
            best.update({"score": float(canonical), "config": config, "model": model})
        return float(canonical)

    n_init = max(2 * space.ndim + 1, 3) if n_init is None else int(n_init)
    n_init = min(n_init, n_iter)
    n_acq = max(0, n_iter - n_init)
    minimize(numeric_objective, bounds, n_init=n_init, n_iter=n_acq, seed=seed, maximize=False)
    return best["config"], best["model"], float(best["score"]), history


def _search_evolutionary(
    space: Space,
    train: list[Any],
    val: list[Any],
    objective: Objective,
    build_fn: Callable[[dict[str, Any], list[Any]], Any],
    *,
    n_iter: int,
    seed: int,
    mu: int = 4,
    lam: int = 8,
) -> tuple[dict[str, Any], Any, float, list[dict[str, Any]]]:
    """A (mu + lambda) evolutionary loop over ``Space.sample`` / ``Space.neighbors``.

    Maintains up to ``mu`` parents; each generation spawns up to ``lam`` offspring (a random neighbor
    of a random parent), evaluates them, and keeps the best ``mu`` of (parents + offspring).
    Categoricals are handled natively (no numeric rounding), so this is the backend for spaces BO
    encodes lossily.

    ``n_iter`` is the TOTAL evaluation budget -- the same meaning :func:`search` documents and the
    ``bo`` backend already honors -- NOT a generation count. It used to be the latter, so the loop
    spent ``mu + lam * n_iter`` evaluations: with ``n_iter=1, mu=4, lam=8`` the receipt reported 12
    successful evaluations against a stated budget of one. Every evaluation is now reserved against
    the budget before it is spent, so the initial parents are capped at ``min(mu, n_iter)`` and a
    generation stops mid-way once the budget is exhausted. ``n_iter <= 0`` buys nothing and is
    rejected outright.
    """
    n_iter = _positive_int(n_iter, "n_iter")
    mu = _positive_int(mu, "mu")
    lam = _positive_int(lam, "lam")

    rng = np.random.RandomState(seed)
    history: list[dict[str, Any]] = []

    def evaluate(config: dict[str, Any]) -> tuple[float, Any]:
        canonical, model, error = _held_out_score(config, build_fn, train, val, objective)
        history.append({"config": config, "score": float(canonical), "failed": error is not None, "error": error})
        return canonical, model

    # initial parents: random samples, capped by the budget (a population is worth nothing if seeding
    # it already overspends what the caller authorized).
    population: list[tuple[float, dict[str, Any], Any]] = []
    for _ in range(min(mu, n_iter)):
        cfg = space.sample(rng)
        score, model = evaluate(cfg)
        population.append((score, cfg, model))
    population.sort(key=lambda t: t[0])

    while len(history) < n_iter:
        offspring: list[tuple[float, dict[str, Any], Any]] = []
        for _ in range(lam):
            if len(history) >= n_iter:
                break
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
    train: list[Any],
    val: list[Any],
    objective: Objective,
    build_fn: Callable[[dict[str, Any], list[Any]], Any],
    *,
    n_iter: int,
    seed: int,
    operators: Sequence[ImprovementOperator] | None = None,
    size: int = 8,
    n_seeds: int = 3,
) -> SearchResult:
    """Delegate to an :class:`~mixle.evolve.population.OperatorBandit` via a :class:`Population`.

    The "space" here is *which operator to apply*, not a parameter box: ``build_fn`` instantiates a few
    seed structures from random configs (given ``train`` -- never ``val``, matching the ``bo`` /
    ``evolutionary`` backends), and the bandit-driven :class:`Population` evolves them, learning which
    operators pay off. Every generation fits challengers on ``train`` and gates + scores them on the
    disjoint ``val`` split (:meth:`~mixle.evolve.population.Population.step`'s ``verify_data``) -- a
    challenger is never verified against the same data it was just fit on. The returned
    :class:`SearchResult` carries the population champion.
    """
    from mixle.evolve.population import OperatorBandit, Population

    rng = np.random.RandomState(seed)
    seeds = []
    for _ in range(max(1, n_seeds)):
        cfg = space.sample(rng)
        try:
            seeds.append(build_fn(cfg, train))
        except Exception:  # noqa: BLE001
            continue
    if not seeds:
        raise ValueError("search(method='bandit'): build_fn produced no valid seed models from the space.")

    from mixle.evolve.operators import default_operators

    ops = list(operators) if operators is not None else default_operators()
    bandit = OperatorBandit(ops, seed=seed)
    pop = Population(seeds, objective=objective, operators=ops, bandit=bandit, size=size, seed=seed)
    return pop.run(train, val, generations=n_iter)


__all__ = ["auto_select", "search", "SearchResult"]
