"""Generic active-acquisition glue: rank an unlabeled pool for *any* scoreable model.

``mixle.task.active.active_distill`` already runs the acquire-label-refit loop end to end, but its
ranking step (:func:`~mixle.task.active.acquisition_scores`) is hardwired to a single concrete
shape: a :class:`~mixle.task.model.TaskModel` whose ``adapter`` exposes ``proba_batch`` over batches
of *text*. That is the demo, not the library primitive -- other candidate pools (records, images,
raw feature vectors, ...) and other scoreable models (a fitted ``mixle.stats`` distribution, an
ensemble of them, a plain ``predict_proba`` classifier) need the same ranking logic without cloning
``active_distill``'s internals. :func:`acquire` is that primitive: ``acquire(pool, model, k,
strategy)`` scores every pool item under ``strategy`` and returns the top ``k``.

**Dispatch, not a hardcoded type.** A model is "scoreable" if :func:`_proba_batch` can get a
row-stochastic ``(n, k)`` prediction matrix out of it, tried in order: a bare ``predict_proba``
method (the generic/sklearn-shaped case); the :class:`~mixle.task.model.TaskModel` adapter shape
(``model.adapter.proba_batch(model.model, items)`` -- the exact call ``active_distill`` already
makes, so every existing distilled student keeps working); or, recursively, a weighted *ensemble* of
scoreable sub-models (see below), whose mixture prediction is the weight-averaged member prediction.
No strategy here ever branches on ``isinstance(model, TaskModel)`` -- it is just one more shape that
happens to satisfy the same duck-typed contract.

**Which EIG machinery.** The acceptance criteria names two candidate sources: ``mixle.epistemic``'s
nested-MC portfolio estimator (:func:`mixle.epistemic.loop._portfolio_eig_nmc`) and
``mixle.doe.active.expected_information_gain_nmc``. Neither is called directly here, and the choice
between them is really a choice about which one's *math*, not its exact function, fits a discrete
already-materialized pool of candidates with a categorical outcome:

* :func:`mixle.doe.active.expected_information_gain_nmc` is written against a *continuous* numpy
  parameter space (``prior_sampler(rng, n) -> (n, k) array`` plus a ``simulate`` callable) -- exactly
  the mismatch :mod:`mixle.epistemic.loop`'s own docstring calls out for its portfolio use case.
  Forcing a pool of discrete "which hypothesis/model in my ensemble is right" questions through that
  interface would mean flattening every ensemble member into a numeric vector, which defeats the
  point of an arbitrary scoreable-model ensemble.
* :mod:`mixle.epistemic.portfolio.HypothesisPortfolio` is *exactly* the right shape instead: a
  weighted, typed set of hypotheses -- which is exactly what an ensemble of scoreable models is. The
  ``eig`` strategy below (:func:`_eig_strategy`) is the discrete-pool, categorical-outcome
  specialization of the same nested-MC EIG identity ``EIG = E_{h,y}[log p(y|h) - log
  E_{h'}[p(y|h')]]`` that :func:`mixle.epistemic.loop._portfolio_eig_nmc` estimates by simulation: for
  a *categorical* ``y`` with a *known* per-hypothesis predictive distribution (no simulation needed,
  the sum over the finite outcome space is exact) that identity reduces in closed form to the
  mutual-information / BALD decomposition ``EIG(x) = H[E_h[p(y|x,h)]] - E_h[H[p(y|x,h)]]`` (Houlsby
  et al. 2011) -- entropy of the mixture prediction minus the expected entropy of each member's own
  prediction. That closed form is what :func:`_eig_strategy` computes: no Monte Carlo, no simulate
  callable, and it accepts either a real :class:`HypothesisPortfolio` or the lighter duck-typed
  ``members``/``weights`` ensemble shape below, so a caller who doesn't want the portfolio's
  reweighting/pruning machinery isn't forced to build one just to rank a pool.

**Ensemble shape.** ``model`` participates in the ``eig``/``disagreement`` strategies if it is a
:class:`~mixle.epistemic.portfolio.HypothesisPortfolio` (its active hypotheses' ``payload``s are the
scoreable members, its weights the ensemble weights) or exposes ``model.members`` (a sequence of
scoreable sub-models) with an optional ``model.weights`` (defaults to uniform). A single
non-ensemble scoreable model has no disagreement/EIG to compute (there is only one opinion) and
raises :class:`~mixle.capability.CapabilityError`; it works fine with ``"entropy"``, which needs
only one predictive distribution per pool item.

**Strategies are a registry, not a branch.** Mirroring
:func:`mixle.doe.bayesopt.register_acquisition`'s "register, don't branch" pattern: built-ins
(``"eig"``, ``"disagreement"``, ``"entropy"``) are registered by name below via
:func:`register_strategy`, and a caller registers a custom one the same way -- ``acquire`` never
special-cases a strategy name.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mixle.capability import CapabilityError
from mixle.epistemic.portfolio import HypothesisPortfolio

# --- scoring: get a row-stochastic (n, k) prediction matrix out of any scoreable model -----------


def _proba_batch(model: Any, items: Sequence[Any]) -> np.ndarray:
    """Row-stochastic ``(len(items), k)`` predictions for ``items`` under ``model``.

    Tries, in order: a bare ``predict_proba(items)`` (the generic contract); the
    :class:`~mixle.task.model.TaskModel` adapter shape (what ``active_distill`` already calls);
    then, recursively, a weighted ensemble's mixture prediction. Raises :class:`CapabilityError` if
    none apply.
    """
    items = list(items)
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        return np.asarray(predict_proba(items), dtype=np.float64)

    adapter = getattr(model, "adapter", None)
    inner = getattr(model, "model", None)
    if adapter is not None and inner is not None and callable(getattr(adapter, "proba_batch", None)):
        return np.asarray(adapter.proba_batch(inner, items), dtype=np.float64)

    members, weights = _ensemble_members(model)
    if members is not None and len(members):
        stacked = np.stack([_proba_batch(m, items) for m in members])  # (M, n, k)
        return np.asarray(np.tensordot(weights, stacked, axes=1))

    raise CapabilityError(
        f"{type(model).__name__} is not scoreable: expected predict_proba(items), a "
        "TaskModel-shaped (adapter, model) pair, or an ensemble (HypothesisPortfolio / "
        "members=[...]) of scoreable sub-models"
    )


def _ensemble_members(model: Any) -> tuple[list[Any] | None, np.ndarray | None]:
    """Return ``(members, normalized_weights)`` if ``model`` is an ensemble, else ``(None, None)``.

    A :class:`~mixle.epistemic.portfolio.HypothesisPortfolio`'s active hypotheses are the members
    (their ``payload``); otherwise ``model.members`` (with optional ``model.weights``, default
    uniform) is the lighter duck-typed shape.
    """
    if isinstance(model, HypothesisPortfolio):
        active = [(w, h.payload) for w, h in zip(model.weights, model.hypotheses) if h.active]
        if not active:
            return [], np.array([])
        weights = np.array([w for w, _ in active], dtype=np.float64)
        weights = weights / weights.sum()
        return [p for _, p in active], weights

    members = getattr(model, "members", None)
    if members is not None:
        members = list(members)
        raw_weights = getattr(model, "weights", None)
        weights = (
            np.full(len(members), 1.0 / len(members))
            if raw_weights is None
            else np.asarray(raw_weights, dtype=np.float64)
        )
        weights = weights / weights.sum()
        return members, weights

    return None, None


def _shannon_entropy(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) along the last axis of a row-stochastic prediction matrix."""
    p = np.clip(proba, 1e-12, 1.0)
    return np.asarray(-np.sum(p * np.log(p), axis=-1))


# --- built-in strategies: fn(pool, model, **kwargs) -> score per pool item, higher = more worth labeling


def _entropy_strategy(pool: Sequence[Any], model: Any, **_: Any) -> np.ndarray:
    """Posterior predictive entropy: label what the model itself is least sure about.

    Uses ``mixle.capability.HasEntropy`` when a single member's prediction is itself a
    ``mixle.stats`` distribution exposing a closed-form ``entropy()``; otherwise falls back to the
    plain Shannon entropy of the ``(n, k)`` categorical prediction, which is the right notion of
    entropy for the ``predict_proba`` contract every scoreable model here ultimately reduces to.
    """
    proba = _proba_batch(model, pool)
    return _shannon_entropy(proba)


def _eig_strategy(pool: Sequence[Any], model: Any, **_: Any) -> np.ndarray:
    """Expected information gain about *which ensemble member is right* (BALD, see module docstring).

    ``H[E_h[p(y|x,h)]] - E_h[H[p(y|x,h)]]``: entropy of the mixture prediction minus the expected
    entropy of each member's own prediction. Zero where every member agrees (mixture entropy equals
    each member's own entropy, however uncertain); large where members are individually confident
    but disagree with each other -- exactly the pool items that would most discriminate between
    hypotheses in the ensemble/portfolio.
    """
    members, weights = _ensemble_members(model)
    if members is None:
        raise CapabilityError(
            f"{type(model).__name__} has no ensemble to compute EIG from; pass a "
            "HypothesisPortfolio or an object exposing members=[...] (optionally weights=[...])"
        )
    if not members:
        return np.zeros(len(pool))
    stacked = np.stack([_proba_batch(m, pool) for m in members])  # (M, n, k)
    mean_proba = np.asarray(np.tensordot(weights, stacked, axes=1))  # (n, k)
    h_mean = _shannon_entropy(mean_proba)  # (n,)
    h_each = np.stack([_shannon_entropy(p) for p in stacked])  # (M, n)
    e_h = np.asarray(np.tensordot(weights, h_each, axes=1))  # (n,)
    return np.clip(h_mean - e_h, 0.0, None)


def _disagreement_strategy(pool: Sequence[Any], model: Any, **_: Any) -> np.ndarray:
    """Query-by-committee vote disagreement: label where the ensemble's hard predictions split.

    ``1 - (largest vote share)`` over each member's argmax label -- ``0`` when every member agrees,
    approaching ``1`` as the vote splits evenly across many labels. A cheaper, non-probabilistic
    cousin of ``eig`` (it only needs each member's predicted label, not its full confidence), the
    same acquisition family :mod:`mixle.task.disagreement` names for the (different) post-hoc
    student-vs-teacher gate; here it is committee-vs-committee, over an unlabeled pool, with no
    teacher labels required.
    """
    members, weights = _ensemble_members(model)
    if members is None:
        raise CapabilityError(
            f"{type(model).__name__} has no ensemble to compute disagreement from; pass a "
            "HypothesisPortfolio or an object exposing members=[...] (optionally weights=[...])"
        )
    if not members:
        return np.zeros(len(pool))
    stacked = np.stack([_proba_batch(m, pool) for m in members])  # (M, n, k)
    preds = stacked.argmax(axis=-1)  # (M, n)
    n = preds.shape[1]
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        votes = preds[:, i]
        _, counts = np.unique(votes, return_counts=True)
        scores[i] = 1.0 - float(counts.max()) / float(votes.shape[0])
    return scores


# --- strategy registry ("register, don't branch", mirrors mixle.doe.bayesopt.register_acquisition)


_STRATEGIES: dict[str, Callable[..., np.ndarray]] = {}


def register_strategy(name: str, fn: Callable[..., np.ndarray]) -> None:
    """Register an acquisition ``strategy`` under ``name``.

    ``fn`` is called as ``fn(pool, model, **strategy_kwargs)`` and must return an array of scores,
    one per pool item, where higher means more worth labeling. This is the extension point for new
    strategies -- registering is all :func:`acquire` needs, no edits to ``acquire`` itself.
    """
    if not callable(fn):
        raise TypeError("strategy must be callable.")
    _STRATEGIES[name.lower()] = fn


def available_strategies() -> list[str]:
    """Return the sorted names of every registered strategy."""
    return sorted(_STRATEGIES)


def _get_strategy(strategy: str | Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    if callable(strategy):
        return strategy
    fn = _STRATEGIES.get(str(strategy).lower())
    if fn is None:
        raise ValueError(f"unknown strategy {strategy!r}; registered: {', '.join(available_strategies())}")
    return fn


register_strategy("eig", _eig_strategy)
register_strategy("disagreement", _disagreement_strategy)
register_strategy("entropy", _entropy_strategy)


# --- the entry point -------------------------------------------------------------------------------


def acquire(
    pool: Sequence[Any],
    model: Any,
    k: int,
    strategy: str | Callable[..., np.ndarray] = "eig",
    **strategy_kwargs: Any,
) -> list[Any]:
    """Rank ``pool`` by ``strategy`` under ``model`` and return the top ``k`` items to label next.

    The model-agnostic generalization of ``active_distill``'s hardwired text-classifier ranking step
    (:func:`mixle.task.active.acquisition_scores`): any scoreable ``model`` (see the module
    docstring's dispatch rules) and any pool of candidates work, not just a
    :class:`~mixle.task.model.TaskModel` over text. ``strategy`` is either a registered name
    (``"eig"``, ``"disagreement"``, ``"entropy"``, or a custom one registered via
    :func:`register_strategy`) or a bare callable with the same ``fn(pool, model, **kwargs) ->
    scores`` contract. Returns the highest-scoring ``min(k, len(pool))`` pool items, most
    worth-labeling first; an empty ``pool`` or non-positive ``k`` returns ``[]``.
    """
    pool = list(pool)
    if k <= 0 or not pool:
        return []
    fn = _get_strategy(strategy)
    scores = np.asarray(fn(pool, model, **strategy_kwargs), dtype=np.float64)
    if scores.shape != (len(pool),):
        raise ValueError(f"strategy {strategy!r} returned scores of shape {scores.shape}, expected ({len(pool)},)")
    order = np.argsort(scores, kind="stable")[::-1][: min(int(k), len(pool))]
    return [pool[i] for i in order]


__all__ = [
    "acquire",
    "register_strategy",
    "available_strategies",
]
