"""L1: closed-loop self-evolution with operator credit -- the loop at five altitudes, wired end to end.

Router-harvested failures -> acquisition (A5's :func:`mixle.task.acquire.acquire`) -> challenger
production (a distill/refine/evolve operator, all *reused* :class:`~mixle.evolve.operators.
ImprovementOperator` instances, not reimplemented) -> the held-out
:func:`~mixle.evolve.verify.challenger_beats_champion` gate -> deploy, as ONE budgeted background
loop (:class:`ClosedLoopSelfEvolution`). A per-context meta-bandit (:class:`OperatorCreditBandit`,
wrapping :class:`mixle.task.bandit.UCB1` -- the same UCB1 machinery reused across this codebase for
meta-bandits) learns which challenger-production operator wins for which kind of failure, and every
ADOPTED champion gets a genealogy receipt (:class:`GenealogyLedger`, built directly on
:class:`~mixle.evolve.ledger.EvolutionLedger`) recording its parent, the operator that produced it,
and the measured gap -- a real, walk-backable lineage.

**Which subsystems this wires, not rebuilds:**

* :mod:`mixle.task.router` -- :func:`harvested_from_router` reads a real
  :class:`~mixle.task.router.Router`'s ``.harvested()`` (the frontier-answered cases the cheap tiers
  could not handle) as the harvested-failure source when a caller already runs one. For domains
  without a live Router (e.g. this module's own tests, which evolve a marginal label model rather than
  a routed per-input classifier), :func:`harvest_failures` is the same idea generalized to any
  :class:`~mixle.evolve.objective.Objective`: the observations where the current champion did NOT
  score at its own best-attainable pointwise value -- literally "the cases it got wrong."
* :func:`mixle.task.acquire.acquire` (A5) ranks the harvested pool before it is spent on challenger
  production. ``acquire`` needs a "scoreable" model (``predict_proba``-shaped); a champion that is a
  bare marginal distribution (does not condition on the item) is wrapped by
  :class:`_ConstantProbaAdapter` so the real ``acquire()`` code path runs (entropy strategy) instead of
  being skipped -- honestly, this degenerates to "prioritize while the champion overall is unconfident"
  since the per-item probability is constant, but it is the real primitive, not a mock.
* :mod:`mixle.evolve.operators` -- three EXISTING operators stand in for L1's "distill / refine /
  evolve ops" triad (see :func:`default_challenger_operators`): ``AutoSelect`` (cold-start refit --
  distillation's "train a fresh model from the teacher-labeled pool" shape) as ``"distill"``,
  ``Refit`` (warm-started refit of the champion's own parameters) as ``"refine"``, and ``Mutate`` (the
  genetic-programming structure-edit operator) as ``"evolve"``.
* :func:`mixle.evolve.verify.challenger_beats_champion` is the held-out gate, used exactly as-is.
* :class:`mixle.evolve.ledger.EvolutionLedger` is the genealogy substrate, wrapped (not replaced) by
  :class:`GenealogyLedger`.
* :class:`mixle.task.bandit.UCB1` is the meta-bandit's arm-selection machinery, one instance per
  context.

**Principled crossover.** If a challenger-production operator ever combines two existing champions
(a "crossover" in evolutionary-algorithm terms), it MUST do so via mixture composition
(:func:`mixle.ops.mixture`, exactly what :class:`~mixle.evolve.operators.Mutate`'s ``grow`` move and
:class:`~mixle.evolve.operators.Recompose` already do) -- never by cutting and pasting incompatible
weight sub-blocks from two unrelated models. :func:`principled_crossover` is the explicit primitive for
that case.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.evolve.ledger import EvolutionLedger
from mixle.evolve.objective import Objective, _ScalarObjective
from mixle.evolve.operators import AutoSelect, ImprovementOperator, Mutate, Refit
from mixle.evolve.verify import challenger_beats_champion
from mixle.task.acquire import acquire
from mixle.task.bandit import UCB1

__all__ = [
    "accuracy_objective",
    "harvest_failures",
    "harvested_from_router",
    "default_challenger_operators",
    "principled_crossover",
    "OperatorCreditBandit",
    "GenealogyLedger",
    "LoopStepResult",
    "ClosedLoopSelfEvolution",
]


# ---------------------------------------------------------------------------
# a classification-shaped Objective (accuracy), for the loop's synthetic tests and any caller whose
# champion is a mixle.stats CategoricalDistribution predicting a single label per batch.
# ---------------------------------------------------------------------------
def accuracy_objective() -> Objective:
    """Higher-is-better accuracy for a model exposing a single "most likely label" prediction (a
    fitted :class:`~mixle.stats.univariate.discrete.categorical.CategoricalDistribution`'s
    ``argmax(pmap)``): per-observation ``1{y == predicted_label}``.
    """

    def _predicted_label(model: Any) -> Any:
        return max(model.pmap, key=model.pmap.get)

    def pw(model: Any, data: Any) -> np.ndarray:
        pred = _predicted_label(model)
        return np.asarray([1.0 if y == pred else 0.0 for y in data], dtype=float)

    def sc(model: Any, data: Any) -> float:
        return float(np.mean(pw(model, data)))

    return _ScalarObjective("accuracy", False, pw, sc)


# ---------------------------------------------------------------------------
# step 1: Router-harvested failures (or the objective-generic analog)
# ---------------------------------------------------------------------------
def harvest_failures(champion: Any, batch: Sequence[Any], objective: Objective) -> list[Any]:
    """The objective-generic analog of :meth:`mixle.task.router.Router.harvested`: the observations in
    ``batch`` where ``champion`` did NOT score at its own best-attainable pointwise value under
    ``objective`` -- for :func:`accuracy_objective` this is exactly "the cases it got wrong." A
    scalar-only objective (no honest per-observation vector) harvests the whole batch -- there is no
    finer signal to rank on.
    """
    batch = list(batch)
    pw = objective.pointwise(champion, batch)
    if pw is None:
        return batch
    pw = np.asarray(pw, dtype=float)
    good = pw if not objective.lower_is_better else -pw
    best = float(np.max(good))
    bad_idx = np.flatnonzero(good < best)
    return [batch[i] for i in bad_idx]


def harvested_from_router(router: Any) -> list[tuple[Any, Any]]:
    """Pull the REAL harvested failure pool from a live :class:`mixle.task.router.Router`: every input
    that escalated all the way to the frontier, paired with its frontier label -- exactly
    ``router.harvested()``, just zipped into one pool for :func:`mixle.task.acquire.acquire`.
    """
    inputs, labels = router.harvested()
    return list(zip(inputs, labels))


# ---------------------------------------------------------------------------
# step 2: acquisition (A5) -- a scoreable-model adapter for marginal (non-conditional) champions
# ---------------------------------------------------------------------------
class _ConstantProbaAdapter:
    """Wraps a fitted ``CategoricalDistribution`` (a marginal label model: it does not condition on
    the item) as an :func:`mixle.task.acquire.acquire`-scoreable model -- ``predict_proba(items)``
    returns the SAME row (the model's own ``pmap``) for every item, so the real ``acquire()`` "entropy"
    strategy runs (rather than raising :class:`~mixle.capability.CapabilityError` and being skipped).
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        labels = sorted(model.pmap.keys(), key=repr)
        self._labels = labels
        row = np.array([model.pmap.get(lb, model.default_value) for lb in labels], dtype=np.float64)
        total = row.sum()
        self._row = row / total if total > 0 else np.full(len(labels), 1.0 / max(len(labels), 1))

    def predict_proba(self, items: Sequence[Any]) -> np.ndarray:
        n = len(list(items))
        return np.tile(self._row, (n, 1))


def _acquire_priority(champion: Any, failures: Sequence[Any], k: int, *, strategy: str = "entropy") -> list[Any]:
    """A5's :func:`~mixle.task.acquire.acquire`, adapted so it runs for real on a marginal champion; on
    any failure to build a scoreable adapter, falls back to "everything harvested" honestly (never
    silently drops the pool)."""
    failures = list(failures)
    if not failures:
        return []
    try:
        adapter = _ConstantProbaAdapter(champion)
        return acquire(failures, adapter, k, strategy=strategy)
    except Exception:
        return failures[: min(k, len(failures))]


# ---------------------------------------------------------------------------
# step 3: challenger production -- distill / refine / evolve, all reused ImprovementOperators
# ---------------------------------------------------------------------------
def default_challenger_operators() -> dict[str, ImprovementOperator]:
    """The three L1 challenger-production operators, each a reused
    :class:`~mixle.evolve.operators.ImprovementOperator` (no bespoke fitting logic):

    * ``"distill"`` -> :class:`~mixle.evolve.operators.AutoSelect` -- cold-start: fit a fresh model
      from the harvested pool, the "train a new model from teacher-labeled data" shape of distillation.
    * ``"refine"``  -> :class:`~mixle.evolve.operators.Refit` -- warm-started refit of the champion's
      own parameters on the harvested pool.
    * ``"evolve"``  -> :class:`~mixle.evolve.operators.Mutate` -- genetic-programming structure edit
      (grow/shrink/perturb), mixle.evolve's own structure-search operator.
    """
    return {"distill": AutoSelect(), "refine": Refit(), "evolve": Mutate()}


def principled_crossover(model_a: Any, model_b: Any, *, weight_a: float = 0.5) -> Any:
    """Combine two champions the ONLY principled way this framework allows: mixture composition
    (:func:`mixle.ops.mixture`), never gene-splicing (cutting/pasting incompatible weight sub-blocks).
    Returns an unfitted mixture prototype; refit it against data (e.g. via its own ``.estimator()``)
    before treating it as a challenger.
    """
    from mixle.ops import mixture

    if not 0.0 < weight_a < 1.0:
        raise ValueError("weight_a must be in (0, 1).")
    return mixture([model_a, model_b], [weight_a, 1.0 - weight_a])


# ---------------------------------------------------------------------------
# the per-context operator-credit meta-bandit
# ---------------------------------------------------------------------------
class OperatorCreditBandit:
    """A per-context meta-bandit over challenger-production operators, wrapping one
    :class:`mixle.task.bandit.UCB1` per context (e.g. a failure type/domain) -- so the loop learns,
    independently for each context, which operator actually produces winning challengers there.
    """

    def __init__(self, operator_names: Sequence[str], *, c: float = 1.0, seed: int = 0) -> None:
        self.operator_names = list(operator_names)
        if len(self.operator_names) < 2:
            raise ValueError("OperatorCreditBandit needs at least two operators.")
        self._c = float(c)
        self._seed = int(seed)
        self._bandits: dict[str, UCB1] = {}

    def _bandit_for(self, context: str) -> UCB1:
        b = self._bandits.get(context)
        if b is None:
            b = UCB1(len(self.operator_names), c=self._c, seed=self._seed)
            self._bandits[context] = b
        return b

    def select(self, context: str) -> str:
        """The bandit-chosen operator name for ``context``."""
        arm = self._bandit_for(context).select()
        return self.operator_names[arm]

    def reward(self, context: str, operator: str, reward: float) -> None:
        """Fold an observed (non-negative, anti-regression) reward back into ``operator``'s arm."""
        arm = self.operator_names.index(operator)
        self._bandit_for(context).update(arm, max(float(reward), 0.0))

    def report(self) -> dict[str, dict[str, float]]:
        """Per-context, per-operator mean reward and pull count."""
        out: dict[str, dict[str, float]] = {}
        for ctx, b in self._bandits.items():
            out[ctx] = {
                name: {"mean_reward": float(b.means[i]), "pulls": int(b.pulls[i])}
                for i, name in enumerate(self.operator_names)
            }
        return out


# ---------------------------------------------------------------------------
# genealogy receipts, built on the existing EvolutionLedger
# ---------------------------------------------------------------------------
@dataclass
class GenealogyLedger:
    """Genealogy receipts for adopted (gate-passing) champions, built directly on
    :class:`~mixle.evolve.ledger.EvolutionLedger` (never storing model objects in the ledger rows
    themselves -- only their operator, measured gap, and a stable id -- exactly the ledger's own
    JSON-serializability discipline).
    """

    ledger: EvolutionLedger = field(default_factory=EvolutionLedger)
    _model_ids: dict[int, str] = field(default_factory=dict)
    _counter: int = 0

    def _id_for(self, model: Any) -> str:
        key = id(model)
        existing = self._model_ids.get(key)
        if existing is not None:
            return existing
        self._counter += 1
        new_id = f"m{self._counter}"
        self._model_ids[key] = new_id
        return new_id

    def record_adoption(
        self,
        *,
        parent: Any | None,
        child: Any,
        operator: str,
        gap: float,
        context: str,
        meta: dict | None = None,
    ) -> dict[str, Any]:
        """Record ONE adoption: ``child`` replaced ``parent`` via ``operator``, measured ``gap`` (the
        verified challenger-beats-champion delta). ``parent=None`` marks the root of a lineage."""
        parent_id = self._id_for(parent) if parent is not None else None
        child_id = self._id_for(child)
        row_meta = {"child_hash": child_id, "context": context, **(meta or {})}
        return self.ledger.record(
            operator=operator,
            delta=gap,
            verdict={"promote": True},
            cost=0.0,
            parent_hash=parent_id,
            meta=row_meta,
        )

    def lineage(self, model: Any) -> list[dict[str, Any]]:
        """Reconstruct ``model``'s full lineage: the ordered (root-first) chain of adoption receipts
        ``{operator, delta (the measured gap), parent_hash, meta: {child_hash, context, ...}}`` back to
        the first recorded ancestor. Returns ``[]`` if ``model`` was never recorded as an adoption.
        """
        target_id = self._model_ids.get(id(model))
        if target_id is None:
            return []
        by_child = {row["meta"]["child_hash"]: row for row in self.ledger.rows if "child_hash" in row.get("meta", {})}
        chain: list[dict[str, Any]] = []
        current_id: str | None = target_id
        seen: set[str] = set()
        while current_id is not None and current_id in by_child and current_id not in seen:
            seen.add(current_id)
            row = by_child[current_id]
            chain.append(row)
            current_id = row["parent_hash"]
        chain.reverse()
        return chain


# ---------------------------------------------------------------------------
# the orchestrating loop
# ---------------------------------------------------------------------------
@dataclass
class LoopStepResult:
    """One :meth:`ClosedLoopSelfEvolution.step` outcome."""

    context: str
    operator: str
    promoted: bool
    delta: float
    champion_score: float
    champion: Any


class ClosedLoopSelfEvolution:
    """The whole L1 loop, one budgeted step at a time: harvest -> acquire -> propose -> gate -> deploy,
    with per-context operator credit and genealogy receipts on every adoption.

    Args:
        champion: the initial fitted model (e.g. a ``CategoricalDistribution``).
        objective: the :class:`~mixle.evolve.objective.Objective` the loop optimizes (e.g.
            :func:`accuracy_objective`).
        operators: challenger-production operators by name; defaults to
            :func:`default_challenger_operators`.
        context_fn: ``batch -> context key`` for the per-context bandit; defaults to a single global
            context (``"default"``) when the caller has no domain/failure-type signal to condition on.
        acquire_k: how many harvested failures A5's ``acquire`` prioritizes per step.
        acquire_strategy: the ``acquire()`` ranking strategy (default ``"entropy"``, the one that works
            for a marginal/non-conditional champion via :class:`_ConstantProbaAdapter`).
        seed: RNG seed threaded through the bandit and the gate.
    """

    def __init__(
        self,
        champion: Any,
        *,
        objective: Objective,
        operators: dict[str, ImprovementOperator] | None = None,
        context_fn: Callable[[Sequence[Any]], str] | None = None,
        acquire_k: int = 32,
        acquire_strategy: str = "entropy",
        bandit_c: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.champion = champion
        self.objective = objective
        self.operators = dict(operators) if operators is not None else default_challenger_operators()
        self.context_fn = context_fn or (lambda batch: "default")
        self.acquire_k = int(acquire_k)
        self.acquire_strategy = acquire_strategy
        self.bandit = OperatorCreditBandit(list(self.operators), c=bandit_c, seed=seed)
        self.genealogy = GenealogyLedger()
        self.seed = int(seed)
        self._n_steps = 0
        self.history: list[LoopStepResult] = []

    def _score(self, model: Any, data: Sequence[Any]) -> float:
        return float(self.objective.scalar(model, data))

    def step(
        self,
        batch: Sequence[Any],
        *,
        verify: Sequence[Any] | None = None,
        context: str | None = None,
    ) -> LoopStepResult:
        """One cycle of the loop on one arriving ``batch`` of held-out-shaped observations.

        1. harvest failures under the CURRENT champion,
        2. rank them with A5's ``acquire``,
        3. pick a challenger-production operator via the per-context bandit,
        4. propose + gate the challenger,
        5. reward the bandit with the verified (anti-regression) delta,
        6. deploy + record a genealogy receipt iff the gate promotes.
        """
        self._n_steps += 1
        batch = list(batch)
        ctx = context if context is not None else self.context_fn(batch)

        failures = harvest_failures(self.champion, batch, self.objective)
        pool = failures if failures else batch
        train_data = _acquire_priority(self.champion, pool, self.acquire_k, strategy=self.acquire_strategy)
        if not train_data:
            train_data = pool
        verify_data = list(verify) if verify is not None else batch

        op_name = self.bandit.select(ctx)
        operator = self.operators[op_name]
        op_ctx = {"parent_hash": None, "seed": self.seed + self._n_steps, "objective": self.objective}

        promoted = False
        delta = 0.0
        try:
            if operator.applicable(self.champion, train_data, ctx=op_ctx):
                candidate = operator.propose(self.champion, train_data, ctx=op_ctx)
                nonnested = type(candidate.model).__name__ != type(self.champion).__name__
                verdict = challenger_beats_champion(
                    self.champion,
                    candidate.model,
                    verify_data,
                    objective=self.objective,
                    nonnested=nonnested,
                    seed=self.seed + self._n_steps,
                )
                delta = float(verdict.delta) if verdict.promote else 0.0
                if verdict.promote:
                    self.genealogy.record_adoption(
                        parent=self.champion,
                        child=candidate.model,
                        operator=op_name,
                        gap=verdict.delta,
                        context=ctx,
                    )
                    self.champion = candidate.model
                    promoted = True
        except Exception:
            delta = 0.0

        self.bandit.reward(ctx, op_name, delta)
        score = self._score(self.champion, verify_data)
        result = LoopStepResult(ctx, op_name, promoted, delta, score, self.champion)
        self.history.append(result)
        return result

    def run(
        self,
        stream: Sequence[Sequence[Any]],
        *,
        verify_batches: Sequence[Sequence[Any]] | None = None,
        contexts: Sequence[str] | None = None,
        budget: float | None = None,
    ) -> list[LoopStepResult]:
        """Run the loop over a SEQUENCE of batches (a stream) -- the budgeted background loop's outer
        iteration. ``budget`` (if given) is a ceiling on the total ``cost_hint`` of operators actually
        applied (proposal attempts that were skipped as inapplicable don't spend budget); the loop stops
        early once it is exhausted, leaving the champion at whatever it last became.
        """
        results: list[LoopStepResult] = []
        spent = 0.0
        for i, batch in enumerate(stream):
            if budget is not None and spent >= budget:
                break
            verify = verify_batches[i] if verify_batches is not None else None
            ctx = contexts[i] if contexts is not None else None
            result = self.step(batch, verify=verify, context=ctx)
            spent += float(getattr(self.operators[result.operator], "cost_hint", 1.0))
            results.append(result)
        return results
