"""``mixle.evolve`` -- the self-improvement core: measure -> propose -> verify -> promote.

A pure, serving-agnostic *orchestration algebra* over capabilities that already exist in the library.
It adds no new modeling capability: the proper scores and calibration diagnostics
(:mod:`mixle.inference.scoring` / :mod:`mixle.inference.calibration`), the comparison tests
(:mod:`mixle.inference.model_comparison`), the fitters (:func:`mixle.inference.estimation.optimize`,
the streaming estimators, :func:`mixle.utils.automatic.get_estimator`), and the decision layer
(:func:`mixle.inference.decision.bayes_action`) are all pre-existing. ``evolve`` wires them into a
single anti-regression loop.

Core surface:

* **measure** -- :class:`Objective` + ``nll`` / ``log_score`` / ``crps`` / ``interval`` / ``calibration``
  / ``decision_regret`` builders.
* **propose** -- :class:`ImprovementOperator` + :class:`Refit`, :class:`OnlineUpdate`,
  :class:`AutoSelect`, :class:`Recalibrate`, a scoped operator registry.
* **verify** -- :func:`challenger_beats_champion` + :class:`Verdict` (the anti-regression gate).
* **drive** -- :func:`improve` + :class:`ImprovementResult`, :func:`auto_select`, and the
  :class:`EvolutionLedger` telemetry.

Search and population surface:

* **search** -- :func:`search` over a typed :class:`Space` (:class:`Real` / :class:`Integer` /
  :class:`Categorical`), ``method='evolutionary'`` / ``'bandit'`` / ``'bo'``, returning a :class:`SearchResult`.
* **meta-search** -- :class:`Population` + :class:`OperatorBandit` (the policy that learns which operators help).
* **structure search** -- :class:`Recompose` / :class:`Mutate` (genetic-programming structural moves over the
  model's compositional tree) + :func:`structural_distance` (a tree-edit genotype distance) driving the
  population's diversity; registered but off by default (structural + expensive).

L1 surface (closed-loop self-evolution, :mod:`mixle.evolve.closed_loop`):

* :class:`ClosedLoopSelfEvolution` -- harvest -> A5 acquire -> distill/refine/evolve challenger
  production -> :func:`challenger_beats_champion` -> deploy, as one budgeted background loop.
* :class:`OperatorCreditBandit` -- a per-context meta-bandit (over :class:`mixle.task.bandit.UCB1`)
  crediting WHICH challenger-production operator wins for which kind of failure.
* :class:`GenealogyLedger` -- parent/operator/measured-gap receipts on every adopted champion, with a
  real ``lineage(model)`` walk-back.
"""

from __future__ import annotations

from mixle.evolve.closed_loop import (
    ClosedLoopSelfEvolution,
    GenealogyLedger,
    LoopStepResult,
    OperatorCreditBandit,
    accuracy_objective,
    default_challenger_operators,
    harvest_failures,
    harvested_from_router,
    principled_crossover,
)
from mixle.evolve.concept_discovery import (
    AdmissionEvent,
    ConceptLibrary,
    TaskResult,
    run_concept_discovery_loop,
    task_signature,
)
from mixle.evolve.improve import ImprovementResult, improve
from mixle.evolve.ledger import EvolutionLedger
from mixle.evolve.objective import (
    Objective,
    calibration_objective,
    crps_objective,
    decision_regret_objective,
    interval_objective,
    log_score_objective,
    nll_objective,
)
from mixle.evolve.operators import (
    AutoSelect,
    Candidate,
    ImprovementOperator,
    Mutate,
    OnlineUpdate,
    Recalibrate,
    Recompose,
    Refit,
    default_operators,
    register_operator,
    registered_operators,
    unregister_operator,
)
from mixle.evolve.population import OperatorBandit, Population
from mixle.evolve.search import SearchResult, auto_select, search
from mixle.evolve.space import Categorical, Integer, Real, Space
from mixle.evolve.structure import model_signature, structural_distance, tree_edit_distance
from mixle.evolve.verify import Verdict, challenger_beats_champion

__all__ = [
    # one-shot loop
    "improve",
    "ImprovementResult",
    # data -> better mixle model (held-out gated)
    "auto_select",
    # measure
    "Objective",
    "nll_objective",
    "log_score_objective",
    "crps_objective",
    "interval_objective",
    "calibration_objective",
    "decision_regret_objective",
    # verify gate
    "challenger_beats_champion",
    "Verdict",
    # propose contract + operators
    "ImprovementOperator",
    "Candidate",
    "Refit",
    "OnlineUpdate",
    "AutoSelect",
    "Recalibrate",
    "Recompose",
    "Mutate",
    "structural_distance",
    "model_signature",
    "tree_edit_distance",
    "register_operator",
    "unregister_operator",
    "registered_operators",
    "default_operators",
    # telemetry
    "EvolutionLedger",
    # Search over a typed space + the meta-search that learns which operators help.
    "search",
    "SearchResult",
    "Space",
    "Real",
    "Integer",
    "Categorical",
    "Population",
    "OperatorBandit",
    # L1: closed-loop self-evolution with operator credit + genealogy
    "ClosedLoopSelfEvolution",
    "LoopStepResult",
    "OperatorCreditBandit",
    "GenealogyLedger",
    "accuracy_objective",
    "harvest_failures",
    "harvested_from_router",
    "default_challenger_operators",
    "principled_crossover",
    # CARD L6: concept discovery -- the library of families itself under selection
    "ConceptLibrary",
    "AdmissionEvent",
    "TaskResult",
    "run_concept_discovery_loop",
    "task_signature",
]
