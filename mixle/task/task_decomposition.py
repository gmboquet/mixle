"""Task decomposition as structure learning (CARD L4): does the loop's five altitudes.

A task is a joint over ``(inputs, proposed intermediates, output)``. "Decomposition" is not a
heuristic split -- it is DEPENDENCY-FOREST DISCOVERY over that joint, reusing the exact machinery
:mod:`mixle.inference.structure` already built for heterogeneous records: :func:`dependency_gain`
scores whether an edge (parent field -> child field) is worth its BIC complexity in nats, the same
model-based, family-agnostic test used for category->real, count->binary, or any other pair. A task's
"is this decomposition good" question is the SAME question that module already answers for record
fields -- routing ``output`` through a candidate intermediate is exactly a dependency edge, and MDL
gain (this module's acceptance metric) IS the sum of :func:`~mixle.inference.structure.dependency_gain`
along the discovered edges, by construction: a good decomposition is one whose edges pay for their own
complexity in compressed nats.

Three pieces of existing machinery are integrated here, not rebuilt:

* :mod:`mixle.inference.structure` (``dependency_gain`` / ``regression_gain`` / ``fit_linear_gaussian_edge``)
  -- scores and fits every edge (input -> intermediate, intermediate -> output) this module considers.
* :mod:`mixle.task.plan_model` / :mod:`mixle.task.outcome_decomposer` -- a "decomposition" (which
  intermediates were used, in what order) is exactly a "plan" (which tool types were used, in what
  order): :class:`~mixle.task.plan_model.PlanModel` fits a distribution over decompositions the same
  way it fits one over tool sequences, and :class:`DecompositionProposer` refits on high-outcome
  decompositions the same round-based way :func:`~mixle.task.outcome_decomposer.train_outcome_decomposer`
  refits a plan model on successful traces -- so as ``(decomposition, outcome)`` pairs get logged from
  real task instances, FUTURE proposals shift toward what actually worked.
* :mod:`mixle.task.design_prior` (``record_accepted_recipe`` / ``rank_design_families``) -- a decomposed
  vs. monolithic recipe is recorded under a ``family`` tag on the existing design ledger, so which
  approach has actually won persists across tasks the same way a structural-family prior does elsewhere.

    forest = discover_decomposition(examples, candidate_intermediates)
    forest.chosen                                  # ["m1", "m2"], not [] (monolithic) -- for a genuinely
                                                     # decomposable task
    forest.mdl_gain                                 # nats: positive means the decomposition compresses
    proposer = record_decomposition_outcome(proposer, forest.chosen, outcome)  # trains the proposer
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from mixle.inference.structure import (
    LinearGaussianEdge,
    dependency_gain,
    fit_linear_gaussian_edge,
    regression_gain,
)
from mixle.task.design_prior import record_accepted_recipe
from mixle.task.edge import DesignModel
from mixle.task.plan_model import PlanModel, fit_plan_model
from mixle.task.traces import AgentTrace

MONOLITHIC = "__monolithic__"  # the "no decomposition, solve output directly from inputs" family tag


@dataclass
class TaskExample:
    """One observed instance of a task: named inputs and the realized output. The joint this module
    reasons over is ``(inputs, proposed_intermediates, output)`` -- ``proposed_intermediates`` are not
    stored here, they are RECOMPUTED per candidate by :func:`discover_decomposition` (a candidate
    intermediate is a function of ``inputs``, not a fixed observed field)."""

    inputs: Mapping[str, float]
    output: float


CandidateIntermediates = Mapping[str, Callable[[Mapping[str, float]], float]]


def _materialize(examples: Sequence[TaskExample], fn: Callable[[Mapping[str, float]], float]) -> list[float]:
    return [float(fn(ex.inputs)) for ex in examples]


def _input_columns(examples: Sequence[TaskExample]) -> dict[str, list[float]]:
    keys = sorted({k for ex in examples for k in ex.inputs})
    return {k: [float(ex.inputs[k]) for ex in examples] for k in keys}


def _output_column(examples: Sequence[TaskExample]) -> list[float]:
    return [float(ex.output) for ex in examples]


def _best_gain(
    candidates: Mapping[str, list[float]],
    residual: list[float],
    *,
    exclude: set[str],
    max_its: int,
    rng: np.random.RandomState,
) -> tuple[str | None, float, LinearGaussianEdge | None]:
    """The single candidate field with the highest :func:`regression_gain` (falling back to
    :func:`dependency_gain` when a regression edge is undefined, e.g. a near-constant column) against
    ``residual`` -- one greedy forward-selection step. Reuses the fitted edge (no re-fitting) so the
    caller can residualize immediately."""
    from mixle.stats import GaussianEstimator

    best_name, best_gain, best_edge = None, 0.0, None
    for name, col in candidates.items():
        if name in exclude:
            continue
        gain = regression_gain(col, residual, GaussianEstimator(), max_its=max_its, rng=rng)
        edge = None
        if np.isfinite(gain):
            edge = fit_linear_gaussian_edge(list(zip(col, residual)))
        else:
            gain = dependency_gain([round(v, 6) for v in col], residual, GaussianEstimator(), max_its=max_its, rng=rng)
        if gain > best_gain:
            best_name, best_gain, best_edge = name, gain, edge
    return best_name, best_gain, best_edge


@dataclass
class DependencyForest:
    """A discovered decomposition of ``output``: the ordered list of parent fields it was routed
    through (``chosen``, e.g. ``["m1", "m2"]``; empty means monolithic -- no candidate intermediate
    or input cleared ``min_gain``), each step's own gain, and the total ``mdl_gain`` -- the
    description-length gain (nats) of this decomposition over solving ``output`` directly from the
    raw inputs. Positive ``mdl_gain`` means the decomposition COMPRESSES; by construction it is the
    sum of the chosen edges' own :func:`~mixle.inference.structure.dependency_gain`/
    :func:`~mixle.inference.structure.regression_gain` scores."""

    chosen: list[str]
    edge_gains: list[float]
    mdl_gain: float
    edges: list[LinearGaussianEdge | None] = field(default_factory=list)

    @property
    def is_decomposed(self) -> bool:
        return len(self.chosen) > 0

    def predict(self, inputs: Mapping[str, float], candidate_intermediates: CandidateIntermediates) -> float:
        """Sum of each chosen edge's prediction from its own parent field -- the decomposed model's
        point estimate, used to compare predictive accuracy against the monolithic baseline."""
        total = 0.0
        for name, edge in zip(self.chosen, self.edges):
            if edge is None:
                continue
            value = float(inputs[name]) if name in inputs else float(candidate_intermediates[name](inputs))
            total += edge.a + edge.b * value
        return total


def discover_decomposition(
    task_examples: Sequence[TaskExample],
    candidate_intermediates: CandidateIntermediates,
    *,
    max_parents: int = 4,
    min_gain: float = 0.0,
    max_its: int = 30,
    seed: int = 0,
) -> DependencyForest:
    """Discover which candidate intermediates ``output`` should be routed through, by greedy forward
    selection scored with :func:`~mixle.inference.structure.regression_gain` /
    :func:`~mixle.inference.structure.dependency_gain` -- the SAME model-based description-length test
    :func:`~mixle.inference.structure.learn_structure` uses for record fields, applied here per step
    against the current RESIDUAL so multiple intermediates (``output = f(g(a), h(b))``) can each earn
    their own edge, not just the single best one (a plain :class:`~mixle.inference.structure.DependencyTreeDistribution`
    forest allows one parent per field; a task's output routinely needs several).

    Every raw input is itself a candidate parent, so a task with NO real decomposable structure
    correctly comes back with ``chosen == []`` (monolithic: the raw inputs already explain ``output``
    as well as anything) rather than inventing intermediates that do not pay for themselves.
    """
    rng = np.random.RandomState(seed)
    inputs_cols = _input_columns(task_examples)
    intermediate_cols = {name: _materialize(task_examples, fn) for name, fn in candidate_intermediates.items()}
    candidates: dict[str, list[float]] = {**inputs_cols, **intermediate_cols}
    residual = _output_column(task_examples)

    chosen: list[str] = []
    gains: list[float] = []
    edges: list[LinearGaussianEdge | None] = []
    for _ in range(max_parents):
        name, gain, edge = _best_gain(candidates, residual, exclude=set(chosen), max_its=max_its, rng=rng)
        if name is None or gain <= min_gain:
            break
        chosen.append(name)
        gains.append(gain)
        edges.append(edge)
        if edge is not None:
            col = np.asarray(candidates[name], dtype=float)
            pred = edge.a + edge.b * col
            residual = (np.asarray(residual, dtype=float) - pred).tolist()

    return DependencyForest(chosen=chosen, edge_gains=gains, mdl_gain=float(sum(gains)), edges=edges)


def fit_decomposition(
    task_examples: Sequence[TaskExample],
    decomposition: Sequence[str],
    candidate_intermediates: CandidateIntermediates,
    *,
    max_its: int = 30,
    seed: int = 0,
) -> DependencyForest:
    """Fit a SPECIFIC, given ``decomposition`` (in order) rather than discovering one -- every named
    field is forced in, in order, scored and residualized the same way :func:`discover_decomposition`'s
    forward selection does. Lets a caller both score (:attr:`DependencyForest.mdl_gain`) and predict
    with (:meth:`DependencyForest.predict`) a decomposition it did not necessarily search for -- e.g.
    a deliberately-worse candidate, for the MDL-gain/outcome correlation check."""
    rng = np.random.RandomState(seed)
    inputs_cols = _input_columns(task_examples)
    intermediate_cols = {name: _materialize(task_examples, fn) for name, fn in candidate_intermediates.items()}
    candidates: dict[str, list[float]] = {**inputs_cols, **intermediate_cols}
    residual = _output_column(task_examples)

    from mixle.stats import GaussianEstimator

    gains: list[float] = []
    edges: list[LinearGaussianEdge | None] = []
    for name in decomposition:
        col = candidates[name]
        gain = regression_gain(col, residual, GaussianEstimator(), max_its=max_its, rng=rng)
        if not np.isfinite(gain):
            gain = dependency_gain([round(v, 6) for v in col], residual, GaussianEstimator(), max_its=max_its, rng=rng)
            gains.append(gain)
            edges.append(None)
            continue
        edge = fit_linear_gaussian_edge(list(zip(col, residual)))
        gains.append(gain)
        edges.append(edge)
        pred = edge.a + edge.b * np.asarray(col, dtype=float)
        residual = (np.asarray(residual, dtype=float) - pred).tolist()
    return DependencyForest(chosen=list(decomposition), edge_gains=gains, mdl_gain=float(sum(gains)), edges=edges)


def mdl_score(
    task_examples: Sequence[TaskExample],
    decomposition: Sequence[str],
    candidate_intermediates: CandidateIntermediates,
    *,
    max_its: int = 30,
    seed: int = 0,
) -> float:
    """The MDL gain (nats) of routing ``output`` through a SPECIFIC, given ``decomposition`` -- a thin
    accessor over :func:`fit_decomposition` for callers that only want the score (e.g. ranking several
    candidate decompositions for the MDL-gain/outcome correlation check)."""
    return fit_decomposition(task_examples, decomposition, candidate_intermediates, max_its=max_its, seed=seed).mdl_gain


def monolithic_predict(train: Sequence[TaskExample], test: Sequence[TaskExample]) -> list[float]:
    """OLS fit of ``output`` on the raw inputs (every field jointly, closed form) -- the "solve as one
    black box" baseline :func:`discover_decomposition` is compared against. Matched compute against
    the decomposed model: both are single closed-form linear solves over the same ``n`` examples."""
    keys = sorted({k for ex in train for k in ex.inputs})
    x = np.asarray([[1.0] + [float(ex.inputs[k]) for k in keys] for ex in train], dtype=float)
    y = np.asarray([ex.output for ex in train], dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    x_test = np.asarray([[1.0] + [float(ex.inputs[k]) for k in keys] for ex in test], dtype=float)
    return (x_test @ beta).tolist()


def decomposed_predict(
    forest: DependencyForest, candidate_intermediates: CandidateIntermediates, test: Sequence[TaskExample]
) -> list[float]:
    return [forest.predict(ex.inputs, candidate_intermediates) for ex in test]


# --- design_prior wiring: which family (decomposed vs. monolithic) has actually won, persisted -----------------


def log_decomposition_recipe(design: DesignModel, mdl_gain: float, *, family: str) -> None:
    """Record one decomposition attempt's MDL gain into the existing design ledger under ``family``
    (``"decomposed"`` / :data:`MONOLITHIC`) -- a thin wrapper over
    :func:`~mixle.task.design_prior.record_accepted_recipe` so :func:`~mixle.task.design_prior.rank_design_families`
    and :func:`~mixle.task.design_prior.best_family` answer "has decomposing this kind of task actually
    paid off" from real history, the same what-worked prior every other structural family search uses."""
    record_accepted_recipe(design, [0.0], mdl_gain, [], family=family)


# --- outcome_decomposer wiring: (decomposition, outcome) logs train the proposer --------------------------------


def _decomposition_traces(decompositions: Sequence[Sequence[str]]) -> list[AgentTrace]:
    """A decomposition (an ordered list of field names) as an :class:`~mixle.task.traces.AgentTrace`
    plan -- the same shape :func:`~mixle.task.plan_model.fit_plan_model` already fits a Markov chain
    over for tool-name sequences, reused unchanged for intermediate-name sequences."""
    return [AgentTrace(request="", plan=[{"tool": name} for name in seq]) for seq in decompositions]


@dataclass
class DecompositionProposer:
    """An outcome-trained proposer over decompositions: :attr:`plan_model` scores/samples which
    intermediates (in what order) to route a task's output through, and shifts toward
    higher-``outcome`` decompositions as they get logged -- :func:`~mixle.task.outcome_decomposer.train_outcome_decomposer`'s
    refit-on-successes loop, applied to decomposition proposals instead of tool-call plans."""

    plan_model: PlanModel
    log: list[tuple[list[str], float]] = field(default_factory=list)


def init_decomposition_proposer(seed_decompositions: Sequence[Sequence[str]]) -> DecompositionProposer:
    """Fit the round-0 (imitation) proposer on a seed corpus of decompositions -- e.g. every
    ``chosen`` a few :func:`discover_decomposition` calls returned on early task instances."""
    model = fit_plan_model(_decomposition_traces(seed_decompositions))
    return DecompositionProposer(plan_model=model)


def record_decomposition_outcome(
    proposer: DecompositionProposer,
    decomposition: Sequence[str],
    outcome: float,
    *,
    success_quantile: float = 0.6,
    min_log: int = 4,
) -> DecompositionProposer:
    """Log one ``(decomposition, outcome)`` pair from a REAL task instance, and once at least
    ``min_log`` outcomes are on file, refit :attr:`~DecompositionProposer.plan_model` on the
    decompositions scoring at or above this round's own ``success_quantile`` -- literally
    :func:`~mixle.task.outcome_decomposer.train_outcome_decomposer`'s keep-the-successes-and-refit
    step, so future :meth:`~mixle.task.plan_model.PlanModel.sample` calls favor what actually worked,
    not just what the seed corpus imitated."""
    proposer.log.append((list(decomposition), float(outcome)))
    if len(proposer.log) < min_log:
        return proposer
    scores = [o for _, o in proposer.log]
    threshold = float(np.quantile(scores, success_quantile))
    kept = [d for d, o in proposer.log if o >= threshold and d]
    if kept:
        proposer.plan_model = fit_plan_model(_decomposition_traces(kept))
    return proposer


__all__ = [
    "MONOLITHIC",
    "CandidateIntermediates",
    "DecompositionProposer",
    "DependencyForest",
    "TaskExample",
    "decomposed_predict",
    "discover_decomposition",
    "fit_decomposition",
    "init_decomposition_proposer",
    "log_decomposition_recipe",
    "mdl_score",
    "monolithic_predict",
    "record_decomposition_outcome",
]
