"""Learned non-myopic probing policy versus myopic expected information gain.

The head-to-head comparison is deliberately conservative. If the learned non-myopic policy does not
beat myopic EIG on solve-rate at matched oracle budget in the exploration world, the result should be
treated as evidence to keep the simpler myopic policy. Myopic is often near-optimal; the learned path
is useful only when delayed or combinatorial payoff makes one-step EIG miss valuable probes.

Two policies compared head-to-head, same action menu, same budget:

* :func:`myopic_eig_policy` -- greedy, ONE STEP of lookahead: at every decision, picks the action
  (across ALL undrilled cells, survey or drill) with the highest per-cost expected information gain
  about that cell's target status. A cell's current belief uncertainty is its survey-noise; a read's
  "borderline-ness" (how close to the decision boundary between target/non-target reads) governs how
  much resolving it is actually worth -- a read far from the boundary is already confidently
  classified, so probing it teaches little.
* the outcome-trained decomposer's plan model (:mod:`~mixle.task.outcome_decomposer`) --
  trained via expert iteration against verifier-grounded reward (terminal world score, no learned
  reward model): propose whole action-type sequences, execute, keep the verifiably-successful ones,
  refit, iterate. Optimizing for a whole sequence's terminal score (rather than one greedy step) is
  the non-myopic half of this comparison -- it can trade an immediately-worse-looking step for a
  better final outcome, which pure one-step EIG structurally cannot.

    directed = train_outcome_decomposer(...)
    result = head_to_head_probe(directed.plan_model, held_out_seeds=range(...), ...)
    result.non_myopic_wins
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mixle.task.explore_world import ExplorationWorld, run_episode
from mixle.task.outcome_decomposer import evaluate_plan_model
from mixle.task.plan_model import PlanModel

# the world's own generative gap: a target cell's geology reads systematically +2.0 higher than a
# non-target's (see ExplorationWorld.__post_init__); a read near the midpoint is the most ambiguous.
# Known from the world's own construction, not fit from data -- a myopic policy operating on
# synthetic ground truth is allowed to know its own world's scale.
_DECISION_BOUNDARY = 1.0
_DRILL_CONFIDENCE = 0.65  # p(target) above which drilling (exploit) beats surveying (explore) further


def _p_target(read: float, noise: float) -> float:
    """Logistic belief that a cell is a target, from its current noisy read: centered on the world's
    own decision boundary, flattened by the read's own uncertainty (more noise = less confident)."""
    z = (read - _DECISION_BOUNDARY) / max(noise, 0.3)
    return float(1.0 / (1.0 + math.exp(-z)))


def _entropy(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(-(p * math.log(p) + (1 - p) * math.log(1 - p)))


def myopic_eig_policy(world: ExplorationWorld) -> dict | None:
    """One-step-lookahead policy, explicitly information-theoretic: EXPLOIT (drill) the most
    confident current target-candidate once its belief clears ``_DRILL_CONFIDENCE``; otherwise
    EXPLORE (survey) the single most UNCERTAIN cell -- maximum current entropy, i.e. the read closest
    to the decision boundary relative to its own noise, the textbook expected-information-gain
    target. No lookahead beyond this one step -- by construction, it cannot see that an
    apparently-mediocre probe now sets up a better probe later."""
    undrilled = [c for c in range(world.n_cells) if not world._drilled[c]]
    if not undrilled:
        return None

    beliefs = {c: _p_target(world.prospectivity(c), float(world._survey_noise[c])) for c in undrilled}
    best_drill = max(undrilled, key=lambda c: beliefs[c])
    if beliefs[best_drill] >= _DRILL_CONFIDENCE and world.remaining_budget >= 5:
        return {"type": "drill", "cell": best_drill}

    if world.remaining_budget < 1:
        return None
    most_uncertain = max(undrilled, key=lambda c: _entropy(beliefs[c]))
    return {"type": "survey", "cell": most_uncertain}


@dataclass
class ProbeHeadToHead:
    """Held-out comparison between the non-myopic probe policy and a myopic baseline."""

    non_myopic_score: float
    myopic_score: float
    non_myopic_wins: bool


def head_to_head_probe(
    plan_model: PlanModel,
    *,
    held_out_seeds,
    n_cells: int,
    n_targets: int,
    budget: int,
) -> ProbeHeadToHead:
    """Compare the non-myopic (outcome-trained) plan model against the myopic EIG policy on the same
    held-out seeds at matched budget."""
    non_myopic_score = evaluate_plan_model(
        plan_model, seeds=held_out_seeds, n_cells=n_cells, n_targets=n_targets, budget=budget
    )
    myopic_scores = [
        run_episode(myopic_eig_policy, n_cells=n_cells, n_targets=n_targets, budget=budget, seed=s).score
        for s in held_out_seeds
    ]
    myopic_score = float(np.mean(myopic_scores)) if myopic_scores else 0.0
    return ProbeHeadToHead(
        non_myopic_score=non_myopic_score, myopic_score=myopic_score, non_myopic_wins=non_myopic_score > myopic_score
    )


__all__ = ["ProbeHeadToHead", "head_to_head_probe", "myopic_eig_policy"]
