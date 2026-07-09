"""The meta-search that *learns which improvement operators help*: a bandit + a diversity population.

The operator-choice problem is a **non-stationary bandit**: each step, pick an operator (arm), apply it
through the propose-and-verify gate, observe the *verified* gate delta as reward (0 if the
challenger was rejected), and update the arm's value. Because the reward is the anti-regression
verified delta, the policy cannot be fooled by overfit in-sample gains.

* :class:`OperatorBandit` -- Thompson or UCB over a fixed operator pool. Reward is the verified delta;
  cost is tracked for a report. Non-stationary: a forgetting factor decays stale arm statistics so the
  policy can follow a problem whose best operator changes over the run.
* :class:`Population` -- a diversity-preserving population of model structures evolved by the bandit:
  select operators, apply them to parents, gate the challengers, reward the bandit, and keep the
  verified-best plus a coarse capability-diversity quota. ``run`` returns a
  :class:`~mixle.evolve.search.SearchResult`; ``champion`` is the incumbent.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.capability import capabilities
from mixle.evolve.objective import Objective
from mixle.evolve.operators import ImprovementOperator, default_operators
from mixle.evolve.structure import structural_distance
from mixle.evolve.verify import challenger_beats_champion


# ---------------------------------------------------------------------------
# OperatorBandit -- a non-stationary bandit over ImprovementOperators
# ---------------------------------------------------------------------------
@dataclass
class _Arm:
    """Per-operator sufficient statistics for the bandit (decayed for non-stationarity)."""

    pulls: float = 0.0
    wins: float = 0.0  # count of verified-positive rewards (Thompson Beta successes)
    reward_sum: float = 0.0  # sum of (clipped) rewards
    reward_sq: float = 0.0  # sum of squared rewards, retained for UCB variance diagnostics
    cost_sum: float = 0.0


class OperatorBandit:
    """A non-stationary bandit over a fixed pool of :class:`ImprovementOperator`.

    ``select(k)`` returns the ``k`` highest-value operators under the chosen policy; ``reward`` folds a
    verified delta + cost back into the chosen arm; ``report`` is the "which operators help" artifact.

    Policies:
      * ``'thompson'`` -- Beta-Bernoulli Thompson sampling on the *win indicator* (reward > 0), scaled by
        the mean positive reward, so an operator that wins rarely but big and one that wins often but
        small are compared on expected verified delta.
      * ``'ucb'``     -- UCB1 on the mean reward with a ``sqrt(2 ln N / n)`` exploration bonus.

    Non-stationarity: each ``reward`` first multiplies every arm's statistics by ``decay`` (a forgetting
    factor in ``(0, 1]``), so old evidence fades and the policy can track a shifting best operator.
    """

    def __init__(
        self,
        operators: Sequence[ImprovementOperator],
        *,
        policy: str = "thompson",
        decay: float = 0.97,
        prior_cost_aware: bool = True,
        seed: int = 0,
    ) -> None:
        ops = list(operators)
        if not ops:
            raise ValueError("OperatorBandit needs at least one operator.")
        if policy not in ("thompson", "ucb"):
            raise ValueError(f"policy must be 'thompson' or 'ucb' (got {policy!r}).")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1].")
        self.operators: dict[str, ImprovementOperator] = {op.name: op for op in ops}
        self.policy = policy
        self.decay = float(decay)
        self.prior_cost_aware = bool(prior_cost_aware)
        self.rng = np.random.RandomState(seed)
        self.arms: dict[str, _Arm] = {name: _Arm() for name in self.operators}
        self._total_pulls = 0

    # -- value of each arm ---------------------------------------------------
    def _thompson_value(self, name: str) -> float:
        arm = self.arms[name]
        op = self.operators[name]
        # cost-aware prior: cheaper operators get a slightly more optimistic prior (explored sooner).
        prior_a = 1.0
        prior_b = 1.0 + (float(getattr(op, "cost_hint", 1.0)) if self.prior_cost_aware else 0.0)
        a = prior_a + arm.wins
        b = prior_b + max(arm.pulls - arm.wins, 0.0)
        win_prob = float(self.rng.beta(a, b))
        # scale the win probability by the mean positive reward so big-but-rare beats small-but-frequent.
        mean_reward = (arm.reward_sum / arm.pulls) if arm.pulls > 0 else 0.0
        scale = max(mean_reward, 1.0e-9)
        return win_prob * scale

    def _ucb_value(self, name: str) -> float:
        arm = self.arms[name]
        if arm.pulls <= 0.0:
            return math.inf  # always try an unpulled arm first
        mean = arm.reward_sum / arm.pulls
        total = max(self._total_pulls, 1)
        bonus = math.sqrt(2.0 * math.log(total) / arm.pulls)
        return mean + bonus

    def value(self, name: str) -> float:
        """The current policy value of operator ``name`` (a Thompson draw or the UCB index)."""
        return self._thompson_value(name) if self.policy == "thompson" else self._ucb_value(name)

    def select(self, k: int = 1) -> list[ImprovementOperator]:
        """Return the ``k`` operators with the highest policy value (a fresh Thompson draw each call)."""
        if k < 1:
            raise ValueError("k must be positive.")
        scored = sorted(self.operators, key=self.value, reverse=True)
        return [self.operators[name] for name in scored[: min(k, len(scored))]]

    def reward(self, op_name: str, delta: float, cost: float) -> None:
        """Fold a verified ``delta`` (0 if the challenger was rejected) and ``cost`` into the arm.

        Decays every arm first (non-stationarity), then updates the chosen arm. A ``delta`` is clipped at
        0 below -- a rejected challenger is a zero reward, never negative, matching the anti-regression
        guarantee (we never *punish* an operator for a rejected proposal beyond not rewarding it).
        """
        if op_name not in self.arms:
            raise KeyError(f"unknown operator {op_name!r}.")
        for arm in self.arms.values():
            arm.pulls *= self.decay
            arm.wins *= self.decay
            arm.reward_sum *= self.decay
            arm.reward_sq *= self.decay
            arm.cost_sum *= self.decay
        r = max(float(delta), 0.0)
        arm = self.arms[op_name]
        arm.pulls += 1.0
        arm.reward_sum += r
        arm.reward_sq += r * r
        arm.cost_sum += float(cost)
        if r > 0.0:
            arm.wins += 1.0
        self._total_pulls += 1

    def report(self) -> dict[str, Any]:
        """Per-operator win-rate, mean verified delta, mean cost, and (decayed) pull count."""
        rows: dict[str, dict[str, float]] = {}
        for name, arm in self.arms.items():
            pulls = arm.pulls
            rows[name] = {
                "pulls": float(pulls),
                "win_rate": float(arm.wins / pulls) if pulls > 0 else 0.0,
                "mean_delta": float(arm.reward_sum / pulls) if pulls > 0 else 0.0,
                "mean_cost": float(arm.cost_sum / pulls) if pulls > 0 else 0.0,
            }
        return {"policy": self.policy, "decay": self.decay, "operators": rows, "total_pulls": self._total_pulls}


# ---------------------------------------------------------------------------
# Population -- a diversity-preserving population evolved by the bandit
# ---------------------------------------------------------------------------
@dataclass
class GenerationReport:
    """One :meth:`Population.step`: which operators ran, the verified wins, and the new champion score."""

    proposals: int = 0
    verified: int = 0
    best_score: float = float("nan")
    operators_used: list[str] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


@dataclass
class _Member:
    """A population member: a fitted model plus its cached objective score and capability fingerprint."""

    model: Any
    score: float
    caps: frozenset[str]


# diversity now uses the real genotype distance (tree-edit over the model's compositional structure); the
# ``caps`` fingerprint is kept as low-cost cached metadata.


class Population:
    """A diversity-preserving population of model structures, evolved by the :class:`OperatorBandit`.

    ``seeds`` are fitted models (the starting structures). Each :meth:`step` selects operators via the
    bandit, applies them to parents chosen by fitness, gates the challengers with the Phase-1
    champion/challenger rule (Benjamini-Hochberg multiplicity, since a generation produces many
    challengers at once), rewards the bandit with the verified deltas, and keeps the verified-best plus a
    coarse capability-diversity quota.

    Args:
        seeds: the initial fitted models (at least one).
        objective: the :class:`~mixle.evolve.objective.Objective` to optimize (lower-is-better aware).
        operators: the proposal-move pool; defaults to the Phase-1 safe set.
        bandit: an :class:`OperatorBandit` over ``operators`` (built with the default policy if omitted).
        size: the carrying capacity of the population.
        diversity_quota: how many of ``size`` slots are reserved for capability-diverse members (the rest
            go to the fittest); the quota keeps the search from collapsing onto one structure too early.
        seed: RNG seed for parent sampling and the bandit.
    """

    def __init__(
        self,
        seeds: Sequence[Any],
        *,
        objective: Objective,
        operators: Sequence[ImprovementOperator] | None = None,
        bandit: OperatorBandit | None = None,
        size: int = 12,
        diversity_quota: int = 2,
        seed: int = 0,
    ) -> None:
        seeds = list(seeds)
        if not seeds:
            raise ValueError("Population needs at least one seed model.")
        if size < 1:
            raise ValueError("size must be positive.")
        self.objective = objective
        self.operators = list(operators) if operators is not None else default_operators()
        self.bandit = bandit if bandit is not None else OperatorBandit(self.operators, seed=seed)
        self.size = int(size)
        self.diversity_quota = max(0, int(diversity_quota))
        self.seed = int(seed)
        self.rng = np.random.RandomState(seed)
        self._gen = 0
        self._eval_data: Any = None
        # members are scored lazily on the first step (we need data); cache raw seeds until then.
        self._members: list[_Member] = []
        self._raw_seeds = seeds
        # the incumbent over the whole run (anti-regression: never replaced by a worse model).
        self._champion: Any = seeds[0]
        self._champion_score: float = float("nan")

    # -- scoring helpers -----------------------------------------------------
    def _score(self, model: Any, data: Any) -> float:
        """Objective scalar normalized so *smaller is always better* (lower-is-better canonical form)."""
        s = float(self.objective.scalar(model, data))
        return s if self.objective.lower_is_better else -s

    def _member(self, model: Any, data: Any) -> _Member:
        return _Member(model, self._score(model, data), capabilities(model))

    def _ensure_initialized(self, data: Any) -> None:
        if self._members:
            return
        self._members = [self._member(m, data) for m in self._raw_seeds]
        best = min(self._members, key=lambda m: m.score)
        self._champion = best.model
        self._champion_score = best.score

    # -- selection -----------------------------------------------------------
    def _select_parents(self, k: int) -> list[_Member]:
        """Pick ``k`` parents biased toward fitness (rank-weighted), with replacement."""
        members = sorted(self._members, key=lambda m: m.score)
        n = len(members)
        weights = np.asarray([n - i for i in range(n)], dtype=float)  # best gets the most weight
        weights /= weights.sum()
        idx = self.rng.choice(n, size=k, replace=True, p=weights)
        return [members[i] for i in idx]

    def _survivors(self) -> list[_Member]:
        """Keep the fittest ``size - quota`` plus a structurally-diverse quota (greedy farthest-first over the
        tree-edit genotype distance)."""
        members = sorted(self._members, key=lambda m: m.score)
        if len(members) <= self.size:
            return members
        n_fit = max(1, self.size - self.diversity_quota)
        kept = members[:n_fit]
        pool = members[n_fit:]
        # greedily add the members whose STRUCTURE is farthest (tree-edit) from those already kept.
        while len(kept) < self.size and pool:
            kept_models = [m.model for m in kept]
            far = max(pool, key=lambda m: min(structural_distance(m.model, k) for k in kept_models))
            kept.append(far)
            pool.remove(far)
        return kept[: self.size]

    # -- the generation step -------------------------------------------------
    def step(self, data: Any) -> GenerationReport:
        """Run one generation: select -> propose -> gate -> reward -> survivor selection."""
        self._ensure_initialized(data)
        report = GenerationReport(best_score=self._champion_score)
        ctx = {"parent_hash": None, "seed": self.seed + self._gen, "objective": self.objective}

        # one operator per parent; how many parents to spawn this generation.
        n_offspring = max(1, self.size // 2)
        parents = self._select_parents(n_offspring)
        ops = self.bandit.select(k=n_offspring)

        new_members: list[_Member] = []
        for parent, op in zip(parents, ops):
            report.operators_used.append(op.name)
            cost = float(getattr(op, "cost_hint", 1.0))
            try:
                if not op.applicable(parent.model, data, ctx=ctx):
                    self.bandit.reward(op.name, 0.0, cost)
                    report.rewards.append(0.0)
                    continue
                candidate = op.propose(parent.model, data, ctx=ctx)
            except Exception:
                self.bandit.reward(op.name, 0.0, cost)
                report.rewards.append(0.0)
                continue
            report.proposals += 1

            nonnested = type(candidate.model).__name__ != type(parent.model).__name__
            verdict = challenger_beats_champion(
                parent.model,
                candidate.model,
                data,
                objective=self.objective,
                multiplicity="bh",  # many simultaneous challengers per generation
                nonnested=nonnested,
                seed=self.seed + self._gen,
            )
            delta = verdict.delta if verdict.promote else 0.0
            self.bandit.reward(op.name, delta, cost)
            report.rewards.append(delta)
            if verdict.promote:
                report.verified += 1
                new_members.append(self._member(candidate.model, data))

        # fold survivors + new verified offspring back into the population.
        self._members = self._survivors_with(new_members)

        # update the run-level incumbent (anti-regression: only replace on a strict improvement).
        best = min(self._members, key=lambda m: m.score)
        if math.isnan(self._champion_score) or best.score < self._champion_score:
            self._champion = best.model
            self._champion_score = best.score
        report.best_score = self._champion_score
        self._gen += 1
        return report

    def _survivors_with(self, new_members: list[_Member]) -> list[_Member]:
        self._members = self._members + new_members
        return self._survivors()

    def run(self, data: Any, generations: int = 5) -> Any:
        """Evolve for ``generations`` steps; return a :class:`~mixle.evolve.search.SearchResult`.

        The returned ``best_model`` is the run incumbent, guaranteed no worse than the best seed on the
        objective (anti-regression). ``history`` is one row per generation (proposals / verified / score).
        """
        from mixle.evolve.search import SearchResult

        self._ensure_initialized(data)
        history: list[dict[str, Any]] = []
        for _ in range(int(generations)):
            rep = self.step(data)
            history.append(
                {
                    "proposals": rep.proposals,
                    "verified": rep.verified,
                    "best_score": rep.best_score,
                    "operators_used": list(rep.operators_used),
                    "rewards": list(rep.rewards),
                }
            )
        # report best_score back in the objective's native orientation.
        native_best = self._champion_score if self.objective.lower_is_better else -self._champion_score
        return SearchResult(
            best_config={},  # population searches structures, not a config vector
            best_model=self._champion,
            best_score=float(native_best),
            history=history,
        )

    @property
    def champion(self) -> Any:
        """The current run incumbent (the best model seen, anti-regression guaranteed)."""
        return self._champion


__all__ = ["OperatorBandit", "Population", "GenerationReport"]
