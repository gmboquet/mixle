"""The meta-search that *learns which improvement operators help*: a bandit + a diversity population.

The operator-choice problem is a **non-stationary bandit**: each step, pick an operator (arm), apply it
through the propose-and-verify gate, observe the *verified* gate delta as reward (0 if the
challenger was rejected), and update the arm's value. Because the reward is the anti-regression
verified delta rather than an in-sample fit score, the policy cannot be fooled by a challenger
overfitting the data it was FIT on.

It can still be fooled by a subtler failure: adaptive reuse of the data it is VERIFIED on. A
:class:`Population` run queries the SAME reusable validation set every single generation, and both
survivor selection and :class:`OperatorBandit` adapt to what they saw there -- exactly the setting
where a held-out set stops being a fair test the more times you look at it and act on what you saw
(the same reason a test set quietly becomes a training signal if you tune against it repeatedly). See
:class:`Population`'s docstring ("Data roles", MXR-080-0042) for the three roles this module keeps
separate -- fit data, the reused validation signal (a heuristic, not evidence), and the genuine
one-shot final holdout (:meth:`Population.evaluate_holdout`).

* :class:`OperatorBandit` -- Thompson or UCB over a fixed operator pool. Reward is the verified delta
  from that reused validation signal (see above); cost is tracked for a report. Non-stationary: a
  forgetting factor decays stale arm statistics so the policy can follow a problem whose best operator
  changes over the run.
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
from mixle.evolve.improve import _split
from mixle.evolve.objective import Objective, _exact_int
from mixle.evolve.operators import Candidate, ImprovementOperator, default_operators
from mixle.evolve.structure import structural_distance
from mixle.evolve.verify import Verdict, challenger_beats_champion
from mixle.inference.multiple_testing import adjust_pvalues


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
        names = [op.name for op in ops]
        if len(set(names)) != len(names):
            # `{op.name: op for op in ops}` silently collapsed same-named operators into one arm, so
            # two supplied operators shared a single set of statistics, only one of them was ever
            # selectable, and reward() credited whichever survived the collapse. An operator name is
            # the arm identity here and in every report -- an ambiguous one has no correct resolution.
            dupes = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"OperatorBandit operator names must be unique, got duplicate(s): {dupes!r}")
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

        ``delta`` and ``cost`` must be finite (and ``cost`` non-negative). ``max(nan, 0.0)`` returns
        ``nan``, so a single non-finite update used to poison the arm permanently: mean delta, mean
        cost, and every subsequent Thompson draw for it came back ``nan``, and a ``nan`` policy value
        sorts unpredictably against real ones. Policy state is not a place to absorb an invalid
        measurement.
        """
        if op_name not in self.arms:
            raise KeyError(f"unknown operator {op_name!r}.")
        delta = float(delta)
        cost = float(cost)
        if not math.isfinite(delta):
            raise ValueError(f"reward for operator {op_name!r} must be a finite delta, got {delta!r}")
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(f"reward for operator {op_name!r} must have a finite, non-negative cost, got {cost!r}")
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


@dataclass
class _GenerationSlot:
    """One parent/operator pairing drawn for a generation, tracked from proposal through to reward.

    ``candidate``/``verdict`` stay ``None`` when the operator was inapplicable or ``propose`` raised --
    that slot never got a comparison, and is rewarded 0.0 without further ado. A slot that WAS compared
    holds its RAW (uncorrected) :class:`~mixle.evolve.verify.Verdict`; :meth:`Population.step` finalizes
    its promotion only after every slot's raw p-value has been pooled and corrected together (see the
    multiplicity note there) -- kept as its own list, walked twice, so the bandit is still rewarded in
    the same slot order the parents/operators were drawn in.
    """

    op: ImprovementOperator
    cost: float
    candidate: Candidate | None = None
    verdict: Verdict | None = None


class Population:
    """A diversity-preserving population of model structures, evolved by the :class:`OperatorBandit`.

    ``seeds`` are fitted models (the starting structures). Each :meth:`step` selects operators via the
    bandit, applies them to parents chosen by fitness, gates the challengers with the Phase-1
    champion/challenger rule -- pooling the whole generation's p-values for a single Benjamini-Hochberg
    correction before the per-candidate gate, since a generation produces many challengers at once --
    rewards the bandit with the verified deltas, and keeps the verified-best plus a coarse
    capability-diversity quota.

    **Data roles (MXR-080-0042).** This class keeps three distinct data roles separate:

    1. **Fit data** -- what operators actually fit challengers on (``data`` in :meth:`step` / :meth:`run`,
       or its auto-derived fit-split when ``verify_data`` is omitted -- see below).
    2. **Reusable validation data** (``verify_data``) -- read by :meth:`step` every single generation, to
       gate every challenger (:func:`~mixle.evolve.verify.challenger_beats_champion`) AND to score
       population fitness / drive survivor selection. When omitted, :meth:`step` / :meth:`run` derive a
       genuinely disjoint split from ``data`` (:meth:`_auto_split`) rather than the old default of
       reusing ``data`` itself verbatim -- but disjointness from the fit data is only half the fix.
       This SAME ``verify_data`` is then queried again, adaptively, by EVERY generation of a
       :meth:`run`: survivor selection keeps whoever currently scores best against it, and
       :class:`OperatorBandit` learns from whichever operators currently win against it. The
       per-generation Benjamini-Hochberg pool (see :meth:`step`) only controls multiplicity WITHIN one
       generation's simultaneous comparisons; it does nothing to correct for that same set being
       re-queried, and acted on, by every OTHER generation -- the textbook way a held-out set stops
       being a fair test once you repeatedly tune against it. Treat every ``Verdict``/score this class
       computes against a reused ``verify_data`` as a NOISY HEURISTIC that steers the search, never as
       a promotion-grade statistical claim on its own.
    3. **One-shot final holdout** -- data never passed to this population as ``data`` or ``verify_data``
       at any point, evaluated exactly ONCE via :meth:`evaluate_holdout` after every intended
       :meth:`step` / :meth:`run` call is done. This is the only comparison this class makes that
       carries real evidentiary weight -- a caller that needs a defensible promotion decision (as
       opposed to search guidance) must use it rather than trusting :attr:`champion` or a
       :class:`GenerationReport` alone.

    Args:
        seeds: the initial fitted models (at least one).
        objective: the :class:`~mixle.evolve.objective.Objective` to optimize (lower-is-better aware).
        operators: the proposal-move pool; defaults to the Phase-1 safe set.
        bandit: an :class:`OperatorBandit` over ``operators`` (built with the default policy if omitted).
        size: the carrying capacity of the population.
        diversity_quota: how many of ``size`` slots are reserved for capability-diverse members (the rest
            go to the fittest); the quota keeps the search from collapsing onto one structure too early.
        holdout: fraction of ``data`` reserved for the auto-derived ``verify_data`` split
            (:meth:`_auto_split`) whenever a caller omits ``verify_data`` -- irrelevant if the caller
            always passes an explicit, already-disjoint ``verify_data``.
        seed: RNG seed for parent sampling, the bandit, and the auto-derived split.
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
        holdout: float = 0.25,
        seed: int = 0,
    ) -> None:
        seeds = list(seeds)
        if not seeds:
            raise ValueError("Population needs at least one seed model.")
        if not 0.0 < holdout < 1.0:
            raise ValueError("holdout must be in (0, 1).")
        # Exact controls, validated before anything is stored or handed to the bandit/RNG
        # (MXR-080-1902). `int(size)` is truncation, not validation: `size=7.9` silently became a
        # population of 7 and -- because `bool` is an `int` subclass -- `size=True` became a
        # population of ONE, which also collapses `n_offspring = max(1, size // 2)` to a single
        # proposal per generation. `max(0, int(diversity_quota))` additionally absorbed a negative
        # quota as 0, so a sign error in a caller's arithmetic read as "no diversity reserved" rather
        # than as the mistake it was. A quota of exactly 0 stays legal -- it means "keep the fittest
        # only", which several callers pass on purpose.
        size = _exact_int(size, "size", minimum=1)
        diversity_quota = _exact_int(diversity_quota, "diversity_quota", minimum=0)
        seed = _exact_int(seed, "seed", minimum=0)
        self.objective = objective
        self.operators = list(operators) if operators is not None else default_operators()
        self.bandit = bandit if bandit is not None else OperatorBandit(self.operators, seed=seed)
        self.size = size
        self.diversity_quota = diversity_quota
        self.holdout = float(holdout)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._gen = 0
        self._eval_data: Any = None
        # members are scored lazily on the first step (we need data); cache raw seeds until then.
        self._members: list[_Member] = []
        self._raw_seeds = seeds
        # the incumbent over the whole run (anti-regression: never replaced by a worse model).
        self._champion: Any = seeds[0]
        self._champion_score: float = float("nan")
        # the incumbent BEFORE any step()/run() call -- the one-shot evaluate_holdout()'s default
        # comparison point (data role 3 in the class docstring above).
        self._initial_champion: Any = seeds[0]

    # -- scoring helpers -----------------------------------------------------
    def _score(self, model: Any, data: Any) -> float:
        """Objective scalar normalized so *smaller is always better* (lower-is-better canonical form).

        Fitness must be finite. Nothing required that before, so a NaN-scored seed produced a
        population whose ``min()`` selection is order-dependent (every NaN comparison is ``False``, so
        ``min`` simply returns whichever element it saw first), a champion chosen by that accident,
        and a :class:`~mixle.evolve.search.SearchResult` reporting ``best_score=nan`` while looking
        like an ordinary successful search. Survivor and champion comparisons downstream have no
        ordering to work with either. This mirrors
        :func:`mixle.evolve.search._held_out_score`, which already treats a non-finite score as a
        failed evaluation rather than a value.
        """
        s = float(self.objective.scalar(model, data))
        s = s if self.objective.lower_is_better else -s
        if not math.isfinite(s):
            raise ValueError(
                f"objective {self.objective.name!r} scored {type(model).__name__} as {s!r}; population "
                "fitness must be finite -- a non-finite score defines no ordering for survivor "
                "selection or the champion comparison."
            )
        return s

    def _member(self, model: Any, data: Any) -> _Member:
        return _Member(model, self._score(model, data), capabilities(model))

    def _ensure_initialized(self, data: Any) -> None:
        if self._members:
            return
        self._members = [self._member(m, data) for m in self._raw_seeds]
        best = min(self._members, key=lambda m: m.score)
        self._champion = best.model
        self._champion_score = best.score
        # the run's starting incumbent, snapshotted the ONE time this runs -- evaluate_holdout()'s
        # default reference point (data role 3: the one-shot final holdout).
        self._initial_champion = best.model

    def _auto_split(self, data: Any) -> tuple[Any, Any]:
        """(fit, verify) split of ``data``, used by :meth:`step` / :meth:`run` whenever a caller omits
        ``verify_data`` (data role 2 in the class docstring).

        Delegates to :func:`mixle.evolve.improve._split` -- the same helper
        :func:`~mixle.evolve.search.search` / :func:`~mixle.evolve.improve.improve` use for their own
        train/verify splits -- seeded by ``self.seed`` (deliberately NOT ``self._gen``): calling this
        repeatedly with data of the same length reproduces the exact SAME split every time. That
        determinism is load-bearing, not incidental -- :meth:`run` relies on every generation seeing
        the SAME auto-derived ``verify_data``, so a champion's score is comparable across generations;
        re-splitting on every call would score generation N and generation N+1 against different
        yardsticks, corrupting the anti-regression ``champion_score`` bookkeeping far worse than the
        train/verify overlap this replaces.

        Raises ``ValueError`` (from ``_split``) if ``data`` has fewer than 4 observations -- too little
        to split honestly. Pass an explicit ``verify_data`` for those cases instead of relying on the
        default.
        """
        return _split(data, self.holdout, self.seed)

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
    def step(self, data: Any, verify_data: Any | None = None) -> GenerationReport:
        """Run one generation: select -> propose -> gate -> reward -> survivor selection.

        Args:
            data: the pool operators FIT candidates from (data role 1, "fit data", in the class
                docstring). When ``verify_data`` is omitted, ``data`` is split (:meth:`_auto_split`)
                and only the fit-half is actually fit on -- the other half becomes the auto-derived
                ``verify_data`` below, so a challenger is never fit and scored on the same rows.
            verify_data: the held-out data used to score population fitness and to gate every
                challenger against its parent (:func:`~mixle.evolve.verify.challenger_beats_champion`)
                -- data role 2, "reusable validation data", in the class docstring: read again every
                generation, and its repeated, adaptive use across a :meth:`run` is a HEURISTIC search
                signal, not evidence on its own -- see :meth:`evaluate_holdout` for the genuine
                one-shot check. Defaults to the verify-half of :meth:`_auto_split` applied to ``data``
                when omitted (never ``data`` itself -- MXR-080-0042: the old default silently made
                verification equal training data). Pass an explicit, already-disjoint split here (e.g.
                :func:`~mixle.evolve.search.search` does) to control the split yourself instead.

        A generation proposes many challengers at once, so the gate applies a genuine population-wide
        Benjamini-Hochberg correction: every candidate is first compared on its own RAW p-value (pass 1
        below), the whole generation's raw p-values are then pooled and corrected together in ONE
        :func:`~mixle.inference.multiple_testing.adjust_pvalues` call (pass 2), and only then is each
        candidate's promotion finalized against its adjusted p-value (pass 3). Correcting each
        candidate's single p-value in isolation -- family size 1 -- is the identity transform for every
        method in :mod:`~mixle.inference.multiple_testing`, which is why
        :func:`~mixle.evolve.verify.challenger_beats_champion` refuses to do that itself. This pooled
        correction operates WITHIN one generation only -- it does not, and cannot, correct for the same
        ``verify_data`` being queried again by every OTHER generation of a :meth:`run` (see the class
        docstring's "Data roles" note and :meth:`evaluate_holdout`).

        A scalar-only ``self.objective`` (:func:`~mixle.evolve.objective.calibration_objective`,
        :func:`~mixle.evolve.objective.decision_regret_objective`) has no paired test to run at all --
        every candidate's :class:`~mixle.evolve.verify.Verdict` then carries ``p_value = nan`` by
        design (see that function's scalar-only branch) -- so pass 2 excludes non-finite p-values from
        the pooled family entirely rather than handing them to ``adjust_pvalues`` (which rejects
        non-finite input outright); such a candidate's own unadjusted ``verdict.promote`` stands as its
        final promotion decision, since there is no p-value for a population-wide correction to act on
        -- which, per :attr:`~mixle.evolve.verify.Verdict.promote`, is now ALWAYS ``False`` for a
        scalar-only objective regardless of ``favored`` (no sampling-uncertainty estimate backs a bare
        scalar delta -- see :attr:`~mixle.evolve.verify.Verdict.has_statistical_evidence`).
        """
        fit_data, verify = (data, verify_data) if verify_data is not None else self._auto_split(data)
        self._ensure_initialized(verify)
        report = GenerationReport(best_score=self._champion_score)
        ctx = {"parent_hash": None, "seed": self.seed + self._gen, "objective": self.objective}
        alpha = 0.05  # must match challenger_beats_champion's own default: the per-candidate gate and
        # the population-wide correction below have to agree on one significance level.

        # one operator per parent; how many parents to spawn this generation.
        n_offspring = max(1, self.size // 2)
        parents = self._select_parents(n_offspring)
        ops = self.bandit.select(k=n_offspring)

        # -- pass 1: propose + compare every candidate on its own RAW (uncorrected) p-value -----------
        slots: list[_GenerationSlot] = []
        for parent, op in zip(parents, ops):
            report.operators_used.append(op.name)
            cost = float(getattr(op, "cost_hint", 1.0))
            slot = _GenerationSlot(op=op, cost=cost)
            slots.append(slot)
            try:
                if not op.applicable(parent.model, fit_data, ctx=ctx):
                    continue
                candidate = op.propose(parent.model, fit_data, ctx=ctx)
            except Exception:  # noqa: BLE001
                continue
            report.proposals += 1

            nonnested = type(candidate.model).__name__ != type(parent.model).__name__
            slot.candidate = candidate
            slot.verdict = challenger_beats_champion(
                parent.model,
                candidate.model,
                verify,
                objective=self.objective,
                alpha=alpha,
                nonnested=nonnested,
                seed=self.seed + self._gen,
            )

        # -- pass 2: pool this generation's raw p-values and correct them ONCE, together ---------------
        # exclude non-finite p-values (scalar-only objectives' verdicts -- see the docstring above):
        # adjust_pvalues' _prep step rejects any non-finite entry outright, so pooling even one of
        # these in unfiltered used to crash pass 2 and, with it, promotion for the WHOLE generation --
        # scalar-only candidates and ordinary paired ones alike, since this loop has not run yet.
        compared = [s for s in slots if s.verdict is not None]
        pvalued = [s for s in compared if math.isfinite(s.verdict.p_value)]
        if pvalued:
            raw_pvals = np.asarray([s.verdict.p_value for s in pvalued], dtype=float)
            adjusted = adjust_pvalues(raw_pvals, method="bh", alpha=alpha)["pvals_adjusted"]
            adjusted_by_slot = {id(s): p for s, p in zip(pvalued, adjusted)}
        else:
            adjusted_by_slot = {}

        # -- pass 3: STAGE each compared candidate's final promotion against its adjusted p-value,
        #    scoring every promoted candidate into a population member. Adjustment only ever raises a
        #    p-value, never lowers it, so this can only REVOKE a raw "challenger" verdict -- it can
        #    never promote a candidate the raw, uncorrected test itself refused. A slot excluded from
        #    the pool above (no finite p-value to adjust) keeps its raw, unadjusted verdict.promote
        #    as-is instead.
        #
        #    Nothing durable is written here (MXR-080-1902). The bandit used to be rewarded inside
        #    this loop, one slot at a time, BEFORE `self._member(...)` scored that slot's promoted
        #    candidate -- and `_member` -> `_score` raises on a non-finite fitness by design. A
        #    generation whose third candidate scored non-finite therefore left the first three arms
        #    permanently credited for a generation that produced no report, no survivors, no champion
        #    update and no `_gen` advance; retrying the generation credited them a second time. The
        #    rewards are collected as values here and folded in below, once every candidate this
        #    generation promoted has been successfully scored.
        staged_rewards: list[tuple[str, float, float]] = []  # (operator name, delta, cost), slot order
        new_members: list[_Member] = []
        for slot in slots:
            if slot.verdict is None:
                staged_rewards.append((slot.op.name, 0.0, slot.cost))
                continue
            adj_p = adjusted_by_slot.get(id(slot))
            promote = slot.verdict.promote if adj_p is None else (slot.verdict.promote and bool(adj_p < alpha))
            delta = slot.verdict.delta if promote else 0.0
            staged_rewards.append((slot.op.name, delta, slot.cost))
            if promote:
                report.verified += 1
                new_members.append(self._member(slot.candidate.model, verify))

        # -- COMMIT: every reward is folded in, in the SAME slot order the parents/operators were
        #    drawn in, and only now that the whole generation's work has succeeded.
        for op_name, delta, cost in staged_rewards:
            self.bandit.reward(op_name, delta, cost)
            report.rewards.append(delta)

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

    def run(self, data: Any, verify_data: Any | None = None, generations: int = 5) -> Any:
        """Evolve for ``generations`` steps; return a :class:`~mixle.evolve.search.SearchResult`.

        Args:
            data: the pool operators fit candidates from each generation (data role 1, "fit data", in
                the class docstring). When ``verify_data`` is omitted, ``data`` is split ONCE, up front
                (:meth:`_auto_split`), and only the fit-half is fit on for every generation of this run.
            verify_data: the held-out data used to score fitness and gate every challenger -- data role
                2, "reusable validation data" (see the class docstring's "Data roles" note): the SAME
                split is reused for EVERY generation, and both survivor selection and
                :class:`OperatorBandit` adapt to it, which is exactly the sequential/adaptive reuse the
                class docstring warns is a heuristic search signal, not evidence. Defaults to the
                verify-half of :meth:`_auto_split` applied to ``data`` when omitted (never ``data``
                itself -- MXR-080-0042). Pass an explicit, already-disjoint split for a caller-controlled
                gate (e.g. :func:`~mixle.evolve.search.search` always does).
            generations: number of :meth:`step` calls.

        The returned ``best_model`` is the run incumbent, guaranteed no worse than the best seed *on
        whatever ``verify_data`` this run used* (anti-regression is relative to that reused validation
        signal, not an unconditional guarantee about unseen data -- see :meth:`evaluate_holdout` for the
        one-shot check that actually earns that stronger claim). ``history`` is one row per generation
        (proposals / verified / score).
        """
        from mixle.evolve.search import SearchResult

        # exact, non-truncated generation count (MXR-080-1902): `range(int(generations))` ran 2
        # generations for `generations=2.9` and 1 for `generations=True`, so the search reported a
        # `history` shorter than the run the caller asked for with nothing to say it had been
        # reinterpreted. `generations=0` stays legal -- it is the documented "score the seeds and
        # stop" case that evolve_population_test.py exercises directly.
        generations = _exact_int(generations, "generations", minimum=0)
        fit_data, verify = (data, verify_data) if verify_data is not None else self._auto_split(data)
        self._ensure_initialized(verify)
        history: list[dict[str, Any]] = []
        for _ in range(generations):
            # the SAME (fit_data, verify) pair every generation -- resolved ONCE above, not re-derived
            # per call, so every generation is scored against one consistent yardstick (see
            # _auto_split's docstring) and step() never re-triggers its own (redundant, and here
            # unreachable since verify is always explicit below) default-split path.
            rep = self.step(fit_data, verify)
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
        # Same unsuccessful/incomplete semantics the bo/evolutionary backends report, rather than
        # leaving them at defaults that read as a successful search of zero evaluations: every seed
        # scoring plus every proposal this run made is real, budgeted work, and each one succeeded
        # (a non-finite fitness now raises in _score rather than becoming a member).
        n_evaluations = len(self._raw_seeds) + sum(int(row["proposals"]) for row in history)
        return SearchResult(
            best_config={},  # population searches structures, not a config vector
            best_model=self._champion,
            best_score=float(native_best),
            history=history,
            search_failed=not self._members,
            n_evaluations=n_evaluations,
            n_successes=n_evaluations,
        )

    def evaluate_holdout(
        self,
        holdout_data: Any,
        *,
        reference: Any | None = None,
        **verdict_kwargs: Any,
    ) -> Verdict:
        """The one genuinely evidence-bearing check this class can make (data role 3, "one-shot final
        holdout", in the class docstring) -- call this EXACTLY ONCE, after every :meth:`step` /
        :meth:`run` call you intend to make is already done, on ``holdout_data`` that was never passed
        to this population as ``data`` or ``verify_data`` at any point.

        Every ``Verdict`` / score :meth:`step` computes internally along the way is checked against the
        SAME reused ``verify_data``, generation after generation -- a heuristic search signal, not a
        defensible promotion claim (see the class docstring's "Data roles" note). This method is the
        genuine, single-look alternative: ONE :func:`~mixle.evolve.verify.challenger_beats_champion`
        comparison between ``reference`` (defaults to this population's incumbent BEFORE its first
        :meth:`step` / :meth:`run` call -- i.e. the best seed) and :attr:`champion` (the current
        incumbent).

        Calling this more than once against the same ``holdout_data`` reintroduces exactly the
        adaptive-reuse problem it exists to avoid -- treat ``holdout_data`` as spent after one call, the
        same as any other held-out test set.

        Args:
            holdout_data: fresh data this population has never scored anything on. Verifying that is
                the caller's responsibility -- ``Population`` has no way to check it from the object
                alone.
            reference: the champion to compare :attr:`champion` against; defaults to the pre-run
                incumbent (see above). Pass the currently-deployed production model explicitly if that
                differs.
            verdict_kwargs: forwarded to :func:`~mixle.evolve.verify.challenger_beats_champion` (e.g.
                ``alpha``, ``min_effect``, ``seed``); ``nonnested`` defaults to whether ``reference`` and
                :attr:`champion` are different model types, but can be overridden.

        Returns:
            The one-shot :class:`~mixle.evolve.verify.Verdict`; ``verdict.promote`` is the genuine,
            evidence-backed promotion decision.
        """
        ref = self._initial_champion if reference is None else reference
        verdict_kwargs.setdefault("nonnested", type(self._champion).__name__ != type(ref).__name__)
        return challenger_beats_champion(ref, self._champion, holdout_data, objective=self.objective, **verdict_kwargs)

    @property
    def champion(self) -> Any:
        """The current run incumbent (the best model seen, anti-regression guaranteed)."""
        return self._champion


__all__ = ["OperatorBandit", "Population", "GenerationReport"]
