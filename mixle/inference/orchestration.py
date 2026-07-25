"""Learned orchestration from telemetry.

The static placement policy decides local-versus-pool execution from rules.
:class:`LearnedPolicy` can improve that decision from historical telemetry rows
of the form ``(features, choice, outcome)``. For a new feature vector it looks
up nearby historical decisions, estimates which choice had lower realized cost,
and uses the learned choice only when there is enough comparable evidence.

When the nearby history is sparse or ambiguous, the policy defers to the static
fallback. Promotion is a separate operation: it requires isolated realized
holdout outcomes, common support, paired or propensity-aware evaluation, and an
uncertainty-adjusted improvement threshold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


def _featurize(features: dict[str, Any], keys: list[str]) -> np.ndarray:
    """A numeric vector from a decision's feature dict, in a fixed key order (bools -> 0/1, else float)."""
    row = []
    for k in keys:
        v = features.get(k, 0.0)
        row.append(float(v) if isinstance(v, (int, float, bool, np.integer, np.floating)) else 0.0)
    return np.asarray(row, dtype=np.float64)


def _context_key(features: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Stable equality key used to keep repeated action outcomes in one split."""
    if not isinstance(features, dict):
        raise ValueError("telemetry features must be dictionaries")
    return tuple(sorted((str(key), repr(value)) for key, value in features.items()))


def _realized_cost(outcome: dict[str, Any], cost_key: str) -> float:
    if not isinstance(outcome, dict) or cost_key not in outcome:
        raise ValueError(f"every telemetry outcome must contain {cost_key!r}")
    cost = float(outcome[cost_key])
    if not np.isfinite(cost):
        raise ValueError(f"telemetry {cost_key!r} values must be finite")
    return cost


def _outcome_panel(
    rows: list[tuple[dict[str, Any], str, dict[str, Any]]], cost_key: str
) -> list[tuple[dict[str, Any], dict[str, float]]]:
    panels: dict[tuple[tuple[str, str], ...], tuple[dict[str, Any], dict[str, list[float]]]] = {}
    for features, choice, outcome in rows:
        if not isinstance(choice, str) or not choice:
            raise ValueError("telemetry choices must be non-empty strings")
        key = _context_key(features)
        if key not in panels:
            panels[key] = (dict(features), {})
        panels[key][1].setdefault(choice, []).append(_realized_cost(outcome, cost_key))
    return [
        (features, {choice: float(np.mean(costs)) for choice, costs in outcomes.items()})
        for features, outcomes in panels.values()
    ]


def _paired_policy_costs(
    rows: list[tuple[dict[str, Any], str, dict[str, Any]]],
    learned_pick: Callable[[dict[str, Any]], str],
    static_pick: Callable[[dict[str, Any]], str],
    cost_key: str,
) -> dict[str, Any]:
    panels = _outcome_panel(rows, cost_key)
    learned_costs: list[float] = []
    static_costs: list[float] = []
    for features, outcomes in panels:
        learned_choice = learned_pick(features)
        static_choice = static_pick(features)
        if learned_choice in outcomes and static_choice in outcomes:
            learned_costs.append(outcomes[learned_choice])
            static_costs.append(outcomes[static_choice])
    total = len(panels)
    return {
        "method": "paired_realized_outcomes",
        "learned": np.asarray(learned_costs, dtype=float),
        "static": np.asarray(static_costs, dtype=float),
        "n_total": total,
        "n_supported": len(learned_costs),
        "overlap_fraction": len(learned_costs) / total if total else 0.0,
    }


def _ips_policy_costs(
    rows: list[tuple[dict[str, Any], str, dict[str, Any]]],
    learned_pick: Callable[[dict[str, Any]], str],
    static_pick: Callable[[dict[str, Any]], str],
    cost_key: str,
    propensity_key: str,
    min_propensity: float,
) -> dict[str, Any]:
    learned_costs: list[float] = []
    static_costs: list[float] = []
    supported = 0
    for features, logged_choice, outcome in rows:
        propensities = outcome.get(propensity_key) if isinstance(outcome, dict) else None
        if not isinstance(propensities, dict):
            continue
        try:
            probabilities = {str(choice): float(value) for choice, value in propensities.items()}
        except (TypeError, ValueError):
            continue
        if (
            not probabilities
            or not np.all(np.isfinite(list(probabilities.values())))
            or any(value < 0 for value in probabilities.values())
            or not np.isclose(sum(probabilities.values()), 1.0, rtol=1e-8, atol=1e-10)
        ):
            continue
        learned_choice = learned_pick(features)
        static_choice = static_pick(features)
        logged_probability = probabilities.get(logged_choice, 0.0)
        if (
            probabilities.get(learned_choice, 0.0) < min_propensity
            or probabilities.get(static_choice, 0.0) < min_propensity
            or logged_probability < min_propensity
        ):
            continue
        cost = _realized_cost(outcome, cost_key)
        learned_costs.append(cost / logged_probability if learned_choice == logged_choice else 0.0)
        static_costs.append(cost / logged_probability if static_choice == logged_choice else 0.0)
        supported += 1
    total = len(rows)
    return {
        "method": "inverse_propensity_weighted",
        "learned": np.asarray(learned_costs, dtype=float),
        "static": np.asarray(static_costs, dtype=float),
        "n_total": total,
        "n_supported": supported,
        "overlap_fraction": supported / total if total else 0.0,
    }


def _learning_controls(k: int, min_neighbors: int) -> tuple[int, int]:
    for value, name in ((k, "k"), (min_neighbors, "min_neighbors")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    return int(k), int(min_neighbors)


@dataclass
class LearnedPolicy:
    """A history-based placement policy that defers to a static teacher where it lacks evidence."""

    keys: list[str]  # feature key order
    vecs: np.ndarray  # (n, d) standardized historical feature vectors
    choices: list[str]  # the choice taken on each historical row
    costs: np.ndarray  # (n,) the realized cost/outcome of each historical row (lower is better)
    static: Callable[[dict[str, Any]], str]  # the fallback policy
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scale: np.ndarray = field(default_factory=lambda: np.ones(0))
    k: int = 8
    min_neighbors: int = 4
    margin: float = 0.02  # required cost gap between the best and next-best choice to trust the learned pick

    def _neighbors(self, vec: np.ndarray) -> np.ndarray:
        z = (vec - self.mean) / self.scale
        d = np.linalg.norm(self.vecs - z[None, :], axis=1)
        return np.argsort(d)[: self.k]

    def decide(self, features: dict[str, Any]) -> tuple[str, bool]:
        """Return ``(choice, learned)`` -- the learned pick when confident, else the static fallback."""
        if len(self.costs) < self.min_neighbors:
            return self.static(features), False
        idx = self._neighbors(_featurize(features, self.keys))
        if len(idx) < self.min_neighbors:
            return self.static(features), False
        by_choice: dict[str, list[float]] = {}
        for i in idx:
            by_choice.setdefault(self.choices[i], []).append(float(self.costs[i]))
        means = {c: float(np.mean(v)) for c, v in by_choice.items() if len(v) >= 2}
        if len(means) < 2:  # only one choice seen nearby -> not enough to compare, defer
            return self.static(features), False
        ordered = sorted(means.items(), key=lambda t: t[1])
        best, best_cost = ordered[0]
        if ordered[1][1] - best_cost < self.margin:  # the choices are effectively tied -> defer
            return self.static(features), False
        return best, True

    def evaluate(
        self, rows: list[tuple[dict[str, Any], str, dict[str, Any]]], *, cost_key: str = "cost"
    ) -> dict[str, Any]:
        """Compare policies on the same contexts with actually observed action costs.

        A context contributes only when both the learned and static selected
        actions have realized outcomes in the supplied holdout panel. No
        training-neighbor estimate is substituted for a missing outcome.
        """
        paired = _paired_policy_costs(rows, lambda f: self.decide(f)[0], self.static, cost_key)
        panels = _outcome_panel(rows, cost_key)
        deferred = sum(not self.decide(features)[1] for features, _ in panels)
        learned_costs = paired["learned"]
        static_costs = paired["static"]
        fixed_mean: dict[str, float | None] = {}
        fixed_support: dict[str, int] = {}
        for choice in set(self.choices):
            costs = [outcomes[choice] for _features, outcomes in panels if choice in outcomes]
            fixed_mean[choice] = float(np.mean(costs)) if costs else None
            fixed_support[choice] = len(costs)
        return {
            "n": paired["n_supported"],
            "n_holdout_contexts": paired["n_total"],
            "overlap_fraction": paired["overlap_fraction"],
            "evaluation_method": paired["method"],
            "learned_mean_cost": float(np.mean(learned_costs)) if learned_costs.size else None,
            "static_mean_cost": float(np.mean(static_costs)) if static_costs.size else None,
            "fixed_mean_cost": fixed_mean,
            "fixed_support": fixed_support,
            "deferred_fraction": deferred / len(panels) if panels else 0.0,
        }


def _expand_action_features(feats: dict[str, Any]) -> dict[str, float]:
    """One-hot the categorical ``kind`` so the numeric featurizer keeps the action-type signal."""
    out: dict[str, float] = {}
    for kk, vv in feats.items():
        if kk == "kind":
            out[f"kind={vv}"] = 1.0
        elif isinstance(vv, (int, float, bool, np.integer, np.floating)):
            out[kk] = float(vv)
    return out


@dataclass
class LearnedAcquisition:
    """A history-based action scorer for the reasoner: learns which actions pay off, else defers.

    Drop-in for :func:`mixle.substrate.act.score_action` (call it as ``scorer=policy`` in ``investigate``).
    From ``route`` telemetry -- each row a fired action's ``(features={kind,cost,overlap}, value)`` -- it
    estimates the expected *yield* of an action in a query's feature region and scores it ``yield / cost``.
    Where nearby history is too thin, it falls back to the static lexical scorer.
    This is an abstention rule, not a guarantee that learned decisions are never worse."""

    keys: list[str]
    vecs: np.ndarray  # (n, d) standardized historical action-feature vectors
    values: np.ndarray  # (n,) realized yield of each historical action (higher is better)
    static: Callable[[Any, str], float]  # fallback scorer (action, question) -> float
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scale: np.ndarray = field(default_factory=lambda: np.ones(0))
    k: int = 8
    min_neighbors: int = 4

    def _neighbors(self, vec: np.ndarray) -> np.ndarray:
        z = (vec - self.mean) / self.scale
        d = np.linalg.norm(self.vecs - z[None, :], axis=1)
        return np.argsort(d)[: self.k]

    def expected_yield(self, features: dict[str, Any]) -> float | None:
        """Estimated yield of an action with these features, or None when history is too thin to say."""
        if len(self.values) < self.min_neighbors:
            return None
        idx = self._neighbors(_featurize(_expand_action_features(features), self.keys))
        if len(idx) < self.min_neighbors:
            return None
        return float(np.mean(self.values[idx]))

    def __call__(self, action: Any, question: str) -> float:
        from mixle.substrate.act import action_features

        feats = action_features(action, question)
        ey = self.expected_yield(feats)
        if ey is None:
            return self.static(action, question)  # defer where evidence is thin
        return ey / max(float(feats.get("cost", 1.0)), 1e-9)


def learn_action_policy(
    rows: list[tuple[dict[str, Any], str, dict[str, Any]]],
    static_scorer: Callable[[Any, str], float] | None = None,
    *,
    value_key: str = "value",
    k: int = 8,
    min_neighbors: int = 4,
) -> LearnedAcquisition:
    """Learn a reasoner acquisition policy from ``route`` telemetry ``(features, kind, outcome)`` rows.

    ``static_scorer`` is the fall-back when history is thin (default :func:`mixle.substrate.act.score_action`).
    ``value_key`` names the outcome field to MAXIMIZE (default ``"value"`` -- did the action yield
    evidence). Returns a :class:`LearnedAcquisition` usable directly as ``investigate(..., scorer=policy)``.
    """
    if not rows:
        raise ValueError("learn_action_policy needs telemetry rows")
    k, min_neighbors = _learning_controls(k, min_neighbors)
    if static_scorer is None:
        from mixle.substrate.act import score_action as static_scorer  # noqa: N806
    if not callable(static_scorer):
        raise TypeError("static_scorer must be callable")
    expanded = [_expand_action_features(feats) for feats, _c, _o in rows]
    keys = sorted({k2 for feats in expanded for k2 in feats})
    vecs = np.stack([_featurize(feats, keys) for feats in expanded])
    try:
        values = np.asarray([float(o[value_key]) for _f, _c, o in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"every action telemetry outcome must contain numeric {value_key!r}") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("action telemetry values must be finite")
    mean = vecs.mean(axis=0)
    scale = vecs.std(axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    z = (vecs - mean) / scale
    return LearnedAcquisition(
        keys=keys,
        vecs=z,
        values=values,
        static=static_scorer,
        mean=mean,
        scale=scale,
        k=k,
        min_neighbors=min_neighbors,
    )


def learn_placement_policy(
    rows: list[tuple[dict[str, Any], str, dict[str, Any]]],
    static_policy: Callable[[dict[str, Any]], str],
    *,
    cost_key: str = "cost",
    k: int = 8,
    min_neighbors: int = 4,
) -> LearnedPolicy:
    """Learn a placement policy from telemetry ``(features, choice, outcome)`` rows (see module docstring).

    ``static_policy`` maps a feature dict to a choice and is the fall-back when history is too thin.
    ``cost_key`` names the outcome field to minimize (default ``"cost"``). Feature standardization and
    the neighbor index are built from the rows; :meth:`LearnedPolicy.decide` and ``evaluate`` follow.
    """
    if not rows:
        raise ValueError("learn_placement_policy needs telemetry rows")
    if not callable(static_policy):
        raise TypeError("static_policy must be callable")
    k, min_neighbors = _learning_controls(k, min_neighbors)
    keys = sorted({k2 for feats, _c, _o in rows for k2 in feats})
    vecs = np.stack([_featurize(feats, keys) for feats, _c, _o in rows]) if rows else np.zeros((0, len(keys)))
    choices = [c for _f, c, _o in rows]
    if any(not isinstance(choice, str) or not choice for choice in choices):
        raise ValueError("telemetry choices must be non-empty strings")
    costs = np.asarray([_realized_cost(outcome, cost_key) for _features, _choice, outcome in rows], dtype=np.float64)
    mean = vecs.mean(axis=0) if len(vecs) else np.zeros(len(keys))
    scale = vecs.std(axis=0) if len(vecs) else np.ones(len(keys))
    scale = np.where(scale < 1e-9, 1.0, scale)
    z = (vecs - mean) / scale if len(vecs) else vecs
    return LearnedPolicy(
        keys=keys,
        vecs=z,
        choices=choices,
        costs=costs,
        static=static_policy,
        mean=mean,
        scale=scale,
        k=k,
        min_neighbors=min_neighbors,
    )


def learn_schedule_policy(
    rows: list[tuple[dict[str, Any], str, dict[str, Any]]],
    static_policy: Callable[[dict[str, Any]], str],
    *,
    latency_key: str = "latency",
    k: int = 8,
    min_neighbors: int = 4,
) -> LearnedPolicy:
    """Learned pool scheduling (J4): when work is pool-eligible, learn where and when it actually runs fastest.

    The same evidence-gated shape as placement, keyed on realized latency instead of dollar cost: rows are
    ``(features, choice, outcome)`` where features describe the moment (queue depth, job size, local
    load), choice is the scheduling decision ("run_local" / "queue_pool" / "defer"), and the outcome's
    ``latency`` is what the decision actually cost in wall-clock. Where nearby history is thin, the
    returned policy defers to the static scheduler."""
    return learn_placement_policy(rows, static_policy, cost_key=latency_key, k=k, min_neighbors=min_neighbors)


def meta_improve(
    rows: list[tuple[dict[str, Any], str, dict[str, Any]]],
    static_policy: Callable[[dict[str, Any]], str],
    *,
    holdout_rows: list[tuple[dict[str, Any], str, dict[str, Any]]] | None = None,
    cost_key: str = "cost",
    holdout_frac: float = 0.3,
    seed: int = 0,
    k: int = 8,
    min_neighbors: int = 4,
    propensity_key: str = "propensities",
    min_propensity: float = 0.05,
    min_overlap: float = 0.8,
    confidence_level: float = 0.95,
    min_improvement: float = 0.0,
) -> dict[str, Any]:
    """Learn from telemetry and promote only with isolated, supported holdout evidence.

    Repeated outcomes for the same feature context are kept in one split. Evaluation
    first uses paired realized outcomes when both policies' selected actions were
    observed for the same context. If a complete propensity map is logged instead,
    inverse-propensity evaluation is available. Promotion requires minimum overlap
    and the one-sided confidence bound on ``learned - static`` to beat the requested
    improvement threshold. Returns::

        {promoted, policy, receipt: {learned_mean_cost, static_mean_cost,
         mean_cost_difference, upper_confidence_bound, overlap_fraction, n}}

    ``policy`` is learned only when promoted; otherwise it is exactly
    ``static_policy``. The receipt records either decision.
    """
    if not callable(static_policy):
        raise TypeError("static_policy must be callable")
    k, min_neighbors = _learning_controls(k, min_neighbors)
    for value, name in (
        (holdout_frac, "holdout_frac"),
        (min_propensity, "min_propensity"),
        (min_overlap, "min_overlap"),
        (confidence_level, "confidence_level"),
        (min_improvement, "min_improvement"),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 < holdout_frac < 1:
        raise ValueError("holdout_frac must be strictly between 0 and 1")
    if not 0 < min_propensity <= 1:
        raise ValueError("min_propensity must lie in (0, 1]")
    if not 0 < min_overlap <= 1:
        raise ValueError("min_overlap must lie in (0, 1]")
    if not 0.5 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0.5, 1)")
    if min_improvement < 0:
        raise ValueError("min_improvement must be >= 0")
    if not rows:
        raise ValueError("meta_improve needs non-empty training telemetry")

    if holdout_rows is None:
        grouped: dict[tuple[tuple[str, str], ...], list[tuple[dict[str, Any], str, dict[str, Any]]]] = {}
        for row in rows:
            grouped.setdefault(_context_key(row[0]), []).append(row)
        context_keys = list(grouped)
        if len(context_keys) < 4:
            raise ValueError("meta_improve needs at least 4 distinct contexts to split train/holdout")
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(context_keys))
        n_hold = min(len(context_keys) - 1, max(1, int(round(holdout_frac * len(context_keys)))))
        hold_keys = {context_keys[index] for index in order[:n_hold]}
        train = [row for key, group in grouped.items() if key not in hold_keys for row in group]
        holdout = [row for key, group in grouped.items() if key in hold_keys for row in group]
        split_method = "context_group_split"
    else:
        train = list(rows)
        holdout = list(holdout_rows)
        if not holdout:
            raise ValueError("holdout_rows must be non-empty when supplied")
        train_contexts = {_context_key(row[0]) for row in train}
        holdout_contexts = {_context_key(row[0]) for row in holdout}
        if train_contexts & holdout_contexts:
            raise ValueError("explicit holdout contexts must be disjoint from training contexts")
        split_method = "external_isolated_holdout"

    learned = learn_placement_policy(train, static_policy, cost_key=cost_key, k=k, min_neighbors=min_neighbors)
    learned_pick = lambda features: learned.decide(features)[0]
    paired = _paired_policy_costs(holdout, learned_pick, static_policy, cost_key)
    ips = _ips_policy_costs(
        holdout,
        learned_pick,
        static_policy,
        cost_key,
        propensity_key,
        min_propensity,
    )
    eligible = [
        estimate
        for estimate in (paired, ips)
        if estimate["n_supported"] >= 2 and estimate["overlap_fraction"] >= min_overlap
    ]
    if eligible:
        evaluation = max(eligible, key=lambda estimate: (estimate["overlap_fraction"], estimate["n_supported"]))
    else:
        evaluation = max((paired, ips), key=lambda estimate: (estimate["overlap_fraction"], estimate["n_supported"]))

    learned_costs = evaluation["learned"]
    static_costs = evaluation["static"]
    learned_mean = float(np.mean(learned_costs)) if learned_costs.size else None
    static_mean = float(np.mean(static_costs)) if static_costs.size else None
    mean_difference = standard_error = upper_bound = None
    if evaluation["n_supported"] < 2 or evaluation["overlap_fraction"] < min_overlap:
        promoted = False
        reason = (
            "insufficient common holdout support: "
            f"{evaluation['n_supported']}/{evaluation['n_total']} supported "
            f"({evaluation['overlap_fraction']:.1%}, need {min_overlap:.1%})"
        )
    else:
        differences = learned_costs - static_costs
        mean_difference = float(np.mean(differences))
        standard_error = float(np.std(differences, ddof=1) / np.sqrt(differences.size))
        upper_bound = float(mean_difference + stats.norm.ppf(confidence_level) * standard_error)
        promoted = bool(upper_bound < -min_improvement)
        if promoted:
            reason = (
                f"promoted: same held-out contexts give upper {confidence_level:.1%} bound "
                f"{upper_bound:.4g} < {-min_improvement:.4g}"
            )
        else:
            reason = (
                f"not promoted: same held-out contexts give upper {confidence_level:.1%} bound "
                f"{upper_bound:.4g}, threshold {-min_improvement:.4g}"
            )

    receipt = {
        "learned_mean_cost": learned_mean,
        "static_mean_cost": static_mean,
        "mean_cost_difference": mean_difference,
        "standard_error": standard_error,
        "upper_confidence_bound": upper_bound,
        "confidence_level": confidence_level,
        "min_improvement": min_improvement,
        "evaluation_method": evaluation["method"],
        "split_method": split_method,
        "n_matched_learned": evaluation["n_supported"],
        "n_matched_static": evaluation["n_supported"],
        "n_evaluated": evaluation["n_supported"],
        "n_holdout": evaluation["n_total"],
        "n_holdout_rows": len(holdout),
        "overlap_fraction": evaluation["overlap_fraction"],
        "min_overlap": min_overlap,
        "min_propensity": min_propensity,
        "same_holdout": True,
        "reason": reason,
    }
    if promoted:
        policy: Callable[[dict[str, Any]], str] = lambda feats: learned.decide(feats)[0]  # noqa: E731
    else:
        policy = static_policy  # the receipt said no: keep the teacher
    return {"promoted": bool(promoted), "policy": policy, "learned": learned, "receipt": receipt}
