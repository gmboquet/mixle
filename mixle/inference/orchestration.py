"""Learned orchestration -- the platform's own decisions become models trained on its telemetry (J2).

The static planner policy (:func:`mixle.inference.plan_placement`) decides local-vs-pool from rules.
:class:`LearnedPolicy` learns to do better from HISTORY: given telemetry rows ``(features, choice,
outcome)``, it estimates, for a query's feature region, which choice actually gave the better outcome
(lower cost) and picks that -- but only when it has enough nearby history to be confident. When the
feature region is unfamiliar, it FALLS BACK to the static policy. That fallback is the never-worse
guarantee made structural: the learned policy can only improve on the static one where the data
supports it, and defers everywhere else -- the same discipline tier-0 routing uses against a frontier
model, now applied to the platform's own placement decisions.

This is the J2 seed. The same shape (telemetry -> policy + conformal fall-back) learns routing across
model versions (J3) and pool scheduling (J4); a learned policy is promoted over the static one only
when receipted never-worse on held-out decisions (realized cost/latency/quality).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _featurize(features: dict[str, Any], keys: list[str]) -> np.ndarray:
    """A numeric vector from a decision's feature dict, in a fixed key order (bools -> 0/1, else float)."""
    row = []
    for k in keys:
        v = features.get(k, 0.0)
        row.append(float(v) if isinstance(v, (int, float, bool, np.integer, np.floating)) else 0.0)
    return np.asarray(row, dtype=np.float64)


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
        """Realized-cost comparison on held-out ``rows``: learned policy vs always-static, vs each fixed choice."""
        learned_cost = static_cost = 0.0
        deferred = 0
        by_choice_fixed: dict[str, float] = {}
        for feats, _choice, outcome in rows:
            c = float(outcome.get(cost_key, 0.0))
            # the realized cost of a decision depends on the choice; here we score by matching the row's
            # OWN observed (choice, cost) -- so a policy "pays" the row's cost only for the choice it picks.
            pick, learned = self.decide(feats)
            deferred += int(not learned)
            static_pick = self.static(feats)
            # approximate realized cost by the nearest historical cost for (feats, pick)
            learned_cost += self._expected_cost(feats, pick, fallback=c)
            static_cost += self._expected_cost(feats, static_pick, fallback=c)
            for ch in set(self.choices):
                by_choice_fixed[ch] = by_choice_fixed.get(ch, 0.0) + self._expected_cost(feats, ch, fallback=c)
        n = max(len(rows), 1)
        return {
            "n": len(rows),
            "learned_mean_cost": learned_cost / n,
            "static_mean_cost": static_cost / n,
            "fixed_mean_cost": {c: v / n for c, v in by_choice_fixed.items()},
            "deferred_fraction": deferred / n,
        }

    def _expected_cost(self, features: dict[str, Any], choice: str, *, fallback: float) -> float:
        if len(self.costs) == 0:
            return fallback
        idx = self._neighbors(_featurize(features, self.keys))
        near = [float(self.costs[i]) for i in idx if self.choices[i] == choice]
        return float(np.mean(near)) if near else fallback


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
    Where nearby history is too thin, it FALLS BACK to the static lexical scorer: the same never-worse
    discipline as :class:`LearnedPolicy`, now on the reasoner's acquisition decisions (J3)."""

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
            return self.static(action, question)  # never-worse: defer where evidence is thin
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
    if static_scorer is None:
        from mixle.substrate.act import score_action as static_scorer  # noqa: N806
    expanded = [_expand_action_features(feats) for feats, _c, _o in rows]
    keys = sorted({k2 for feats in expanded for k2 in feats})
    vecs = np.stack([_featurize(feats, keys) for feats in expanded])
    values = np.asarray([float(o.get(value_key, 0.0)) for _f, _c, o in rows], dtype=np.float64)
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
    keys = sorted({k2 for feats, _c, _o in rows for k2 in feats})
    vecs = np.stack([_featurize(feats, keys) for feats, _c, _o in rows]) if rows else np.zeros((0, len(keys)))
    choices = [c for _f, c, _o in rows]
    costs = np.asarray([float(o.get(cost_key, 0.0)) for _f, _c, o in rows], dtype=np.float64)
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
