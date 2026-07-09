"""``do`` -- interventions on a learned heterogeneous Bayesian network (graph-surgery semantics).

The causality front door over :func:`mixle.inference.bayesian_network.learn_bayesian_network`. An
intervention ``do(net, {field: value})`` clamps the intervened fields during ancestral sampling —
their own factors (and hence their parents) are cut out of the generation, which is exactly Pearl's
graph surgery — and everything downstream flows through the fitted conditional factors::

    net = learn_bayesian_network(records)
    world = do(net, {0: 2.0})                    # the world where field 0 is set to 2.0
    world.sample(1000)                            # interventional draws
    world.expectation(2)                          # E[field 2 | do(field 0 = 2.0)]
    average_causal_effect(net, 0, 2.0, 0.0, outcome=2)   # E[Y|do(a)] - E[Y|do(b)]

The signature difference from conditioning: intervening on a DOWNSTREAM field leaves its ancestors
at their marginal law (observing it would have shifted them). ``do`` gives interventional
distributions; counterfactuals (abduction over exogenous noise) are a separate, harder rung and are
deliberately not claimed here.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


class InterventionalNetwork:
    """A Bayesian network under ``do(...)``: sample and summarize the post-intervention world."""

    def __init__(self, net: Any, interventions: dict[int, Any]) -> None:
        self.net = net
        self.interventions = dict(interventions)
        n_fields = len(net.factors)
        for k in self.interventions:
            if not (0 <= int(k) < n_fields):
                raise ValueError(f"intervened field {k} is out of range (network has {n_fields} fields)")

    def sample(self, size: int = 1, *, seed: int | None = None) -> list[tuple]:
        """Ancestral sampling with the intervened fields clamped (their factors are never consulted)."""
        rng = np.random.RandomState(seed)
        by_child = {f.child: f for f in self.net.factors}
        rows: list[tuple] = []
        for _ in range(int(size)):
            vals: list[Any] = [None] * len(self.net.factors)
            for i in self.net.order:
                vals[i] = self.interventions[i] if i in self.interventions else by_child[i].sample(vals, rng)
            rows.append(tuple(vals))
        return rows

    def expectation(self, field: int, *, n: int = 4000, seed: int = 0) -> float:
        """Monte-Carlo ``E[field | do(...)]`` for a numeric field."""
        draws = [row[field] for row in self.sample(n, seed=seed)]
        return float(np.mean(np.asarray(draws, dtype=np.float64)))

    def distribution(self, field: int, *, n: int = 4000, seed: int = 0) -> dict[Any, float]:
        """Interventional marginal of a discrete field as ``{value: probability}``."""
        draws = [row[field] for row in self.sample(n, seed=seed)]
        counts = Counter(draws)
        return {k: v / len(draws) for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))}


def do(net: Any, interventions: dict[int, Any]) -> InterventionalNetwork:
    """Return the network under Pearl's ``do`` operator (see module docstring)."""
    if not hasattr(net, "factors") or not hasattr(net, "order"):
        raise TypeError("do() expects a learned HeterogeneousBayesianNetwork")
    return InterventionalNetwork(net, interventions)


def average_causal_effect(
    net: Any, treatment: int, a: Any, b: Any, outcome: int, *, n: int = 4000, seed: int = 0
) -> float:
    """``E[outcome | do(treatment=a)] - E[outcome | do(treatment=b)]`` (numeric outcome)."""
    ea = do(net, {treatment: a}).expectation(outcome, n=n, seed=seed)
    eb = do(net, {treatment: b}).expectation(outcome, n=n, seed=seed)
    return float(ea - eb)


def counterfactual(net: Any, observed: tuple, interventions: dict[int, Any]) -> tuple:
    """What this observed record would have been under the intervention (abduction-action-prediction).

    Per Pearl's three steps, walked in topological order:

      * **abduction** -- a linear-Gaussian field's exogenous noise is point-identified from the row:
        its residual ``eps = observed - coef @ parents_observed``;
      * **action** -- intervened fields take their ``do`` values;
      * **prediction** -- the same residual replays through the counterfactual parents:
        ``cf = coef @ parents_cf + eps``.

    Boundaries: (1) a field that is not linear-Gaussian keeps its observed value only while its
    parents are unchanged under the intervention (that much IS identified); if its parents change, its
    exogenous noise cannot be recovered from one observation and this raises — use
    :func:`average_causal_effect` for the population answer instead of a guessed individual one.
    (2) The counterfactual is relative to the network's DAG **as given**: purely observational structure
    learning cannot orient Markov-equivalent edges (x -> y and y -> x fit equally well), so if the
    causal direction matters, assert it from domain knowledge rather than trusting the learned arrow.
    """
    from mixle.inference.bayesian_network import _LinearGaussianFactor

    if not hasattr(net, "factors") or not hasattr(net, "order"):
        raise TypeError("counterfactual() expects a learned HeterogeneousBayesianNetwork")
    observed = tuple(observed)
    if len(observed) != len(net.factors):
        raise ValueError(f"observed record has {len(observed)} fields; the network has {len(net.factors)}")
    fixed = {int(k): v for k, v in interventions.items()}
    by_child = {f.child: f for f in net.factors}
    cf: list[Any] = [None] * len(net.factors)
    for i in net.order:
        if i in fixed:
            cf[i] = fixed[i]
            continue
        f = by_child[i]
        if isinstance(f, _LinearGaussianFactor):
            mu_obs = float(f._row([observed[p] for p in f.parents]) @ f.coef)
            eps = float(observed[f.child]) - mu_obs  # abduction
            mu_cf = float(f._row([cf[p] for p in f.parents]) @ f.coef)
            cf[i] = mu_cf + eps  # action + prediction
            continue
        if any(not _same_value(cf[p], observed[p]) for p in getattr(f, "parents", [])):
            raise ValueError(
                f"counterfactual for field {i} is not point-identified: it is not linear-Gaussian and its "
                f"parents changed under the intervention; use average_causal_effect for the population answer."
            )
        cf[i] = observed[i]
    return tuple(cf)


def _same_value(a: Any, b: Any) -> bool:
    try:
        return bool(np.isclose(float(a), float(b)))
    except (TypeError, ValueError):
        return a == b
