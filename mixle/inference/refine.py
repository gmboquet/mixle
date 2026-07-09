"""Diagnosis-directed correction compared with blind structure search.

If diagnosis-directed correction does not reach the same held-out target in
fewer trials than blind structure search (``learn_bayesian_network`` run with no
diagnosis, over growing prefixes of data) on the planted-fault benchmark, keep
blind search as the baseline.

Only the `add_edge` fix is translated to a concrete structural edit here, because that is the only fix
`mixle.inference.explain.diagnose` actually detects today (`upgrade_leaf`/`split_region`/`add_factor` are
recognized vocabulary with no detector wired yet -- see that module's docstring). A `FaultReport` naming
any other fix is reported as "not actionable," never guessed at.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.inference.bayesian_network import HeterogeneousBayesianNetwork, _LinearGaussianFactor, learn_bayesian_network
from mixle.inference.explain import FaultReport, diagnose


def held_out_log_likelihood(model: Any, data: Sequence[tuple]) -> float:
    """Total log-density of a fitted network (or anything exposing ``dist_to_encoder``/``seq_log_density``)
    over held-out ``data`` -- the metric both paths below are compared on."""
    enc = model.dist_to_encoder().seq_encode(list(data))
    return float(np.sum(model.seq_log_density(enc)))


def apply_add_edge_fix(
    model: HeterogeneousBayesianNetwork, fault: FaultReport, data: Sequence[tuple]
) -> HeterogeneousBayesianNetwork | None:
    """Apply only the fix ``diagnose`` actually named: add a linear-Gaussian edge between the two fields
    in ``fault.dominant`` (parsed from its ``"field[i]|...field[j]|..."`` shape), refit that one factor on
    ``data``, and return the corrected network -- every other factor is left untouched.

    Returns ``None`` (never guesses) when ``fault.suggested_fix`` is not ``"add_edge"``, or ``dominant``
    does not name exactly two fields (nothing dominant, or a shape this translation doesn't understand).
    """
    if fault.suggested_fix != "add_edge":
        return None
    idx = sorted({int(m) for m in re.findall(r"field\[(\d+)\]", fault.dominant)})
    if len(idx) != 2:
        return None
    parent, child = idx
    cols = [[row[i] for row in data] for i in range(len(data[0]))]
    new_factor = _LinearGaussianFactor.fit(child, [parent], cols, {})
    factors = [f for f in model.factors if f.child != child] + [new_factor]
    return HeterogeneousBayesianNetwork(factors)


@dataclass
class TrialsToTarget:
    """How many trials a path needed to reach its target (``None`` = never reached), plus its score history."""

    n_trials: int | None
    final_model: Any
    history: list[float] = field(default_factory=list)


def directed_correction(
    model: Any,
    cases: Sequence[tuple],
    data: Sequence[tuple],
    held_out: Sequence[tuple],
    *,
    background: Sequence[tuple] | None = None,
) -> TrialsToTarget:
    """One diagnosis, one targeted edit, one verification.

    Costs exactly 1 trial if ``diagnose`` names an actionable fix and it verifiably improves held-out
    score over the original model; costs 0 (unreached) if the fix isn't actionable or doesn't verify --
    a correction that does not improve held-out is a failed diagnosis, logged as such, never silently
    kept anyway.
    """
    fault = diagnose(model, cases, background=background)
    fixed = apply_add_edge_fix(model, fault, data)
    before = held_out_log_likelihood(model, held_out)
    if fixed is None:
        return TrialsToTarget(n_trials=None, final_model=model, history=[before])
    after = held_out_log_likelihood(fixed, held_out)
    if after <= before:
        return TrialsToTarget(n_trials=None, final_model=model, history=[before, after])
    return TrialsToTarget(n_trials=1, final_model=fixed, history=[before, after])


def blind_search_trials_to_target(
    data: Sequence[tuple],
    held_out: Sequence[tuple],
    target_score: float,
    *,
    round_size: int = 10,
    max_rounds: int = 20,
    max_parents: int = 1,
    seed: int = 0,
) -> TrialsToTarget:
    """The blind baseline: NO diagnosis. Re-run ``learn_bayesian_network`` from scratch on growing
    prefixes of ``data`` (``round_size`` more examples each round, shuffled once up front) until the
    discovered structure's held-out score reaches ``target_score`` (typically
    :func:`directed_correction`'s own held-out score) or ``max_rounds`` is exhausted.
    """
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(data))
    shuffled = [data[i] for i in order]
    history: list[float] = []
    model = None
    for round_idx in range(1, max_rounds + 1):
        n = min(round_idx * round_size, len(shuffled))
        model = learn_bayesian_network(shuffled[:n], max_parents=max_parents)
        score = held_out_log_likelihood(model, held_out)
        history.append(score)
        if score >= target_score:
            return TrialsToTarget(n_trials=round_idx, final_model=model, history=history)
        if n >= len(shuffled):
            break
    return TrialsToTarget(n_trials=None, final_model=model, history=history)
