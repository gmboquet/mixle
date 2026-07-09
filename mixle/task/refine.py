"""Diagnosis-directed correction versus blind structure search.

``diagnose`` turns failing cases into a :class:`~mixle.inference.explain.FaultReport` naming a
structural element and a suggested fix. This module compares acting on that diagnosis with blind
search over the same edit space. Every candidate edge, whether diagnosed or blindly tried, is refit
from the same training data and verified against held-out data before acceptance. A candidate that
does not clear the held-out bar is recorded as a failed trial rather than silently kept.

The comparison is intentionally conservative: if diagnosis-directed correction does not reach the
held-out target in fewer trials than blind search on the planted-fault benchmark, the recorded result
shows that the diagnosis did not justify using the directed refinement path for that case.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.inference.bayesian_network import HeterogeneousBayesianNetwork, _LinearGaussianFactor, _MarginalFactor
from mixle.inference.explain import diagnose
from mixle.stats import GaussianDistribution


def _columns(data: Sequence[tuple]) -> list[list[Any]]:
    return [list(col) for col in zip(*data)]


def _mean_log_density(model: HeterogeneousBayesianNetwork, data: Sequence[tuple]) -> float:
    return float(np.mean([model.log_density(x) for x in data]))


def fit_independent_baseline(train_data: Sequence[tuple]) -> HeterogeneousBayesianNetwork:
    """Fit an independent network with one marginal Gaussian per field."""
    cols = _columns(train_data)
    factors = []
    for i, col in enumerate(cols):
        arr = np.asarray(col, dtype=np.float64)
        factors.append(_MarginalFactor(i, GaussianDistribution(float(arr.mean()), float(max(arr.var(), 1e-9)))))
    return HeterogeneousBayesianNetwork(factors)


def apply_edge(
    model: HeterogeneousBayesianNetwork, edge: tuple[int, int], train_data: Sequence[tuple]
) -> HeterogeneousBayesianNetwork:
    """Refit the named child factor as a linear-Gaussian conditional.

    Every other factor is kept unchanged, so the returned model represents only
    the proposed edge edit.
    """
    parent, child = edge
    cols = _columns(train_data)
    new_factor = _LinearGaussianFactor.fit(child, [parent], cols, discrete={})
    factors = [f if f.child != child else new_factor for f in model.factors]
    return HeterogeneousBayesianNetwork(factors)


@dataclass
class EditTrial:
    """Held-out result for one proposed graph edit."""

    edge: tuple[int, int]
    held_out_score: float
    verified: bool  # cleared the held-out target


@dataclass
class SearchOutcome:
    """Final refinement state plus the verified edit-search history."""

    trials: int
    found_edge: tuple[int, int] | None
    final_model: HeterogeneousBayesianNetwork
    history: list[EditTrial] = field(default_factory=list)


def _try_edge(
    model: HeterogeneousBayesianNetwork,
    edge: tuple[int, int],
    train_data: Sequence[tuple],
    held_out: Sequence[tuple],
    *,
    target: float,
) -> EditTrial:
    candidate = apply_edge(model, edge, train_data)
    score = _mean_log_density(candidate, held_out)
    return EditTrial(edge=edge, held_out_score=score, verified=score >= target)


def _dominant_pair(dominant: str) -> tuple[int, int] | None:
    """Parse the two field indices out of a ``diagnose`` dominant string like
    ``"field[1]|parents()+field[2]|parents()"``; ``None`` if it does not name a two-field pair."""
    parts = dominant.split("+")
    if len(parts) != 2:
        return None
    try:
        return tuple(int(p.split("[", 1)[1].split("]", 1)[0]) for p in parts)  # type: ignore[return-value]
    except (IndexError, ValueError):
        return None


def diagnosis_directed_correction(
    model: HeterogeneousBayesianNetwork,
    train_data: Sequence[tuple],
    failing_cases: Sequence[tuple],
    held_out: Sequence[tuple],
    *,
    background: Sequence[tuple] | None = None,
    target: float,
) -> SearchOutcome:
    """Diagnose the fault from ``failing_cases``, apply only its suggested edge (trying both parent-child
    orientations of the named pair, since ``diagnose`` reports an undirected co-anomaly), and verify held-out
    improvement before accepting -- one trial if the diagnosis names the right pair and orientation,
    honestly more (or a refusal) if it does not."""
    report = diagnose(model, failing_cases, background=background)
    history: list[EditTrial] = []
    if report.suggested_fix != "add_edge":
        return SearchOutcome(trials=0, found_edge=None, final_model=model, history=history)

    pair = _dominant_pair(report.dominant)
    if pair is None:
        return SearchOutcome(trials=0, found_edge=None, final_model=model, history=history)

    for edge in (pair, (pair[1], pair[0])):
        trial = _try_edge(model, edge, train_data, held_out, target=target)
        history.append(trial)
        if trial.verified:
            return SearchOutcome(
                trials=len(history), found_edge=edge, final_model=apply_edge(model, edge, train_data), history=history
            )
    return SearchOutcome(trials=len(history), found_edge=None, final_model=model, history=history)


def blind_structure_search(
    model: HeterogeneousBayesianNetwork,
    train_data: Sequence[tuple],
    held_out: Sequence[tuple],
    edit_space: Sequence[tuple[int, int]],
    *,
    target: float,
) -> SearchOutcome:
    """Try candidate edges in order and accept only verified held-out gains."""
    history: list[EditTrial] = []
    for edge in edit_space:
        trial = _try_edge(model, edge, train_data, held_out, target=target)
        history.append(trial)
        if trial.verified:
            return SearchOutcome(
                trials=len(history), found_edge=edge, final_model=apply_edge(model, edge, train_data), history=history
            )
    return SearchOutcome(trials=len(history), found_edge=None, final_model=model, history=history)
