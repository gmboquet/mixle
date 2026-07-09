"""Fit plans as models over harvested agent traces.

A plan is the ordered sequence of tool NAMES an agent called for a request. Fitting a Markov chain
over those sequences (via the ordinary ``optimize`` entry point every mixle model goes through, not a
hand-rolled counter) turns "which plans look like what this agent usually does" into a real, scoreable
distribution: ``PlanModel.log_prob(plan)`` is exact, ``PlanModel.sample(rng)`` draws a plausible plan,
and ``PlanModel.is_typical(plan)`` flags a plan whose probability falls below the training traces' own
log-prob quantile -- an escalation signal, not a silent guess, the same discipline
:func:`~mixle.task.sft_plan.sample_plans` uses for its generative sibling.

    model = fit_plan_model(harvest_agent_traces())
    model.log_prob(["lookup_order", "notify"])
    model.is_typical(candidate_plan)   # False -> escalate; this plan does not look like the traces
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mixle.task.traces import AgentTrace, AgentTraces


def _tool_names(plan: Sequence[Any]) -> list[str]:
    """A plan is either already a list of tool-name strings, or the ``AgentTrace``/teacher shape
    ``[{"tool": name, "args": {...}}, ...]`` -- accept both so a harvested trace's ``plan`` and a
    freshly proposed candidate plan score the same way."""
    plan = list(plan)
    if all(isinstance(p, str) for p in plan):
        return plan
    return [str(step["tool"]) for step in plan]


@dataclass
class PlanModel:
    """A fitted Markov chain over tool-name sequences, plus the training traces' own log-prob spread."""

    dist: Any  # a fitted mixle.stats.MarkovChainDistribution over tool-name sequences
    training_log_probs: np.ndarray  # log_prob of every training trace, for is_typical's quantile

    def log_prob(self, plan: Sequence[Any]) -> float:
        """Exact log-probability of ``plan`` (a tool-name list, or the ``[{"tool":...}, ...]`` shape)."""
        return float(self.dist.log_density(_tool_names(plan)))

    def sample(self, rng: np.random.RandomState | None = None) -> list[str]:
        """Draw one plausible tool-name sequence from the fitted chain.

        The underlying sampler draws a length from ``len_dist`` first, then walks the chain; once the
        walk reaches an absorbing state (no fitted outgoing transition -- typically the tool that
        always ends a workflow), the remaining, unreachable slots are returned as ``None``. Truncate
        there rather than exposing that padding: only known, actually-reached tool names are emitted.
        """
        rng = rng if rng is not None else np.random.RandomState()
        seed = int(rng.randint(0, 2**31 - 1))
        raw = list(self.dist.sampler(seed).sample())
        out: list[str] = []
        for tool in raw:
            if tool is None:
                break
            out.append(str(tool))
        return out

    def is_typical(self, plan: Sequence[Any], *, quantile: float = 0.05) -> bool:
        """False when ``plan`` scores below the training traces' own ``quantile`` log-prob -- the
        escalation signal: a plan that does not look like what this agent usually does."""
        floor = float(np.quantile(self.training_log_probs, quantile))
        return self.log_prob(plan) >= floor


def fit_plan_model(
    traces: AgentTraces | Sequence[AgentTrace], *, smoothing: float = 0.5, init_p: float = 1.0
) -> PlanModel:
    """Fit a :class:`PlanModel` on harvested traces' tool-name sequences.

    ``smoothing`` is the Markov chain's Dirichlet pseudo-count (higher = smoother transition
    estimates, matters most with few traces). Fits via :func:`mixle.inference.optimize` on the
    existing :class:`~mixle.stats.sequences.markov_chain.MarkovChainEstimator` -- the same
    declare-an-estimator/call-optimize path every other mixle model uses, not hand-rolled counting.

    ``init_p`` defaults to ``1.0`` (use every trace for the init pass), not ``optimize``'s own
    ``init_p=0.1`` default: that Bernoulli-subsamples observations for a low-cost init estimate, sized
    for large corpora, but a trace corpus here is typically tens to a few hundred sequences -- with
    that few, a 10% subsample has a real chance of drawing ZERO sequences, which crashes
    ``MarkovChainEstimator.estimate1`` (``all_keys`` ends up empty, dividing by zero). Using the full
    corpus for this small an init pass is low-overhead and more reliable; override down only for corpora
    large enough that subsampling actually matters.
    """
    from mixle.inference import optimize
    from mixle.stats import IntegerCategoricalEstimator, MarkovChainEstimator

    trace_list = list(traces.traces) if isinstance(traces, AgentTraces) else list(traces)
    sequences = [_tool_names(t.plan) for t in trace_list]

    est = MarkovChainEstimator(pseudo_count=float(smoothing), len_estimator=IntegerCategoricalEstimator())
    dist = optimize(sequences, est, out=None, init_p=float(init_p))
    log_probs = np.asarray([dist.log_density(seq) for seq in sequences], dtype=float)
    return PlanModel(dist=dist, training_log_probs=log_probs)


__all__ = ["PlanModel", "fit_plan_model"]
