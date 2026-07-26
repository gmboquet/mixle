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
    """A fitted Markov chain plus an untouched calibration slice's log-probability spread."""

    dist: Any  # a fitted mixle.stats.MarkovChainDistribution over tool-name sequences
    calibration_log_probs: np.ndarray
    calibration_receipt: dict[str, Any]

    @property
    def training_log_probs(self) -> np.ndarray:
        """Compatibility alias; these scores now come from held-out calibration traces."""
        return self.calibration_log_probs

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
        if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be finite and in [0, 1]")
        floor = float(np.quantile(self.calibration_log_probs, quantile))
        return self.log_prob(plan) >= floor


def fit_plan_model(
    traces: AgentTraces | Sequence[AgentTrace],
    *,
    smoothing: float = 0.5,
    init_p: float = 1.0,
    calibration_frac: float = 0.2,
    seed: int = 0,
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

    if not np.isfinite(smoothing) or smoothing < 0.0:
        raise ValueError("smoothing must be finite and nonnegative")
    if not np.isfinite(init_p) or not 0.0 < init_p <= 1.0:
        raise ValueError("init_p must be finite and in (0, 1]")
    if not np.isfinite(calibration_frac) or not 0.0 < calibration_frac < 1.0:
        raise ValueError("calibration_frac must be finite and in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    trace_list = list(traces.traces) if isinstance(traces, AgentTraces) else list(traces)
    if len(trace_list) < 2 or any(not isinstance(trace, AgentTrace) for trace in trace_list):
        raise ValueError("at least two AgentTrace records are required")
    sequences = [_tool_names(t.plan) for t in trace_list]
    order = np.random.RandomState(seed).permutation(len(sequences))
    n_cal = min(len(sequences) - 1, max(1, int(round(len(sequences) * calibration_frac))))
    cal_idx = [int(i) for i in order[:n_cal]]
    fit_idx = [int(i) for i in order[n_cal:]]
    fit_sequences = [sequences[i] for i in fit_idx]
    calibration_sequences = [sequences[i] for i in cal_idx]

    est = MarkovChainEstimator(pseudo_count=float(smoothing), len_estimator=IntegerCategoricalEstimator())
    dist = optimize(fit_sequences, est, out=None, init_p=float(init_p))
    log_probs = np.asarray([dist.log_density(seq) for seq in calibration_sequences], dtype=float)
    if log_probs.shape != (len(calibration_sequences),) or not np.all(np.isfinite(log_probs)):
        raise ValueError("plan model returned invalid held-out calibration scores")
    return PlanModel(
        dist=dist,
        calibration_log_probs=log_probs,
        calibration_receipt={
            "kind": "seeded_holdout",
            "seed": seed,
            "fit_indices": fit_idx,
            "calibration_indices": cal_idx,
            "fit_count": len(fit_idx),
            "calibration_count": len(cal_idx),
        },
    )


__all__ = ["PlanModel", "fit_plan_model"]
