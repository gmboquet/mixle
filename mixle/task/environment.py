"""Environment protocol + belief-driven interaction loop (roadmap M1) -- the generic
act-observe-update spine that on-the-fly simulators (M2), inversion (M3), the language<->belief
bridge (M5), and environments-as-selection-pressure (L1) build on.

:func:`interact` drives ANY object satisfying the :class:`Environment` protocol against a
streaming BELIEF over its latents (:mod:`mixle.inference.streaming`), picking actions by EIG
(reusing :func:`mixle.task.probe_policy.myopic_eig_policy` unchanged), a belief-driven greedy
heuristic, or a caller-supplied callable -- then hands back a replayable
:class:`InteractionLog` (:mod:`mixle.task.replay`).

:class:`~mixle.task.explore_world.ExplorationWorld` becomes the first environment:
:class:`ExplorationEnvironment` below is a THIN wrapper -- it holds episode config and adapts
``reset``/``step``/``action_space`` onto the world; ``ExplorationWorld`` itself is untouched, so
existing callers (``run_episode``, ``probe_policy``) keep working exactly as before.

Belief math (see ``notes/designs/M1.md`` for the full writeup): a cell's underlying latent
("geology") is fixed for the episode; each accepted ``survey`` observation is one more noisy
read of it. :class:`GaussianStreamingBelief` folds those reads one at a time through
:class:`mixle.inference.streaming.StreamingEstimator` with a ``harmonic(1.0)`` schedule -- which
is exactly the textbook incremental-mean/-variance recursion (``rho_t = 1/t``), so the running
``GaussianDistribution`` is the exact batch MLE over all reads so far, not merely an
approximation. Streaming's own ``nobs`` bookkeeping is a *decayed effective count* (it does not
grow with ``t`` under a stationary rho schedule), so credible intervals track their own read
count separately for the standard-error term.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mixle.inference.estimation import harmonic
from mixle.inference.streaming import StreamingEstimator
from mixle.stats import GaussianDistribution, GaussianEstimator
from mixle.task.explore_world import ExplorationWorld
from mixle.task.probe_policy import myopic_eig_policy
from mixle.task.replay import ExecutionTrace, is_bit_identical_replay, record_step

__all__ = [
    "Environment",
    "ExplorationEnvironment",
    "GaussianStreamingBelief",
    "InteractionLog",
    "interact",
]


@runtime_checkable
class Environment(Protocol):
    """Generic act-observe world.

    ``reset`` starts (or restarts) an episode from a seed and returns an initial observation;
    ``step`` applies one action and returns ``(observation, cost)``; ``action_space`` lists the
    actions currently legal to take. Costs are returned per step (not tracked internally) so
    :func:`interact` can enforce ONE budget semantics uniformly across arbitrary environments.
    """

    def reset(self, seed: int | None = None) -> Any: ...

    def step(self, action: Any) -> tuple[Any, float]: ...

    def action_space(self) -> list[Any]: ...


@dataclass
class ExplorationEnvironment:
    """Thin :class:`Environment` wrapper over :class:`~mixle.task.explore_world.ExplorationWorld`.

    Holds the episode config (cell/target/budget counts); ``reset(seed)`` builds a fresh
    ``ExplorationWorld`` and keeps it as ``self.world`` (so a caller -- or the ``"eig"`` policy
    below, which reads ``ExplorationWorld`` internals exactly the way
    :func:`~mixle.task.probe_policy.myopic_eig_policy` already does -- can still get at the raw
    world). ``ExplorationWorld``'s own public API is unmodified; this class only adapts it.
    """

    n_cells: int
    n_targets: int
    budget: int
    world: ExplorationWorld | None = field(default=None, init=False, repr=False)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.world = ExplorationWorld(
            n_cells=self.n_cells, n_targets=self.n_targets, budget=self.budget, seed=0 if seed is None else seed
        )
        # a plain, JSON-serializable observation (not the ExplorationWorld object itself) --
        # replay's diff() compares recorded results via json.dumps, so the observation surface
        # must stay JSON-safe; the raw world is still reachable via self.world for the "eig"
        # policy and other privileged callers.
        return {"n_cells": self.n_cells, "n_targets": self.n_targets, "remaining_budget": self.world.remaining_budget}

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float]:
        if self.world is None:
            raise RuntimeError("ExplorationEnvironment.step called before reset().")
        before = self.world.remaining_budget
        obs = self.world.step(action)
        cost = float(before - self.world.remaining_budget) if obs.get("accepted") else 0.0
        return obs, cost

    def action_space(self) -> list[dict[str, Any]]:
        if self.world is None:
            raise RuntimeError("ExplorationEnvironment.action_space called before reset().")
        return self.world.action_menu()


_Z_BY_LEVEL = {0.80: 1.2815515655446008, 0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


def _z_for(level: float) -> float:
    z = _Z_BY_LEVEL.get(level)
    if z is not None:
        return z
    from scipy.stats import norm  # local import: keep the hot path free of the scipy dependency

    return float(norm.ppf(0.5 + level / 2.0))


@dataclass
class GaussianStreamingBelief:
    """Per-cell streaming posterior over a scalar continuous latent (``ExplorationWorld``'s
    per-cell "geology" value), folded in one accepted ``survey`` observation at a time via
    :class:`mixle.inference.streaming.StreamingEstimator` -- the generic online sufficient-
    statistic machinery M0's ``condition()`` is built to consume once a fitted model exists.
    One independent :class:`~mixle.stats.GaussianDistribution` per cell; an unsurveyed cell
    reports the shared prior.
    """

    prior_mu: float = 0.0
    prior_sigma2: float = 4.0
    min_covar: float = 0.05
    # Bayesian pseudo-count blending the prior variance into the (otherwise degenerate-at-n=1)
    # sample variance for credible_interval's standard error: eff_var = (k0*prior_sigma2 +
    # n*sample_var) / (k0+n). Picked by calibration sweep (notes/designs/M1.md): k0=0.05 keeps
    # 90%-nominal coverage close to 90% (measured ~93% at n=200 draws) without either the
    # near-total undercoverage of raw sample variance at low n (k0=0, ~80%) or gross
    # overcoverage from over-trusting the prior (k0>=0.5, ~100%).
    belief_pseudo_count: float = 0.05
    _streams: dict[int, StreamingEstimator] = field(default_factory=dict, init=False, repr=False)
    _n: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def _stream_for(self, cell: int) -> StreamingEstimator:
        stream = self._streams.get(cell)
        if stream is None:
            # harmonic(1.0) -> rho_t = 1/t, the exact incremental-mean/-variance recursion: the
            # running GaussianDistribution equals the batch MLE over every read folded in so far.
            stream = StreamingEstimator(
                GaussianEstimator(min_covar=self.min_covar),
                schedule=harmonic(1.0),
                model=GaussianDistribution(self.prior_mu, self.prior_sigma2),
            )
            self._streams[cell] = stream
        return stream

    def update(self, obs: dict[str, Any]) -> None:
        """Fold one accepted ``survey`` observation's prospectivity read into that cell's belief.
        Drill/rejected/other observations carry no continuous read and are not folded in here --
        a drill resolves ground truth directly, it needs no posterior (v1 scope)."""
        if obs.get("type") != "survey" or not obs.get("accepted", False):
            return
        cell = int(obs["cell"])
        self._stream_for(cell).update(np.asarray([float(obs["prospectivity"])]))
        self._n[cell] = self._n.get(cell, 0) + 1

    def n(self, cell: int) -> int:
        return self._n.get(cell, 0)

    def mean(self, cell: int) -> float:
        stream = self._streams.get(cell)
        return self.prior_mu if stream is None else float(stream.model.mu)

    def credible_interval(self, cell: int, level: float = 0.9) -> tuple[float, float]:
        """A ``level``-credible interval for the cell's latent: the running Gaussian's own mean,
        and a standard error of the mean built from a prior/sample-variance blend (see
        ``belief_pseudo_count``) over this belief's own read count -- not the raw per-cell
        sample variance alone, which is degenerate (zero, before ``min_covar`` clamps it) at a
        single read and undercovers badly until several reads accumulate."""
        n = self.n(cell)
        if n == 0:
            mu, var = self.prior_mu, self.prior_sigma2
        else:
            stream = self._streams[cell]
            mu = float(stream.model.mu)
            k0 = self.belief_pseudo_count
            var = (k0 * self.prior_sigma2 + n * stream.model.sigma2) / (k0 + n)
        se = math.sqrt(var / max(n, 1))
        half = _z_for(level) * se
        return (mu - half, mu + half)


@dataclass
class InteractionLog:
    """One episode's action/observation/cost trace, replayable via :mod:`mixle.task.replay`.

    Each recorded ``"act"`` step bundles POLICY DECISION + ``env.step`` + belief update as one
    unit (rather than recording the chosen action alone and replaying it against a bare
    ``env.step``) because a world-peeking policy like ``"eig"`` (:func:`myopic_eig_policy` reads
    ``ExplorationWorld``'s own RNG-backed ``prospectivity()`` while DECIDING) consumes the same
    environment randomness the eventual observation depends on -- replaying only the action list
    would silently desync that RNG stream and stop reproducing bit-for-bit. Bundling the policy
    call into the replayed unit keeps the two draws in the same relative order both times.
    """

    seed: int | None
    budget: float
    policy: str
    trace: ExecutionTrace
    total_cost: float
    n_actions: int

    def is_deterministic(self, env: Environment, belief_model: Any) -> bool:
        """Replay this log against a fresh ``env``/``belief_model`` pair (same policy name, same
        seed) and confirm every recorded step reproduces exactly -- the M1 replay receipt.
        Only named policies (``"eig"``, ``"greedy"``) can be reconstructed for replay; a log
        built from an arbitrary callable policy cannot (the callable itself is not serialized).
        """
        policy_fn, policy_name = _resolve_policy(self.policy)
        if policy_name != self.policy:
            raise ValueError(f"cannot reconstruct a callable policy for replay (recorded name {self.policy!r}).")
        tools, _state = _build_tools(env, belief_model, policy_fn, self.budget)
        return is_bit_identical_replay(self.trace, tools)


def _build_tools(
    env: Environment, belief_model: Any, policy_fn: Callable[[Environment, Any, list[Any]], Any], budget: float
) -> tuple[dict[str, Callable[..., Any]], dict[str, float]]:
    state = {"remaining": float(budget), "n_actions": 0}

    def _reset(seed: int | None = None) -> Any:
        return env.reset(seed=seed)

    def _act() -> dict[str, Any] | None:
        if state["remaining"] <= 0:
            return None
        menu = env.action_space()
        if not menu:
            return None
        action = policy_fn(env, belief_model, menu)
        if action is None:
            return None
        obs, cost = env.step(action)
        cost = float(cost)
        accepted = bool(obs.get("accepted", True)) and cost <= state["remaining"]
        if accepted:
            state["remaining"] -= cost
            belief_model.update(obs)
            state["n_actions"] += 1
        return {"action": action, "obs": obs, "cost": cost, "accepted": accepted}

    return {"reset": _reset, "act": _act}, state


def _eig_policy_fn(env: Environment, belief: Any, menu: list[Any]) -> Any:
    world = getattr(env, "world", env)
    return myopic_eig_policy(world)


def _greedy_belief_policy_fn(env: Environment, belief: GaussianStreamingBelief, menu: list[Any]) -> Any:
    """Belief-driven (not world-peeking) baseline: survey every cell that has not been read yet,
    then drill the undrilled cell with the highest current belief mean."""
    survey_actions = [a for a in menu if a.get("type") == "survey"]
    drill_actions = [a for a in menu if a.get("type") == "drill"]
    unsurveyed = [a for a in survey_actions if belief.n(a["cell"]) < 1]
    if unsurveyed:
        return unsurveyed[0]
    if not drill_actions:
        return None
    return max(drill_actions, key=lambda a: belief.mean(a["cell"]))


def _resolve_policy(
    policy: str | Callable[[Environment, Any, list[Any]], Any],
) -> tuple[Callable[[Environment, Any, list[Any]], Any], str]:
    if callable(policy) and not isinstance(policy, str):
        return policy, getattr(policy, "__name__", "callable")
    if policy == "eig":
        return _eig_policy_fn, "eig"
    if policy == "greedy":
        return _greedy_belief_policy_fn, "greedy"
    raise ValueError(f"unknown interact() policy {policy!r}; expected 'eig', 'greedy', or a callable")


def interact(
    env: Environment,
    belief_model: Any,
    *,
    policy: str | Callable[[Environment, Any, list[Any]], Any] = "eig",
    budget: float,
    seed: int | None = None,
) -> InteractionLog:
    """Drive the act-observe-update loop.

    Resets ``env``, then repeatedly: pick an action over ``env.action_space()`` (EIG / belief-
    greedy / a caller callable), execute it via ``env.step``, fold the observation into
    ``belief_model.update(obs)``, until the summed action cost would exceed ``budget`` or the
    policy/environment stops (``action_space()`` empty, policy returns ``None``, or the
    environment refuses the action). Every reset/act is recorded as a :class:`TraceStep` (see
    :class:`InteractionLog` for why policy decision + step are bundled into one ``"act"`` unit)
    so the returned :class:`InteractionLog` replays deterministically via :mod:`mixle.task.replay`.
    """
    policy_fn, policy_name = _resolve_policy(policy)
    tools, state = _build_tools(env, belief_model, policy_fn, budget)
    trace = ExecutionTrace(request="interact")
    trace.steps.append(record_step(tools, "reset", {}, seed=seed))

    while state["remaining"] > 0:
        step_result = record_step(tools, "act", {})
        trace.steps.append(step_result)
        if step_result.result is None or not step_result.result.get("accepted", True):
            break

    return InteractionLog(
        seed=seed,
        budget=float(budget),
        policy=policy_name,
        trace=trace,
        total_cost=float(budget) - state["remaining"],
        n_actions=state["n_actions"],
    )
