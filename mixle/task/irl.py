"""Maximum-entropy inverse reinforcement learning (Ziebart et al., 2008): recover a REWARD from
expert demonstrations, rather than assume one is given.

This is the complementary direction to :mod:`mixle.task.rl` (which learns a POLICY given a KNOWN
reward): given expert trajectories -- state sequences produced by an expert acting optimally under
some UNKNOWN reward -- :func:`max_ent_irl` recovers linear reward weights over a state feature map
such that the maximum-entropy (Boltzmann-rational) policy induced by those weights matches the
expert's empirical feature expectations. That match is the algorithm's own optimality certificate
(feature-expectation matching at convergence), not a proxy metric graded after the fact.

Differs from :mod:`mixle.task.plan_model`, which fits a Markov chain directly over observed action
sequences and models what the expert did. This module additionally explains why by recovering the
reward the expert's behavior is consistent with, over the same
:class:`~mixle.task.rl.GridWorld` environment shape.

    world = GridWorld(size=5, goal=(4, 4))
    demos = [rollout_states(world, expert_policy, start=(0, 0)) for _ in range(20)]
    result = max_ent_irl(world, demos)
    result.reward_weights.reshape(world.size, world.size)   # recovered per-cell reward
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from mixle.task.rl import ACTIONS, GridWorld, State


def state_features(env: GridWorld) -> np.ndarray:
    """Default feature map: a one-hot indicator per grid cell (``n_states x n_states``) -- a fully
    expressive tabular basis, so recovering per-feature weights is equivalent to recovering the
    per-state reward directly."""
    return np.eye(env.n_states)


def rollout_states(env: GridWorld, policy: dict[State, str], *, start: State = (0, 0)) -> list[State]:
    """The state-only trace of a deterministic policy from ``start`` (the demonstration format
    :func:`max_ent_irl` expects: what the expert visited, not what it was thinking)."""
    state = env.reset(start)
    trace = [state]
    for _ in range(env.max_steps):
        if state == env.goal:
            break
        state, _, done = env.step(policy[state])
        trace.append(state)
        if done:
            break
    return trace


def _transition_table(env: GridWorld) -> np.ndarray:
    """Precompute the deterministic ``next_state[s, a]`` index table for every state/action."""
    next_state = np.zeros((env.n_states, len(ACTIONS)), dtype=int)
    for s_idx in range(env.n_states):
        state = env.index_state(s_idx)
        for a_idx, action in enumerate(ACTIONS):
            next_state[s_idx, a_idx] = env.state_index(env.transition(state, action))
    return next_state


def _expert_feature_expectation(env: GridWorld, features: np.ndarray, trajectories: list[list[State]]) -> np.ndarray:
    """Mean feature-count vector per demonstration, retaining variable trajectory lengths."""
    total = np.zeros(features.shape[1])
    for traj in trajectories:
        for state in traj:
            total += features[env.state_index(state)]
    return total / len(trajectories)


def _soft_value_iteration(
    reward: np.ndarray,
    next_state: np.ndarray,
    *,
    gamma: float,
    terminal_index: int,
    iterations: int = 200,
) -> np.ndarray:
    """Soft (log-sum-exp) Bellman backup -> the Boltzmann-rational policy ``pi(a|s) ~ exp(Q(s,a))``,
    the maximum-entropy-optimal policy for ``reward``."""
    n_states = next_state.shape[0]
    v = np.zeros(n_states)
    for _ in range(iterations):
        q = reward[:, None] + gamma * v[next_state]
        q_max = q.max(axis=1, keepdims=True)
        v_new = np.log(np.sum(np.exp(q - q_max), axis=1)) + q_max.squeeze(-1)
        # Termination yields its reward once. It does not collect reward or entropy forever merely
        # because the transition representation uses an absorbing self-loop.
        v_new[terminal_index] = reward[terminal_index]
        if np.max(np.abs(v_new - v)) < 1e-8:
            v = v_new
            break
        v = v_new
    q = reward[:, None] + gamma * v[next_state]
    q[terminal_index] = reward[terminal_index]
    q = q - q.max(axis=1, keepdims=True)
    policy = np.exp(q)
    policy /= policy.sum(axis=1, keepdims=True)
    return policy


def _expected_state_visitation(
    policy: np.ndarray,
    next_state: np.ndarray,
    *,
    start_indices: list[int],
    horizons: list[int],
    terminal_index: int,
) -> np.ndarray:
    """Mean state-visitation counts over each demonstration's own start and horizon.

    Terminal mass is counted once and then removed instead of being propagated through the
    transition table's absorbing goal self-loop.
    """
    n_states, n_actions = policy.shape
    visitation = np.zeros(n_states)
    for start_index, horizon in zip(start_indices, horizons, strict=True):
        distribution = np.zeros(n_states)
        distribution[start_index] = 1.0
        for step in range(horizon):
            visitation += distribution
            if step + 1 >= horizon:
                break
            distribution = distribution.copy()
            distribution[terminal_index] = 0.0
            next_distribution = np.zeros(n_states)
            for state in np.nonzero(distribution)[0]:
                for action in range(n_actions):
                    next_distribution[next_state[state, action]] += distribution[state] * policy[state, action]
            distribution = next_distribution
    return visitation / len(start_indices)


@dataclass
class MaxEntIRLResult:
    """The recovered reward, its induced Boltzmann-rational policy, and the convergence trace
    (``||expert_feature_expectation - policy_feature_expectation||`` per iteration -- should
    decrease toward zero as the algorithm's own certificate of fit)."""

    reward_weights: np.ndarray
    policy: np.ndarray
    history: list[float]

    def reward(self, features: np.ndarray) -> np.ndarray:
        """Evaluate the learned linear reward on feature rows."""
        return features @ self.reward_weights


def max_ent_irl(
    env: GridWorld,
    expert_trajectories: list[list[State]],
    *,
    start: State | None = None,
    gamma: float = 0.9,
    iterations: int = 150,
    lr: float = 0.5,
    features: np.ndarray | None = None,
) -> MaxEntIRLResult:
    """Recover linear reward weights whose maximum-entropy-optimal policy matches the expert's
    empirical feature expectations, via gradient ascent on trajectory likelihood:
    ``weights += lr * (expert_feature_expectation - policy_feature_expectation)``. Requires only
    ``expert_trajectories`` (state sequences); never sees the expert's true reward or the actions
    that produced them."""
    if not isinstance(env, GridWorld):
        raise ValueError("env must be a GridWorld")
    if expert_trajectories is None:
        raise ValueError("max_ent_irl requires at least one expert trajectory.")
    try:
        trajectories = [list(trajectory) for trajectory in expert_trajectories]
    except TypeError as exc:
        raise ValueError("expert_trajectories must be an iterable of state trajectories") from exc
    if not trajectories:
        raise ValueError("max_ent_irl requires at least one expert trajectory.")
    if any(not trajectory for trajectory in trajectories):
        raise ValueError("expert trajectories must be nonempty")
    for trajectory in trajectories:
        normalized = [env._validate_occupiable_state(state, name="demonstration state") for state in trajectory]
        for index, state in enumerate(normalized):
            if state == env.goal and index != len(normalized) - 1:
                raise ValueError("a demonstration cannot continue after reaching the goal")
        for current, following in zip(normalized, normalized[1:]):
            if following not in {env.transition(current, action) for action in ACTIONS}:
                raise ValueError("consecutive demonstration states must be connected by one action")
        trajectory[:] = normalized
    if start is not None:
        asserted_start = env._validate_occupiable_state(start, name="start")
        if any(trajectory[0] != asserted_start for trajectory in trajectories):
            raise ValueError("start does not match every demonstration's initial state")
    if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    try:
        gamma = float(gamma)
    except (TypeError, ValueError) as exc:
        raise ValueError("gamma must be finite and in [0, 1)") from exc
    if not np.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    try:
        lr = float(lr)
    except (TypeError, ValueError) as exc:
        raise ValueError("lr must be finite and positive") from exc
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")

    features = np.asarray(state_features(env) if features is None else features, dtype=float)
    if features.ndim != 2 or features.shape[0] != env.n_states or features.shape[1] == 0:
        raise ValueError("features must have shape (env.n_states, n_features) with at least one feature")
    if not np.all(np.isfinite(features)):
        raise ValueError("features must contain only finite values")
    next_state = _transition_table(env)
    expert_fe = _expert_feature_expectation(env, features, trajectories)
    horizons = [len(trajectory) for trajectory in trajectories]
    start_indices = [env.state_index(trajectory[0]) for trajectory in trajectories]
    terminal_index = env.state_index(env.goal)

    weights = np.zeros(features.shape[1])
    history = []
    for _ in range(iterations):
        reward = features @ weights
        policy = _soft_value_iteration(reward, next_state, gamma=gamma, terminal_index=terminal_index)
        visitation = _expected_state_visitation(
            policy,
            next_state,
            start_indices=start_indices,
            horizons=horizons,
            terminal_index=terminal_index,
        )
        expected_fe = (visitation[:, None] * features).sum(axis=0)
        grad = expert_fe - expected_fe
        weights = weights + lr * grad
        history.append(float(np.linalg.norm(grad)))
    # The loop's policy preceded its final weight update. Recompute so both returned artifacts
    # describe the same committed reward.
    policy = _soft_value_iteration(
        features @ weights,
        next_state,
        gamma=gamma,
        terminal_index=terminal_index,
    )
    return MaxEntIRLResult(reward_weights=weights, policy=policy, history=history)
