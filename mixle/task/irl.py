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
    total = np.zeros(features.shape[1])
    count = 0
    for traj in trajectories:
        for state in traj:
            total += features[env.state_index(state)]
            count += 1
    return total / count


def _soft_value_iteration(
    reward: np.ndarray, next_state: np.ndarray, *, gamma: float, iterations: int = 200
) -> np.ndarray:
    """Soft (log-sum-exp) Bellman backup -> the Boltzmann-rational policy ``pi(a|s) ~ exp(Q(s,a))``,
    the maximum-entropy-optimal policy for ``reward``."""
    n_states = next_state.shape[0]
    v = np.zeros(n_states)
    for _ in range(iterations):
        q = reward[:, None] + gamma * v[next_state]
        q_max = q.max(axis=1, keepdims=True)
        v_new = np.log(np.sum(np.exp(q - q_max), axis=1)) + q_max.squeeze(-1)
        if np.max(np.abs(v_new - v)) < 1e-8:
            v = v_new
            break
        v = v_new
    q = reward[:, None] + gamma * v[next_state]
    q = q - q.max(axis=1, keepdims=True)
    policy = np.exp(q)
    policy /= policy.sum(axis=1, keepdims=True)
    return policy


def _expected_state_visitation(
    policy: np.ndarray, next_state: np.ndarray, *, start_index: int, horizon: int
) -> np.ndarray:
    """Forward pass: expected visitation count per state over ``horizon`` steps from ``start_index``,
    under ``policy`` and the deterministic ``next_state`` transition (the MaxEnt IRL algorithm's own
    state-visitation-frequency computation)."""
    n_states, n_actions = policy.shape
    d = np.zeros((horizon, n_states))
    d[0, start_index] = 1.0
    for t in range(1, horizon):
        prev = d[t - 1]
        active = np.nonzero(prev)[0]
        for s in active:
            for a in range(n_actions):
                d[t, next_state[s, a]] += prev[s] * policy[s, a]
    return d.sum(axis=0)


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
    start: State = (0, 0),
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
    if not expert_trajectories:
        raise ValueError("max_ent_irl requires at least one expert trajectory.")
    features = state_features(env) if features is None else features
    next_state = _transition_table(env)
    expert_fe = _expert_feature_expectation(env, features, expert_trajectories)
    horizon = max(len(t) for t in expert_trajectories)
    start_index = env.state_index(start)

    weights = np.zeros(features.shape[1])
    history = []
    policy = np.ones((env.n_states, len(ACTIONS))) / len(ACTIONS)
    for _ in range(iterations):
        reward = features @ weights
        policy = _soft_value_iteration(reward, next_state, gamma=gamma)
        visitation = _expected_state_visitation(policy, next_state, start_index=start_index, horizon=horizon)
        expected_fe = (visitation[:, None] * features).sum(axis=0) / visitation.sum()
        grad = expert_fe - expected_fe
        weights = weights + lr * grad
        history.append(float(np.linalg.norm(grad)))
    return MaxEntIRLResult(reward_weights=weights, policy=policy, history=history)
