"""Reinforcement learning: tabular Q-learning over a discrete MDP with a KNOWN reward.

This is the classical value-based counterpart to sequence-level optimization routines such as
``outcome_decomposer`` and ``probe_policy``: those optimize a whole action sequence's terminal,
oracle-verified score by propose/filter/refit, with no per-step value estimate. This module is the
other end of the design space -- a per-step Bellman backup,
``Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]`` -- for problems that ARE
naturally modeled as a finite MDP with a known step reward. It composes with :mod:`mixle.task.irl`
(the complementary direction: reward FROM demonstrations, not policy from a known reward) via the
shared :class:`GridWorld` environment.

    world = GridWorld(size=5, goal=(4, 4))
    result = tabular_q_learning(world, episodes=500, seed=0)
    policy = result.greedy_policy(world)     # {state: best action}, the recovered optimal policy
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

Action = str
State = tuple[int, int]

_DELTA: dict[Action, tuple[int, int]] = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ACTIONS: tuple[Action, ...] = ("up", "down", "left", "right")


@dataclass
class GridWorld:
    """A deterministic ``size`` x ``size`` grid MDP: a goal cell worth ``goal_reward``, a per-step
    cost of ``step_cost``, and optional impassable ``obstacles`` (moving into a wall or obstacle
    leaves the agent in place, still paying the step cost). The optimal policy is the shortest
    obstacle-free path to the goal -- computable independently via :meth:`optimal_path_length`
    (BFS), which is what makes this a closed-form-known-optimum test environment."""

    size: int
    goal: State
    obstacles: frozenset[State] = field(default_factory=frozenset)
    step_cost: float = -1.0
    goal_reward: float = 10.0
    max_steps: int = 100

    def __post_init__(self) -> None:
        self._state: State = (0, 0)
        self._steps = 0

    @property
    def n_states(self) -> int:
        """Return the number of states in the square grid."""
        return self.size * self.size

    def state_index(self, state: State) -> int:
        """Map a ``(row, column)`` state to its row-major integer index."""
        return state[0] * self.size + state[1]

    def index_state(self, index: int) -> State:
        """Map a row-major integer state index back to ``(row, column)``."""
        r, c = divmod(index, self.size)
        return (r, c)

    def states(self) -> list[State]:
        """Return every grid state in row-major order."""
        return [(r, c) for r in range(self.size) for c in range(self.size)]

    def transition(self, state: State, action: Action) -> State:
        """The deterministic next state for ``action`` at ``state`` (walls/obstacles are a no-op)."""
        dr, dc = _DELTA[action]
        r, c = state
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.size and 0 <= nc < self.size) or (nr, nc) in self.obstacles:
            return state
        return (nr, nc)

    def reset(self, start: State = (0, 0)) -> State:
        """Reset the environment to ``start`` and return the initial state."""
        self._state = start
        self._steps = 0
        return self._state

    def step(self, action: Action) -> tuple[State, float, bool]:
        """Apply one action and return ``(next_state, reward, done)``."""
        self._state = self.transition(self._state, action)
        self._steps += 1
        at_goal = self._state == self.goal
        reward = self.goal_reward if at_goal else self.step_cost
        done = at_goal or self._steps >= self.max_steps
        return self._state, reward, done

    def optimal_path_length(self, start: State = (0, 0)) -> int:
        """BFS shortest obstacle-free path length from ``start`` to ``goal`` -- ground truth for
        tests, computed independently of any learning algorithm in this module."""
        if start == self.goal:
            return 0
        visited = {start}
        queue: deque[tuple[State, int]] = deque([(start, 0)])
        while queue:
            state, dist = queue.popleft()
            for action in ACTIONS:
                nxt = self.transition(state, action)
                if nxt == state or nxt in visited:
                    continue
                if nxt == self.goal:
                    return dist + 1
                visited.add(nxt)
                queue.append((nxt, dist + 1))
        raise ValueError("goal is unreachable from start given the obstacles.")


@dataclass
class QLearningResult:
    """The fitted Q-table plus the per-episode return trace (the learning curve)."""

    q_table: np.ndarray
    rewards_per_episode: list[float]

    def greedy_action_index(self, state_index: int) -> int:
        """Return the index of the highest-valued action for ``state_index``."""
        return int(np.argmax(self.q_table[state_index]))

    def greedy_policy(self, env: GridWorld) -> dict[State, Action]:
        """The recovered deterministic policy: the argmax action at every non-goal state."""
        return {s: ACTIONS[self.greedy_action_index(env.state_index(s))] for s in env.states() if s != env.goal}


def tabular_q_learning(
    env: GridWorld,
    *,
    episodes: int = 500,
    alpha: float = 0.3,
    gamma: float = 0.95,
    epsilon: float = 0.2,
    seed: int | None = None,
) -> QLearningResult:
    """Epsilon-greedy tabular Q-learning: ``episodes`` full rollouts from ``env.reset()``, each step
    updating ``Q(s, a)`` toward the observed one-step Bellman target."""
    rng = np.random.RandomState(seed)
    q = np.zeros((env.n_states, len(ACTIONS)))
    rewards_per_episode = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            s_idx = env.state_index(state)
            if rng.random_sample() < epsilon:
                a_idx = int(rng.randint(len(ACTIONS)))
            else:
                a_idx = int(np.argmax(q[s_idx]))
            next_state, reward, done = env.step(ACTIONS[a_idx])
            ns_idx = env.state_index(next_state)
            target = reward + (0.0 if done else gamma * np.max(q[ns_idx]))
            q[s_idx, a_idx] += alpha * (target - q[s_idx, a_idx])
            total_reward += reward
            state = next_state
        rewards_per_episode.append(total_reward)
    return QLearningResult(q_table=q, rewards_per_episode=rewards_per_episode)


def rollout(env: GridWorld, policy: dict[State, Action], *, start: State = (0, 0)) -> list[tuple[State, Action]]:
    """Roll out a deterministic state -> action ``policy`` from ``start``; the ``(state, action)``
    trace (stops at the goal or ``env.max_steps``, whichever first)."""
    state = env.reset(start)
    trace: list[tuple[State, Action]] = []
    for _ in range(env.max_steps):
        if state == env.goal:
            break
        action = policy[state]
        trace.append((state, action))
        state, _, done = env.step(action)
        if done:
            break
    return trace


def random_policy(env: GridWorld, rng: np.random.RandomState) -> dict[State, Action]:
    """Baseline: an independent uniform-random action at every non-goal state."""
    return {s: ACTIONS[int(rng.randint(len(ACTIONS)))] for s in env.states() if s != env.goal}
