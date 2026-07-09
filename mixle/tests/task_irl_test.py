"""Maximum-entropy IRL: recover a reward from expert demonstrations, verify it reproduces the
expert's optimal behavior -- never checked against the true reward the expert used to generate them
(IRL never sees it; the recovered reward only ever sees expert STATE trajectories)."""

import numpy as np

from mixle.task.irl import max_ent_irl, rollout_states, state_features
from mixle.task.rl import ACTIONS, GridWorld, tabular_q_learning


def _expert_policy(world: GridWorld, seed: int = 0):
    result = tabular_q_learning(world, episodes=800, seed=seed)
    return result.greedy_policy(world)


def test_convergence_history_decreases():
    world = GridWorld(size=4, goal=(3, 3))
    expert_policy = _expert_policy(world)
    demos = [rollout_states(world, expert_policy)]

    result = max_ent_irl(world, demos, iterations=150, lr=0.5)
    early = np.mean(result.history[:10])
    late = np.mean(result.history[-10:])
    assert late < early


def test_recovered_reward_peaks_at_the_goal_among_visited_states():
    world = GridWorld(size=4, goal=(3, 3))
    expert_policy = _expert_policy(world)
    demos = [rollout_states(world, expert_policy)]

    result = max_ent_irl(world, demos, iterations=150, lr=0.5)
    reward_map = result.reward(state_features(world))

    visited = {s for traj in demos for s in traj}
    visited_indices = [world.state_index(s) for s in visited]
    goal_index = world.state_index(world.goal)
    assert reward_map[goal_index] == max(reward_map[i] for i in visited_indices)


def test_recovered_policy_matches_expert_actions_at_every_visited_state():
    world = GridWorld(size=5, goal=(4, 4), obstacles=frozenset({(1, 1), (2, 1), (1, 2)}))
    expert_policy = _expert_policy(world, seed=1)
    trajectory = rollout_states(world, expert_policy, start=(0, 0))
    demos = [trajectory]

    result = max_ent_irl(world, demos, iterations=150, lr=0.5, gamma=0.9)
    recovered_policy = {
        s: ACTIONS[int(np.argmax(result.policy[world.state_index(s)]))] for s in world.states() if s != world.goal
    }
    # the recovered Boltzmann policy's argmax action at every EXPERT-VISITED state matches the
    # expert's own action there -- the algorithm's behavioral fidelity check.
    for state in trajectory[:-1]:
        assert recovered_policy[state] == expert_policy[state]


def test_max_ent_irl_requires_at_least_one_trajectory():
    world = GridWorld(size=3, goal=(2, 2))
    try:
        max_ent_irl(world, [])
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_determinism_given_same_demonstrations():
    world = GridWorld(size=4, goal=(3, 3))
    expert_policy = _expert_policy(world)
    demos = [rollout_states(world, expert_policy)]

    result_a = max_ent_irl(world, demos, iterations=100, lr=0.5)
    result_b = max_ent_irl(world, demos, iterations=100, lr=0.5)
    np.testing.assert_array_equal(result_a.reward_weights, result_b.reward_weights)
