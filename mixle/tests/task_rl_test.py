"""Tabular Q-learning over GridWorld: known-optimum recovery, beats-random, determinism."""

import numpy as np

from mixle.task.rl import GridWorld, random_policy, rollout, tabular_q_learning


def test_optimal_path_length_matches_manual_bfs_with_obstacles():
    world = GridWorld(size=4, goal=(3, 3), obstacles=frozenset({(1, 1), (2, 1), (1, 2)}))
    assert world.optimal_path_length() == 6


def test_optimal_path_length_no_obstacles_is_manhattan_distance():
    world = GridWorld(size=5, goal=(4, 4))
    assert world.optimal_path_length() == 8


def test_q_learning_recovers_the_optimal_policy():
    world = GridWorld(size=5, goal=(4, 4))
    result = tabular_q_learning(world, episodes=800, seed=0)
    policy = result.greedy_policy(world)
    trace = rollout(world, policy)
    assert len(trace) == world.optimal_path_length()


def test_q_learning_recovers_the_optimal_policy_with_obstacles():
    world = GridWorld(size=5, goal=(4, 4), obstacles=frozenset({(1, 1), (2, 1), (3, 1), (1, 2)}))
    result = tabular_q_learning(world, episodes=1500, seed=1)
    policy = result.greedy_policy(world)
    trace = rollout(world, policy)
    assert len(trace) == world.optimal_path_length()


def _episode_return(world: GridWorld, policy: dict) -> float:
    state = world.reset()
    total = 0.0
    for _ in range(world.max_steps):
        state, reward, done = world.step(policy[state])
        total += reward
        if done:
            break
    return total


def test_q_learning_beats_random_on_average_return():
    world = GridWorld(size=5, goal=(4, 4))
    result = tabular_q_learning(world, episodes=500, seed=2)
    learned_policy = result.greedy_policy(world)
    learned_return = _episode_return(world, learned_policy)

    random_returns = [_episode_return(world, random_policy(world, np.random.RandomState(seed))) for seed in range(20)]
    assert learned_return > float(np.mean(random_returns))


def test_q_learning_is_deterministic_given_seed():
    world_a = GridWorld(size=5, goal=(4, 4))
    world_b = GridWorld(size=5, goal=(4, 4))
    result_a = tabular_q_learning(world_a, episodes=200, seed=7)
    result_b = tabular_q_learning(world_b, episodes=200, seed=7)
    np.testing.assert_array_equal(result_a.q_table, result_b.q_table)
    assert result_a.rewards_per_episode == result_b.rewards_per_episode


def test_greedy_policy_covers_every_non_goal_state():
    world = GridWorld(size=3, goal=(2, 2))
    result = tabular_q_learning(world, episodes=200, seed=0)
    policy = result.greedy_policy(world)
    assert set(policy.keys()) == set(world.states()) - {world.goal}
