"""Tabular Q-learning over GridWorld: known-optimum recovery, beats-random, determinism."""

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": 0, "goal": (0, 0)},
        {"size": 3, "goal": (3, 0)},
        {"size": 3, "goal": (2, 2), "obstacles": {(2, 2)}},
        {"size": 3, "goal": (2, 2), "obstacles": {(0, 0)}},
        {"size": 3, "goal": (2, 2), "obstacles": {(-1, 0)}},
        {"size": 3, "goal": (2, 2), "max_steps": 0},
        {"size": 3, "goal": (2, 2), "step_cost": float("nan")},
        {"size": 3, "goal": (2, 2), "goal_reward": float("inf")},
    ],
)
def test_invalid_worlds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        GridWorld(**kwargs)


def test_reset_and_coordinate_conversions_validate_states():
    world = GridWorld(size=3, goal=(2, 2), obstacles={(1, 1)})
    for start in ((3, 0), (-1, 0), (1, 1), (True, 0)):
        with pytest.raises(ValueError):
            world.reset(start)
    for index in (-1, world.n_states, 1.5, True):
        with pytest.raises(ValueError):
            world.index_state(index)
    with pytest.raises(ValueError):
        world.transition((0, 0), "diagonal")


def test_goal_is_absorbing_after_terminal_transition():
    world = GridWorld(size=2, goal=(0, 1))
    world.reset((0, 0))
    state, reward, done = world.step("right")
    assert (state, reward, done) == ((0, 1), world.goal_reward, True)
    assert world.transition(world.goal, "left") == world.goal
    assert world.step("left") == (world.goal, 0.0, True)


def test_step_limit_is_absorbing():
    world = GridWorld(size=3, goal=(2, 2), max_steps=1)
    state, _, done = world.step("right")
    assert done
    assert world.step("down") == (state, 0.0, True)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("episodes", 0),
        ("episodes", 1.5),
        ("alpha", 0.0),
        ("alpha", float("nan")),
        ("gamma", -0.1),
        ("gamma", float("inf")),
        ("epsilon", -0.1),
        ("epsilon", 1.1),
        ("seed", -1),
        ("seed", 2**32),
    ],
)
def test_q_learning_rejects_invalid_hyperparameters(name, value):
    world = GridWorld(size=3, goal=(2, 2))
    kwargs = {name: value}
    with pytest.raises(ValueError):
        tabular_q_learning(world, **kwargs)
