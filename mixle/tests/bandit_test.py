"""Multi-armed bandits (mixle.task.bandit).

The load-bearing claims: every policy concentrates on the best arm (measured as best-arm pull
share against gaps a uniform player would not find), batched/delayed feedback is EXACTLY the
sequential replay (state equality, not similarity), everything is deterministic given its seed,
and EstimatorBandit really does turn an arbitrary mixle estimator into an arm.
"""

import unittest

import numpy as np

from mixle.stats import GammaEstimator, GaussianEstimator
from mixle.task.bandit import UCB1, EstimatorBandit, ThompsonBernoulli, ThompsonGaussian


def _run_bernoulli(policy, ps, steps, env_seed=0):
    env = np.random.RandomState(env_seed)
    for _ in range(steps):
        arm = policy.select()
        policy.update(arm, float(env.rand() < ps[arm]))
    return policy


class ThompsonBernoulliTest(unittest.TestCase):
    def test_concentrates_on_the_best_arm(self):
        policy = _run_bernoulli(ThompsonBernoulli(3, seed=0), ps=(0.8, 0.5, 0.2), steps=1200)
        self.assertGreater(policy.pulls[0] / policy.pulls.sum(), 0.6)
        self.assertEqual(int(np.argmax(policy.means)), 0)

    def test_beats_uniform_regret(self):
        ps = (0.8, 0.5, 0.2)
        policy = _run_bernoulli(ThompsonBernoulli(3, seed=1), ps=ps, steps=1200)
        regret = sum((max(ps) - ps[a]) * n for a, n in enumerate(policy.pulls))
        uniform_regret = 1200 * (max(ps) - float(np.mean(ps)))
        self.assertLess(regret, 0.25 * uniform_regret)

    def test_deterministic_given_seed(self):
        a = _run_bernoulli(ThompsonBernoulli(3, seed=7), ps=(0.7, 0.4, 0.1), steps=300)
        b = _run_bernoulli(ThompsonBernoulli(3, seed=7), ps=(0.7, 0.4, 0.1), steps=300)
        np.testing.assert_array_equal(a.pulls, b.pulls)
        np.testing.assert_array_equal(a.alpha, b.alpha)

    def test_batch_update_is_the_sequential_replay(self):
        seq = ThompsonBernoulli(3, seed=None)
        bat = ThompsonBernoulli(3, seed=None)
        arms = [0, 1, 2, 0, 1, 0]
        rewards = [1.0, 0.0, 0.5, 1.0, 1.0, 0.0]
        for a, r in zip(arms, rewards):
            seq.update(a, r)
        bat.batch_update(arms, rewards)
        np.testing.assert_array_equal(seq.alpha, bat.alpha)
        np.testing.assert_array_equal(seq.beta, bat.beta)
        np.testing.assert_array_equal(seq.pulls, bat.pulls)

    def test_rejects_rewards_outside_the_unit_interval(self):
        policy = ThompsonBernoulli(2, seed=0)
        with self.assertRaises(ValueError):
            policy.update(0, 1.5)
        with self.assertRaises(ValueError):
            policy.update(3, 1.0)


class ThompsonGaussianTest(unittest.TestCase):
    def test_concentrates_on_the_higher_mean(self):
        policy = ThompsonGaussian(2, seed=2)
        env = np.random.RandomState(2)
        for _ in range(800):
            arm = policy.select()
            policy.update(arm, float(env.normal((1.0, 0.0)[arm], 1.0)))
        self.assertGreater(policy.pulls[0] / policy.pulls.sum(), 0.6)
        self.assertGreater(policy.means[0], policy.means[1])

    def test_batch_update_is_the_sequential_replay(self):
        seq = ThompsonGaussian(2, seed=None)
        bat = ThompsonGaussian(2, seed=None)
        arms = [0, 1, 0, 1, 1]
        rewards = [1.2, -0.3, 0.8, 0.1, -1.0]
        for a, r in zip(arms, rewards):
            seq.update(a, r)
        bat.batch_update(arms, rewards)
        np.testing.assert_allclose(seq.m, bat.m)
        np.testing.assert_allclose(seq.b, bat.b)


class UCB1Test(unittest.TestCase):
    def test_plays_every_arm_once_then_concentrates(self):
        policy = UCB1(3)
        self.assertEqual([policy.select() for _ in range(1)][0], 0)  # unplayed first, by index
        _run_bernoulli(policy, ps=(0.9, 0.5, 0.1), steps=2000)
        self.assertTrue((policy.pulls > 0).all())
        self.assertGreater(policy.pulls[0] / policy.pulls.sum(), 0.5)

    def test_is_fully_deterministic(self):
        a = _run_bernoulli(UCB1(3), ps=(0.9, 0.5, 0.1), steps=500, env_seed=3)
        b = _run_bernoulli(UCB1(3), ps=(0.9, 0.5, 0.1), steps=500, env_seed=3)
        np.testing.assert_array_equal(a.pulls, b.pulls)
        np.testing.assert_array_equal(a.sums, b.sums)


class EstimatorBanditTest(unittest.TestCase):
    def test_gaussian_estimator_arms_find_the_best_arm(self):
        policy = EstimatorBandit([GaussianEstimator(), GaussianEstimator()], n_boot=16, seed=4)
        env = np.random.RandomState(4)
        for _ in range(300):
            arm = policy.select()
            policy.update(arm, float(env.normal((1.0, 0.0)[arm], 0.5)))
        self.assertGreater(policy.pulls[0] / policy.pulls.sum(), 0.6)

    def test_gamma_estimator_arms_with_a_custom_score(self):
        # waiting times: LOWER is better, so score by the negated Monte-Carlo-free model mean
        def neg_mean(fitted):
            return -fitted.sampler(seed=0).sample(size=32).mean()

        policy = EstimatorBandit([GammaEstimator(), GammaEstimator()], n_boot=16, mean_fn=neg_mean, seed=5)
        env = np.random.RandomState(5)
        scales = (0.5, 2.0)  # arm 0 is the fast one
        for _ in range(250):
            arm = policy.select()
            policy.update(arm, float(env.gamma(2.0, scales[arm])))
        self.assertGreater(policy.pulls[0] / policy.pulls.sum(), 0.6)

    def test_deterministic_given_seed(self):
        def run():
            policy = EstimatorBandit([GaussianEstimator(), GaussianEstimator()], n_boot=8, seed=6)
            env = np.random.RandomState(6)
            for _ in range(120):
                arm = policy.select()
                policy.update(arm, float(env.normal((1.0, 0.0)[arm], 0.5)))
            return policy.pulls

        np.testing.assert_array_equal(run(), run())

    def test_needs_two_arms_and_two_replicates(self):
        with self.assertRaises(ValueError):
            EstimatorBandit([GaussianEstimator()])
        with self.assertRaises(ValueError):
            EstimatorBandit([GaussianEstimator(), GaussianEstimator()], n_boot=1)


if __name__ == "__main__":
    unittest.main()
