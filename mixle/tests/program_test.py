"""mixle.program -- optimization programs (moves + combinators) covering the validated patterns.

minimize/maximize + weighted (multi-objective) + alternate (adversarial/coordinate); constraints as a
primal-dual game; REINFORCE policy gradient; LoRA scoped fine-tuning; continual learning by replay; and the
``em`` bridge that makes a mixle estimator a first-class move.
"""

import unittest

import numpy as np

from mixle.program import (
    Stream,
    alternate,
    constrain,
    em,
    ewc,
    fisher_diagonal,
    fit,
    lora,
    maximize,
    minimize,
    reinforce,
    snapshot,
    trainable,
    weighted,
)

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ProgramGradientTest(unittest.TestCase):
    def test_minimize_quadratic(self):
        theta = torch.tensor(0.0, requires_grad=True)
        fit(minimize(lambda: (theta - 3.0) ** 2, over=[theta]), steps=500, lr=0.05)
        self.assertAlmostEqual(theta.item(), 3.0, places=2)

    def test_weighted_multi_objective(self):
        theta = torch.zeros(2, requires_grad=True)
        prog = weighted([(lambda: (theta[0] - 1.0) ** 2, 1.0), (lambda: (theta[1] - 2.0) ** 2, 1.0)], over=[theta])
        fit(prog, steps=500, lr=0.05)
        self.assertAlmostEqual(theta[0].item(), 1.0, places=2)
        self.assertAlmostEqual(theta[1].item(), 2.0, places=2)

    def test_alternate_independent_moves_both_converge(self):
        a = torch.tensor(0.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        fit(alternate(minimize(lambda: (a - 1.0) ** 2, [a]), minimize(lambda: (b + 2.0) ** 2, [b])), steps=500, lr=0.05)
        self.assertAlmostEqual(a.item(), 1.0, places=2)
        self.assertAlmostEqual(b.item(), -2.0, places=2)

    def test_constraints_are_a_primal_dual_game(self):
        # minimize (theta-5)^2 s.t. theta <= 2  ->  the binding optimum is theta*=2, lambda*=6
        theta = torch.tensor(0.0, requires_grad=True)
        fit(
            minimize(lambda: (theta - 5.0) ** 2, over=[theta]),
            constraints=[constrain(lambda: theta, 2.0, "<=")],
            steps=4000,
            lr=0.02,
        )
        self.assertAlmostEqual(theta.item(), 2.0, places=1)
        self.assertLessEqual(theta.item(), 2.0 + 1e-2)  # constraint satisfied

    def test_reinforce_policy_gradient(self):
        torch.manual_seed(0)
        logits = torch.zeros(4, requires_grad=True)
        reward = torch.tensor([0.1, 0.9, 0.3, 0.5])  # arm 1 is best

        def sample_and_reward():
            pi = torch.softmax(logits, 0)
            a = torch.multinomial(pi, 256, replacement=True)
            return torch.log_softmax(logits, 0)[a], reward[a]

        fit(maximize(reinforce(sample_and_reward), over=[logits]), steps=300, lr=0.1)
        self.assertGreater(torch.softmax(logits, 0)[1].item(), 0.8)

    def test_lora_trains_only_adapters(self):
        torch.manual_seed(0)
        net = _mlp([4, 16, 16, 1])
        adapter_params = lora(net, rank=2)
        self.assertGreater(len(adapter_params), 0)
        # base weights frozen, adapters trainable
        self.assertTrue(all(not p.requires_grad for n, p in net.named_parameters() if "base" in n))
        self.assertTrue(all(p.requires_grad for p in adapter_params))
        X, y = torch.randn(128, 4), torch.randn(128, 1)
        loss0 = ((net(X) - y) ** 2).mean().item()
        fit(minimize(lambda: ((net(X) - y) ** 2).mean(), over=adapter_params), steps=300, lr=0.05)
        self.assertLess(((net(X) - y) ** 2).mean().item(), loss0)  # adapters reduced the loss

    def test_continual_learning_replay_prevents_forgetting(self):
        torch.manual_seed(0)
        np.random.seed(0)
        f = lambda x: np.sin(2 * x)

        def region(lo, hi, n=400):
            x = np.random.uniform(lo, hi, n).astype("float32")
            return torch.tensor(x[:, None]), torch.tensor(f(x)[:, None].astype("float32"))

        xa, ya = region(-3, 0)
        xb, yb = region(0, 3)
        xat, yat = region(-3, 0, 1500)
        mse = lambda net, x, y: ((net(x) - y) ** 2).mean().item()

        def train(net, objective, steps=500):
            fit(minimize(objective, over=trainable(net)), steps=steps, lr=0.01)

        import copy

        net = _mlp([1, 64, 64, 1])
        train(net, lambda: ((net(xa) - ya) ** 2).mean())  # pretrain on region A
        naive = copy.deepcopy(net)
        replay = copy.deepcopy(net)
        train(naive, lambda: ((naive(xb) - yb) ** 2).mean())  # continue on B (forgets A)
        train(replay, lambda: ((replay(xb) - yb) ** 2).mean() + ((replay(xa) - ya) ** 2).mean())  # B + replay(A)
        self.assertLess(mse(replay, xat, yat), 0.5 * mse(naive, xat, yat))  # replay retains A far better


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ContinuousPretrainingTest(unittest.TestCase):
    def test_streaming_fit_tracks_a_shifting_stream(self):
        # CPT loop: data arrives as chunks from a shifting target; the model adapts to each (warm-started).
        torch.manual_seed(0)
        net = _mlp([1, 32, 1])

        def chunks():
            for c in range(5):
                x = torch.linspace(c, c + 1, 64)[:, None]
                yield (x, torch.sin(2 * x))

        s = Stream(chunks())
        cur = lambda: ((net(s.current[0]) - s.current[1]) ** 2).mean()
        losses = []
        fit(minimize(cur, over=trainable(net)), data=s, steps_per_chunk=150, lr=0.02, callback=lambda i, p: losses.append(cur().item()))
        self.assertEqual(len(losses), 5)  # consumed every chunk
        self.assertLess(max(losses), 0.1)  # tracked each one

    def test_ewc_with_model_fisher_prevents_forgetting(self):
        # EWC as a program term, with the proper MODEL-sampled Fisher -- retains the old task far better than naive.
        torch.manual_seed(0)
        np.random.seed(0)
        f = lambda x: np.sin(2 * x)

        def region(lo, hi, n=400):
            x = np.random.uniform(lo, hi, n).astype("float32")
            return torch.tensor(x[:, None]), torch.tensor(f(x)[:, None].astype("float32"))

        xa, ya = region(-3, 0)
        xb, yb = region(0, 3)
        xat, yat = region(-3, 0, 1500)
        mse = lambda net, x, y: ((net(x) - y) ** 2).mean().item()

        import copy

        net = _mlp([1, 64, 64, 1])
        fit(minimize(lambda: ((net(xa) - ya) ** 2).mean(), over=trainable(net)), steps=500, lr=0.01)  # pretrain A
        fisher = fisher_diagonal(net, [xa], kind="regression")
        anchor = snapshot(trainable(net))
        naive = copy.deepcopy(net)
        ewc_net = copy.deepcopy(net)
        fit(minimize(lambda: ((naive(xb) - yb) ** 2).mean(), over=trainable(naive)), steps=500, lr=0.01)  # naive
        p = trainable(ewc_net)
        fit(weighted([(lambda: ((ewc_net(xb) - yb) ** 2).mean(), 1.0), (ewc(p, fisher, anchor, 1.0), 1.0)], over=p), steps=500, lr=0.01)
        self.assertLess(mse(ewc_net, xat, yat), 0.7 * mse(naive, xat, yat))  # EWC retains A much better


class ProgramEMBridgeTest(unittest.TestCase):
    def test_em_step_is_a_move(self):
        from mixle.stats import MixtureDistribution, MixtureEstimator
        from mixle.stats import MultivariateGaussianDistribution as MVN

        rng = np.random.RandomState(0)
        data = [list(rng.randn(2)) for _ in range(150)] + [list(rng.randn(2) + 6.0) for _ in range(150)]
        est = MixtureEstimator([MVN(np.zeros(2), np.eye(2)).estimator() for _ in range(2)])
        # deliberately off-center init so EM has work to do
        init = MixtureDistribution([MVN(np.array([1.0, 1.0]), np.eye(2)), MVN(np.array([3.0, 3.0]), np.eye(2))], [0.5, 0.5])
        move = em(est, data, init)
        ll0 = sum(init.log_density(x) for x in data)
        fit(alternate(move), steps=15)  # the EM move runs inside the same program runner
        ll1 = sum(move.model.log_density(x) for x in data)
        self.assertGreater(ll1, ll0 + 1.0)  # EM (as a move) improved the fit


if __name__ == "__main__":
    unittest.main()
