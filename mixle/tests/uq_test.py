"""uq(): one verb, method auto-selected -- Laplace posterior / split conformal / semantic entropy."""

import unittest

import numpy as np

import mixle.stats as st
from mixle.inference import optimize, uq

try:
    import torch  # noqa: F401
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class MixleModelUQTest(unittest.TestCase):
    def test_parameter_posterior_covers_the_truth(self):
        data = [float(x) for x in np.random.RandomState(0).normal(5.0, 2.0, 300)]
        model = optimize(data, st.GaussianEstimator(), out=None)
        r = uq(model, data)
        self.assertEqual(r.kind, "parameter_posterior")
        lo, hi = r.credible_interval(lambda d: d.mean(), alpha=0.1, n=400)
        self.assertLess(lo, 5.0)
        self.assertGreater(hi, 5.0)
        self.assertLess(hi - lo, 1.5)  # 300 points -> a tight posterior, not a vacuous interval

    def test_needs_data_for_the_posterior(self):
        model = optimize([float(x) for x in np.random.RandomState(0).randn(100)], st.GaussianEstimator(), out=None)
        with self.assertRaises(ValueError):
            uq(model)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TorchPredictorUQTest(unittest.TestCase):
    def _trained_net(self):
        torch.manual_seed(0)
        x = np.random.RandomState(1).uniform(-3, 3, (500, 1)).astype("float32")
        y = (2.0 * x[:, 0] + 1.0 + 0.5 * np.random.RandomState(2).randn(500)).astype("float32")
        net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
        opt = torch.optim.Adam(net.parameters(), lr=0.05)
        for _ in range(300):
            opt.zero_grad()
            loss = ((net(torch.tensor(x)).squeeze(1) - torch.tensor(y)) ** 2).mean()
            loss.backward()
            opt.step()
        return net

    def test_split_conformal_covers_on_fresh_data(self):
        net = self._trained_net()
        xc = np.random.RandomState(3).uniform(-3, 3, (300, 1)).astype("float32")
        yc = 2.0 * xc[:, 0] + 1.0 + 0.5 * np.random.RandomState(4).randn(300)
        r = uq(net, data=(list(xc), list(yc)), alpha=0.1)
        self.assertEqual(r.kind, "conformal_regressor")
        xt = np.random.RandomState(5).uniform(-3, 3, (400, 1)).astype("float32")
        yt = 2.0 * xt[:, 0] + 1.0 + 0.5 * np.random.RandomState(6).randn(400)
        covered = 0
        for xi, yi in zip(xt, yt):
            lo, hi = r.interval(xi)
            covered += int(lo[0] <= yi <= hi[0])
        self.assertGreaterEqual(covered / len(xt), 0.85)  # >= 1 - alpha minus finite-sample slack

    def test_ensemble_reports_epistemic_spread(self):
        nets = [self._trained_net(), self._trained_net()]
        xc = np.random.RandomState(3).uniform(-3, 3, (200, 1)).astype("float32")
        yc = 2.0 * xc[:, 0] + 1.0 + 0.5 * np.random.RandomState(4).randn(200)
        r = uq(nets, data=(list(xc), list(yc)), alpha=0.1)
        self.assertEqual(r.kind, "ensemble_regressor")
        self.assertEqual(r.epistemic_std(np.float32([[1.0]])).shape, (1,))  # a spread per output


class LLMUQTest(unittest.TestCase):
    def test_ambiguous_generator_has_higher_semantic_entropy(self):
        def determinate(_prompt):
            return "the capital is paris"

        rng = np.random.RandomState(0)

        def ambiguous(_prompt):
            return rng.choice(["yes", "no", "maybe", "unclear"])

        rd, ra = uq(determinate), uq(ambiguous)
        self.assertEqual(rd.kind, "llm_semantic")
        self.assertLess(rd.semantic_entropy("q", n=8), ra.semantic_entropy("q", n=16))

    def test_calibrated_abstention_threshold(self):
        rng = np.random.RandomState(1)
        pool = [["paris"], ["paris"], ["london"], ["yes", "no"]]

        def gen(prompt):
            return rng.choice(pool[prompt % len(pool)])

        r = uq(gen, data=[0, 1, 2], alpha=0.2)  # calibrate on determinate prompts
        self.assertTrue(np.isfinite(r.payload["max_entropy"]))
        self.assertTrue(r.confident(0, n=8))  # a determinate prompt is confident
        self.assertFalse(r.confident(3, n=16))  # the two-meaning prompt is not


class DispatchTest(unittest.TestCase):
    def test_unknown_type_raises(self):
        with self.assertRaises(TypeError):
            uq(42)


if __name__ == "__main__":
    unittest.main()
