"""uq(): one verb, method auto-selected -- Laplace curvature / split conformal / semantic entropy."""

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
    def test_parameter_likelihood_approximation_covers_the_truth(self):
        data = [float(x) for x in np.random.RandomState(0).normal(5.0, 2.0, 300)]
        model = optimize(data, st.GaussianEstimator(), out=None)
        r = uq(model, data)
        self.assertEqual(r.kind, "parameter_likelihood_approximation")
        lo, hi = r.parameter_interval(lambda d: d.mean(), alpha=0.1, n=400)
        self.assertLess(lo, 5.0)
        self.assertGreater(hi, 5.0)
        self.assertLess(hi - lo, 1.5)  # 300 points -> tight local likelihood curvature, not a vacuous interval
        with self.assertRaises(ValueError):
            r.credible_interval(lambda d: d.mean(), alpha=0.1, n=20)

    def test_needs_data_for_the_approximation(self):
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
        self.assertFalse(r.confident(3, n=8))  # the two-meaning prompt is not
        # STAT-RR19-12: the calibrated threshold covers only the calibrated draw count
        with self.assertRaisesRegex(ValueError, "calibrated draw count"):
            r.confident(3, n=16)


class DispatchTest(unittest.TestCase):
    def test_unknown_type_raises(self):
        with self.assertRaises(TypeError):
            uq(42)


if __name__ == "__main__":
    unittest.main()


class SemanticEntropyScaleAlignmentTest(unittest.TestCase):
    """STAT-RR19-12: calibration quantiled PLUG-IN entropies while serving gated on the
    Miller-Madow estimate -- the reviewer's [a,a,b,c,d,e,f,g] pattern calibrated a threshold of
    1.9062 and was then rejected at its own serving value 2.2812; acceptance on a uniform
    ten-class generator moved 98.2% -> 41.0% purely from the scale mismatch. Both sides now use
    the corrected statistic at a recorded draw count."""

    @staticmethod
    def _generator():
        import zlib

        classes = [f"c{i}" for i in range(10)]

        def gen(prompt):
            # zlib.crc32 is process-stable, unlike salted hash(): the calibrated quantile must
            # be reproducible for the scale-identity assertions below
            rs = np.random.RandomState(zlib.crc32(f"g|{prompt}|{gen.calls}".encode()) % (2**31))
            gen.calls += 1
            return rs.choice(classes)

        gen.calls = 0
        return gen

    def test_calibration_and_serving_share_the_corrected_scale(self):
        from mixle.inference.uncertainty import semantic_entropy_receipt
        from mixle.inference.uq import uq

        result = uq(self._generator(), data=[f"p{i}" for i in range(120)], alpha=0.2)
        self.assertEqual(result.payload["calibration_n"], 8)
        pattern = semantic_entropy_receipt(["a", "a", "b", "c", "d", "e", "f", "g"])
        self.assertAlmostEqual(pattern["entropy_miller_madow"], 2.2811547465, places=9)
        # the calibrated threshold is the (1 - alpha)-quantile OF THE SERVED STATISTIC: recompute
        # the same Miller-Madow receipts over the calibration prompts and the quantile must be
        # identical -- with the old plug-in-scale calibration it sat a full (K-1)/(2n) below
        replay = self._generator()
        entropies = [
            semantic_entropy_receipt([replay(p) for _ in range(8)])["entropy_miller_madow"]
            for p in [f"p{i}" for i in range(120)]
        ]
        self.assertAlmostEqual(result.payload["max_entropy"], float(np.quantile(entropies, 0.8)), places=9)
        accepted = np.mean([result.confident(f"q{i}") for i in range(200)])
        self.assertGreaterEqual(accepted, 0.78)  # calibration promises >= 1 - alpha up to MC noise

    def test_calibrated_threshold_refuses_a_different_draw_count(self):
        from mixle.inference.uq import uq

        result = uq(self._generator(), data=[f"p{i}" for i in range(30)], alpha=0.2)
        with self.assertRaisesRegex(ValueError, "calibrated draw count"):
            result.confident("x", n=16)
        # an explicit override carries no calibration claim and may pick its own n
        self.assertIn(result.confident("x", n=16, max_entropy=50.0), (True, False))

    def test_receipt_carries_a_standard_error(self):
        from mixle.inference.uncertainty import semantic_entropy_receipt

        receipt = semantic_entropy_receipt(["a", "a", "b", "c", "d", "e", "f", "g"])
        self.assertIsNotNone(receipt["entropy_se_estimate"])
        self.assertGreater(receipt["entropy_se_estimate"], 0.0)
