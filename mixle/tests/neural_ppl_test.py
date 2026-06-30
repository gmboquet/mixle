"""The declarative neural surface: a Net predictor in a PPL slot, fit by the standard estimate() loop.

``Categorical(logits=Net(out=K)).fit(y, given={"x": X})`` and ``Normal(Net(out=1), free).fit(y, given={"x": X})``
are neural classification / regression in 3 closure-free lines; ``SoftmaxNeuralLeaf`` composes into a mixture of
experts via ordinary EM. No loss function, no training loop, no lambda in any of it.
"""

import unittest

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class NeuralPPLTest(unittest.TestCase):
    def _toy_classes(self, seed):
        rng = np.random.RandomState(seed)
        x = rng.randn(200, 4).astype("float32")
        return x, (x @ rng.randn(4, 3)).argmax(1)

    def test_softmax_leaf_fits_and_composes_in_a_mixture(self):
        import torch.nn as nn

        from mixle.inference import estimate
        from mixle.models import SoftmaxNeuralLeaf
        from mixle.stats import MixtureDistribution, MixtureEstimator

        torch.manual_seed(0)
        x, y = self._toy_classes(0)
        data = list(zip(x, y))

        def mlp():
            return nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))

        # the leaf fits via the standard estimate(data, est) contract -- no closures
        fit = estimate(data, SoftmaxNeuralLeaf(mlp(), m_steps=200, lr=0.02).estimator())
        self.assertGreater(np.mean([fit.predict(xx) == int(yy) for xx, yy in data]), 0.9)

        # and it composes: a mixture of neural experts trains via real (monotone) EM
        torch.manual_seed(1)
        experts = [SoftmaxNeuralLeaf(mlp(), m_steps=10, lr=0.03) for _ in range(2)]
        est = MixtureEstimator([e.estimator() for e in experts])
        model = MixtureDistribution(experts, [0.5, 0.5])
        enc = model.dist_to_encoder().seq_encode(data)
        lls = []
        for _ in range(6):
            model = estimate(data, est, model)
            lls.append(float(model.seq_log_density(enc).sum()))
        self.assertTrue(all(lls[i] <= lls[i + 1] + 1e-3 for i in range(len(lls) - 1)))

    def test_declarative_categorical_logits_net(self):
        from mixle.ppl import Categorical, Net

        torch.manual_seed(0)
        x, y = self._toy_classes(1)
        fit = Categorical(logits=Net(hidden=[16], out=3)).fit(y, given={"x": x}, epochs=200)
        self.assertGreater(fit.score(y, given={"x": x}), 0.9)
        self.assertEqual(len(np.atleast_1d(fit.predict(given={"x": x[:5]}))), 5)

    def test_declarative_neural_regression_blend(self):
        from mixle.ppl import Net, Normal, free

        torch.manual_seed(0)
        rng = np.random.RandomState(2)
        x = rng.uniform(-2, 2, (200, 1)).astype("float32")
        y = (2 * x[:, 0] + 0.3 * rng.randn(200)).astype("float32")
        fit = Normal(Net(hidden=[16], out=1), free).fit(y, given={"x": x}, epochs=150)
        self.assertGreater(fit.score(y, given={"x": x}), 0.9)

    def test_declarative_conv_classifier_minibatch(self):
        # a conv net over image covariates, trained by minibatch SGD -- still three closure-free lines
        from mixle.ppl import Categorical, Conv

        torch.manual_seed(0)
        rng = np.random.RandomState(3)
        imgs = rng.randn(300, 3, 8, 8).astype("float32")
        y = (imgs[:, 0].mean((1, 2)) + 0.5 * imgs[:, 1].mean((1, 2)) > 0).astype(int)
        fit = Categorical(logits=Conv(channels=[8, 16], out=2)).fit(y, given={"x": imgs}, epochs=40, batch_size=64)
        self.assertGreater(fit.score(y, given={"x": imgs}), 0.9)
        self.assertEqual(len(np.atleast_1d(fit.predict(given={"x": imgs[:5]}))), 5)


if __name__ == "__main__":
    unittest.main()
