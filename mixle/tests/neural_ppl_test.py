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

    def test_declarative_autoregressive_transformer_lm(self):
        # an autoregressive Transformer LM, trained through the UNCHANGED estimate() loop -- one declarative line
        from mixle.ppl import Categorical, Transformer

        torch.manual_seed(0)
        text = "the quick brown fox jumps over the lazy dog. " * 20
        chars = sorted(set(text))
        v = len(chars)
        stoi = {c: i for i, c in enumerate(chars)}
        ids = np.array([stoi[c] for c in text])
        b = 16
        ctx = np.stack([ids[i : i + b] for i in range(len(ids) - b)]).astype("float32")
        nxt = ids[b:]
        # epochs=15 already drives nll to ~0.0001 on this highly repetitive corpus (verified empirically;
        # the pass/fail boundary is between epochs=7 and 8), a large margin under the < 0.5 threshold
        # while cutting training cost ~3x vs. the original epochs=40.
        fit = Categorical(logits=Transformer(out=v, d_model=64, n_layer=2, n_head=4)).fit(
            nxt, given={"x": ctx}, epochs=15, batch_size=128, lr=0.003
        )
        nll = -np.mean(fit.dist.seq_log_density((ctx, nxt)))
        self.assertLess(nll, 0.5)  # learned next-token prediction (random would be ~log(v) ~ 2.8 nats)

    def test_streaming_transformer_leaf_does_not_buffer_the_corpus(self):
        # the keystone: seq_update IS a train step; value() is (loss_sum, tokens) telemetry, NEVER the corpus
        from mixle.data.stream_token_source import stream_token_source
        from mixle.models.streaming_transformer_leaf import stream_fit
        from mixle.models.transformer import build_causal_lm

        torch.manual_seed(0)
        text = "the quick brown fox jumps over the lazy dog. " * 25
        chars = sorted(set(text))
        v = len(chars)
        stoi = {c: i for i, c in enumerate(chars)}
        ids = np.array([stoi[c] for c in text])
        b = 16
        module = build_causal_lm(v, d_model=64, n_layer=2, n_head=4, block=b)
        # epochs=10 already drives nll to ~0.0001 on this highly repetitive corpus (verified empirically;
        # the pass/fail boundary is between epochs=5 and 6), a large margin under the < 0.5 threshold
        # while cutting training cost ~3x vs. the original epochs=30.
        src = stream_token_source(ids, block=b, batch_size=128, epochs=10, seed=0)  # a generator, not a buffered list
        leaf, payload = stream_fit(module, src, lr=3e-3)
        self.assertEqual(len(payload), 2)  # (loss_sum, tokens) -- the accumulator never held the corpus
        ctx = np.stack([ids[i : i + b] for i in range(64)]).astype("float32")
        nll = -np.mean(leaf.seq_log_density((ctx, ids[b : b + 64])))
        self.assertLess(nll, 0.5)  # the streamed model learned next-token prediction

    def test_sft_loss_mask_ignores_masked_observations(self):
        # SFT-style masking: weight-0 (prompt) observations must not affect the model
        from mixle.ppl import Categorical, Net

        torch.manual_seed(0)
        rng = np.random.RandomState(0)
        x = rng.randn(200, 4).astype("float32")
        y = (x @ rng.randn(4, 3)).argmax(1)
        mask = (np.arange(200) % 2 == 0).astype(float)
        yc = y.copy()
        yc[mask == 0] = rng.randint(0, 3, (mask == 0).sum())  # masked-out half carries WRONG labels
        fit = Categorical(logits=Net(hidden=[32], out=3)).fit(yc, given={"x": x}, epochs=250, weights=mask)
        self.assertGreater(np.mean(fit.predict(given={"x": x})[mask == 1] == y[mask == 1]), 0.9)  # learned the unmasked

    def test_cpt_ewc_retains_the_old_task(self):
        # continued pretraining with EWC retains task A better than plain continuation does
        from mixle.models.continual import ewc, fisher_diagonal, snapshot
        from mixle.ppl import Categorical, Net

        rng = np.random.RandomState(1)
        xa = rng.randn(300, 4).astype("float32")
        ya = (xa @ rng.randn(4, 3)).argmax(1)
        xb = rng.randn(300, 4).astype("float32")
        yb = (xb @ rng.randn(4, 3)).argmax(1)

        def pre(s):
            torch.manual_seed(s)
            return Categorical(logits=Net(hidden=[32], out=3)).fit(ya, given={"x": xa}, epochs=250)

        p1, p2 = pre(0), pre(0)
        anc, fish = snapshot(p1.dist), fisher_diagonal(p1.dist, xa, ya)
        no = Categorical(logits=Net(hidden=[32], out=3)).fit(yb, given={"x": xb}, epochs=250, init=p1)
        yes = Categorical(logits=Net(hidden=[32], out=3)).fit(
            yb, given={"x": xb}, epochs=250, init=p2, ewc=ewc(anc, fish, lam=3e4)
        )
        acc_a_no = np.mean(no.predict(given={"x": xa}) == ya)
        acc_a_yes = np.mean(yes.predict(given={"x": xa}) == ya)
        self.assertGreater(acc_a_yes, acc_a_no + 0.08)  # EWC anti-forgetting retains task A

    def test_dpo_aligns_policy_to_preferences(self):
        # DPO: the policy learns to prefer chosen over rejected -- no reward model, no RL
        import copy

        import torch.nn as nn

        from mixle.inference import estimate
        from mixle.models.dpo_leaf import DPOLeaf

        rng = np.random.RandomState(0)
        x = rng.randn(300, 4).astype("float32")
        good = (x @ rng.randn(4, 3)).argmax(1)  # the preferred action per context
        rej = (good + rng.randint(1, 3, len(good))) % 3
        torch.manual_seed(0)
        policy = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
        leaf = DPOLeaf(policy, copy.deepcopy(policy), beta=0.1, m_steps=400, lr=1e-2)  # frozen ref = initial policy
        before = np.mean(leaf.prefers(x) == good)
        fit = estimate(list(zip(x, good, rej)), leaf.estimator())  # (x, chosen, rejected) triples
        after = np.mean(fit.prefers(x) == good)
        self.assertGreater(after, before + 0.4)  # alignment shifted the policy toward the preferred action

    def test_language_model_surface_fit_generate_nll(self):
        # the LM surface: one object trains (streaming), evaluates, and generates
        from mixle.models.language_model import LM

        torch.manual_seed(0)
        text = "the quick brown fox jumps over the lazy dog. " * 25
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        ids = np.array([stoi[c] for c in text])
        lm = LM(len(chars), d_model=64, n_layer=2, n_head=4, block=16)
        # epochs=10 already drives nll to ~0.0001 on this highly repetitive corpus (verified empirically;
        # the pass/fail boundary is between epochs=5 and 6), a large margin under the < 0.5 threshold
        # while cutting training cost ~3x vs. the original epochs=30.
        lm.fit(ids, epochs=10, batch_size=128, lr=3e-3)
        self.assertLess(lm.nll(ids), 0.5)  # learned next-token prediction
        gen = lm.generate(ids[:16].tolist(), n=20, greedy=True)
        self.assertEqual(len(gen), 36)  # generation extends the 16-token prompt by 20


if __name__ == "__main__":
    unittest.main()
