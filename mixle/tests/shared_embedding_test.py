"""Shared learned embedding (mixle.models.CategoricalEmbedding / mixle.ppl.Embedding): declare once, tie everywhere.

Several language models can reference one word embedding so they train the same token vectors jointly -- the
neural analogue of the PPL's ``name=`` scalar tying. The tie must be a real shared parameter (same tensor, one
gradient), reachable from the plain estimator (``TransformerLMEstimator``), the ``LM`` convenience, and the PPL
``Transformer`` token.
"""

import unittest

import pytest

torch = pytest.importorskip("torch")

from mixle.models import LM, CategoricalEmbedding, TransformerLMEstimator, build_causal_lm  # noqa: E402
from mixle.ppl import Embedding, Transformer  # noqa: E402


class SharingTest(unittest.TestCase):
    def test_lms_share_the_same_embedding_tensor(self):
        emb = CategoricalEmbedding(40, 16, name="word")
        a = LM(vocab=40, d_model=16, n_layer=2, block=8, embedding=emb)
        b = LM(vocab=40, d_model=16, n_layer=2, block=8, embedding=emb)
        self.assertIs(a.module.tok, b.module.tok)
        self.assertIs(a.module.tok.weight, b.module.tok.weight)
        self.assertIs(a.module.head.weight, a.module.tok.weight)  # tied head follows the shared embedding
        self.assertIsNot(a.module.blocks[0].attn.qkv.weight, b.module.blocks[0].attn.qkv.weight)  # rest is its own

    def test_a_gradient_step_on_one_updates_the_shared_embedding(self):
        emb = CategoricalEmbedding(40, 16)
        a = LM(vocab=40, d_model=16, n_layer=1, block=8, embedding=emb)
        b = LM(vocab=40, d_model=16, n_layer=1, block=8, embedding=emb)
        before = emb.module().weight.detach().clone()
        x = torch.randint(0, 40, (4, 8)).float()
        y = torch.randint(0, 40, (4,))
        opt = torch.optim.SGD(a.module.parameters(), lr=1.0)
        torch.nn.functional.cross_entropy(a.module(x), y).backward()
        opt.step()
        self.assertFalse(torch.allclose(before, b.module.tok.weight))  # b sees a's update

    def test_no_embedding_means_independent(self):
        a = LM(vocab=40, d_model=16, n_layer=1, block=8)
        b = LM(vocab=40, d_model=16, n_layer=1, block=8)
        self.assertIsNot(a.module.tok.weight, b.module.tok.weight)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            build_causal_lm(40, 32, embedding=CategoricalEmbedding(40, 16))  # dim 16 != d_model 32
        with self.assertRaises(ValueError):
            build_causal_lm(50, 16, embedding=CategoricalEmbedding(40, 16))  # num_categories 40 != vocab 50


class PPLTest(unittest.TestCase):
    def test_ppl_transformer_tokens_share_embedding(self):
        emb = Embedding(40, 16)  # mixle.ppl.Embedding is CategoricalEmbedding
        t1 = Transformer(out=40, d_model=16, embedding=emb)
        t2 = Transformer(out=40, d_model=16, embedding=emb)
        m1, m2 = t1.build(8), t2.build(8)
        self.assertIs(m1.tok.weight, m2.tok.weight)
        m3 = Transformer(out=40, d_model=16).build(8)  # no shared embedding -> its own
        self.assertIsNot(m3.tok.weight, m1.tok.weight)


class EstimatorSyntaxTest(unittest.TestCase):
    def test_transformer_lm_estimator_shares_and_trains(self):
        # the clean estimator surface: TransformerLMEstimator(vocab, ...) -- no Leaf, no .estimator(), no raw module
        import numpy as np

        from mixle.inference import optimize
        from mixle.stats import MixtureEstimator

        v, d, block = 60, 24, 8
        emb = CategoricalEmbedding(v, d, name="word")
        experts = [
            TransformerLMEstimator(v, d_model=d, n_layer=2, n_head=2, block=block, embedding=emb) for _ in range(3)
        ]
        est = MixtureEstimator(experts)
        self.assertEqual(len({id(e.module.tok.weight) for e in experts}), 1)  # one shared tensor before fit

        rng = np.random.RandomState(0)
        data = [(list(rng.randint(0, v, size=block)), int(rng.randint(0, v))) for _ in range(48)]
        before = emb.module().weight.detach().clone()
        model = optimize(data, est, max_its=2, out=None)
        self.assertFalse(torch.allclose(before, emb.module().weight))  # the shared embedding trained under EM
        self.assertEqual(len({id(c.module.tok.weight) for c in model.components}), 1)  # still shared after fit


if __name__ == "__main__":
    unittest.main()
