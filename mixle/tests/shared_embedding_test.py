"""Shared learned embedding (mixle.models.SharedEmbedding / mixle.ppl.Embedding): declare once, tie everywhere.

Several language models can reference one word embedding so they train the same token vectors jointly -- the
neural analogue of the PPL's ``name=`` scalar tying. The tie must be a real shared parameter (same tensor, one
gradient), reachable from both the stats ``LM`` and the PPL ``Transformer`` token.
"""

import unittest

import pytest

torch = pytest.importorskip("torch")

from mixle.models import LM, SharedEmbedding, build_causal_lm  # noqa: E402
from mixle.ppl import Embedding, Transformer  # noqa: E402


class SharingTest(unittest.TestCase):
    def test_lms_share_the_same_embedding_tensor(self):
        emb = SharedEmbedding(vocab=40, dim=16, name="word")
        a = LM(vocab=40, d_model=16, n_layer=2, block=8, embedding=emb)
        b = LM(vocab=40, d_model=16, n_layer=2, block=8, embedding=emb)
        self.assertIs(a.module.tok, b.module.tok)
        self.assertIs(a.module.tok.weight, b.module.tok.weight)
        self.assertIs(a.module.head.weight, a.module.tok.weight)  # tied head follows the shared embedding
        # the rest of each model is its own
        self.assertIsNot(a.module.blocks[0].attn.qkv.weight, b.module.blocks[0].attn.qkv.weight)

    def test_a_gradient_step_on_one_updates_the_shared_embedding(self):
        emb = SharedEmbedding(40, 16)
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
            build_causal_lm(40, 32, embedding=SharedEmbedding(40, 16))  # dim 16 != d_model 32
        with self.assertRaises(ValueError):
            build_causal_lm(50, 16, embedding=SharedEmbedding(40, 16))  # vocab 40 != 50


class PPLTest(unittest.TestCase):
    def test_ppl_transformer_tokens_share_embedding(self):
        emb = Embedding(40, 16)
        t1 = Transformer(out=40, d_model=16, embedding=emb)
        t2 = Transformer(out=40, d_model=16, embedding=emb)
        m1, m2 = t1.build(8), t2.build(8)
        self.assertIs(m1.tok.weight, m2.tok.weight)
        # a Transformer without a shared embedding keeps its own
        m3 = Transformer(out=40, d_model=16).build(8)
        self.assertIsNot(m3.tok.weight, m1.tok.weight)


class MixtureExpertsTest(unittest.TestCase):
    def test_mixture_of_lms_shares_word_embedding(self):
        # per-cluster experts (the README's mixture) that tie one word embedding across components
        from mixle.models import StreamingTransformerLeaf
        from mixle.stats import MixtureEstimator

        emb = SharedEmbedding(vocab=64, dim=24, name="word")
        experts = [LM(vocab=64, d_model=24, n_layer=2, block=16, embedding=emb) for _ in range(3)]
        est = MixtureEstimator([StreamingTransformerLeaf(e.module).estimator() for e in experts])
        self.assertEqual(len(est.estimators), 3)
        # every expert's token embedding is the one shared tensor
        weights = {id(e.module.tok.weight) for e in experts}
        self.assertEqual(len(weights), 1)


if __name__ == "__main__":
    unittest.main()
