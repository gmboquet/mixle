"""LM + training-leaf serialization: a trained model must survive a process boundary.

Regression cover for GROUP F1: the causal-LM ``nn.Module`` classes were function-local (unpicklable), so
``LM`` / ``StreamingTransformer`` / ``DPOModel`` could not be saved once trained. These tests pickle each trained
object and assert the scoring surface (``seq_log_density`` / ``nll`` / ``to_dict``+``from_dict``) is bit-identical
after the round trip, plus the input-validation guards on short sequences, out-of-vocab ids, and empty pair lists.
"""

import copy
import pickle
import unittest

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class LMSerializationTest(unittest.TestCase):
    def _lm(self):
        from mixle.models.language_model import LM

        torch.manual_seed(0)
        return LM(vocab=10, d_model=32, n_layer=2, n_head=4, block=8)

    def test_lm_pickle_round_trip_preserves_nll(self):
        lm = self._lm()
        lm.fit_pairs([([1, 4, 5, 2], [4, 5, 3])] * 8, epochs=3, batch_size=8, seed=0)
        ids = np.array([1, 4, 5, 2, 4, 5, 3, 1, 4, 5, 2, 4])
        n0 = lm.nll(ids)
        lm2 = pickle.loads(pickle.dumps(lm))
        self.assertEqual(lm2.nll(ids), n0)

    def test_lm_to_dict_from_dict_preserves_nll(self):
        lm = self._lm()
        lm.fit_pairs([([1, 4, 5, 2], [4, 5, 3])] * 8, epochs=3, batch_size=8, seed=0)
        ids = np.array([1, 4, 5, 2, 4, 5, 3, 1, 4, 5, 2, 4])
        n0 = lm.nll(ids)
        from mixle.models.language_model import LM

        lm2 = LM.from_dict(lm.to_dict())
        self.assertEqual(lm2.nll(ids), n0)
        self.assertEqual((lm2.vocab, lm2.d_model, lm2.n_layer, lm2.n_head, lm2.block), (10, 32, 2, 4, 8))

    def test_lm_save_load_round_trip(self):
        import tempfile

        lm = self._lm()
        ids = np.arange(9) % 10
        n0 = lm.nll(ids)
        from mixle.models.language_model import LM

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        lm.save(path)
        lm2 = LM.load(path)
        self.assertEqual(lm2.nll(ids), n0)

    def test_nll_on_short_sequence_raises_clear_error(self):
        lm = self._lm()  # block=8
        with self.assertRaises(ValueError) as cm:
            lm.nll(np.arange(8) % 10)  # exactly block tokens -> zero targets
        self.assertIn("block=8", str(cm.exception))

    def test_oov_token_raises_naming_the_id(self):
        lm = self._lm()  # vocab=10
        with self.assertRaises(ValueError) as cm:
            lm.nll(np.array([1, 2, 3, 4, 5, 6, 7, 8, 42]))
        self.assertIn("42", str(cm.exception))
        with self.assertRaises(ValueError):
            lm.generate([1, 2, 99], n=1)
        with self.assertRaises(ValueError):
            lm.fit_pairs([([1, 2], [3, 55])], epochs=1)

    def test_fit_pairs_empty_returns_self(self):
        lm = self._lm()
        self.assertIs(lm.fit_pairs([], epochs=3), lm)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class LeafSerializationTest(unittest.TestCase):
    def test_streaming_transformer_pickle_preserves_seq_log_density(self):
        from mixle.models.streaming_transformer_leaf import StreamingTransformer
        from mixle.models.transformer import build_causal_lm

        torch.manual_seed(0)
        st = StreamingTransformer(build_causal_lm(10, d_model=32, n_layer=2, n_head=4, block=8))
        enc = (np.zeros((4, 8), dtype=float), np.array([1, 2, 3, 4]))
        d0 = st.seq_log_density(enc)
        st2 = pickle.loads(pickle.dumps(st))
        np.testing.assert_array_equal(st2.seq_log_density(enc), d0)

    def test_streaming_transformer_to_dict_and_json_preserve_seq_log_density(self):
        from mixle.models.streaming_transformer_leaf import StreamingTransformer
        from mixle.models.transformer import build_causal_lm
        from mixle.utils.serialization import from_json, to_json, trusted_deserialization

        torch.manual_seed(0)
        st = StreamingTransformer(build_causal_lm(10, d_model=32, n_layer=2, n_head=4, block=8))
        enc = (np.zeros((4, 8), dtype=float), np.array([1, 2, 3, 4]))
        d0 = st.seq_log_density(enc)

        with trusted_deserialization():  # embedded torch module: a self-produced, trusted round-trip
            st_d = StreamingTransformer.from_dict(st.to_dict())
            np.testing.assert_array_equal(st_d.seq_log_density(enc), d0)

            st_j = from_json(to_json(st))
        self.assertIsInstance(st_j, StreamingTransformer)
        np.testing.assert_array_equal(st_j.seq_log_density(enc), d0)

    def test_dpo_model_pickle_preserves_seq_log_density(self):
        from mixle.models.dpo_leaf import DPOModel
        from mixle.models.transformer import build_causal_lm

        torch.manual_seed(0)
        policy = build_causal_lm(10, d_model=32, n_layer=2, n_head=4, block=8)
        dm = DPOModel(policy, copy.deepcopy(policy), beta=0.1)
        enc = (np.zeros((3, 8), dtype=float), np.array([1, 2, 3]), np.array([4, 5, 6]))
        e0 = dm.seq_log_density(enc)
        dm2 = pickle.loads(pickle.dumps(dm))
        np.testing.assert_array_equal(dm2.seq_log_density(enc), e0)

    def test_dpo_model_to_dict_and_json_preserve_seq_log_density(self):
        from mixle.models.dpo_leaf import DPOModel
        from mixle.models.transformer import build_causal_lm
        from mixle.utils.serialization import from_json, to_json, trusted_deserialization

        torch.manual_seed(0)
        policy = build_causal_lm(10, d_model=32, n_layer=2, n_head=4, block=8)
        dm = DPOModel(policy, copy.deepcopy(policy), beta=0.1)
        enc = (np.zeros((3, 8), dtype=float), np.array([1, 2, 3]), np.array([4, 5, 6]))
        e0 = dm.seq_log_density(enc)

        with trusted_deserialization():  # embedded torch module: a self-produced, trusted round-trip
            dm_d = DPOModel.from_dict(dm.to_dict())
            self.assertEqual(dm_d.beta, 0.1)
            np.testing.assert_array_equal(dm_d.seq_log_density(enc), e0)

            dm_j = from_json(to_json(dm))
        self.assertIsInstance(dm_j, DPOModel)
        np.testing.assert_array_equal(dm_j.seq_log_density(enc), e0)


if __name__ == "__main__":
    unittest.main()
