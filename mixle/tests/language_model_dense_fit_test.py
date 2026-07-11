"""LM.fit(dense=True) -- packed all-position teacher forcing (worklist N9.3), checked on held-out data.

The corpus is a deterministic cycle over the payload alphabet, so the next token is always knowable.
The tests pin the N9.3 claim structurally rather than through loss thresholds: the dense path must
supervise every forwarded token exactly once (density 1.0) while the streaming path supervises ~1/block
of them, and at equal corpus/epochs the dense path must run ~block-times fewer token-forwards. Learning
is checked on a HELD-OUT stream, not training loss.
"""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_VOCAB = 10
_BLOCK = 8


def _cycle_corpus(repeats):
    """Deterministic cycle 4,5,6,7,8,9,4,5,... -- every next token is exactly predictable."""
    return np.tile(np.arange(4, 10, dtype=np.int64), repeats)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class DenseFitTest(unittest.TestCase):
    def _lm(self):
        from mixle.models.language_model import LM

        torch.manual_seed(0)
        return LM(vocab=_VOCAB, d_model=48, n_layer=2, n_head=4, block=_BLOCK)

    @staticmethod
    def _count_token_forwards(lm):
        """Attach a pre-hook on the first transformer block counting batch*positions per forward."""
        counter = [0]

        def hook(_mod, args):
            h = args[0]
            counter[0] += int(h.shape[0]) * int(h.shape[1])

        handle = lm.module.blocks[0].register_forward_pre_hook(hook)
        return counter, handle

    def test_dense_supervises_every_token_and_streaming_pays_the_block_factor(self):
        corpus = _cycle_corpus(400)  # 2400 tokens

        lm_dense = self._lm()
        counter_d, handle_d = self._count_token_forwards(lm_dense)
        lm_dense.fit(corpus, dense=True, epochs=1, batch_size=32)
        handle_d.remove()
        n_rows = len(corpus) // (_BLOCK + 1)
        dense_targets = n_rows * _BLOCK
        # every forwarded token is a supervised target: density exactly 1.0
        self.assertEqual(counter_d[0], dense_targets)

        lm_stream = self._lm()
        counter_s, handle_s = self._count_token_forwards(lm_stream)
        lm_stream.fit(corpus, epochs=1, batch_size=32)
        handle_s.remove()
        stream_targets = len(corpus) - _BLOCK  # one target per block-length window
        # streaming forwards a full window per target: density ~1/block
        self.assertGreaterEqual(counter_s[0], stream_targets * _BLOCK)
        # the N9.3 block factor: same corpus, same epochs, ~block-times more compute
        self.assertGreater(counter_s[0] / counter_d[0], _BLOCK / 2)

    def test_dense_fit_learns_the_cycle_on_held_out_data(self):
        lm = self._lm()
        held_out = _cycle_corpus(50)
        nll_before = lm.nll(held_out)

        losses = []
        lm.fit(_cycle_corpus(400), dense=True, epochs=3, batch_size=32, log=lambda e, v: losses.append(v))
        nll_after = lm.nll(held_out)

        self.assertEqual(len(losses), 3)
        self.assertLess(losses[-1], losses[0])
        self.assertLess(nll_after, 0.5 * nll_before)

    def test_dense_fit_validation(self):
        lm = self._lm()
        with self.assertRaises(ValueError):
            lm.fit(np.arange(4, 10, dtype=np.int64)[:_BLOCK], dense=True)  # < block+1 tokens
        with self.assertRaises(NotImplementedError):
            lm.fit(_cycle_corpus(10), dense=True, distributed=True)
        with self.assertRaises(ValueError):
            lm.fit([0, 1, _VOCAB], dense=True)  # id outside vocabulary

    def test_dense_fit_is_deterministic_under_seed(self):
        corpus = _cycle_corpus(100)
        lm_a = self._lm()
        lm_a.fit(corpus, dense=True, epochs=1, batch_size=16, seed=7)
        lm_b = self._lm()
        lm_b.fit(corpus, dense=True, epochs=1, batch_size=16, seed=7)
        for pa, pb in zip(lm_a.module.parameters(), lm_b.module.parameters()):
            self.assertTrue(torch.equal(pa, pb))


if __name__ == "__main__":
    unittest.main()
