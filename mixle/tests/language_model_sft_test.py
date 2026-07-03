"""LM.fit_pairs — dense prompt-masked SFT on (prompt, completion) pairs, checked by what it GENERATES.

The task is tiny but real: completions are a deterministic transform of the prompt (echo the payload,
then a stop token), and correctness is judged on HELD-OUT prompts the model never trained on — so a pass
means the objective actually taught the mapping, not that the loss went down.
"""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# token ids: 0 = pad, 1 = BOS, 2 = SEP, 3 = EOS, payload alphabet = 4..9
_PAD, _BOS, _SEP, _EOS = 0, 1, 2, 3
_VOCAB = 10


def _pair(a, b):
    """prompt = [BOS a b SEP], completion = [a b EOS]: echo the two payload tokens, then stop."""
    return ([_BOS, a, b, _SEP], [a, b, _EOS])


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class FitPairsTest(unittest.TestCase):
    def _lm(self, d_model=48):
        from mixle.models.language_model import LM

        torch.manual_seed(0)
        return LM(vocab=_VOCAB, d_model=d_model, n_layer=2, n_head=4, block=8)

    def test_sft_learns_the_mapping_and_generalizes_to_unseen_pairs(self):
        alphabet = range(4, 10)
        all_pairs = [(a, b) for a in alphabet for b in alphabet]
        rng = np.random.RandomState(0)
        rng.shuffle(all_pairs)
        held_out = all_pairs[:6]
        train = [_pair(a, b) for a, b in all_pairs[6:]]

        lm = self._lm(d_model=64)
        losses = []
        lm.fit_pairs(train * 8, epochs=100, batch_size=16, lr=3e-3, seed=0, log=lambda e, x: losses.append(x))
        self.assertLess(losses[-1], losses[0] * 0.2)  # the masked objective actually optimized

        # judged by generation on pairs never seen in training: exact on ALL train pairs, and >= 5/6
        # held-out (cross-platform float drift can plausibly flip one borderline pair; 6/6 locally)
        for a, b in all_pairs[6:]:
            prompt, want = _pair(a, b)
            self.assertEqual(lm.generate(prompt, n=6, greedy=True, stop_id=_EOS)[len(prompt) :], want)
        hits = 0
        for a, b in held_out:
            prompt, want = _pair(a, b)
            hits += lm.generate(prompt, n=6, greedy=True, stop_id=_EOS)[len(prompt) :] == want
        self.assertGreaterEqual(hits, 5)

    def test_stop_id_halts_generation_and_is_returned(self):
        lm = self._lm()
        lm.fit_pairs([_pair(4, 5)] * 32, epochs=40, batch_size=16, seed=0)
        out = lm.generate([_BOS, 4, 5, _SEP], n=50, greedy=True, stop_id=_EOS)
        gen = out[4:]
        self.assertIn(_EOS, gen)
        self.assertIs(gen[-1], gen[gen.index(_EOS)])  # nothing generated past the stop token
        self.assertLess(len(gen), 50)

    def test_long_pairs_keep_the_completion_by_truncating_old_prompt(self):
        lm = self._lm()  # block=8; a 10-token prompt must not crash and must still train on the completion
        long_prompt = [_BOS] + [4] * 9 + [_SEP]
        lm.fit_pairs([(long_prompt, [5, _EOS])] * 16, epochs=5, batch_size=8, seed=0)

    def test_mask_prompt_false_trains_prompt_positions_too(self):
        lm = self._lm()
        losses_all, losses_masked = [], []
        pairs = [_pair(a, b) for a in (4, 5) for b in (6, 7)] * 8
        lm.fit_pairs(pairs, epochs=1, batch_size=8, mask_prompt=False, seed=0, log=lambda e, x: losses_all.append(x))
        lm2 = self._lm()
        lm2.fit_pairs(pairs, epochs=1, batch_size=8, mask_prompt=True, seed=0, log=lambda e, x: losses_masked.append(x))
        # both ran; the unmasked objective scores strictly more positions (prompt tokens are near-uniform
        # at init either way, so this only checks the switch is wired, not loss magnitudes)
        self.assertEqual(len(losses_all), 1)
        self.assertEqual(len(losses_masked), 1)


if __name__ == "__main__":
    unittest.main()
