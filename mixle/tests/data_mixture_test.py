"""DoReMi-style data mixture optimization (mixle.task.data_mixture, roadmap F8).

The load-bearing claim: a mixture learned by :func:`optimize_mixture` from cheap proxy runs beats a
uniform mixture at MATCHED total token budget on held-out data from every domain -- and, separately,
the optimizer really does discover that an informative domain deserves most of the weight when the
alternative domains are pure noise. All training here is a real (tiny) transformer LM, not a mock.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.task.data_mixture import (  # noqa: E402
    SyntheticDomain,
    estimate_near_duplicate_rate,
    optimize_mixture,
    proxy_run_score,
)

VOCAB = 20
BLOCK = 16


def _difficulty_domains() -> list[SyntheticDomain]:
    """Four domains of increasing pattern difficulty (short/clean -> longer/noisier), all comfortably
    learnable within ``BLOCK`` tokens of context so difficulty comes from sample-efficiency, not from
    whether the pattern fits in the attention window at all."""
    return [
        SyntheticDomain(name="easy", vocab=VOCAB, period=2, noise_p=0.0, pattern_seed=1),
        SyntheticDomain(name="medium", vocab=VOCAB, period=3, noise_p=0.1, pattern_seed=2),
        SyntheticDomain(name="hard", vocab=VOCAB, period=4, noise_p=0.2, pattern_seed=3),
        SyntheticDomain(name="hardest", vocab=VOCAB, period=6, noise_p=0.35, pattern_seed=4),
    ]


class SyntheticDomainTest(unittest.TestCase):
    def test_pure_noise_domain_is_iid_uniform(self):
        d = SyntheticDomain(name="noise", vocab=VOCAB, period=None)
        a = d.sample(2000, seed=0)
        b = d.sample(2000, seed=0)
        np.testing.assert_array_equal(a, b)  # deterministic given seed
        c = d.sample(2000, seed=1)
        self.assertFalse(np.array_equal(a, c))  # different seed, different draw
        # roughly uniform over the vocab
        counts = np.bincount(a, minlength=VOCAB)
        self.assertLess(counts.std() / counts.mean(), 0.3)

    def test_periodic_domain_is_deterministic_without_noise(self):
        d = SyntheticDomain(name="clean", vocab=VOCAB, period=4, noise_p=0.0, pattern_seed=0)
        a = d.sample(40, seed=0)
        b = d.sample(40, seed=7)  # noise_p=0 -> no randomness at all, any seed matches
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[:4], a[4:8])  # repeats with the stated period


class OptimizerSanityTest(unittest.TestCase):
    """One domain is informative (a short, clean, learnable pattern); the rest are pure noise. A
    DoReMi search should discover this and push most of the weight onto the informative domain."""

    def test_optimizer_finds_the_informative_domain_bandit(self):
        domains = [
            SyntheticDomain(name="informative", vocab=VOCAB, period=2, noise_p=0.0, pattern_seed=0),
            SyntheticDomain(name="noise_a", vocab=VOCAB, period=None),
            SyntheticDomain(name="noise_b", vocab=VOCAB, period=None),
        ]
        weights = optimize_mixture(
            domains,
            proxy_steps=20,
            budget=10,
            method="bandit",
            proxy_kwargs={"batch_size": 16, "block": BLOCK, "eval_tokens": 256},
            seed=0,
        )
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertEqual(int(np.argmax(weights)), 0)  # the informative domain wins
        self.assertGreater(weights[0], 1.0 / len(domains) + 0.1)  # clearly above uniform's share

    def test_optimizer_finds_the_informative_domain_doe(self):
        domains = [
            SyntheticDomain(name="informative", vocab=VOCAB, period=2, noise_p=0.0, pattern_seed=0),
            SyntheticDomain(name="noise_a", vocab=VOCAB, period=None),
            SyntheticDomain(name="noise_b", vocab=VOCAB, period=None),
        ]
        weights = optimize_mixture(
            domains,
            proxy_steps=20,
            budget=10,
            method="doe",
            proxy_kwargs={"batch_size": 16, "block": BLOCK, "eval_tokens": 256},
            seed=0,
        )
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertEqual(int(np.argmax(weights)), 0)


class LearnedMixtureBeatsUniformTest(unittest.TestCase):
    """The F8 acceptance criterion: a mixture LEARNED via proxy runs beats UNIFORM weights at matched
    total token budget, measured as mean held-out NLL across all domains (lower is better)."""

    def test_learned_mixture_beats_uniform_at_matched_tokens(self):
        domains = _difficulty_domains()
        n = len(domains)

        # method="doe" (continuous simplex search via softmax-reparameterized BayesianOptimizer) --
        # unlike the discrete bandit-lattice arms in OptimizerSanityTest, this never drives a domain's
        # weight all the way to zero, which matters here: every domain in `_difficulty_domains` is
        # genuinely learnable and starving one outright (a real risk with coarse lattice arms) would
        # leave it stuck near the log(vocab) baseline and easily lose to uniform on aggregate loss.
        learned = optimize_mixture(
            domains,
            proxy_steps=40,
            budget=16,
            method="doe",
            proxy_kwargs={"batch_size": 16, "block": BLOCK, "eval_tokens": 384},
            seed=0,
        )
        uniform = np.full(n, 1.0 / n)

        # the "real" run: same total token budget (proxy_steps * batch_size, matched) for both mixtures.
        final_kwargs = dict(proxy_steps=40, batch_size=16, block=BLOCK, eval_tokens=512, seed=42)

        learned_loss, learned_detail = proxy_run_score(learned, domains, return_detail=True, **final_kwargs)
        uniform_loss, uniform_detail = proxy_run_score(uniform, domains, return_detail=True, **final_kwargs)

        print("\n[F8] learned mixture:", dict(zip((d.name for d in domains), learned.round(3))))
        print("[F8] uniform mixture:", dict(zip((d.name for d in domains), uniform.round(3))))
        print("[F8] learned per-domain held-out NLL:", {k: round(v, 4) for k, v in learned_detail.items()})
        print("[F8] uniform per-domain held-out NLL:", {k: round(v, 4) for k, v in uniform_detail.items()})
        print(f"[F8] aggregate held-out NLL -- learned: {learned_loss:.4f}  uniform: {uniform_loss:.4f}")

        self.assertLess(learned_loss, uniform_loss)


class NearDuplicateReceiptTest(unittest.TestCase):
    def test_planted_duplicates_are_detected(self):
        base = "the quick brown fox jumps over the lazy dog near the river bank at dawn"
        corpus = [
            base,
            base,  # exact duplicate
            base.replace("lazy", "sleepy"),  # near-duplicate (one word changed)
            "completely unrelated sentence about quarterly revenue and inventory forecasts",
            "another unrelated sentence discussing orbital mechanics and rocket propulsion systems",
        ]
        rate = estimate_near_duplicate_rate(corpus, shingle_size=3, num_hashes=64, threshold=0.5, seed=0)
        # 3 of 5 documents (the base + its exact and near duplicate) have a near-duplicate partner
        self.assertAlmostEqual(rate, 3.0 / 5.0, delta=1.0e-9)

    def test_no_duplicates_gives_zero_rate(self):
        corpus = [
            "alpha beta gamma delta epsilon zeta eta theta",
            "quarterly revenue grew steadily across every region this year",
            "the rocket achieved stable orbit after a nominal ascent profile",
            "a slow cooker recipe for lentil soup with cumin and lemon",
        ]
        rate = estimate_near_duplicate_rate(corpus, shingle_size=3, num_hashes=64, threshold=0.9, seed=0)
        self.assertEqual(rate, 0.0)

    def test_single_document_corpus_has_zero_rate(self):
        self.assertEqual(estimate_near_duplicate_rate(["only one document here"]), 0.0)
        self.assertEqual(estimate_near_duplicate_rate([]), 0.0)

    def test_invalid_minhash_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "shingle_size"):
            estimate_near_duplicate_rate(["one", "two"], shingle_size=0)
        with self.assertRaisesRegex(ValueError, "num_hashes"):
            estimate_near_duplicate_rate(["one", "two"], num_hashes=0)
        with self.assertRaisesRegex(ValueError, "threshold"):
            estimate_near_duplicate_rate(["one", "two"], threshold=float("nan"))


if __name__ == "__main__":
    unittest.main()
