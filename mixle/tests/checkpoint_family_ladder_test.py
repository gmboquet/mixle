"""Acceptance tests for mixle.task.checkpoint_family_ladder (roadmap J2: checkpoint -> family ladder).

1. ``FamilyLadderAcceptanceTest`` -- the core J2 acceptance criterion: build a real small headline model
   (trained, not random, so it has genuine capability compression could degrade), run
   ``build_checkpoint_family`` through four decreasing target sizes, confirm EVERY rung stayed within its
   stated eval budget of the PREVIOUS rung (F10's own ``track_regression``, reference="prior"), and confirm
   the TOTAL real calibration samples spent across the whole ladder is small and measured -- reported
   against what full sampling-KD at every rung would have cost.
2. ``EvalBudgetViolationTest`` -- an artificially strict eval budget on one rung is correctly detected and
   halts the ladder, rather than silently accepted.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.eval_harness import markov_transition_matrix
from mixle.models.transformer import build_causal_lm
from mixle.task.checkpoint_family_ladder import RungSpec, build_checkpoint_family, count_params


def _train_headline_model(
    seed: int, vocab: int, d_model: int, n_layer: int, n_head: int, block: int, steps: int = 8000
):
    """A real (not random-init) small headline model, trained round-robin on the SAME FOUR task formats
    F10's own suite scores (mirroring ``mixle/models/eval_harness.py``'s private ``_held_out_perplexity_task``
    /``_arithmetic_task``/``_parity_task``/``_induction_task`` generators exactly -- same vocab slot layout,
    same modulus, same bit length, same plant-position scheme), so the headline model has genuine,
    ABOVE-CHANCE capability on every F10 axis. This matters for regression tracking specifically: a
    never-trained axis scores near its chance floor, where finite-example accuracy is highly quantized
    (1/n_examples per flipped prediction) and a handful of prediction flips -- whether from real compression
    or plain model-to-model variation -- produces enormous RELATIVE swings unrelated to any genuine
    capability loss (confirmed empirically: an untrained axis showed 40%+ "regressions" under compression
    that vanished once the model was actually trained on that axis). Training every axis above chance is
    what makes "did this rung regress F10's eval scores" a meaningful question rather than sampling noise.
    """
    torch.manual_seed(seed)
    model = build_causal_lm(vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block)

    trans = markov_transition_matrix(vocab)
    rng = np.random.default_rng(seed + 1000)
    ctx_len = min(block, 16)
    modulus = min(vocab - 2, 10)
    plus_id, eq_id = vocab - 2, vocab - 1
    bit_len = min(block, 5)

    def batch_perplexity(bs: int):
        seqs = np.empty((bs, ctx_len), dtype=np.int64)
        cur = rng.integers(0, vocab, size=bs)
        seqs[:, 0] = cur
        for t in range(1, ctx_len):
            nxt = np.array([rng.choice(vocab, p=trans[c]) for c in cur])
            seqs[:, t] = nxt
            cur = nxt
        return seqs[:, :-1], seqs[:, -1]

    def batch_arithmetic(bs: int):
        a = rng.integers(0, modulus, size=bs)
        b = rng.integers(0, modulus, size=bs)
        target = (a + b) % modulus
        seq = np.stack([a, np.full(bs, plus_id), b, np.full(bs, eq_id)], axis=1)
        return seq, target

    def batch_parity(bs: int):
        bits = rng.integers(0, 2, size=(bs, bit_len))
        target = bits.sum(axis=1) % 2
        return bits, target

    def batch_induction(bs: int):
        seqs = rng.integers(0, vocab, size=(bs, ctx_len))
        a = rng.integers(0, vocab, size=bs)
        b = rng.integers(0, vocab, size=bs)
        b = np.where(b == a, (b + 1) % vocab, b)
        plant_pos = rng.integers(0, ctx_len - 3, size=bs)
        for i in range(bs):
            p = int(plant_pos[i])
            seqs[i, p] = a[i]
            seqs[i, p + 1] = b[i]
            seqs[i, ctx_len - 1] = a[i]
        return seqs, b

    batchers = (batch_perplexity, batch_arithmetic, batch_parity, batch_induction)
    opt = torch.optim.Adam(model.parameters(), lr=4e-3)
    for step in range(steps):
        x_np, y_np = batchers[step % len(batchers)](64)
        x = torch.as_tensor(x_np.astype(np.int64))
        y = torch.as_tensor(y_np.astype(np.int64))
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    return model


class FamilyLadderAcceptanceTest(unittest.TestCase):
    """The core J2 acceptance criterion, on a real trained headline model."""

    def test_every_rung_within_budget_and_total_calibration_is_small(self):
        vocab, d_model, n_layer, n_head, block = 23, 16, 8, 2, 12
        model = _train_headline_model(seed=7, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block)

        calib_rng = np.random.RandomState(7)
        n_calib = 200
        calibration_data = torch.as_tensor(calib_rng.randint(0, vocab, size=(n_calib, block)), dtype=torch.long)

        # An increasing divergence budget per rung -> a decreasing (non-increasing) real model-size ladder,
        # standing in for the roadmap's real "70B -> 8B -> 1B -> edge" progression. See this module's own
        # docstring for why each rung compress()-es the ORIGINAL headline model rather than chaining.
        rung_specs = [
            RungSpec(name="rung_a", real_target="70B-equivalent stand-in", budget=0.1, trust_region=0.1, seed=0),
            RungSpec(name="rung_b", real_target="8B-equivalent stand-in", budget=0.3, trust_region=0.3, seed=0),
            RungSpec(name="rung_c", real_target="1B-equivalent stand-in", budget=0.6, trust_region=0.6, seed=0),
            RungSpec(name="rung_d", real_target="edge-equivalent stand-in", budget=2.0, trust_region=2.0, seed=0),
        ]

        result = build_checkpoint_family(model, rung_specs, calibration_data=calibration_data, eval_n_examples=1024)

        print(f"\n[J2 ladder] headline n_params = {result.headline_n_params}")
        print(f"[J2 ladder] headline scores = {result.headline_eval.scores()}")
        for rung in result.rungs:
            print(
                f"[J2 ladder] rung={rung.name!r} real_target={rung.real_target!r} "
                f"n_params={rung.n_params} ratio={rung.compression_ratio:.4f} "
                f"method={rung.compression_receipt.method!r} "
                f"calib_samples={rung.calibration_samples_spent} "
                f"within_budget={rung.within_eval_budget} reason={rung.reason!r}"
            )
            print(f"    scores = {rung.eval_report.scores()}")

        print(
            f"[J2 ladder] total_calibration_samples = {result.total_calibration_samples} "
            f"(pool={result.calibration_pool_size}, "
            f"full_kd_equivalent={result.full_kd_equivalent_samples()}, "
            f"fraction={result.total_calibration_fraction():.4%})"
        )

        # (a) every rung was actually attempted (none halted the ladder early) and stayed within budget.
        self.assertIsNone(result.halted_at)
        self.assertEqual(len(result.rungs), len(rung_specs))
        for rung in result.rungs:
            self.assertTrue(rung.within_eval_budget, rung.reason)
        self.assertEqual(result.passed_rungs(), [s.name for s in rung_specs])

        # real sizes were measured, and the ladder is genuinely non-increasing in size.
        n_params_sequence = [result.headline_n_params] + [r.n_params for r in result.rungs]
        for a, b in zip(n_params_sequence, n_params_sequence[1:]):
            self.assertLessEqual(b, a)
        self.assertLess(n_params_sequence[-1], n_params_sequence[0], "the ladder must shrink the model somewhere")

        # (b) total calibration data spent across the WHOLE ladder is small and measured -- report the
        # real number, and show it is a small fraction of what full sampling-KD at every rung would cost.
        self.assertGreater(result.calibration_pool_size, 0)
        self.assertGreaterEqual(result.total_calibration_samples, 0)
        self.assertLessEqual(result.total_calibration_fraction(), 0.02, "ladder should stay near J1's own <=1% bar")

    def test_count_params_is_real_and_shrinks_with_depth(self):
        torch.manual_seed(0)
        big = build_causal_lm(vocab=23, d_model=16, n_layer=8, n_head=2, block=12)
        small = build_causal_lm(vocab=23, d_model=16, n_layer=4, n_head=2, block=12)
        self.assertGreater(count_params(big), count_params(small))


class EvalBudgetViolationTest(unittest.TestCase):
    """An artificially strict eval budget must be DETECTED and halt the ladder, not silently accepted --
    mirrors F7's own GO/NO-GO halting behavior."""

    def test_impossibly_strict_budget_is_flagged_and_halts_the_ladder(self):
        # A threshold=0.0 budget is unsatisfiable regardless of model quality (any real compression
        # pass perturbs at least one score by a nonzero amount), so this rung does not need the full
        # multi-task training curriculum -- a handful of perplexity-only steps is enough for a real,
        # non-random model to exercise the same detection path much faster.
        vocab, d_model, n_layer, n_head, block = 23, 16, 8, 2, 12
        model = _train_headline_model(
            seed=11, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block, steps=300
        )

        calib_rng = np.random.RandomState(11)
        calibration_data = torch.as_tensor(calib_rng.randint(0, vocab, size=(200, block)), dtype=torch.long)

        rung_specs = [
            # threshold=0.0 -- ANY nonzero relative movement on ANY task versus the headline is a
            # regression; a real compression pass essentially always perturbs at least one task's score
            # by a nonzero amount, so this rung is expected to be flagged, not passed.
            RungSpec(
                name="impossible_rung",
                real_target="deliberately-unachievable eval budget",
                budget=2.0,
                trust_region=2.0,
                seed=0,
                max_relative_eval_regression=0.0,
            ),
            # a second rung that would otherwise be perfectly fine -- must never be attempted because the
            # ladder halts at the first failing rung.
            RungSpec(name="never_reached_rung", real_target="unreachable", budget=0.1, trust_region=0.1, seed=0),
        ]

        result = build_checkpoint_family(model, rung_specs, calibration_data=calibration_data)

        print(f"\n[J2 NO-GO] halted_at = {result.halted_at!r}")
        print(f"[J2 NO-GO] rung reason = {result.rungs[0].reason!r}")
        print(f"[J2 NO-GO] rung flags = {result.rungs[0].regression_flags}")

        self.assertEqual(result.halted_at, "impossible_rung")
        self.assertEqual(len(result.rungs), 1, "the ladder must not attempt a rung after a NO-GO")
        self.assertFalse(result.rungs[0].within_eval_budget)
        self.assertGreater(len(result.rungs[0].regression_flags), 0)
        self.assertNotIn("never_reached_rung", [r.name for r in result.rungs])


if __name__ == "__main__":
    unittest.main()
