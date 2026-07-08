"""F10 general-capability eval harness: one-command-per-checkpoint receipts + regression tracking.

Three things are tested against a real (tiny) mixle.models.transformer.CausalLM, no cluster needed:

1. `evaluate_checkpoint` runs end-to-end and produces a complete, sane report.
2. `track_regression` flags a genuine, deliberately-injected regression beyond threshold, and does *not*
   flag a sequence with only noise-level fluctuation.
3. The suite's task axes actually discriminate: a model trained on the harness's own synthetic task
   families scores measurably better than a randomly-initialized model on every axis -- proof the evals
   are not vacuous / always-same-score.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.eval_harness import (  # noqa: E402
    EvalReport,
    TaskResult,
    evaluate_checkpoint,
    markov_transition_matrix,
    track_regression,
)
from mixle.models.transformer import build_causal_lm  # noqa: E402

VOCAB = 16
BLOCK = 8


def _tiny_model(seed: int = 0, d_model: int = 32, n_layer: int = 2, n_head: int = 2):
    torch.manual_seed(seed)
    return build_causal_lm(vocab=VOCAB, d_model=d_model, n_layer=n_layer, n_head=n_head, block=BLOCK)


class EvaluateCheckpointTest(unittest.TestCase):
    def test_one_command_produces_a_complete_sane_report(self):
        model = _tiny_model()
        report = evaluate_checkpoint(model, checkpoint_id="rung-0", seed=1, n_examples=64)

        self.assertIsInstance(report, EvalReport)
        self.assertEqual(report.checkpoint_id, "rung-0")
        names = {t.name for t in report.tasks}
        self.assertEqual(
            names, {"held_out_perplexity", "modular_arithmetic", "parity_reasoning", "in_context_induction"}
        )
        for t in report.tasks:
            self.assertTrue(np.isfinite(t.score), f"{t.name} score not finite: {t.score}")
            self.assertEqual(t.n_examples, 64)

        d = report.report()
        self.assertEqual(d["checkpoint_id"], "rung-0")
        self.assertEqual(len(d["tasks"]), 4)

    def test_rejects_undersized_model(self):
        model = build_causal_lm(vocab=4, d_model=8, n_layer=1, n_head=1, block=4)
        with self.assertRaises(ValueError):
            evaluate_checkpoint(model)

    def test_leaves_model_training_mode_unchanged(self):
        model = _tiny_model()
        model.train()
        evaluate_checkpoint(model, n_examples=16)
        self.assertTrue(model.training)

        model.eval()
        evaluate_checkpoint(model, n_examples=16)
        self.assertFalse(model.training)

    def test_same_seed_is_reproducible(self):
        model = _tiny_model()
        r1 = evaluate_checkpoint(model, seed=7, n_examples=32)
        r2 = evaluate_checkpoint(model, seed=7, n_examples=32)
        self.assertEqual(r1.scores(), r2.scores())


def _make_report(checkpoint_id: str, scores: dict[str, float], directions: dict[str, bool]) -> EvalReport:
    tasks = [
        TaskResult(name=name, score=score, higher_is_better=directions[name], n_examples=100)
        for name, score in scores.items()
    ]
    return EvalReport(checkpoint_id=checkpoint_id, tasks=tasks, seed=0)


class RegressionTrackingTest(unittest.TestCase):
    _DIRECTIONS = {"accuracy_metric": True, "perplexity_metric": False}

    def test_flags_a_genuine_injected_regression(self):
        reports = [
            _make_report("rung-0", {"accuracy_metric": 0.80, "perplexity_metric": 5.0}, self._DIRECTIONS),
            _make_report("rung-1", {"accuracy_metric": 0.82, "perplexity_metric": 4.8}, self._DIRECTIONS),
            # deliberately corrupted/undertrained checkpoint: accuracy collapses, perplexity spikes
            _make_report("rung-2", {"accuracy_metric": 0.40, "perplexity_metric": 9.0}, self._DIRECTIONS),
        ]
        result = track_regression(reports, threshold=0.10)

        self.assertTrue(result.has_regressions)
        flagged_tasks = {f.task for f in result.flags if f.checkpoint_id == "rung-2"}
        self.assertEqual(flagged_tasks, {"accuracy_metric", "perplexity_metric"})

        acc_flag = next(f for f in result.flags if f.task == "accuracy_metric")
        self.assertEqual(acc_flag.reference_checkpoint_id, "rung-1")  # best-so-far, not just immediately-prior
        self.assertLess(acc_flag.relative_delta, -0.10)

    def test_noise_level_fluctuation_does_not_false_flag(self):
        rng = np.random.default_rng(42)
        base_acc, base_ppl = 0.75, 5.0
        reports = []
        for i in range(8):
            # +/- 2% relative noise, well under the 10% threshold
            acc = base_acc * (1.0 + rng.uniform(-0.02, 0.02))
            ppl = base_ppl * (1.0 + rng.uniform(-0.02, 0.02))
            reports.append(
                _make_report(f"rung-{i}", {"accuracy_metric": acc, "perplexity_metric": ppl}, self._DIRECTIONS)
            )

        result = track_regression(reports, threshold=0.10)
        self.assertFalse(result.has_regressions, result.report())

    def test_prior_reference_mode_only_compares_to_immediately_previous(self):
        # A slow monotonic decline that never drops >10% step-to-step should not flag under "prior",
        # even though checkpoint 2 is well below checkpoint 0 (which "best" mode would catch).
        reports = [
            _make_report("rung-0", {"accuracy_metric": 1.00}, {"accuracy_metric": True}),
            _make_report("rung-1", {"accuracy_metric": 0.95}, {"accuracy_metric": True}),
            _make_report("rung-2", {"accuracy_metric": 0.90}, {"accuracy_metric": True}),
        ]
        prior_result = track_regression(reports, threshold=0.10, reference="prior")
        self.assertFalse(prior_result.has_regressions)

        # A single sharp step-to-step drop of >10% must still be caught under "prior".
        reports_with_drop = reports + [_make_report("rung-3", {"accuracy_metric": 0.5}, {"accuracy_metric": True})]
        prior_result_2 = track_regression(reports_with_drop, threshold=0.10, reference="prior")
        self.assertTrue(prior_result_2.has_regressions)

    def test_rejects_invalid_reference_mode(self):
        with self.assertRaises(ValueError):
            track_regression([], reference="bogus")


# ---------------------------------------------------------------------------
# Discrimination: a trained toy model must score measurably better than random init on every axis.
# ---------------------------------------------------------------------------


def _batch_markov(rng: np.random.Generator, batch: int):
    trans = markov_transition_matrix(VOCAB)
    ctx_len = min(BLOCK, 16)
    seqs = np.empty((batch, ctx_len), dtype=np.int64)
    cur = rng.integers(0, VOCAB, size=batch)
    seqs[:, 0] = cur
    for t in range(1, ctx_len):
        nxt = np.array([rng.choice(VOCAB, p=trans[c]) for c in cur])
        seqs[:, t] = nxt
        cur = nxt
    return torch.as_tensor(seqs[:, :-1]), torch.as_tensor(seqs[:, -1])


def _batch_arithmetic(rng: np.random.Generator, batch: int):
    m = min(VOCAB - 2, 10)
    plus_id, eq_id = VOCAB - 2, VOCAB - 1
    a = rng.integers(0, m, size=batch)
    b = rng.integers(0, m, size=batch)
    target = (a + b) % m
    seq = np.stack([a, np.full(batch, plus_id), b, np.full(batch, eq_id)], axis=1)
    return torch.as_tensor(seq.astype(np.int64)), torch.as_tensor(target.astype(np.int64))


def _batch_parity(rng: np.random.Generator, batch: int):
    bit_len = min(BLOCK, 5)
    bits = rng.integers(0, 2, size=(batch, bit_len))
    target = bits.sum(axis=1) % 2
    return torch.as_tensor(bits.astype(np.int64)), torch.as_tensor(target.astype(np.int64))


def _batch_induction(rng: np.random.Generator, batch: int):
    ctx_len = min(BLOCK, 16)
    seqs = rng.integers(0, VOCAB, size=(batch, ctx_len))
    a = rng.integers(0, VOCAB, size=batch)
    b = rng.integers(0, VOCAB, size=batch)
    b = np.where(b == a, (b + 1) % VOCAB, b)
    plant_pos = rng.integers(0, ctx_len - 3, size=batch)
    for i in range(batch):
        p = int(plant_pos[i])
        seqs[i, p] = a[i]
        seqs[i, p + 1] = b[i]
        seqs[i, ctx_len - 1] = a[i]
    return torch.as_tensor(seqs.astype(np.int64)), torch.as_tensor(b.astype(np.int64))


# encoded exactly as the harness's own private task functions encode them (see mixle/models/eval_harness.py)
_TRAIN_BATCH_FNS = (_batch_markov, _batch_arithmetic, _batch_parity, _batch_induction)


def _train_step(model, opt, fn, rng, batch=64):
    x, y = fn(rng, batch)
    opt.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    opt.step()


class DiscriminationTest(unittest.TestCase):
    def test_trained_model_beats_random_init_on_every_axis(self):
        random_model = _tiny_model(seed=0)
        random_report = evaluate_checkpoint(random_model, checkpoint_id="random-init", seed=1, n_examples=256)

        trained_model = _tiny_model(seed=0, d_model=64, n_layer=3, n_head=4)
        opt = torch.optim.Adam(trained_model.parameters(), lr=4e-3)
        rng = np.random.default_rng(123)

        # Phase 1: warm up equally on the three fast-converging axes (markov, arithmetic, parity).
        for i in range(800):
            fn = _TRAIN_BATCH_FNS[i % 3]
            _train_step(trained_model, opt, fn, rng)

        # Phase 2: induction heads form via a slower "phase transition" -- weight training heavily
        # toward induction while lightly rehearsing the other three so they aren't forgotten.
        weights = [0.1, 0.1, 0.1, 0.7]
        for _ in range(5000):
            fn = rng.choice(_TRAIN_BATCH_FNS, p=weights)
            _train_step(trained_model, opt, fn, rng)

        trained_report = evaluate_checkpoint(trained_model, checkpoint_id="trained", seed=1, n_examples=256)

        random_scores = random_report.scores()
        trained_scores = trained_report.scores()
        chance = {t.name: t.details.get("chance_accuracy") for t in trained_report.tasks}

        # Perplexity: no fixed accuracy scale, so require a large relative drop instead of an absolute margin.
        self.assertLess(
            trained_scores["held_out_perplexity"],
            random_scores["held_out_perplexity"] * 0.5,
            "trained perplexity not measurably better than random init",
        )

        # Accuracy axes: an absolute margin over random init, sized per axis to its own difficulty/chance
        # level (parity's chance level is 0.5, so it needs a much bigger absolute margin than induction's
        # 1/vocab chance level to count as a measurable, non-noise improvement).
        margins = {"modular_arithmetic": 0.5, "parity_reasoning": 0.2, "in_context_induction": 0.15}
        for name, margin in margins.items():
            r, tr = random_scores[name], trained_scores[name]
            self.assertGreater(tr, r + margin, f"{name}: trained={tr} not measurably better than random={r}")
            # Sanity floor: the trained model must also be clearly above the task's own chance level, not
            # just above (a possibly-also-low) random init.
            self.assertGreater(tr, chance[name] + margin, f"{name}: trained={tr} not clearly above chance")


if __name__ == "__main__":
    unittest.main()
