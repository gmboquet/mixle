"""Durability + edge-case hardening for the task/calibrate/quantize spine (Group F3).

Three regressions this locks down:

1. ``solve``/``CalibratedTaskModel`` with a small calibration set (or tight ``alpha``) yields ``qhat=+inf``
   -- a real, callable threshold (every input escalates). Saving previously nulled it to ``None``, so the
   RELOADED model raised ``RuntimeError('call calibrate')`` on every call. The fix round-trips ``+inf``.
2. ``batch([])`` / empty inputs return ``[]`` uniformly across the IO classes instead of a numpy reshape
   ``ValueError`` (``cannot reshape array of size 0 into shape (0, newaxis)``).
3. ``quantize_mlp(..., clip_percentile=...)`` keeps a heavy-tailed layer non-degenerate: one int4 outlier
   weight no longer collapses the quantized layer to a handful of nonzero weights.
"""

import os
import tempfile
import unittest

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mixle.task.calibrate import (  # noqa: E402
    ESCALATE,
    CalibratedTaskModel,
    _qhat_from_json,
    _qhat_to_json,
)
from mixle.task.distill import distill, distill_from_labels, distill_records_from_labels  # noqa: E402


def _easy_corpus(n_per_class=30, seed=0):
    """Two linearly-separable classes: no token overlap, so the student is highly confident."""
    rng = np.random.RandomState(seed)
    a_words = ["alpha", "apex", "acorn", "amber"]
    b_words = ["bravo", "basalt", "birch", "bronze"]
    texts, labels = [], []
    for words, lab in ((a_words, "a"), (b_words, "b")):
        for _ in range(n_per_class):
            toks = list(rng.choice(words, size=rng.randint(3, 6)))
            texts.append(" ".join(toks))
            labels.append(lab)
    order = rng.permutation(len(texts))
    return [texts[i] for i in order], [labels[i] for i in order]


def _teacher_ab(texts):
    return ["a" if any(w.startswith("a") or w in ("apex", "acorn", "amber") for w in t.split()) else "b" for t in texts]


# --- Fix 1: qhat = +inf survives save/load and the reloaded model stays callable ----------------------------


class QhatInfinityRoundTripTest(unittest.TestCase):
    def test_sentinel_helpers_round_trip_inf(self):
        self.assertEqual(_qhat_to_json(float("inf")), "inf")
        self.assertEqual(_qhat_from_json("inf"), float("inf"))
        self.assertIsNone(_qhat_to_json(None))
        self.assertIsNone(_qhat_from_json(None))
        self.assertAlmostEqual(_qhat_from_json(_qhat_to_json(0.37)), 0.37)
        # a bare non-finite float (e.g. legacy Infinity round-trip) also maps back to inf
        self.assertEqual(_qhat_from_json(float("inf")), float("inf"))

    def _small_cal_model(self, alpha=0.01, seed=0):
        train_x, train_y = _easy_corpus(n_per_class=40, seed=seed)
        student = distill(_teacher_ab, train_x, n=3, dim=256, hidden=[32], epochs=200, lr=1e-2, seed=0)
        cal_x, cal_y = _easy_corpus(n_per_class=30, seed=seed + 100)  # ~60 easy examples
        return CalibratedTaskModel(student, alpha=alpha).calibrate(cal_x, cal_y), (cal_x, cal_y)

    def test_small_cal_tight_alpha_gives_infinite_qhat(self):
        model, _ = self._small_cal_model(alpha=0.01)
        self.assertTrue(np.isinf(model.qhat))  # ceil((n+1)(1-alpha)) > n -> conformal quantile is +inf

    def test_infinite_qhat_survives_save_load_and_model_stays_callable(self):
        model, _ = self._small_cal_model(alpha=0.01)
        self.assertTrue(np.isinf(model.qhat))
        test_x, _ = _easy_corpus(n_per_class=10, seed=999)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cal")
            model.save(path)
            loaded = CalibratedTaskModel.load(path)
            # the bug: reloaded qhat was None -> predict_sets raised RuntimeError('call calibrate')
            self.assertTrue(np.isinf(loaded.qhat))
            live = loaded.decide(test_x[0])  # a LIVE call must not raise
            self.assertIs(live, ESCALATE)  # inf threshold admits every label -> honest escalate
            self.assertEqual(loaded.batch_decide(test_x), model.batch_decide(test_x))

    def test_manifest_persists_a_json_safe_sentinel_not_none(self):
        model, _ = self._small_cal_model(alpha=0.01)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cal")
            model.save(path)
            from mixle.task import artifact as _artifact

            cal = _artifact.read_manifest(path).meta["calibration"]
        self.assertEqual(cal["qhat"], "inf")  # explicit sentinel, not None (which would break reload)

    def test_finite_qhat_still_round_trips_exactly(self):
        train_x, train_y = _easy_corpus(n_per_class=120, seed=3)
        student = distill(_teacher_ab, train_x, n=3, dim=256, hidden=[32], epochs=200, lr=1e-2, seed=0)
        cal_x, cal_y = _easy_corpus(n_per_class=120, seed=77)
        model = CalibratedTaskModel(student, alpha=0.1).calibrate(cal_x, cal_y)
        self.assertTrue(np.isfinite(model.qhat))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cal")
            model.save(path)
            loaded = CalibratedTaskModel.load(path)
        self.assertAlmostEqual(loaded.qhat, model.qhat, places=9)


# --- Fix 2: batch([]) returns [] uniformly across the IO classes ---------------------------------------------


class EmptyBatchTest(unittest.TestCase):
    def test_text_classifier_empty_batch(self):
        texts, labels = _easy_corpus(n_per_class=60, seed=0)
        student = distill_from_labels(texts, labels, dim=128, hidden=[16], epochs=40, lr=1e-2, seed=0)
        self.assertEqual(student.batch([]), [])
        self.assertEqual(student.adapter.logits_batch(student.model, []).shape, (0, len(student.adapter.labels)))
        self.assertEqual(student.adapter.proba_batch(student.model, []).shape, (0, len(student.adapter.labels)))

    def test_record_classifier_empty_batch(self):
        rng = np.random.RandomState(0)
        recs = [(float(rng.normal()), "p" if rng.random() < 0.5 else "q") for _ in range(200)]
        labels = ["a" if r[1] == "p" else "b" for r in recs]
        student = distill_records_from_labels(recs, labels, dim=64, hidden=[8], epochs=40, lr=1e-2, seed=0)
        self.assertEqual(student.batch([]), [])

    def test_quantized_classifier_empty_batch(self):
        from mixle.task import quantize_mlp

        rng = np.random.RandomState(1)
        recs = [(float(rng.normal()), "p" if rng.random() < 0.5 else "q") for _ in range(200)]
        labels = ["a" if r[1] == "p" else "b" for r in recs]
        fp32 = distill_records_from_labels(recs, labels, dim=64, hidden=[8], epochs=40, lr=1e-2, seed=0)
        for q in (quantize_mlp(fp32, bits=8), quantize_mlp(fp32, bits=4)):
            self.assertEqual(q.batch([]), [])
            self.assertEqual(q.adapter.logits_batch(q.model, []).shape, (0, len(q.adapter.labels)))

    def test_structured_classifier_empty_batch(self):
        from mixle.task import distill_structured_from_labels

        rng = np.random.RandomState(2)
        recs = [(("red", "green", "blue")[rng.randint(3)], ("s", "l")[rng.randint(2)]) for _ in range(300)]
        labels = ["a" if r[0] == "red" else "b" for r in recs]
        student = distill_structured_from_labels(recs, labels, seed=0)
        self.assertEqual(student.batch([]), [])
        self.assertEqual(student.adapter.logits_batch(student.model, []).shape, (0, len(student.adapter.labels)))
        self.assertEqual(student.adapter.proba_batch(student.model, []).shape, (0, len(student.adapter.labels)))

    def test_lns_classifier_empty_batch(self):
        from mixle.task import distill_structured_from_labels, lns_classifier

        rng = np.random.RandomState(3)
        recs = [(("red", "green", "blue")[rng.randint(3)], ("s", "l")[rng.randint(2)]) for _ in range(300)]
        labels = ["a" if r[0] == "red" else "b" for r in recs]
        lns = lns_classifier(distill_structured_from_labels(recs, labels, seed=0), step=1e-2)
        self.assertEqual(lns.batch([]), [])
        self.assertEqual(lns.adapter.int_logits_batch(lns.model, []).shape, (0, len(lns.adapter.labels)))

    def test_generative_text_empty_batch(self):
        from mixle.task.generative_text import distill_text_generative_from_labels

        texts, labels = _easy_corpus(n_per_class=60, seed=4)
        student = distill_text_generative_from_labels(texts, labels, min_count=1)
        self.assertEqual(student.batch([]), [])
        self.assertEqual(student.adapter.logits_batch(student.model, []).shape, (0, len(student.adapter.labels)))
        self.assertEqual(student.adapter.proba_batch(student.model, []).shape, (0, len(student.adapter.labels)))

    def test_extraction_empty_batch(self):
        from mixle.task.extract import distill_extractor

        def teacher(texts):
            out = []
            for t in texts:
                import re

                m = re.search(r"\d+", t)
                out.append({"id": m.group(0)} if m else {})
            return out

        texts = [f"order {i} shipped" for i in range(40)]
        student = distill_extractor(teacher, texts, ["id"], epochs=10)
        self.assertEqual(student.batch([]), [])

    def test_calibrated_model_empty_batch(self):
        texts, labels = _easy_corpus(n_per_class=120, seed=5)
        student = distill(_teacher_ab, texts, n=3, dim=256, hidden=[32], epochs=150, lr=1e-2, seed=0)
        cal_x, cal_y = _easy_corpus(n_per_class=120, seed=55)
        model = CalibratedTaskModel(student, alpha=0.1).calibrate(cal_x, cal_y)
        self.assertEqual(model.predict_sets([]), [])
        self.assertEqual(model.batch_decide([]), [])
        self.assertEqual(model.escalation_rate([]), 0.0)


# --- Fix 3: heavy-tailed weights stay non-degenerate under clip_percentile -----------------------------------


class QuantizeOutlierRobustnessTest(unittest.TestCase):
    def _student_with_injected_outlier(self, seed=0):
        import torch

        rng = np.random.RandomState(seed)
        recs = [(float(rng.normal()), "p" if rng.random() < 0.5 else "q") for _ in range(240)]
        labels = ["a" if (r[1] == "p") == (r[0] > 0) else "b" for r in recs]
        fp32 = distill_records_from_labels(recs, labels, dim=128, hidden=[16], epochs=120, lr=1e-2, seed=0)
        # inject one heavy-tailed outlier weight into the first Linear layer
        lin = next(m for m in fp32.model.modules() if type(m).__name__ == "Linear")
        with torch.no_grad():
            typical = float(lin.weight.abs().median())
            lin.weight[0, 0] = 40.0 * max(typical, 1e-3)  # ~40x the bulk scale
        return fp32, recs, labels

    def test_int4_outlier_collapses_the_naive_quantized_layer(self):
        from mixle.task import quantize_mlp

        fp32, _, _ = self._student_with_injected_outlier()
        q_naive = quantize_mlp(fp32, bits=4)  # default max-scaling: outlier dictates the scale
        w0 = q_naive.model.layers[0][0]
        nonzero_frac = float(np.mean(w0 != 0))
        self.assertLess(nonzero_frac, 0.10)  # the pathology: nearly the whole layer flattens to 0

    def test_clip_percentile_keeps_the_layer_non_degenerate(self):
        from mixle.task import quantize_mlp

        fp32, _, _ = self._student_with_injected_outlier()
        q = quantize_mlp(fp32, bits=4, clip_percentile=99.0)
        w0 = q.model.layers[0][0]
        nonzero_frac = float(np.mean(w0 != 0))
        self.assertGreater(nonzero_frac, 0.5)  # the bulk of the distribution keeps its resolution
        self.assertLessEqual(int(np.abs(w0).max()), 7)  # still a valid int4 layer

    def test_clip_percentile_defaults_to_bit_identical_max_scaling(self):
        # on WELL-BEHAVED weights (no injected outlier) percentile=100 must equal the default max path
        from mixle.task import quantize_mlp

        rng = np.random.RandomState(7)
        recs = [(float(rng.normal()), "p" if rng.random() < 0.5 else "q") for _ in range(240)]
        labels = ["a" if (r[1] == "p") == (r[0] > 0) else "b" for r in recs]
        fp32 = distill_records_from_labels(recs, labels, dim=128, hidden=[16], epochs=120, lr=1e-2, seed=0)
        q_default = quantize_mlp(fp32, bits=8)
        q_p100 = quantize_mlp(fp32, bits=8, clip_percentile=100.0)
        for (w0, s0, _), (w1, s1, _) in zip(q_default.model.layers, q_p100.model.layers):
            np.testing.assert_array_equal(w0, w1)
            self.assertEqual(s0, s1)

    def test_clip_percentile_bounds_are_validated(self):
        from mixle.task import quantize_mlp

        fp32, _, _ = self._student_with_injected_outlier()
        for bad in (0.0, -1.0, 100.1, 150.0):
            with self.assertRaises(ValueError):
                quantize_mlp(fp32, bits=4, clip_percentile=bad)

    def test_clipped_int4_roundtrips_through_the_artifact(self):
        from mixle.task import TaskModel, quantize_mlp

        fp32, recs, _ = self._student_with_injected_outlier()
        q = quantize_mlp(fp32, bits=4, clip_percentile=99.0)
        with tempfile.TemporaryDirectory() as d:
            q.save(d)
            loaded = TaskModel.load(d)
        self.assertEqual(loaded.batch(recs[:40]), q.batch(recs[:40]))
        self.assertEqual(loaded.meta["quantized"]["clip_percentile"], 99.0)


if __name__ == "__main__":
    unittest.main()
