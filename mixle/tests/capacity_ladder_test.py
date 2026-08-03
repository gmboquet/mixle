"""Capacity ladder (mixle.task.capacity): climb representation families, measure each rung's ceiling.

A paraphrase-style rule teacher is the load-bearing fixture: it labels by a sentiment *word*, and its
canonical vocabulary includes synonyms never shown to the student during training. A hashed character
n-gram featurizer cannot generalize across those synonyms (no shared n-grams); an embedding-head featurizer
given vectors that place synonyms near each other can. That gap is exactly what the ladder should surface.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mixle.task.capacity import (  # noqa: E402
    DEFAULT_RUNGS,
    KNOWN_RUNGS,
    WordEmbeddingFeaturizer,
    _prepare_split,
    _teacher_labels,
    capacity_ladder,
    climb_to,
)

_POS_TRAIN = ["good", "nice"]
_NEG_TRAIN = ["bad", "poor"]
_POS_UNSEEN = ["great", "lovely"]
_NEG_UNSEEN = ["terrible", "awful"]
_FILLER = ["the", "movie", "today", "was", "really", "quite", "very", "a", "this"]

_CANONICAL = {w: "positive" for w in (*_POS_TRAIN, *_POS_UNSEEN)}
_CANONICAL.update({w: "negative" for w in (*_NEG_TRAIN, *_NEG_UNSEEN)})


def _sentences(sentiment_words, n_per_word, rng):
    out = []
    for w in sentiment_words:
        for _ in range(n_per_word):
            k = rng.randint(2, 5)
            toks = list(rng.choice(_FILLER, size=k)) + [w]
            rng.shuffle(toks)
            out.append(" ".join(toks))
    return out


def _teacher(texts):
    labels = []
    for t in texts:
        words = t.lower().split()
        hit = next((w for w in words if w in _CANONICAL), None)
        labels.append(_CANONICAL[hit] if hit is not None else "negative")
    return labels


def _corpus():
    rng = np.random.RandomState(0)
    train_texts = _sentences(_POS_TRAIN, 40, rng) + _sentences(_NEG_TRAIN, 40, rng)
    rng.shuffle(train_texts)
    val_texts = _sentences(_POS_UNSEEN, 20, rng) + _sentences(_NEG_UNSEEN, 20, rng)
    rng.shuffle(val_texts)
    return train_texts, val_texts


def _word_vectors():
    vecs = {}
    for w in (*_POS_TRAIN, *_POS_UNSEEN):
        vecs[w] = [1.0, 0.0]
    for w in (*_NEG_TRAIN, *_NEG_UNSEEN):
        vecs[w] = [-1.0, 0.0]
    return vecs


class CapacityLadderTest(unittest.TestCase):
    def test_hashed_ngram_cannot_generalize_across_synonyms(self):
        train_texts, val_texts = _corpus()
        result = capacity_ladder(
            _teacher,
            train_texts,
            target=0.85,
            rungs=("hashed_ngram",),
            val_texts=val_texts,
            hidden=(32,),
            epochs=150,
            seed=0,
        )
        self.assertEqual(len(result.rungs), 1)
        self.assertIsNotNone(result.ceiling("hashed_ngram"))
        self.assertLess(result.ceiling("hashed_ngram"), 0.7)
        self.assertIsNone(result.winner)
        self.assertEqual(result.outcome, "capacity_ceiling_measured")
        self.assertTrue(result.fully_evaluated())

    def test_embedding_head_generalizes_with_synonym_vectors(self):
        train_texts, val_texts = _corpus()
        result = capacity_ladder(
            _teacher,
            train_texts,
            target=0.85,
            rungs=DEFAULT_RUNGS,
            val_texts=val_texts,
            word_vectors=_word_vectors(),
            hidden=(32,),
            epochs=150,
            seed=0,
        )
        self.assertEqual([r.rung for r in result.rungs], list(DEFAULT_RUNGS))
        self.assertGreaterEqual(result.ceiling("embedding_head"), 0.85)
        self.assertEqual(result.winner, "embedding_head")
        # the ladder picked the smallest rung meeting target, not just any rung that happens to meet it
        self.assertLess(result.ceiling("hashed_ngram"), result.ceiling("embedding_head"))

    def test_target_unmet_returns_honest_none_not_exception(self):
        train_texts, val_texts = _corpus()
        result = capacity_ladder(
            _teacher,
            train_texts,
            target=0.999,
            rungs=("hashed_ngram",),
            val_texts=val_texts,
            hidden=(32,),
            epochs=80,
            seed=0,
        )
        self.assertIsNone(result.winner)
        self.assertIsNotNone(result.ceiling("hashed_ngram"))

    def test_unbuilt_rung_is_skipped_with_a_note_not_a_crash(self):
        train_texts, val_texts = _corpus()
        result = capacity_ladder(
            _teacher,
            train_texts,
            target=0.85,
            rungs=("hashed_ngram", "strong_encoder"),
            val_texts=val_texts,
            hidden=(32,),
            epochs=80,
            seed=0,
        )
        stub = next(r for r in result.rungs if r.rung == "strong_encoder")
        self.assertIsNone(stub.score)
        self.assertIsNone(stub.model)
        self.assertTrue(stub.note)
        self.assertEqual(stub.status, "not_evaluated")
        self.assertEqual(result.outcome, "not_evaluated")
        self.assertFalse(result.fully_evaluated())

    def test_unknown_rung_name_raises(self):
        train_texts, val_texts = _corpus()
        with self.assertRaises(ValueError):
            capacity_ladder(
                _teacher,
                train_texts,
                target=0.85,
                rungs=("not_a_real_rung",),
                val_texts=val_texts,
            )

    def test_determinism_given_seed(self):
        train_texts, val_texts = _corpus()
        kwargs = dict(
            target=0.85,
            rungs=DEFAULT_RUNGS,
            val_texts=val_texts,
            word_vectors=_word_vectors(),
            hidden=(32,),
            epochs=100,
            seed=7,
        )
        r1 = capacity_ladder(_teacher, train_texts, **kwargs)
        r2 = capacity_ladder(_teacher, train_texts, **kwargs)
        self.assertEqual(r1.winner, r2.winner)
        for a, b in zip(r1.rungs, r2.rungs):
            self.assertEqual(a.score, b.score)

    def test_climb_to_returns_next_rung(self):
        self.assertEqual(climb_to("hashed_ngram"), "embedding_head")
        self.assertEqual(climb_to("embedding_head"), "strong_encoder")

    def test_climb_to_accepts_fault_like_object_and_rejects_ceiling(self):
        class _Fault:
            dominant = "small_lm"

        with self.assertRaises(ValueError):
            climb_to(_Fault())  # already at the top of KNOWN_RUNGS

        class _Fault2:
            rung = "hashed_ngram"

        self.assertEqual(climb_to(_Fault2()), "embedding_head")

    def test_climb_to_rejects_unknown_rung(self):
        with self.assertRaises(ValueError):
            climb_to("not_a_rung")
        self.assertEqual(KNOWN_RUNGS[-1], "small_lm")


class CapacityContractTest(unittest.TestCase):
    def test_unavailable_only_ladder_is_not_a_measured_ceiling(self):
        result = capacity_ladder(
            ["a", "b"],
            ["one", "two"],
            target=0.9,
            rungs=("strong_encoder",),
            val_texts=["three"],
            val_labels=["a"],
        )
        self.assertIsNone(result.winner)
        self.assertEqual(result.outcome, "not_evaluated")
        self.assertFalse(result.fully_evaluated())

    def test_rejects_misaligned_or_missing_labels(self):
        with self.assertRaises(ValueError):
            capacity_ladder(["a"], ["one", "two"], target=0.8, rungs=("strong_encoder",))
        with self.assertRaises(ValueError):
            capacity_ladder(
                ["a", "b"],
                ["one", "two"],
                target=0.8,
                rungs=("strong_encoder",),
                val_texts=["three"],
            )
        with self.assertRaises(ValueError):
            capacity_ladder(
                ["a", "b"],
                ["one", "two"],
                target=0.8,
                rungs=("strong_encoder",),
                val_texts=["three"],
                val_labels=["a", "b"],
            )
        with self.assertRaises(ValueError):
            capacity_ladder(
                lambda _texts: ["a"],
                ["one", "two"],
                target=0.8,
                rungs=("strong_encoder",),
            )

    def test_validates_target_split_rungs_and_embedding_family(self):
        base = dict(
            teacher_or_labels=["a", "b"],
            texts=["one", "two"],
            target=0.8,
            rungs=("strong_encoder",),
            val_texts=["three"],
            val_labels=["a"],
        )
        for target in (np.nan, np.inf, -0.1, 1.1):
            with self.assertRaises(ValueError):
                capacity_ladder(**{**base, "target": target})
        for calibration_frac in (np.nan, 0.0, 1.0):
            with self.assertRaises(ValueError):
                capacity_ladder(**base, calibration_frac=calibration_frac)
        for rungs in ((), ("embedding_head", "hashed_ngram"), ("hashed_ngram", "hashed_ngram")):
            with self.assertRaises(ValueError):
                capacity_ladder(**{**base, "rungs": rungs})
        for vectors in (
            {"a": [1.0, 2.0], "b": [1.0]},
            {"a": [1.0, np.nan]},
            {"a": []},
        ):
            with self.assertRaises(ValueError):
                capacity_ladder(**base, word_vectors=vectors)

    def test_embedding_featurizer_copies_and_validates_vectors(self):
        source = np.asarray([1.0, 0.0])
        featurizer = WordEmbeddingFeaturizer({"word": source}, dim=2)
        source[0] = 99.0
        np.testing.assert_array_equal(featurizer.vectors["word"], [1.0, 0.0])
        with self.assertRaises(ValueError):
            WordEmbeddingFeaturizer({"a": [1.0], "b": [1.0, 2.0]}, dim=2)


class SideEffectingTeacherInvocationCountTest(unittest.TestCase):
    """MXR-080-1895: the capacity fallback invoked a side-effecting teacher more than once per text.

    Reproduced: ``_teacher_labels`` discovers a teacher's calling convention by CALLING it, batch
    first. Hand it a per-item teacher and the whole list goes in as one argument, the answer is
    discarded, and every text is labelled again -- three texts produced four invocations. A metered or
    logging teacher is billed for the discarded probe and is called on an argument shape it was never
    written to accept.
    """

    @staticmethod
    def _counting_per_item(counter):
        def teacher(text):
            counter.append(text)
            return "positive" if "good" in str(text) else "negative"

        return teacher

    @staticmethod
    def _counting_batch(counter):
        def teacher(texts):
            counter.append(list(texts))
            return ["positive" if "good" in t else "negative" for t in texts]

        return teacher

    def test_a_declared_item_teacher_is_called_exactly_once_per_text(self):
        calls = []
        labels, batched = _teacher_labels(self._counting_per_item(calls), ["good a", "bad b"], batched=False)
        self.assertEqual(labels, ["positive", "negative"])
        self.assertFalse(batched)
        self.assertEqual(len(calls), 2)  # was 3: one discarded probe on the whole list, then two

    def test_a_declared_batch_teacher_is_called_exactly_once(self):
        calls = []
        labels, batched = _teacher_labels(self._counting_batch(calls), ["good a", "bad b"], batched=True)
        self.assertEqual(labels, ["positive", "negative"])
        self.assertTrue(batched)
        self.assertEqual(len(calls), 1)

    def test_auto_discovery_reports_the_shape_so_it_is_not_rediscovered(self):
        # The discovered shape is what lets _prepare_split label a second split without re-probing.
        calls = []
        _, batched = _teacher_labels(self._counting_per_item(calls), ["good a"], batched=None)
        self.assertFalse(batched)
        probe_cost = len(calls)
        calls.clear()
        _teacher_labels(self._counting_per_item(calls), ["bad b", "good c"], batched=batched)
        self.assertEqual(len(calls), 2)  # no second probe
        self.assertEqual(probe_cost, 2)  # the one unavoidable auto-discovery probe, documented

    def test_labelling_both_splits_costs_at_most_one_discovery_probe(self):
        # The end-to-end shape of the defect: _prepare_split labels train AND holdout, and used to
        # re-discover the convention for each, so a per-item teacher paid two discarded probes.
        calls = []
        teacher = self._counting_per_item(calls)
        _prepare_split(teacher, ["good a", "bad b"], ["good c", "bad d"], None, None, 0.3, 0)
        # 2 train + 2 holdout labels + exactly one discarded discovery probe.
        self.assertEqual(len(calls), 5)

    def test_a_teacher_declared_batch_that_is_not_batch_is_refused_not_retried(self):
        # Negative control against guard overreach in the other direction: a declared-batch teacher
        # returning a scalar is a contract violation, and re-calling it per text would both hide the
        # mistake and spend len(texts) more invocations doing it.
        calls = []
        with self.assertRaises(ValueError):
            _teacher_labels(self._counting_per_item(calls), ["good a", "bad b"], batched=True)
        self.assertEqual(len(calls), 1)  # the single declared batch call, no per-item retry

    def test_capacity_ladder_rejects_an_unknown_teacher_mode(self):
        with self.assertRaises(ValueError):
            capacity_ladder(
                _teacher,
                ["good one", "bad two"],
                target=0.8,
                rungs=("hashed_ngram",),
                teacher_mode="nonsense",
            )


if __name__ == "__main__":
    unittest.main()
