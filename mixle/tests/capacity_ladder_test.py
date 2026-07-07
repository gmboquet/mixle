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


if __name__ == "__main__":
    unittest.main()
