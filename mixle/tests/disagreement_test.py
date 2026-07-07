"""Disagreement gate + active-labeling shrink (mixle.task.disagreement), CARD D4-a.

The corpus is built so the student provably can't follow the teacher on one sub-region: it trains only on
a SEEN set of sentiment synonyms and is evaluated (region B) on a disjoint HELD-OUT set of synonyms never
seen in training -- a hashed bag-of-words student has no representation for genuinely novel tokens, so it
is near-chance there, while region A (the SEEN vocabulary, held out only as fresh examples of the same
words) is easy.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mixle.task.calibrate import CalibratedTaskModel  # noqa: E402
from mixle.task.disagreement import (  # noqa: E402
    UnionGate,
    fit_disagreement_gate,
    measure_disagreement_mass,
)
from mixle.task.distill import agreement, distill_from_labels  # noqa: E402

POS_SEEN = ["fantastic", "wonderful", "excellent", "superb", "stellar"]
POS_HELD = ["marvelous", "terrific", "outstanding", "exceptional", "phenomenal"]
NEG_SEEN = ["terrible", "awful", "dreadful", "atrocious", "abysmal"]
NEG_HELD = ["horrendous", "appalling", "lousy", "subpar", "deficient"]
FILLER = ["the", "movie", "was", "really", "quite", "so", "this", "book", "show", "honestly"]


def _teacher(texts):
    out = []
    for t in texts:
        toks = set(t.split())
        if toks & set(POS_SEEN + POS_HELD):
            out.append("positive")
        elif toks & set(NEG_SEEN + NEG_HELD):
            out.append("negative")
        else:
            out.append("negative")
    return out


def _make_texts(vocab_pos, vocab_neg, n_per_class=60, seed=0):
    rng = np.random.RandomState(seed)
    texts = []
    for vocab in (vocab_pos, vocab_neg):
        for _ in range(n_per_class):
            k = rng.randint(3, 6)
            toks = [rng.choice(vocab)] + list(rng.choice(FILLER, size=k))
            rng.shuffle(toks)
            texts.append(" ".join(toks))
    rng.shuffle(texts)
    return texts


class DisagreementMassTest(unittest.TestCase):
    def test_student_disagrees_far_more_on_held_out_synonyms_than_seen_ones(self):
        train = _make_texts(POS_SEEN, NEG_SEEN, n_per_class=80, seed=1)
        student = distill_from_labels(
            train, _teacher(train), labels=["positive", "negative"], n=2, dim=64, hidden=[16], epochs=60, seed=0
        )

        region_a = _make_texts(POS_SEEN, NEG_SEEN, n_per_class=40, seed=2)  # same vocab, fresh examples
        region_b = _make_texts(POS_HELD, NEG_HELD, n_per_class=40, seed=3)  # never-seen synonyms

        mass_a = measure_disagreement_mass(student, region_a, _teacher(region_a))
        mass_b = measure_disagreement_mass(student, region_b, _teacher(region_b))

        self.assertLess(mass_a, 0.15)
        self.assertGreater(mass_b, 0.3)
        self.assertGreater(mass_b, mass_a)


class DisagreementGateTest(unittest.TestCase):
    def _fit_student_and_gate(self, seed=0):
        train = _make_texts(POS_SEEN, NEG_SEEN, n_per_class=80, seed=seed)
        student = distill_from_labels(
            train, _teacher(train), labels=["positive", "negative"], n=2, dim=64, hidden=[16], epochs=60, seed=0
        )
        # fit the gate on a MIXED sample from both regions so it has examples of true agreement AND
        # disagreement to learn from
        gate_fit_texts = _make_texts(POS_SEEN, NEG_SEEN, n_per_class=30, seed=seed + 10) + _make_texts(
            POS_HELD, NEG_HELD, n_per_class=30, seed=seed + 20
        )
        gate = fit_disagreement_gate(
            student, gate_fit_texts, _teacher(gate_fit_texts), dim=64, hidden=[16], epochs=100, seed=0
        )
        return student, gate

    def test_gate_flags_the_held_out_region_far_more_than_the_seen_region(self):
        student, gate = self._fit_student_and_gate(seed=1)
        region_a = _make_texts(POS_SEEN, NEG_SEEN, n_per_class=40, seed=42)
        region_b = _make_texts(POS_HELD, NEG_HELD, n_per_class=40, seed=43)

        flag_rate_a = float(np.mean(gate.ood_mask(region_a)))
        flag_rate_b = float(np.mean(gate.ood_mask(region_b)))
        self.assertGreater(flag_rate_b, flag_rate_a)

    def test_cascade_with_the_gate_recovers_near_teacher_level_agreement(self):
        student, gate = self._fit_student_and_gate(seed=2)
        calibrated = CalibratedTaskModel(student, alpha=0.2, density_gate=gate).calibrate(
            _make_texts(POS_SEEN, NEG_SEEN, n_per_class=30, seed=77),
            _teacher(_make_texts(POS_SEEN, NEG_SEEN, n_per_class=30, seed=77)),
        )
        test = _make_texts(POS_SEEN, NEG_SEEN, n_per_class=20, seed=88) + _make_texts(
            POS_HELD, NEG_HELD, n_per_class=20, seed=89
        )
        truth = _teacher(test)
        served = []
        for t, y in zip(test, truth):
            decision = calibrated.decide(t)
            served.append(y if decision is None else decision)  # ESCALATE -> teacher answers correctly
        realized_agreement = float(np.mean([s == y for s, y in zip(served, truth)]))
        unaided_agreement = agreement(student, truth, test)
        self.assertGreater(realized_agreement, unaided_agreement)
        self.assertGreater(realized_agreement, 0.85)

    def test_union_gate_ors_two_gates(self):
        class _AlwaysFlag:
            def ood_mask(self, texts):
                return np.ones(len(texts), dtype=bool)

        class _NeverFlag:
            def ood_mask(self, texts):
                return np.zeros(len(texts), dtype=bool)

        union = UnionGate(_NeverFlag(), _AlwaysFlag())
        self.assertTrue(np.all(union.ood_mask(["a", "b", "c"])))
        union_none = UnionGate(_NeverFlag(), _NeverFlag())
        self.assertFalse(np.any(union_none.ood_mask(["a", "b"])))


class ActiveLabelingShrinksTheRegionTest(unittest.TestCase):
    def test_labeling_the_flagged_region_and_redistilling_shrinks_its_disagreement_mass(self):
        train = _make_texts(POS_SEEN, NEG_SEEN, n_per_class=80, seed=5)
        student = distill_from_labels(
            train, _teacher(train), labels=["positive", "negative"], n=2, dim=64, hidden=[16], epochs=60, seed=0
        )
        flagged_region = _make_texts(POS_HELD, NEG_HELD, n_per_class=40, seed=6)
        region_labels = _teacher(flagged_region)

        mass_before = measure_disagreement_mass(student, flagged_region, region_labels)
        self.assertGreater(mass_before, 0.3)

        # sample the disagreement region, label it with the teacher, re-distill including those labels
        augmented_texts = train + flagged_region
        augmented_labels = _teacher(train) + region_labels
        restudent = distill_from_labels(
            augmented_texts,
            augmented_labels,
            labels=["positive", "negative"],
            n=2,
            dim=64,
            hidden=[16],
            epochs=60,
            seed=0,
        )
        mass_after = measure_disagreement_mass(restudent, flagged_region, region_labels)

        self.assertLess(mass_after, mass_before)
        self.assertLess(mass_after, 0.1)


if __name__ == "__main__":
    unittest.main()
