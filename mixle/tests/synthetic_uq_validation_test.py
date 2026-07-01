"""Rigorous validation of the LLM-UQ machinery against KNOWN ground truth (synthetic generative model).

This is deliberately not a rigged mock: a principled 'LLM' (true answer + hidden knowledge level +
each meaning realized by several surface phrasings + a confident-hallucination subset). The machinery
must RECOVER quantities it is never told. The tests assert the *large* effects that settle whether the
method works and where it fails:

  1. marginalize_meaning recovers the true P(meaning) it was never given;
  2. the answer-format confound is real under naive (first-token) clustering and removed by a proper
     semantic-equivalence relation (the exact bug that crippled a real-LLM run, and its fix);
  3. probability calibration genuinely lowers out-of-sample ECE at scale (it needs data — it did
     nothing on 39 real points, it works on ~1000);
  4. the honest limit: confident hallucination defeats consistency-based UQ (confidence AUC collapses
     to chance, and semantic entropy is LOW on confabulations) -- why fact-checking, not consistency,
     is needed for that failure mode.
"""

import unittest

import numpy as np

from mixle.inference import ProbabilityCalibrator, expected_calibration_error, marginalize_meaning
from mixle.inference.uncertainty import semantic_entropy

ANSWERS = [f"a{i}" for i in range(10)]
PHRASINGS = [("{m}", 0.4), ("the answer is {m}", 0.35), ("it is {m} .", 0.25)]


def _meaning_of(s):
    toks = set(str(s).split())
    return next((a for a in ANSWERS if a in toks), str(s))


_eq_proper = lambda a, b: _meaning_of(a) == _meaning_of(b)  # noqa: E731
_eq_naive = lambda a, b: (str(a).split() or [""])[0] == (str(b).split() or [""])[0]  # noqa: E731


def _auc(scores, y):
    pos, neg = np.sum(y == 1), np.sum(y == 0)
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(scores)) + 1.0
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg)


class SyntheticGroundTruthValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(7)
        cls.rng = rng
        n_q, n_samples = 900, 15
        cls.questions = {}
        for i in range(n_q):
            true = rng.choice(ANSWERS)
            if i < int(0.15 * n_q):
                cls.questions[f"q{i}"] = dict(true=true, p=0.0, hall=rng.choice([a for a in ANSWERS if a != true]))
            else:
                cls.questions[f"q{i}"] = dict(true=true, p=float(rng.uniform(0.12, 0.97)), hall=None)

        def sample_one(q):
            d = cls.questions[q]
            if d["hall"] is not None:
                m = d["hall"] if rng.random() < 0.9 else rng.choice(ANSWERS)
            else:
                m = d["true"] if rng.random() < d["p"] else rng.choice([a for a in ANSWERS if a != d["true"]])
            tmpls, w = zip(*PHRASINGS)
            return rng.choice(tmpls, p=np.array(w) / sum(w)).format(m=m)

        cls.sample_one = staticmethod(sample_one)
        rows = []
        for q, d in cls.questions.items():
            s = [sample_one(q) for _ in range(n_samples)]
            mp = marginalize_meaning(s, _eq_proper)
            mn = marginalize_meaning(s, _eq_naive)
            mode = _meaning_of(mp.representatives[int(np.argmax(mp.probs))])
            rows.append(
                dict(
                    cp=float(mp.probs.max()),
                    cn=float(mn.probs.max()),
                    correct=1.0 if mode == d["true"] else 0.0,
                    se=semantic_entropy(s, _eq_proper),
                    length=float(np.mean([len(x.split()) for x in s])),
                    hall=d["hall"] is not None,
                )
            )
        cls.cp = np.array([r["cp"] for r in rows])
        cls.cn = np.array([r["cn"] for r in rows])
        cls.correct = np.array([r["correct"] for r in rows])
        cls.se = np.array([r["se"] for r in rows])
        cls.length = np.array([r["length"] for r in rows])
        cls.hall = np.array([r["hall"] for r in rows])
        cls.gk = ~cls.hall  # genuine-knowledge questions

    def test_marginalization_recovers_true_pmeaning(self):
        q = next(q for q, d in self.questions.items() if d["hall"] is None and d["p"] > 0.85)
        big = [self.sample_one(q) for _ in range(6000)]
        rec = marginalize_meaning(big, _eq_proper)
        i = [_meaning_of(r) for r in rec.representatives].index(self.questions[q]["true"])
        self.assertAlmostEqual(rec.probs[i], self.questions[q]["p"], delta=0.03)  # recovers unseen truth

    def test_proper_equivalence_fixes_the_format_confound(self):
        auc_naive = _auc(self.cn[self.gk], self.correct[self.gk])
        auc_proper = _auc(self.cp[self.gk], self.correct[self.gk])
        self.assertGreater(auc_proper, 0.85)  # proper equivalence: strong signal
        self.assertLess(auc_naive, 0.75)  # naive first-token clustering: crippled (the shipped bug)
        self.assertGreater(auc_proper - auc_naive, 0.2)
        # the confound: naive confidence tracks answer length, proper confidence does not
        self.assertGreater(abs(np.corrcoef(self.length[self.gk], self.cn[self.gk])[0, 1]), 0.3)
        self.assertLess(abs(np.corrcoef(self.length[self.gk], self.cp[self.gk])[0, 1]), 0.2)

    def test_calibration_lowers_out_of_sample_ece_at_scale(self):
        idx = np.where(self.gk)[0]
        np.random.RandomState(1).shuffle(idx)
        tr, te = idx[: len(idx) // 2], idx[len(idx) // 2 :]
        cal = ProbabilityCalibrator("isotonic").fit(self.cp[tr], self.correct[tr])
        raw = float(expected_calibration_error(self.cp[te], self.correct[te]))
        calibrated = float(expected_calibration_error(cal.predict(self.cp[te]), self.correct[te]))
        self.assertLess(calibrated, raw * 0.5)  # calibration genuinely helps given enough data
        self.assertLess(calibrated, 0.08)

    def test_confident_hallucination_defeats_consistency_uq(self):
        # semantic entropy is LOW on confident confabulation -> it misses hallucinations
        self.assertLess(self.se[self.hall].mean(), self.se[self.gk].mean() * 0.6)
        self.assertGreater(self.cp[self.hall].mean(), 0.8)  # confabulations look confident
        # and including them collapses confidence's discrimination toward chance
        auc_without = _auc(self.cp[self.gk], self.correct[self.gk])
        auc_with = _auc(self.cp, self.correct)
        self.assertGreater(auc_without, 0.85)
        self.assertLess(auc_with, 0.65)  # the failure mode consistency-based UQ cannot fix


if __name__ == "__main__":
    unittest.main()
