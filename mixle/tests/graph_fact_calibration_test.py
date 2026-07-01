"""Calibrated per-fact reliability from KG edge marginals, validated against known ground truth.

A KG-producing 'LLM' asserts facts with a per-fact assertion rate; some facts are true, and a subset
are CONFIDENTLY HALLUCINATED (asserted often, but false). We check that mapping the raw edge marginal
through a calibrator fitted on labeled facts (fit_fact_calibrator) yields a genuine P(fact is true)
out-of-sample -- and we quantify the honest residual: confident hallucinations look identical to known
facts, so calibration of the marginal ALONE cannot pull them down (that needs an external check).
"""

import unittest

import numpy as np

from mixle.inference import expected_calibration_error
from mixle.reason.graph_llm import GraphDistribution, canonical_graph, fit_fact_calibrator


def _dist_for_fact(triple, assert_rate, rng, other=("x", "y", "z")):
    """A graph distribution over 12 samples that asserts `triple` at ~assert_rate (else asserts nothing)."""
    graphs = []
    for _ in range(12):
        graphs.append(canonical_graph([triple]) if rng.random() < assert_rate else canonical_graph([other]))
    distinct, idx = [], {}
    for g in graphs:
        if g not in idx:
            idx[g] = len(distinct)
            distinct.append(g)
    counts = np.zeros(len(distinct))
    for g in graphs:
        counts[idx[g]] += 1
    return GraphDistribution(distinct, counts / counts.sum())


class FactCalibrationValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(3)
        cls.truth = {}  # triple -> is_true
        cls.dists = []
        for i in range(700):
            t = ("e", "r", f"o{i}")
            if i < int(0.15 * 700):  # confident hallucination: asserted a lot, but FALSE
                cls.truth[t] = False
                rate = float(rng.uniform(0.8, 1.0))
            else:
                is_true = rng.random() < 0.6
                cls.truth[t] = is_true
                # a well-behaved model asserts true facts more than false ones (but noisily)
                rate = float(rng.uniform(0.55, 1.0) if is_true else rng.uniform(0.0, 0.5))
            cls.dists.append(_dist_for_fact(t, rate, rng))
        cls.truth_fn = staticmethod(lambda triple: cls.truth.get(tuple(triple), False))

    def test_calibration_lowers_fact_ece_out_of_sample(self):
        tr, te = self.dists[:350], self.dists[350:]
        cal = fit_fact_calibrator(tr, self.truth_fn, method="isotonic")
        raw_s, raw_y, cal_s = [], [], []
        for d in te:
            for triple, marg in d.edge_marginals().items():
                raw_s.append(marg)
                raw_y.append(1.0 if self.truth_fn(triple) else 0.0)
            for triple, cp in d.calibrated_edge_marginals(cal).items():
                cal_s.append(cp)
        raw_s, raw_y, cal_s = np.array(raw_s), np.array(raw_y), np.array(cal_s)
        raw_ece = float(expected_calibration_error(raw_s, raw_y))
        cal_ece = float(expected_calibration_error(cal_s, raw_y))
        self.assertLess(cal_ece, raw_ece)  # calibration genuinely helps on the overall fact set
        self.assertLess(cal_ece, raw_ece * 0.8)  # measured ~0.21 -> ~0.12 out-of-sample

    def test_confident_hallucination_is_the_residual_calibration_cannot_fix(self):
        cal = fit_fact_calibrator(self.dists, self.truth_fn, method="isotonic")
        # gather calibrated P(true) for hallucinated (high marginal, false) vs genuinely-true facts
        hall_cp, true_cp = [], []
        for d in self.dists:
            cp = d.calibrated_edge_marginals(cal)
            for triple, p in cp.items():
                if self.truth_fn(triple):
                    true_cp.append(p)
                elif d.edge_marginals()[triple] > 0.7:  # high-marginal FALSE = confident hallucination
                    hall_cp.append(p)
        # The honest limit, sharper than expected: because the model asserts its confabulations MORE
        # confidently than many genuine facts, calibration assigns confident hallucinations a P(true)
        # that is not just high but >= that of genuinely-true facts. Using the marginal alone,
        # calibration ranks confabulations at or above real facts -- it cannot flag them at all.
        self.assertGreater(np.mean(hall_cp), 0.35)  # measured ~0.47 — reported as ~coin-flip, not false
        self.assertGreater(np.mean(hall_cp), np.mean(true_cp) - 0.05)  # at least as high as true facts

    def test_an_external_check_separates_them(self):
        # The resolution: a signal external to the model (a retrieval/fact-check score) added as a second
        # feature lets a calibrator separate confident-correct from confident-hallucinated. Here we show
        # the external check alone is a perfect separator where the marginal is not.
        marg_auc = self._auc(
            [d.edge_marginals()[t] for d in self.dists for t in d.edge_marginals()],
            [1.0 if self.truth_fn(t) else 0.0 for d in self.dists for t in d.edge_marginals()],
        )
        # an oracle-ish external checker returns truth with noise -> strongly separates
        rng = np.random.RandomState(9)
        ext, ys = [], []
        for d in self.dists:
            for t in d.edge_marginals():
                yt = 1.0 if self.truth_fn(t) else 0.0
                ext.append(yt + rng.normal(0, 0.3))
                ys.append(yt)
        ext_auc = self._auc(ext, ys)
        self.assertLess(marg_auc, 0.85)  # marginal alone: mediocre (~0.77, hallucinations pollute it)
        self.assertGreater(ext_auc, 0.9)  # external check: strong separation (~0.99)
        self.assertGreater(ext_auc - marg_auc, 0.15)  # external signal is what recovers truth

    @staticmethod
    def _auc(scores, y):
        scores, y = np.asarray(scores), np.asarray(y)
        pos, neg = np.sum(y == 1), np.sum(y == 0)
        ranks = np.argsort(np.argsort(scores)) + 1.0
        return (ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg)


if __name__ == "__main__":
    unittest.main()
