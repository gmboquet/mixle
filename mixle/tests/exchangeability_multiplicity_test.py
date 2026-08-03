"""The advertised alpha must be the report's error rate, not each test's (MXR-080-1887).

``exchangeability_check`` runs TWO permutation tests per numeric field and compared each raw p-value
to ``alpha`` independently, with any single rejection flipping the whole report. With k fields that is
2k chances to reject at level alpha each, so the aggregate error rate grows with the number of
columns and the advertised alpha describes nothing the caller reads.

Measured on genuinely exchangeable data with 20 columns: the uncorrected rule flagged a violation in
23 of 30 datasets. The whole primary family is now corrected together, once.
"""

import unittest

import numpy as np

from mixle.data.exchangeability import ExchangeabilityReport, exchangeability_check


def _exchangeable_rows(seed: int, columns: int = 20, rows: int = 120) -> list[tuple]:
    rs = np.random.RandomState(seed)
    return [tuple(rs.normal(size=columns)) for _ in range(rows)]


class FamilyWiseErrorTest(unittest.TestCase):
    def test_many_exchangeable_columns_do_not_manufacture_a_violation(self):
        flagged = 0
        trials = 12
        for seed in range(trials):
            report = exchangeability_check(_exchangeable_rows(seed), alpha=0.05, n_perm=120, seed=seed)
            if report.label != "exchangeable":
                flagged += 1
        # Uncorrected this was a large majority of trials; corrected it should be rare.
        self.assertLessEqual(flagged, 1, f"{flagged}/{trials} exchangeable datasets flagged")

    def test_the_uncorrected_comparison_is_still_reported_and_is_the_looser_one(self):
        # Both are published: the corrected p decided the verdict, the raw p is what a reader would
        # otherwise compare against alpha themselves and get a different answer.
        report = exchangeability_check(_exchangeable_rows(0), alpha=0.05, n_perm=120, seed=0)
        tested = [f for f in report.fields if f.get("verdict") != "invalid"]
        self.assertTrue(tested)
        for row in tested:
            self.assertLessEqual(row["trend_p_raw"], row["trend_p"])
            self.assertLessEqual(row["shift_p_raw"], row["shift_p"])

    def test_the_correction_is_recorded_because_the_label_is_read_against_it(self):
        report = exchangeability_check(_exchangeable_rows(0, columns=3), alpha=0.05, n_perm=120, seed=0)
        self.assertEqual(report.multiplicity["method"], "holm")
        self.assertEqual(report.multiplicity["family_size"], 6)  # 2 tests x 3 fields
        self.assertEqual(report.multiplicity["alpha"], 0.05)
        self.assertIn("multiplicity", report.as_dict())


class RealViolationTest(unittest.TestCase):
    """Correcting must not cost the check its power on a violation it exists to find."""

    def test_a_genuine_trend_is_still_detected(self):
        rs = np.random.RandomState(0)
        rows = [(float(i) + rs.normal() * 0.1,) for i in range(200)]
        self.assertEqual(exchangeability_check(rows, alpha=0.05, n_perm=200, seed=0).label, "trend")

    def test_a_genuine_shift_is_still_detected(self):
        rs = np.random.RandomState(1)
        rows = [(rs.normal() + (0.0 if i < 100 else 8.0),) for i in range(200)]
        self.assertIn(exchangeability_check(rows, alpha=0.05, n_perm=200, seed=0).label, ("shift", "trend"))


class ReportLabelTest(unittest.TestCase):
    def test_a_label_outside_the_closed_vocabulary_is_refused(self):
        with self.assertRaisesRegex(ValueError, "label must be one of"):
            ExchangeabilityReport(label="definitely-fine")

    def test_every_real_label_constructs(self):
        for label in ("exchangeable", "trend", "shift", "inconclusive"):
            with self.subTest(label=label):
                self.assertEqual(ExchangeabilityReport(label=label).label, label)

    def test_exchangeable_is_failure_to_reject_not_certification(self):
        # The property's contract, stated where a reader of the code will see it.
        self.assertIn("FAILURE TO REJECT", ExchangeabilityReport.exchangeable.__doc__)


if __name__ == "__main__":
    unittest.main()
