"""Smoke test for ``examples/calibrated_report_demo.py`` (workstream B2): the volume -> claims ->
calibrated report vertical runs end to end on small synthetic data, and the resulting report objects
correctly flag low-confidence claims as abstained and high-confidence claims as accepted.

Both outcomes are made deterministic, not probabilistic: the ambiguous volumes are planted with a much
fainter signature (see ``synthetic_volume(..., ambiguous=True)``), and both the ``CalibratedGenerator``
calibration and the demo's own per-example candidate draws are fully seeded, so a fixed probe pool always
produces the same accept/abstain outcomes -- searching that fixed pool for one of each (mirroring
``CascadeIntegrationTest`` in ``task_calibrated_generator_test.py``) is not flaky across runs.
"""

import sys
import unittest
from pathlib import Path

import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from calibrated_report_demo import (  # noqa: E402
    ABSTAIN,
    build_claim_report,
    build_records,
    build_shape_gate,
    claim_teacher,
    consistency_check,
    solve_structured,
)


class CalibratedReportDemoSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # fitting solve_structured + calibrating the shape gate is the expensive part of this test; every
        # test method reads the same fixture without mutating it, so build it once for the whole class.
        cls.train_records = build_records(n_per_shape=40, seed=0)
        cls.cal_records = build_records(n_per_shape=40, seed=1)
        cls.structured = solve_structured(
            claim_teacher, cls.train_records, tol={"brightness": 0.08}, alpha=0.1, seed=0, epochs=80
        )
        cls.shape_gate = build_shape_gate(cls.cal_records, alpha=0.1, seed=0)
        # a mix of clear and ambiguous volumes: half the probe pool is planted with a faint (ambiguous)
        # signature, so the shape claim has genuinely hard cases to abstain on, not decorative ones.
        cls.probe_records = build_records(n_per_shape=25, seed=2, ambiguous_fraction=0.5)
        cls.reports = [
            build_claim_report(i, r, cls.shape_gate, cls.structured) for i, r in enumerate(cls.probe_records)
        ]

    def test_schema_has_both_claim_kinds(self) -> None:
        self.assertEqual(self.structured.schema, {"shape": "categorical", "brightness": "numeric"})

    def test_at_least_one_shape_claim_abstains(self) -> None:
        # direct check against the gate itself: some ambiguous record must fail to clear the conformal
        # singleton threshold (empty or multi-candidate set), i.e. CalibratedGenerator.serve(...) is ABSTAIN.
        abstained = [r for r in self.probe_records if self.shape_gate.serve(r) is ABSTAIN]
        self.assertGreater(len(abstained), 0, "expected at least one ambiguous volume to abstain on the shape claim")

        report_abstentions = [r for r in self.reports if "shape" in r.abstained()]
        self.assertGreater(len(report_abstentions), 0, "report objects should surface the same abstentions")
        verdict = report_abstentions[0].verdicts["shape"]
        self.assertIsNone(verdict.value)  # an abstained claim never carries a guessed value
        self.assertTrue(verdict.detail)

    def test_at_least_one_claim_is_confidently_accepted(self) -> None:
        accepted_reports = [r for r in self.reports if r.accepted()]
        self.assertGreater(len(accepted_reports), 0)
        # the brightness claim is calibrated with a loose tol relative to the achievable qhat, so it should
        # answer locally for every volume -- a concrete "accepted, with a value" claim to check.
        self.assertTrue(self.structured.fields_num["brightness"].answers_locally)
        for r in self.reports:
            self.assertIn("brightness", r.accepted())
            self.assertIsInstance(r.accepted()["brightness"], float)

        # at least one report should have EVERY claim accepted (a clear, unambiguous volume).
        self.assertTrue(any(r.all_accepted for r in self.reports))

    def test_summary_and_report_shape(self) -> None:
        for r in self.reports:
            summary = r.summary()
            self.assertIn("shape=", summary)
            self.assertIn("brightness=", summary)
            for name in r.verdicts:
                self.assertIn(r.verdicts[name].status, ("accepted", "abstained"))
            # accepted() only ever contains claims marked accepted -- no silently-included guesses.
            self.assertEqual(set(r.accepted()), {n for n, v in r.verdicts.items() if v.status == "accepted"})

    def test_consistency_check_runs_and_reports_a_verdict(self) -> None:
        # optional learn_structure check (workstream B2, "optionally"): just asserts it runs cleanly and
        # returns an inspectable string -- not that it always recovers the edge on this tiny sample size.
        msg = consistency_check(self.train_records)
        self.assertIsInstance(msg, str)
        self.assertIn("learn_structure", msg)


if __name__ == "__main__":
    unittest.main()
