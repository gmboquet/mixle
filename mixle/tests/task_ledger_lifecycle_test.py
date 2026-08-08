"""Lifecycle contract for the claim-bearing honesty ledgers (D-0157, hardening after RR13).

Three findings in this release were one defect -- a claim field written at save time and
reconstructed by hand, or not at all, at load time (STAT-R1's dropped ``tau``, STAT-RR13-1's
uncertified threshold reloading as certified, STAT-RR13-2's selection-reuse count resetting).
These tests are registry-driven: they iterate the SAME ledger declarations that ``save`` and
``load`` iterate, so a newly declared claim field is automatically round-trip-tested, corrupt
values are automatically refusal-tested, and the tests cannot go stale against the registry.

The ledger is a REQUIRED member of the artifact format (STAT-RR14-1): an artifact missing a
claim field has an unknown calibration history, and loading it under fresh-solve defaults
presented a reused-and-spent threshold as clean, single-use, certified evidence. Missing
fields refuse -- live-object initialization defaults are never missing-artifact semantics.
"""

import tempfile
import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

import pytest

from mixle.task._ledger import CLASSIFICATION_LEDGER, REGRESSION_LEDGER, read_ledger, write_ledger


def _route(t):
    if t["amount"] > 500 and t["kind"] == "refund":
        return "review"
    return "auto"


def _tickets(n, seed=0):
    rng = np.random.RandomState(seed)
    kinds = ["refund", "order", "chargeback"]
    return [{"kind": kinds[rng.randint(0, 3)], "amount": float(rng.uniform(0, 1000)), "region": "eu"} for _ in range(n)]


def _price(item):
    return {"refund": 20.0, "order": 80.0, "chargeback": 150.0}[item["kind"]] + 0.5 * item["amount"]


# Sentinels: for each field, a legal non-default value the round trip must preserve, and an
# illegal value the loader must refuse. Keyed by field name so a registry addition FAILS these
# tests until its sentinels are declared -- a new claim field cannot ship untested.
_SENTINELS = {
    "selection_uses": {"legal": 4, "illegal": -1},
    "calibration_evidence": {
        "classification": {"legal": "reused-after-adaptive-harvest", "illegal": "certified-by-vibes"},
        "regression": {"legal": "fresh-harvest", "illegal": "certified-by-vibes"},
    },
    # answered-slice measurement counts (STAT-RR16-2): non-negative integers; booleans and
    # negatives are refused (a bool True would silently count as 1)
    "sel_rows": {"legal": 25, "illegal": -1},
    "answered_sel_n": {"legal": 19, "illegal": -1},
    "answered_sel_correct": {"legal": 17, "illegal": True},
}


def _sentinel(field_name, shape, kind):
    entry = _SENTINELS[field_name]
    if shape in entry:
        entry = entry[shape]
    return entry[kind]


class LedgerRegistryContractTest(unittest.TestCase):
    """The registries themselves: every field has sentinels, and the helpers validate both ways."""

    def test_every_registry_field_declares_test_sentinels(self):
        for shape, ledger in (("classification", CLASSIFICATION_LEDGER), ("regression", REGRESSION_LEDGER)):
            for field in ledger:
                with self.subTest(shape=shape, field=field.name):
                    self.assertEqual(
                        field.validate(_sentinel(field.name, shape, "legal")), _sentinel(field.name, shape, "legal")
                    )
                    with self.assertRaises(ValueError):
                        field.validate(_sentinel(field.name, shape, "illegal"))

    def test_read_ledger_refuses_missing_and_corrupt_fields(self):
        # missing is refused, not defaulted: fresh-solve initialization values are what a NEW
        # object means, never what an artifact with an unrecorded history means (STAT-RR14-1)
        for shape, ledger in (("classification", CLASSIFICATION_LEDGER), ("regression", REGRESSION_LEDGER)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "missing the claim-bearing ledger field"):
                    read_ledger({}, ledger)
                complete = {f.name: _sentinel(f.name, shape, "legal") for f in ledger}
                self.assertEqual(set(read_ledger(complete, ledger)), {f.name for f in ledger})
                for field in ledger:
                    partial = dict(complete)
                    del partial[field.name]
                    with self.assertRaisesRegex(ValueError, field.name):
                        read_ledger(partial, ledger)
                    with self.assertRaises(ValueError):
                        read_ledger({**complete, field.name: _sentinel(field.name, shape, "illegal")}, ledger)

    def test_write_ledger_refuses_a_corrupt_live_object(self):
        class Holder:
            pass

        for shape, ledger in (("classification", CLASSIFICATION_LEDGER), ("regression", REGRESSION_LEDGER)):
            with self.subTest(shape=shape):
                holder = Holder()
                for field in ledger:
                    setattr(holder, field.name, _sentinel(field.name, shape, "legal"))
                self.assertEqual(set(write_ledger(holder, ledger)), {f.name for f in ledger})
                setattr(holder, ledger[0].name, _sentinel(ledger[0].name, shape, "illegal"))
                with self.assertRaises(ValueError):
                    write_ledger(holder, ledger)


@pytest.mark.torch
@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ClassificationLedgerLifecycleTest(unittest.TestCase):
    """Every registry field survives the classification artifact round trip, exactly."""

    @classmethod
    def setUpClass(cls):
        from mixle.task import solve

        cls.solution = solve(_route, _tickets(300), alpha=0.15, seed=0, epochs=120)

    def test_every_ledger_field_round_trips_and_reports(self):
        from mixle.task import Solution

        for field in CLASSIFICATION_LEDGER:
            setattr(self.solution, field.name, _sentinel(field.name, "classification", "legal"))
        with tempfile.TemporaryDirectory() as d:
            back = Solution.load(self.solution.save(d + "/router"), _route)
        report = back.report()
        for field in CLASSIFICATION_LEDGER:
            with self.subTest(field=field.name):
                expected = _sentinel(field.name, "classification", "legal")
                self.assertEqual(getattr(back, field.name), expected)
                if field.name in report:
                    self.assertEqual(report[field.name], expected)
        # the derived honesty flag follows the restored count, not a reset default
        self.assertFalse(back.selection_evidence_is_single_use)

    def test_load_refuses_a_corrupt_ledger(self):
        # A tampered manifest is already rejected by the integrity digest before the ledger is
        # even parsed (defense in depth -- verified below). The ledger validator's own refusal
        # path is for a WELL-FORMED artifact carrying an uninterpretable claim (a different or
        # future writer version), so the corrupt value is re-signed with a valid digest: the
        # refusal must then come from the ledger itself.
        import json
        from pathlib import Path

        from mixle.task import Solution
        from mixle.task.artifact import _manifest_integrity

        with tempfile.TemporaryDirectory() as d:
            path = self.solution.save(d + "/router")
            manifest_path = Path(path) / "manifest.json"
            original = manifest_path.read_text()
            for field in CLASSIFICATION_LEDGER:
                with self.subTest(field=field.name):
                    doc = json.loads(original)
                    doc["meta"]["solve"]["verification"][field.name] = _sentinel(
                        field.name, "classification", "illegal"
                    )
                    # unsigned tampering is caught by the integrity layer, not the ledger
                    manifest_path.write_text(json.dumps(doc))
                    with self.assertRaisesRegex(ValueError, "integrity"):
                        Solution.load(path, _route)
                    # a validly signed manifest with a corrupt claim is the ledger's to refuse
                    doc["integrity_sha256"] = _manifest_integrity(doc)
                    manifest_path.write_text(json.dumps(doc))
                    try:
                        with self.assertRaisesRegex(ValueError, "ledger"):
                            Solution.load(path, _route)
                    finally:
                        manifest_path.write_text(original)

    def test_load_refuses_a_validly_signed_artifact_missing_ledger_fields(self):
        # STAT-RR14-1's exact construction: a legacy-shaped artifact (well-formed, validly
        # signed, ledger fields ABSENT) reloaded as clean solve-split evidence with zero uses
        import json
        from pathlib import Path

        from mixle.task import Solution
        from mixle.task.artifact import _manifest_integrity

        with tempfile.TemporaryDirectory() as d:
            path = self.solution.save(d + "/router")
            manifest_path = Path(path) / "manifest.json"
            original = manifest_path.read_text()
            for field in CLASSIFICATION_LEDGER:
                with self.subTest(field=field.name):
                    doc = json.loads(original)
                    del doc["meta"]["solve"]["verification"][field.name]
                    doc["integrity_sha256"] = _manifest_integrity(doc)
                    manifest_path.write_text(json.dumps(doc))
                    try:
                        with self.assertRaisesRegex(ValueError, "missing the claim-bearing ledger field"):
                            Solution.load(path, _route)
                    finally:
                        manifest_path.write_text(original)
            # and an artifact with NO verification block at all refuses for the same reason
            doc = json.loads(original)
            del doc["meta"]["solve"]["verification"]
            doc["integrity_sha256"] = _manifest_integrity(doc)
            manifest_path.write_text(json.dumps(doc))
            try:
                with self.assertRaisesRegex(ValueError, "ledger is missing"):
                    Solution.load(path, _route)
            finally:
                manifest_path.write_text(original)


@pytest.mark.torch
@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class RegressionLedgerLifecycleTest(unittest.TestCase):
    """Every registry field survives the regression artifact round trip, exactly."""

    def test_every_ledger_field_round_trips_and_reports(self):
        from mixle.task import RegressionSolution, solve_regression

        solution = solve_regression(_price, _tickets(150, seed=2), tol=50.0, alpha=0.1, seed=0, epochs=80)
        for field in REGRESSION_LEDGER:
            setattr(solution, field.name, _sentinel(field.name, "regression", "legal"))
        with tempfile.TemporaryDirectory() as d:
            back = RegressionSolution.load(solution.save(d + "/pricer"), _price)
        report = back.report()
        for field in REGRESSION_LEDGER:
            with self.subTest(field=field.name):
                expected = _sentinel(field.name, "regression", "legal")
                self.assertEqual(getattr(back, field.name), expected)
                if field.name in report:
                    self.assertEqual(report[field.name], expected)


@pytest.mark.torch
@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class RegressionLedgerRefusalTest(unittest.TestCase):
    """The regression artifact refuses missing ledger fields under a valid signature too."""

    def test_load_refuses_a_validly_signed_artifact_missing_ledger_fields(self):
        import json
        from pathlib import Path

        from mixle.task import RegressionSolution, solve_regression
        from mixle.task.artifact import _manifest_integrity

        solution = solve_regression(_price, _tickets(150, seed=2), tol=50.0, alpha=0.1, seed=0, epochs=80)
        with tempfile.TemporaryDirectory() as d:
            path = solution.save(d + "/pricer")
            manifest_path = Path(path) / "manifest.json"
            original = manifest_path.read_text()
            for field in REGRESSION_LEDGER:
                with self.subTest(field=field.name):
                    doc = json.loads(original)
                    del doc["meta"]["regress"][field.name]
                    doc["integrity_sha256"] = _manifest_integrity(doc)
                    manifest_path.write_text(json.dumps(doc))
                    try:
                        with self.assertRaisesRegex(ValueError, "missing the claim-bearing ledger field"):
                            RegressionSolution.load(path, _price)
                    finally:
                        manifest_path.write_text(original)


@pytest.mark.torch
@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class StructuredCompositionLedgerTest(unittest.TestCase):
    """The composition shape inherits the ledger contract through its sub-loaders.

    solve_structured persists one sub-artifact per field through the SAME save/load pairs the
    scalar shapes use, so the required-ledger refusal must hold transitively: a structured
    artifact whose sub-field manifest lost its ledger refuses to load, and an intact round trip
    preserves every sub-solution's ledger fields.
    """

    def test_sub_solution_ledgers_round_trip_and_missing_ones_refuse(self):
        import json
        from pathlib import Path

        from mixle.task import StructuredSolution, solve_structured
        from mixle.task.artifact import _manifest_integrity

        def _enrich(t):
            return {"queue": "finance" if t["amount"] > 500 else "ops", "reserve": t["amount"] * 0.1}

        solution = solve_structured(_enrich, _tickets(150, seed=3), tol={"reserve": 1e6}, alpha=0.15, epochs=60, seed=0)
        cat_key = next(iter(solution.fields_cat))
        solution.fields_cat[cat_key].selection_uses = 4
        solution.fields_cat[cat_key].calibration_evidence = "reused-after-adaptive-harvest"
        with tempfile.TemporaryDirectory() as d:
            path = solution.save(d + "/enricher")
            back = StructuredSolution.load(path, _enrich)
            self.assertEqual(back.fields_cat[cat_key].selection_uses, 4)
            self.assertEqual(back.fields_cat[cat_key].calibration_evidence, "reused-after-adaptive-harvest")
            # strip one sub-field's ledger under a valid signature: the WHOLE structured load refuses
            sub_manifest = Path(path) / "cat" / cat_key / "manifest.json"
            doc = json.loads(sub_manifest.read_text())
            del doc["meta"]["solve"]["verification"]["calibration_evidence"]
            doc["integrity_sha256"] = _manifest_integrity(doc)
            sub_manifest.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "missing the claim-bearing ledger field"):
                StructuredSolution.load(path, _enrich)


@pytest.mark.torch
@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ReportReceiptCompletenessTest(unittest.TestCase):
    """Every reported measurement travels with its receipt (STAT-RR12-2's rule, held everywhere).

    A coverage-like proportion must carry its denominator and a named interval, and every shape
    whose report exposes a conformal quantity must say which regime certified it.
    """

    def test_regression_report_carries_denominator_interval_and_regime(self):
        from mixle.task import solve_regression

        report = solve_regression(_price, _tickets(150, seed=2), tol=50.0, alpha=0.1, seed=0, epochs=80).report()
        self.assertIn("selection_coverage", report)
        self.assertIn("selection_coverage_n", report)
        self.assertIn("selection_coverage_ci95", report)
        self.assertIn("selection_uses", report)
        self.assertIn("calibration_evidence", report)
        if report["selection_coverage"] is not None:
            low, high = report["selection_coverage_ci95"]
            self.assertLessEqual(0.0, low)
            self.assertLessEqual(low, report["selection_coverage"])
            self.assertLessEqual(report["selection_coverage"], high)
            self.assertLessEqual(high, 1.0)
            self.assertGreater(report["selection_coverage_n"], 0)

    def test_classification_report_carries_uses_flag_and_regime(self):
        from mixle.task import solve

        report = solve(_route, _tickets(300), alpha=0.15, seed=0, epochs=120).report()
        self.assertIn("selection_uses", report)
        self.assertIn("selection_evidence_is_single_use", report)
        self.assertIn("calibration_evidence", report)


if __name__ == "__main__":
    unittest.main()
