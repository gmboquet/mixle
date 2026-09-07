"""The rehearsal record that gates promotion is verified fail-closed in both of its forms.

``scripts/verify_rehearsal_record.py`` accepts the automated ``testpypi``-phase artifact, or a
manual record for the one situation the automated phase cannot pass (an earlier candidate of the
same version already occupies TestPyPI, whose bytes can never be replaced). Every binding and the
exemption's live precondition must hold; anything else is refused with a reason.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SHA = "9504451e19b4a2f1be65e38d37ce319c076dbd9c"
_SUMS = "aa" * 32 + "  mixle-0.8.0-py3-none-any.whl\n" + "bb" * 32 + "  mixle-0.8.0.tar.gz\n"


def _load():
    spec = importlib.util.spec_from_file_location(
        "verify_rehearsal_record", _ROOT / "scripts" / "verify_rehearsal_record.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _automated(**overrides):
    record = {
        "artifact": "mixle.testpypi_rehearsal/v1",
        "candidate_commit": _SHA,
        "prepare_run": 34042540432,
        "sums_sha256": hashlib.sha256(_SUMS.encode()).hexdigest(),
        "passed": True,
    }
    record.update(overrides)
    return record


def _manual(**overrides):
    record = _automated(
        artifact="mixle.testpypi_rehearsal_manual/v1",
        reason="version-already-on-testpypi-from-an-earlier-candidate",
        draft_release_files={"mixle-0.8.0-py3-none-any.whl": "aa" * 32, "mixle-0.8.0.tar.gz": "bb" * 32},
        testpypi_files={"mixle-0.8.0-py3-none-any.whl": "cc" * 32, "mixle-0.8.0.tar.gz": "dd" * 32},
        reproducible_builds_passed=True,
        clean_install=True,
        import_sweep_modules=809,
        verified_by="gmboquet",
        verified_at="2026-09-06T16:00:00+00:00",
    )
    record.update(overrides)
    return record


_TESTPYPI = {"mixle-0.8.0-py3-none-any.whl": "cc" * 32, "mixle-0.8.0.tar.gz": "dd" * 32}


class VerifyFunctionTest(unittest.TestCase):
    def setUp(self):
        self.module = _load()

    def _verify(self, record, *, manual=False, testpypi=_TESTPYPI, **kw):
        return self.module.verify(
            record,
            candidate_sha=kw.pop("candidate_sha", _SHA),
            prepare_run=kw.pop("prepare_run", 34042540432),
            sums_text=kw.pop("sums_text", _SUMS),
            manual=manual,
            version=kw.pop("version", "0.8.0"),
            uploader=kw.pop("uploader", "gmboquet"),
            testpypi_files=testpypi if manual else None,
        )

    def test_automated_record_is_accepted_and_bound(self):
        self.assertIn("automated", self._verify(_automated()))
        for bad in (
            _automated(candidate_commit="0" * 40),
            _automated(prepare_run=1),
            _automated(sums_sha256="0" * 64),
            _automated(passed=False),
            _automated(artifact="something/else"),
        ):
            with self.assertRaises(self.module.Refusal):
                self._verify(bad)

    def test_automated_record_with_the_workflows_string_typed_prepare_run_is_accepted(self):
        # publish.yml writes prepare_run from an environment string ("34091612593"); the verifier's
        # --prepare-run is an int. The first automated promote (0.8.1, run 34094030813) was refused
        # on exactly that comparison, so a run id is now matched as text in both forms.
        module = _load()
        summary = module.verify(
            _automated(prepare_run="34042540432"),
            candidate_sha=_SHA,
            prepare_run=34042540432,
            sums_text=_SUMS,
            manual=False,
        )
        self.assertIn("accepted", summary)

    def test_forms_cannot_be_swapped(self):
        with self.assertRaisesRegex(self.module.Refusal, "manual one was declared"):
            self._verify(_automated(), manual=True)
        with self.assertRaisesRegex(self.module.Refusal, "without the manual declaration"):
            self._verify(_manual())

    def test_manual_record_is_accepted_only_with_every_binding_and_the_live_precondition(self):
        self.assertIn("809 modules swept", self._verify(_manual(), manual=True))
        refusals = {
            "reason": _manual(reason="because"),
            "verified_by": _manual(verified_by="someone-else"),
            "verified_at": _manual(verified_at="yesterday"),
            "reproducible": _manual(reproducible_builds_passed=False),
            "clean_install": _manual(clean_install=False),
            "sweep count": _manual(import_sweep_modules=0),
            "sweep bool": _manual(import_sweep_modules=True),
            "draft files": _manual(draft_release_files={"mixle-0.8.0-py3-none-any.whl": "aa" * 32}),
            "testpypi claim": _manual(testpypi_files={"mixle-0.8.0-py3-none-any.whl": "cc" * 32}),
        }
        for label, bad in refusals.items():
            with self.assertRaises(self.module.Refusal, msg=label):
                self._verify(bad, manual=True)

    def test_manual_record_is_refused_when_the_automated_phase_would_apply(self):
        # TestPyPI holds this candidate's exact bytes: nothing stops the automated rehearsal
        same = {"mixle-0.8.0-py3-none-any.whl": "aa" * 32, "mixle-0.8.0.tar.gz": "bb" * 32}
        with self.assertRaisesRegex(self.module.Refusal, "exact bytes"):
            self._verify(_manual(testpypi_files=same), manual=True, testpypi=same)
        # TestPyPI does not hold the version at all
        with self.assertRaisesRegex(self.module.Refusal, "does not match what TestPyPI serves|does not hold"):
            self._verify(_manual(testpypi_files={}), manual=True, testpypi={})

    def test_sums_must_be_well_formed(self):
        with self.assertRaises(self.module.Refusal):
            self._verify(_automated(), sums_text="not a sums file\n")


class CommandLineTest(unittest.TestCase):
    def test_cli_exit_codes_in_both_forms(self):
        module = _load()
        tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        sums = tmp / "SHA256SUMS"
        sums.write_text(_SUMS, encoding="utf-8")
        auto = tmp / "auto.json"
        auto.write_text(json.dumps(_automated()), encoding="utf-8")
        manual = tmp / "testpypi-rehearsal-manual-9504451e.json"
        manual.write_text(json.dumps(_manual()), encoding="utf-8")
        testpypi = tmp / "testpypi.json"
        testpypi.write_text(
            json.dumps({"urls": [{"filename": k, "digests": {"sha256": v}} for k, v in _TESTPYPI.items()]}),
            encoding="utf-8",
        )
        common = ["--candidate-sha", _SHA, "--prepare-run", "34042540432", "--sums", str(sums)]
        self.assertEqual(module.main(["--record", str(auto), *common]), 0)
        self.assertEqual(
            module.main(
                [
                    "--record",
                    str(manual),
                    "--manual",
                    "--version",
                    "0.8.0",
                    "--uploader",
                    "gmboquet",
                    "--testpypi-json",
                    str(testpypi),
                    *common,
                ]
            ),
            0,
        )
        self.assertEqual(module.main(["--record", str(manual), *common]), 1)  # manual record without the declaration
        self.assertEqual(
            module.main(
                [
                    "--record",
                    str(auto),
                    "--manual",
                    "--version",
                    "0.8.0",
                    "--uploader",
                    "gmboquet",
                    "--testpypi-json",
                    str(testpypi),
                    *common,
                ]
            ),
            1,
        )
        self.assertEqual(module.main(["--record", str(tmp / "missing.json"), *common]), 1)


if __name__ == "__main__":
    unittest.main()
