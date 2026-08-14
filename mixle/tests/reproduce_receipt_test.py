"""The reproduction script's claim checks are deterministic and correct (worklist E14).

``scripts/reproduce.py`` emits a receipt an external reviewer runs to reproduce mixle's headline claims. Its
value is only as good as its determinism: if the seeded checks drift between runs, the receipt cannot be
compared. This pins that -- the checks reproduce exactly, and they hold (a Gaussian fit recovers its
parameters, scalar/vectorized scores agree, serialization is score-preserving, automatic selection recovers
the family) -- and that the environment capture has the fields a reviewer needs.
"""

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

_GEN = Path(__file__).resolve().parents[2] / "scripts" / "reproduce.py"


def _load():
    spec = importlib.util.spec_from_file_location("_reproduce", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ReproduceReceiptTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load()

    def test_claim_checks_are_deterministic(self):
        self.assertEqual(self.mod.claim_checks(), self.mod.claim_checks())

    def test_claim_checks_hold(self):
        checks = self.mod.evaluate_claims(self.mod.claim_checks())
        self.assertTrue(all(check["passed"] for check in checks.values()))
        # a Gaussian fit on N(3, 2) data recovers the parameters within sampling error.
        self.assertAlmostEqual(checks["gaussian_fit_mu"]["observed"], 3.0, delta=0.2)
        self.assertAlmostEqual(checks["gaussian_fit_sigma"]["observed"], 2.0, delta=0.2)

    def test_environment_capture_has_required_fields(self):
        env = self.mod.environment()
        for field in ("python", "platform", "machine", "mixle", "numpy", "scipy", "installed_content"):
            self.assertIn(field, env)

    def test_false_or_wrong_claims_fail_the_receipt_and_process(self):
        wrong = self.mod.claim_checks()
        wrong["scalar_vectorized_agree"] = False
        wrong["auto_selects"] = "WrongEstimator"
        evaluated = self.mod.evaluate_claims(wrong)
        self.assertFalse(evaluated["scalar_vectorized_agree"]["passed"])
        self.assertFalse(evaluated["auto_selects"]["passed"])

        receipt_path = Path(self.id().replace(".", "-") + ".json")
        try:
            with (
                patch("mixle.reproduction.claim_checks", return_value=wrong),
                patch(
                    "mixle.reproduction.source_tree_provenance",
                    return_value={"artifact": "test", "verified": True},
                ),
                patch(
                    "mixle.reproduction.installed_content_provenance",
                    return_value={"artifact": "test", "verified": True},
                ),
            ):
                self.assertEqual(self.mod.main(["--source-tree", "--out", str(receipt_path)]), 1)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertFalse(receipt["passed"])
        finally:
            receipt_path.unlink(missing_ok=True)

    def test_ambient_repository_is_not_used_for_source_provenance(self):
        provenance = self.mod.source_tree_provenance()
        self.assertEqual(Path(provenance["repository"]), _GEN.parents[1])
        self.assertNotEqual(provenance["commit"], "unknown")

    def test_wheel_receipt_requires_clean_embedded_source_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "mixle-0.8.0-py3-none-any.whl"

            def write_wheel(source_dirty):
                provenance = {
                    "artifact": "mixle.build_provenance/v1",
                    "source_commit": "a" * 40,
                    "source_tree": "b" * 40,
                    "source_dirty": source_dirty,
                    "source_content_sha256": "c" * 64,
                }
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr("mixle-0.8.0.dist-info/METADATA", "Name: mixle\nVersion: 0.8.0\n")
                    archive.writestr("mixle/_build_provenance.json", json.dumps(provenance))

            write_wheel(False)
            receipt = self.mod.wheel_provenance(wheel)
            self.assertTrue(receipt["verified"])
            self.assertEqual(len(receipt["sha256"]), 64)

            write_wheel(True)
            with self.assertRaisesRegex(ValueError, "clean source"):
                self.mod.wheel_provenance(wheel)


class SubjectBindingTest(unittest.TestCase):
    """SYS-RR7-3/4: the receipt's subject must be the wheel whose code executes, and the receipt
    must say what it covers.

    Version equality is not identity -- the adversarial review presented an older 0.8.0 wheel as
    the subject while a newer 0.8.0 build executed, and the receipt said passed. The binding
    compares the installed package's embedded build provenance and every hashed RECORD entry
    against the subject wheel, and a foreign wheel fails closed.
    """

    @staticmethod
    def _foreign_wheel(directory):
        import base64

        from mixle import reproduction

        wheel = Path(directory) / "mixle-0.8.0-py3-none-any.whl"
        provenance = {
            "artifact": "mixle.build_provenance/v1",
            "source_commit": "a" * 40,
            "source_tree": "b" * 40,
            "source_dirty": False,
            "source_content_sha256": "c" * 64,
        }
        fake_hash = base64.urlsafe_b64encode(bytes.fromhex("d" * 64)).decode("ascii").rstrip("=")
        record = "mixle/__init__.py,sha256=%s,10\nmixle-0.8.0.dist-info/RECORD,,\n" % fake_hash
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("mixle-0.8.0.dist-info/METADATA", "Name: mixle\nVersion: 0.8.0\n")
            archive.writestr("mixle-0.8.0.dist-info/RECORD", record)
            archive.writestr("mixle/_build_provenance.json", json.dumps(provenance))
        return wheel, reproduction

    def test_a_foreign_same_version_wheel_fails_the_binding_and_the_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel, reproduction = self._foreign_wheel(directory)
            artifact = reproduction.wheel_provenance(wheel)
            binding = reproduction.subject_binding(wheel, artifact["build"])
            self.assertFalse(binding["verified"])
            self.assertTrue(binding["mismatches"])
            receipt, passed = reproduction.build_receipt(wheel=wheel, allow_source_tree=False)
            self.assertFalse(passed)
            self.assertFalse(receipt["subject"]["verified"])
            self.assertFalse(receipt["subject"]["installed_binding"]["verified"])

    def test_wheel_record_parsing_is_exact_and_fail_closed(self):
        import base64

        from mixle import reproduction

        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "mixle-0.8.0-py3-none-any.whl"
            digest = "e" * 64
            encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode("ascii").rstrip("=")
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    "mixle-0.8.0.dist-info/RECORD",
                    "mixle/a.py,sha256=%s,5\nmixle-0.8.0.dist-info/RECORD,,\n" % encoded,
                )
            self.assertEqual(reproduction._wheel_record_hashes(wheel), {"mixle/a.py": digest})

            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("mixle-0.8.0.dist-info/RECORD", "mixle/a.py,md5=abc,5\n")
            with self.assertRaisesRegex(ValueError, "unsupported hash"):
                reproduction._wheel_record_hashes(wheel)

            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("mixle-0.8.0.dist-info/RECORD", "mixle-0.8.0.dist-info/RECORD,,\n")
            with self.assertRaisesRegex(ValueError, "no hashed entries"):
                reproduction._wheel_record_hashes(wheel)

    def test_the_receipt_declares_its_scope(self):
        from mixle import reproduction

        receipt, _ = reproduction.build_receipt(wheel=None, allow_source_tree=True)
        self.assertEqual(receipt["scope"]["claim_checks"], sorted(reproduction._EXPECTATIONS))
        self.assertIn("run_repro_entry.py", receipt["scope"]["reproduction_bundle_entries"])


class RecordTamperTest(unittest.TestCase):
    """SYS-RR8-2: stripping a file's RECORD hash excluded it from every integrity comparison.

    The reviewer altered ``mixle/reproduction.py`` inside a copy of the exact wheel and changed its
    RECORD row to ``mixle/reproduction.py,,``. The wheel installed, the altered code imported, and
    the receipt still said passed with 819 compared entries. Only RECORD may lack a hash now;
    anything else unhashed, duplicated, or malformed is refused.
    """

    @staticmethod
    def _wheel(directory, record_body):
        wheel = Path(directory) / "mixle-0.8.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("mixle-0.8.0.dist-info/RECORD", record_body)
        return wheel

    def setUp(self):
        import base64

        from mixle import reproduction

        self.reproduction = reproduction
        self.hash = base64.urlsafe_b64encode(bytes.fromhex("a" * 64)).decode("ascii").rstrip("=")

    def test_an_unhashed_code_row_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = self._wheel(
                directory,
                "mixle/a.py,sha256=%s,5\nmixle/reproduction.py,,\nmixle-0.8.0.dist-info/RECORD,,\n" % self.hash,
            )
            with self.assertRaisesRegex(ValueError, "omits hashes"):
                self.reproduction._wheel_record_hashes(wheel)

    def test_duplicate_malformed_and_missing_self_row_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = "mixle/a.py,sha256={h},5\nmixle/a.py,sha256={h},5\nmixle-0.8.0.dist-info/RECORD,,\n"
            wheel = self._wheel(directory, duplicate.format(h=self.hash))
            with self.assertRaisesRegex(ValueError, "more than once"):
                self.reproduction._wheel_record_hashes(wheel)

            wheel = self._wheel(directory, "mixle/a.py,sha256=!!!notbase64!!!,5\nmixle-0.8.0.dist-info/RECORD,,\n")
            with self.assertRaises(ValueError):
                self.reproduction._wheel_record_hashes(wheel)

            wheel = self._wheel(directory, "mixle/a.py,sha256=%s,5\n" % self.hash)
            with self.assertRaisesRegex(ValueError, "exactly one self-referencing"):
                self.reproduction._wheel_record_hashes(wheel)

    def test_a_well_formed_record_still_parses(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = self._wheel(directory, "mixle/a.py,sha256=%s,5\nmixle-0.8.0.dist-info/RECORD,,\n" % self.hash)
            self.assertEqual(self.reproduction._wheel_record_hashes(wheel), {"mixle/a.py": "a" * 64})


class ShadowInstallationTest(unittest.TestCase):
    """SYS-RR8-3: the receipt never proved the imported code was the distribution's own.

    A PYTHONPATH shadow supplied its own modules while extending ``__path__`` to the real
    installation, and produced a receipt byte-identical to the legitimate one. The binding now
    requires exactly one package root, equal to the distribution's, with the executing module
    inside it.
    """

    def test_extra_package_roots_and_outside_execution_are_reported(self):
        from mixle import reproduction

        real_root = Path(reproduction.__file__).resolve().parent
        shadow = Path("/nonexistent-shadow/mixle")

        class _ShadowReproduction:  # the attack supplied its own wrapper module too
            __file__ = str(shadow / "reproduction.py")

        class _Shadow:
            __file__ = str(shadow / "__init__.py")
            __path__ = [str(shadow), str(real_root)]
            reproduction = _ShadowReproduction

        with patch.dict("sys.modules", {"mixle": _Shadow, "mixle.reproduction": _ShadowReproduction}):
            with tempfile.TemporaryDirectory() as directory:
                wheel = Path(directory) / "mixle-0.8.0-py3-none-any.whl"
                wheel.write_bytes(b"not a wheel")
                result = reproduction.subject_binding(wheel, {})
        self.assertFalse(result["verified"])
        joined = " ".join(result["mismatches"])
        self.assertIn("package root", joined)

    def test_the_legitimate_installation_records_its_paths(self):
        from mixle import reproduction

        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "mixle-0.8.0-py3-none-any.whl"
            wheel.write_bytes(b"not a wheel")
            result = reproduction.subject_binding(wheel, {})
        # the wheel is unreadable, so this fails -- but the import-path half must still be recorded
        self.assertFalse(result["verified"])
        self.assertTrue(result["mismatches"])


class InstallerBytecodeExemptionTest(unittest.TestCase):
    """SYS-02: pip's own byte-compilation must not read as tampering.

    ``pip install`` compiles every module after hashing the distributed files, so it writes those
    ``.pyc`` rows to RECORD with no hash. Treating them as modification made ``mixle-reproduce``
    report ``passed: false`` on every ordinary installation -- 810 of them on this package -- while
    the six claim checks and the exact subject binding all passed. The exemption is narrow, so
    these pin both halves: the installer's own bytecode is exempt, and nothing else is.
    """

    def test_only_pycache_bytecode_is_treated_as_installer_output(self):
        from mixle.reproduction import _is_installer_bytecode

        for name in (
            "mixle/__pycache__/blending.cpython-312.pyc",
            "mixle/stats/__pycache__/dist.cpython-311.pyc",
        ):
            self.assertTrue(_is_installer_bytecode(name), name)
        for name in (
            "mixle/blending.py",  # distributed source
            "mixle/vendored.pyc",  # a .pyc SHIPPED outside __pycache__ is not installer output
            "mixle/__pycache__extra/x.pyc",  # not the __pycache__ directory
            "mixle/stats/kernel.so",
            "manifests/api_manifest.json",
        ):
            self.assertFalse(_is_installer_bytecode(name), name)

    def test_bytecode_maps_back_to_the_source_it_was_compiled_from(self):
        from mixle.reproduction import _bytecode_source_path

        self.assertEqual(_bytecode_source_path("mixle/__pycache__/blending.cpython-312.pyc"), "mixle/blending.py")
        self.assertEqual(
            _bytecode_source_path("mixle/stats/__pycache__/dist.cpython-311.opt-1.pyc"), "mixle/stats/dist.py"
        )
        # an undecodable name returns None so the caller fails it rather than exempting it
        self.assertIsNone(_bytecode_source_path("mixle/blending.py"))
        self.assertIsNone(_bytecode_source_path("__pycache__/.cpython-312.pyc"))

    def test_orphan_bytecode_is_a_failure_not_an_exemption(self):
        """A .pyc whose source is absent (or unhashed) must still fail the check.

        This is the property that keeps the exemption from becoming a hole: an attacker who adds an
        unhashed ``.pyc`` under ``__pycache__`` gets no free pass, because the exemption is
        conditional on a hashed distributed source of the same name.
        """
        from mixle.reproduction import _bytecode_source_path

        hashed = {"mixle/blending.py"}
        self.assertIn(_bytecode_source_path("mixle/__pycache__/blending.cpython-312.pyc"), hashed)
        self.assertNotIn(_bytecode_source_path("mixle/__pycache__/evil.cpython-312.pyc"), hashed)


if __name__ == "__main__":
    unittest.main()


class PartialInstallDetectionTest(unittest.TestCase):
    """SYS-06: a half-unpacked wheel must be reported as broken, not as absent.

    pip writes the ``.dist-info`` last, so an install that dies on a corrupt archive leaves the
    files it already unpacked in place with no metadata behind them. That environment imports and
    answers ``mixle.__version__`` as ``0+unknown``. Reproduced with a CRC-corrupted wheel into an
    empty venv: pip exits nonzero, 643 package files remain, and ``import mixle`` succeeds.
    """

    def test_orphaned_files_are_counted_from_the_importable_package(self):
        from mixle.reproduction import _orphaned_package_files

        # this test process has a real importable mixle, so the count is the honest nonzero one
        self.assertGreater(_orphaned_package_files(), 0)

    def test_absent_and_partial_installs_are_reported_differently(self):
        """The two failure modes need different remedies, so they must not share one message."""
        from unittest.mock import patch

        from mixle import reproduction

        with patch.object(reproduction, "distribution", side_effect=reproduction.PackageNotFoundError):
            with patch.object(reproduction, "_orphaned_package_files", return_value=0):
                absent = reproduction.installed_content_provenance()
            with patch.object(reproduction, "_orphaned_package_files", return_value=643):
                partial = reproduction.installed_content_provenance()

        self.assertFalse(absent["verified"])
        self.assertFalse(partial["verified"])
        self.assertNotIn("orphaned_file_count", absent)
        self.assertEqual(partial["orphaned_file_count"], 643)
        # the partial case must say what to DO -- installing over orphans leaves orphans
        self.assertIn("partial install", partial["reason"])
        self.assertIn("Remove", partial["reason"])
        self.assertNotEqual(absent["reason"], partial["reason"])
