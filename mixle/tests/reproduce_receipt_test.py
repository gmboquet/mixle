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
                    "source_content_file_count": 3,
                    "source_content_universe": "pyproject.toml, setup.py, and mixle/**/*.{json,py,pyx}",
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
            "source_content_file_count": 3,
            "source_content_universe": "pyproject.toml, setup.py, and mixle/**/*.{json,py,pyx}",
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
            import hashlib

            member = b"x = 1"
            digest = hashlib.sha256(member).hexdigest()
            encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode("ascii").rstrip("=")
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("mixle/a.py", member)
                archive.writestr(
                    "mixle-0.8.0.dist-info/RECORD",
                    "mixle/a.py,sha256=%s,%d\nmixle-0.8.0.dist-info/RECORD,,\n" % (encoded, len(member)),
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
        """A RECORD whose claims match the archived members parses; the returned hashes are the
        RECORDED ones, which the parser has now verified against the bytes (SYS4-01)."""
        import base64
        import hashlib

        member = b"x = 1"
        digest = hashlib.sha256(member).hexdigest()
        encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode("ascii").rstrip("=")
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "mixle-0.8.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("mixle/a.py", member)
                archive.writestr(
                    "mixle-0.8.0.dist-info/RECORD",
                    "mixle/a.py,sha256=%s,%d\nmixle-0.8.0.dist-info/RECORD,,\n" % (encoded, len(member)),
                )
            self.assertEqual(self.reproduction._wheel_record_hashes(wheel), {"mixle/a.py": digest})

    def test_a_record_claim_that_the_archive_does_not_satisfy_is_refused(self):
        """SYS4-01: RECORD claims are checked against the members, and members against the RECORD."""
        import base64
        import hashlib

        member = b"x = 1"
        good = base64.urlsafe_b64encode(hashlib.sha256(member).digest()).decode("ascii").rstrip("=")
        wrong = base64.urlsafe_b64encode(bytes.fromhex("f" * 64)).decode("ascii").rstrip("=")
        with tempfile.TemporaryDirectory() as directory:
            # a) member bytes disagree with their RECORD hash (the reviewer's fixture shape)
            wheel = Path(directory) / "a.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("mixle/a.py", member)
                archive.writestr(
                    "mixle-0.8.0.dist-info/RECORD", "mixle/a.py,sha256=%s,5\nmixle-0.8.0.dist-info/RECORD,,\n" % wrong
                )
            with self.assertRaisesRegex(ValueError, "does not match its RECORD hash"):
                self.reproduction._wheel_record_hashes(wheel)
            # b) RECORD claims a member the archive does not hold
            wheel = Path(directory) / "b.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    "mixle-0.8.0.dist-info/RECORD", "mixle/a.py,sha256=%s,5\nmixle-0.8.0.dist-info/RECORD,,\n" % good
                )
            with self.assertRaisesRegex(ValueError, "not in the archive"):
                self.reproduction._wheel_record_hashes(wheel)
            # c) the archive holds a member the RECORD never claims (the neighbouring case)
            wheel = Path(directory) / "c.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("mixle/a.py", member)
                archive.writestr("mixle/_smuggled.py", b"EVIL = 1")
                archive.writestr(
                    "mixle-0.8.0.dist-info/RECORD", "mixle/a.py,sha256=%s,5\nmixle-0.8.0.dist-info/RECORD,,\n" % good
                )
            with self.assertRaisesRegex(ValueError, "does not claim"):
                self.reproduction._wheel_record_hashes(wheel)


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


class CarriedProvenanceVerificationTest(unittest.TestCase):
    """SYS-01 second pass: a carried identity must still describe the tree that carries it.

    The first repair copied the sdist's attestation into any wheel built from it without checking
    that the tree still matched. That turned a missing label into a lying one: unpack, edit,
    rebuild, and the wheel asserted the original commit over modified bytes. This exercises
    ``setup.py``'s ``_carried_provenance`` directly, because the end-to-end proof (build sdist,
    tamper, rebuild) is too slow for the unit tier -- it was run manually and is recorded in the
    commit message.
    """

    @staticmethod
    def _setup_module():
        """Load setup.py's provenance helpers with setuptools stubbed out entirely.

        setup.py imports setuptools and calls setup() at import time. The first version of this
        helper imported the real setuptools to monkeypatch it, which passed locally only because
        the developer venv happened to have setuptools installed -- Python 3.12 does not bundle it,
        and the CI test environment does not have it either, so all four tests in this class failed
        there with ModuleNotFoundError. Stubbing the modules in sys.modules removes the dependency
        instead of relying on the environment to satisfy it.
        """
        import sys
        import types

        path = Path(__file__).resolve().parents[2] / "setup.py"
        spec = importlib.util.spec_from_file_location("_mixle_setup", path)
        module = importlib.util.module_from_spec(spec)

        stub = types.ModuleType("setuptools")
        stub.setup = lambda *args, **kwargs: None
        build_py_module = types.ModuleType("setuptools.command.build_py")
        build_py_module.build_py = type("build_py", (), {"run": lambda self: None})
        sdist_module = types.ModuleType("setuptools.command.sdist")
        sdist_module.sdist = type("sdist", (), {"make_release_tree": lambda self, base_dir, files: None})
        command_pkg = types.ModuleType("setuptools.command")

        names = {
            "setuptools": stub,
            "setuptools.command": command_pkg,
            "setuptools.command.build_py": build_py_module,
            "setuptools.command.sdist": sdist_module,
        }
        saved = {name: sys.modules.get(name) for name in names}
        sys.modules.update(names)
        try:
            spec.loader.exec_module(module)
        finally:
            for name, previous in saved.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
        return module

    def _tree(self, directory, *, commit="a" * 40, with_digest=True, tamper=False):
        setup = self._setup_module()
        root = Path(directory)
        (root / "mixle").mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='mixle'\n", encoding="utf-8")
        (root / "setup.py").write_text("# stub\n", encoding="utf-8")
        (root / "mixle" / "__init__.py").write_text("__version__='0.8.0'\n", encoding="utf-8")
        payload = {
            "artifact": "mixle.build_provenance/v1",
            "source_commit": commit,
            "source_tree": "b" * 40,
            "source_dirty": False,
        }
        if with_digest:
            payload["sdist_content_sha256"] = setup._source_content_digest(root)
        if tamper:  # edit a shipped file AFTER the digest was recorded
            (root / "mixle" / "__init__.py").write_text("__version__='0.8.0'\nEVIL=1\n", encoding="utf-8")
        (root / "mixle" / "_build_provenance.json").write_text(json.dumps(payload), encoding="utf-8")
        return setup, root

    def test_an_untouched_tree_carries_its_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            setup, root = self._tree(directory)
            carried = setup._carried_provenance(root)
            self.assertIsNotNone(carried)
            self.assertEqual(carried["source_commit"], "a" * 40)

    def test_a_modified_tree_does_not_carry_the_original_commit(self):
        """The attack: unpack, edit, rebuild. The wheel must not assert the original commit."""
        with tempfile.TemporaryDirectory() as directory:
            setup, root = self._tree(directory, tamper=True)
            self.assertIsNone(setup._carried_provenance(root), "a tampered tree must not be adoptable")

    def test_a_record_without_a_content_digest_is_not_adoptable(self):
        """An unverifiable record is refused rather than trusted for backward compatibility."""
        with tempfile.TemporaryDirectory() as directory:
            setup, root = self._tree(directory, with_digest=False)
            self.assertIsNone(setup._carried_provenance(root))

    def test_the_provenance_file_is_excluded_from_its_own_digest(self):
        """Otherwise the recorded digest could never be recomputed."""
        with tempfile.TemporaryDirectory() as directory:
            setup, root = self._tree(directory)
            names = [p.name for p in setup._source_content_files(root)]
            self.assertNotIn("_build_provenance.json", names)


class BytecodeAuthenticationTest(unittest.TestCase):
    """SYS-02 second pass: the bytecode that executes must match its verified source.

    Exempting installer bytecode from the hash rule left the executing bytes unverified. CPython's
    default timestamp invalidation does not help: an attacker who keeps the original header, so the
    recorded mtime and size still match the ``.py``, gets the tampered bytecode loaded with no
    recompile. The first repair recorded that limitation in a comment and stopped there.
    """

    def _compile_pair(self, directory, source_text, bytecode_text=None):
        import marshal

        root = Path(directory)
        source = root / "m.py"
        source.write_text(source_text, encoding="utf-8")
        compiled = compile((bytecode_text or source_text).encode(), str(source), "exec")
        pyc = root / "m.cpython-312.pyc"
        pyc.write_bytes(b"\x00" * 16 + marshal.dumps(compiled))
        return pyc, source

    def test_matching_bytecode_is_accepted(self):
        from mixle.reproduction import _bytecode_matches_source

        with tempfile.TemporaryDirectory() as directory:
            pyc, source = self._compile_pair(directory, "VALUE = 1\n")
            self.assertIs(_bytecode_matches_source(pyc, source), True)

    def test_bytecode_compiled_from_different_source_is_rejected(self):
        """The attack: bytecode that does not correspond to the verified .py."""
        from mixle.reproduction import _bytecode_matches_source

        with tempfile.TemporaryDirectory() as directory:
            pyc, source = self._compile_pair(directory, "VALUE = 1\n", bytecode_text="VALUE = 1\nBACKDOOR = True\n")
            self.assertIs(_bytecode_matches_source(pyc, source), False)

    def test_undecidable_bytecode_is_reported_as_undecidable_not_as_matching(self):
        from mixle.reproduction import _bytecode_matches_source

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "m.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            truncated = root / "m.cpython-312.pyc"
            truncated.write_bytes(b"\x00" * 16 + b"not marshal data")
            self.assertIsNone(_bytecode_matches_source(truncated, source))

    def test_code_comparison_is_structural_not_serialized_bytes(self):
        """marshal encodes back-references, so identical code can serialize differently.

        Comparing ``marshal.dumps`` output produced 9 false mismatches over 810 files on a clean
        install -- which would have reinstated the original defect of refusing ordinary
        installations. This pins the structural comparison instead.
        """
        import marshal

        from mixle.reproduction import _code_equal

        source = "def f():\n    a = 'repeated'\n    b = 'repeated'\n    return a, b\n"
        first = compile(source, "m.py", "exec")
        second = marshal.loads(marshal.dumps(first))  # same code, different object sharing
        self.assertTrue(_code_equal(first, second))
        different = compile(source.replace("return a, b", "return b, a"), "m.py", "exec")
        self.assertFalse(_code_equal(first, different))


class ProvenanceAndPartialInstallMediumsTest(unittest.TestCase):
    """Second-pass Mediums: an unrelated repo must not override the artifact, and a partial
    install that retains metadata must still be classified as one."""

    def test_the_artifact_record_beats_an_unrelated_enclosing_repository(self):
        from mixle.inference.production import provenance as module

        embedded = {
            "artifact": "mixle.build_provenance/v1",
            "source_commit": "a" * 40,
            "source_tree": "b" * 40,
            "source_dirty": False,
        }
        ambient = {"git_commit": "f" * 40, "git_dirty": False, "git_worktree_digest": "x"}
        with (
            patch.object(module, "_embedded_build_provenance", return_value=embedded),
            patch.object(module, "_git_state", return_value=ambient),
        ):
            info = module.environment_info()
        # the executing bytes' own attestation wins; the surrounding directory is not evidence
        self.assertEqual(info["git_commit"], "a" * 40)
        self.assertEqual(info["provenance_source"], "installed-artifact-build-provenance")
        # and the disagreement is recorded rather than discarded -- under its OWN key, in full,
        # so the artifact's git_* fields describe one source only (SYS3-08)
        self.assertEqual(info["ambient_repository"]["git_commit"], "f" * 40)
        self.assertIsNone(
            info["git_worktree_digest"], "an ambient worktree digest must not sit beside an artifact commit"
        )

    def test_ambient_repository_is_still_used_when_there_is_no_artifact_record(self):
        from mixle.inference.production import provenance as module

        ambient = {"git_commit": "f" * 40, "git_dirty": False, "git_worktree_digest": "x"}
        with (
            patch.object(module, "_embedded_build_provenance", return_value=None),
            patch.object(module, "_git_state", return_value=ambient),
        ):
            info = module.environment_info()
        self.assertEqual(info["git_commit"], "f" * 40)
        self.assertEqual(info["provenance_source"], "ambient-repository")

    def test_metadata_that_survives_a_partial_install_is_still_classified_as_partial(self):
        """The first repair only recognised a partial install when metadata was absent entirely."""
        from mixle import reproduction

        class _Item(str):
            hash = type("H", (), {"mode": "sha256", "value": "x"})()

        class _Dist:
            version = "0.8.0"
            files = [_Item("mixle/__init__.py"), _Item("mixle/blending.py")]

            @staticmethod
            def locate_file(item):
                return Path("/nonexistent") / str(item)

        with patch.object(reproduction, "distribution", return_value=_Dist()):
            result = reproduction.installed_content_provenance()
        self.assertFalse(result["verified"])
        self.assertIn("partial install", result["reason"])
        self.assertEqual(result["missing_recorded_file_count"], 2)


class SourceContentUniverseIsRequiredTest(unittest.TestCase):
    """SYS3-07: SYS-08's population fields must be REQUIRED, not merely emitted.

    Both fixture wheels in this file omitted ``source_content_file_count`` and
    ``source_content_universe`` and passed ``wheel_provenance`` -- which is precisely the finding:
    a record could drop the fields that say which population its digest covers and still verify.
    """

    def _wheel(self, directory, **overrides):
        wheel = Path(directory) / "mixle-0.8.0-py3-none-any.whl"
        provenance = {
            "artifact": "mixle.build_provenance/v1",
            "source_commit": "a" * 40,
            "source_tree": "b" * 40,
            "source_dirty": False,
            "source_content_sha256": "c" * 64,
            "source_content_file_count": 3,
            "source_content_universe": "pyproject.toml, setup.py, and mixle/**/*.{json,py,pyx}",
        }
        provenance.update(overrides)
        for key in [k for k, v in overrides.items() if v is None]:
            provenance.pop(key, None)
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("mixle-0.8.0.dist-info/METADATA", "Name: mixle\nVersion: 0.8.0\n")
            archive.writestr("mixle/_build_provenance.json", json.dumps(provenance))
        return wheel

    def test_complete_record_verifies(self):
        from mixle import reproduction

        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(reproduction.wheel_provenance(self._wheel(directory))["verified"])

    def test_missing_or_ill_typed_population_fields_are_refused(self):
        from mixle import reproduction

        cases = {
            "count absent": {"source_content_file_count": None},
            "count zero": {"source_content_file_count": 0},
            "count bool": {"source_content_file_count": True},
            "count string": {"source_content_file_count": "3"},
            "universe absent": {"source_content_universe": None},
            "universe empty": {"source_content_universe": "   "},
        }
        for label, overrides in cases.items():
            with tempfile.TemporaryDirectory() as directory, self.subTest(label):
                with self.assertRaisesRegex(ValueError, "source_content_(file_count|universe)"):
                    reproduction.wheel_provenance(self._wheel(directory, **overrides))


class FourthPassNeighboursTest(unittest.TestCase):
    """Neighbours of SYS4-01/02, found by attacking the fixes before shipping them rather than after.

    Three passes had each reopened the previous repairs at their boundaries. This wave the fixes were
    attacked adversarially first; these are what that pass found, closed in the same commit.
    """

    @staticmethod
    def _enc(data: bytes) -> str:
        import base64
        import hashlib

        return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")

    def test_a_record_or_member_path_that_escapes_the_archive_root_is_refused(self):
        from mixle.reproduction import _wheel_record_hashes

        member = b"x = 1"
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "t.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("mixle/a.py", member)
                archive.writestr("../escape.py", member)
                archive.writestr(
                    "mixle-0.8.0.dist-info/RECORD",
                    "mixle/a.py,sha256=%s,5\n../escape.py,sha256=%s,5\nmixle-0.8.0.dist-info/RECORD,,\n"
                    % (self._enc(member), self._enc(member)),
                )
            with self.assertRaisesRegex(ValueError, "escapes the archive root"):
                _wheel_record_hashes(wheel)

    def test_duplicate_zip_member_names_are_refused_even_when_identical(self):
        """A zip may hold one name twice; the installer may extract the one the check did not read."""
        import warnings

        from mixle.reproduction import _wheel_record_hashes

        member = b"x = 1"
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "d.whl"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr("mixle/a.py", member)
                    archive.writestr("mixle/a.py", member)
                    archive.writestr(
                        "mixle-0.8.0.dist-info/RECORD",
                        "mixle/a.py,sha256=%s,5\nmixle-0.8.0.dist-info/RECORD,,\n" % self._enc(member),
                    )
            with self.assertRaisesRegex(ValueError, "more than once"):
                _wheel_record_hashes(wheel)

    def test_identical_code_holding_a_nan_constant_compares_equal(self):
        """``nan != nan`` made identical NaN-bearing modules compare unequal -- a false refusal that
        would have reinstated the original SYS-02 defect through a numeric edge."""
        from mixle.reproduction import _code_equal

        code = compile("X = 1.5\n", "m.py", "exec")
        with_nan = code.replace(co_consts=tuple(float("nan") if c == 1.5 else c for c in code.co_consts))
        self.assertTrue(_code_equal(with_nan, with_nan.replace()))
        with_other = with_nan.replace(co_consts=tuple(2.5 if isinstance(c, float) else c for c in with_nan.co_consts))
        self.assertFalse(_code_equal(with_nan, with_other))

    def test_every_observable_code_field_is_compared(self):
        """SYS4-02 and its neighbours: filename, first line and line table are all readable by the
        program (``f_code.co_filename``, ``f_lineno``), so none may be treated as metadata."""
        import types

        from mixle.reproduction import _code_equal

        src = "def g():\n    import sys\n    return sys._getframe().f_code.co_filename\n"
        a = compile(src, "original.py", "exec")
        b = compile(src, "changed.py", "exec")
        self.assertFalse(_code_equal(a, b), "co_filename is observable and must be compared")
        g = next(c for c in a.co_consts if isinstance(c, types.CodeType))
        self.assertFalse(_code_equal(g, g.replace(co_firstlineno=g.co_firstlineno + 50)))
        self.assertFalse(_code_equal(g, g.replace(co_linetable=b"")))
        self.assertTrue(_code_equal(g, g.replace()))
