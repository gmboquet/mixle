"""The extras verifier must agree with what pip actually installs.

Two independent defects made the whole `numba` extras job red on every platform, and between them
they cover both directions the verifier can be wrong.

* It matched the requirement string with a bare name regex, which DISCARDS the environment marker.
  On any machine that is not x86_64, pip correctly declines to install `tbb; platform_machine ==
  "x86_64"` and the verifier then failed trying to import it — a verification failure invented out
  of a dependency that was never meant to be there.
* On x86_64, where pip does install it, `tbb` ships only shared libraries (`libtbb.so` and friends
  under `.data/data/lib`) and provides no importable Python module at all — verified against
  tbb-2021.13.1. The import map claimed `tbb -> ["tbb"]`, so the verifier failed there too.

The empty import list is therefore a deliberate statement — "this distribution is not
import-verifiable" — not an unfinished mapping.
"""

import ast
import importlib.util
import json
import tomllib
import unittest
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "manifests" / "optional_dependency_imports.json"


def _verifier():
    spec = importlib.util.spec_from_file_location("_verify_extra_profile", ROOT / "scripts" / "verify_extra_profile.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _extras() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]


def _mapping() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["distribution_to_imports"]


class MarkerEvaluationTest(unittest.TestCase):
    def test_a_requirement_excluded_by_its_marker_contributes_no_module(self):
        # tbb is declared `; platform_machine == "x86_64"`. On any other machine pip installs
        # nothing for it, so the verifier must not ask for it either.
        modules_here = _verifier().applicable_modules("numba")
        self.assertIn("numba", modules_here)

    def test_every_profile_resolves_on_this_platform(self):
        # A regex that drops markers only fails on the platforms the marker excludes, so checking
        # one profile is not enough: resolve them all.
        verifier = _verifier()
        for profile in _extras():
            with self.subTest(profile=profile):
                verifier.applicable_modules(profile)

    def test_a_marker_is_actually_consulted(self):
        marked = [
            Requirement(req) for reqs in _extras().values() for req in reqs if Requirement(req).marker is not None
        ]
        self.assertTrue(marked, "expected at least one marked optional requirement to exercise this path")


class ImportMapHonestyTest(unittest.TestCase):
    def test_tbb_declares_no_importable_module(self):
        self.assertEqual(_mapping()["tbb"], [])

    def test_the_empty_mapping_is_explained(self):
        # An empty list is indistinguishable from an oversight without a stated reason, and the next
        # maintainer would "fix" it straight back to ["tbb"].
        notes = json.loads(MANIFEST.read_text(encoding="utf-8")).get("notes", {})
        self.assertIn("tbb", notes)
        self.assertIn("libtbb", notes["tbb"])

    def test_every_optional_distribution_is_mapped(self):
        mapping = _mapping()
        for profile, requirements in _extras().items():
            for requirement in requirements:
                with self.subTest(profile=profile, requirement=requirement):
                    self.assertIn(Requirement(requirement).name, mapping)


class SplitResponsibilityTest(unittest.TestCase):
    """`verify` runs inside the environment under test and must not need `packaging` there."""

    def test_the_import_path_does_not_need_packaging_at_module_scope(self):
        """The `--modules` path must import with the standard library alone.

        `from packaging.requirements import Requirement` sat at MODULE scope, which defeated the
        split entirely: the standard-library-only path died on ModuleNotFoundError before reaching
        main(). It passed every local check because pytest itself depends on `packaging`, so every
        environment used for testing happened to have it -- the workflow's floor environment,
        built from an exact constraint set, installs neither. Asserting on the SOURCE is what
        catches that, because this interpreter cannot help having packaging available.
        """
        source = (ROOT / "scripts" / "verify_extra_profile.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        module_level = {
            alias.name.split(".")[0]
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(name=node.module or "")])
        }
        self.assertNotIn("packaging", module_level)

    def test_verify_imports_a_supplied_module_list(self):
        self.assertEqual(_verifier().verify(["json", "json"]), ["json"])

    def test_verify_reports_a_genuinely_missing_module(self):
        with self.assertRaises(ImportError):
            _verifier().verify(["a_module_that_does_not_exist_anywhere"])


if __name__ == "__main__":
    unittest.main()
