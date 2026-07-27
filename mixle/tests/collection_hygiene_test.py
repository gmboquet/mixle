"""Test collection stays lean in the base environment (worklist T3.2).

Collection imports every test module to read its markers -- before any deselection. If a test module
imported a heavy optional framework (torch, jax, numba, spark, ...) *unconditionally* at module top, the
base-install smoke gate would pay that framework's multi-second import during collection (T3.2's
"collection inflation"), and worse, the module would fail to collect at all in an environment without the
framework installed. The suite avoids both by guarding every such import with ``pytest.importorskip(...)``
earlier in the module, so an absent framework skips the module instantly instead of importing it.

This test enforces that invariant with an AST scan: a bare top-level ``import <heavy>`` (or ``from
<heavy> import ...``) in a test module must be preceded by a top-level ``pytest.importorskip("<heavy>")``.
A ``try/except ImportError`` guard is fine too -- such an import lives inside an ``ast.Try`` node, not at
module top level, so it is not flagged.
"""

import ast
import fnmatch
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# Frameworks whose import is slow and/or optional; a test module must not import them unconditionally.
_HEAVY = {
    "torch",
    "numba",
    "jax",
    "pyspark",
    "dask",
    "ray",
    "transformers",
    "datasets",
    "sentence_transformers",
    "umap",
    "sklearn",
    "mpi4py",
    "zarr",
    "h5py",
    "pandas",
    "pyarrow",
    "networkx",
    "gmpy2",
    "sympy",
    "lightning",
    "safetensors",
    "numpyro",
    "hmmlearn",
}


def _importorskip_target(node):
    """If ``node`` is a top-level ``pytest.importorskip("pkg")`` (or ``importorskip(...)``), return ``pkg``."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        # also allow `x = pytest.importorskip("pkg")`
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
        else:
            return None
    else:
        call = node.value
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name == "importorskip" and call.args and isinstance(call.args[0], ast.Constant):
        return str(call.args[0].value).split(".")[0]
    return None


def _unguarded_heavy_imports(path: Path):
    """Top-level heavy imports in ``path`` not preceded by an importorskip of that package."""
    tree = ast.parse(path.read_text(), filename=str(path))
    skipped: set[str] = set()
    offenders = []
    for node in tree.body:  # module top level only -- try/except-guarded imports live deeper
        target = _importorskip_target(node)
        if target is not None:
            skipped.add(target)
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                pkg = alias.name.split(".")[0]
                if pkg in _HEAVY and pkg not in skipped:
                    offenders.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            pkg = (node.module or "").split(".")[0]
            if pkg in _HEAVY and pkg not in skipped:
                offenders.append(f"from {node.module} import ...")
    return offenders


class CollectionHygieneTest(unittest.TestCase):
    def test_no_unguarded_heavy_imports_at_module_top(self):
        findings = []
        paths = {
            path
            for path in TESTS_DIR.rglob("*.py")
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in ("*_test.py", "test_*.py"))
        }
        for path in sorted(paths):
            for imp in _unguarded_heavy_imports(path):
                findings.append(f"{path.relative_to(TESTS_DIR)}: {imp}")
        self.assertEqual(
            findings,
            [],
            "test modules import a heavy optional framework at top level without a preceding "
            "pytest.importorskip -- this inflates base-install collection and breaks it when the "
            "framework is absent:\n" + "\n".join(findings),
        )

    def test_test_modules_do_not_self_assign_the_fast_tier(self):
        findings = []
        paths = {
            path
            for path in TESTS_DIR.rglob("*.py")
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in ("*_test.py", "test_*.py"))
        }
        for path in sorted(paths):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "fast"
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "mark"
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "pytest"
                ):
                    findings.append(f"{path.relative_to(TESTS_DIR)}:{node.lineno}")
        self.assertEqual(
            findings,
            [],
            "test modules must not self-assign pytest.mark.fast; conftest.py is the central tier authority:\n"
            + "\n".join(findings),
        )

    def test_every_file_marker_registry_entry_names_a_collected_module(self):
        from mixle.tests.conftest import FILE_MARKERS

        discovered = {
            path.name
            for path in TESTS_DIR.rglob("*.py")
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in ("*_test.py", "test_*.py"))
        }
        missing = sorted(set(FILE_MARKERS) - discovered)
        self.assertEqual(missing, [], f"FILE_MARKERS names nonexistent collected modules: {missing}")


if __name__ == "__main__":
    unittest.main()
