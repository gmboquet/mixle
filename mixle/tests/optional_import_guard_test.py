"""Every optional import is mapped from project extras and guarded at executable module scope."""

import ast
import json
import re
import tomllib
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PKG_ROOT = _ROOT / "mixle"
_IMPORT_MAP = _ROOT / "manifests" / "optional_dependency_imports.json"


def _distribution(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    if match is None:
        raise ValueError(f"invalid optional requirement: {requirement!r}")
    return match.group(0)


def _optional_import_names() -> frozenset[str]:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    distributions = {
        _distribution(requirement)
        for requirements in project["optional-dependencies"].values()
        for requirement in requirements
    }
    mapping = json.loads(_IMPORT_MAP.read_text(encoding="utf-8"))
    if mapping.get("artifact") != "mixle.optional_dependency_imports/v1":
        raise ValueError("unsupported optional-dependency import-map schema")
    distribution_to_imports = mapping.get("distribution_to_imports")
    if not isinstance(distribution_to_imports, dict):
        raise ValueError("optional-dependency import map must contain a distribution mapping")
    missing = sorted(distributions - set(distribution_to_imports))
    extra = sorted(set(distribution_to_imports) - distributions)
    if missing or extra:
        raise ValueError(f"optional-dependency import map drift: missing={missing}, extra={extra}")
    return frozenset(
        import_name for distribution in distributions for import_name in distribution_to_imports[distribution]
    )


def _catches_import_error(node: ast.Try) -> bool:
    for handler in node.handlers:
        error = handler.type
        names = []
        if isinstance(error, ast.Name):
            names = [error.id]
        elif isinstance(error, ast.Tuple):
            names = [element.id for element in error.elts if isinstance(element, ast.Name)]
        if {"ImportError", "ModuleNotFoundError"} & set(names):
            return True
    return False


def _type_checking_guard(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Name)
        and test.id == "TYPE_CHECKING"
        or isinstance(test, ast.Attribute)
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
        and test.attr == "TYPE_CHECKING"
    )


def _scan_statements(
    statements: list[ast.stmt],
    optional_names: frozenset[str],
    *,
    guarded: bool = False,
) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for node in statements:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top in optional_names and not guarded:
                    hits.append((node.lineno, top))
            continue
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top = node.module.split(".", 1)[0]
            if top in optional_names and not guarded:
                hits.append((node.lineno, top))
            continue
        if isinstance(node, ast.Try):
            hits.extend(
                _scan_statements(
                    node.body,
                    optional_names,
                    guarded=guarded or _catches_import_error(node),
                )
            )
            hits.extend(_scan_statements(node.orelse, optional_names, guarded=guarded))
            hits.extend(_scan_statements(node.finalbody, optional_names, guarded=guarded))
            for handler in node.handlers:
                hits.extend(_scan_statements(handler.body, optional_names, guarded=guarded))
            continue
        if isinstance(node, ast.If):
            if not _type_checking_guard(node.test):
                hits.extend(_scan_statements(node.body, optional_names, guarded=guarded))
            hits.extend(_scan_statements(node.orelse, optional_names, guarded=guarded))
            continue
        child_blocks = []
        for field in ("body", "orelse", "finalbody"):
            value = getattr(node, field, None)
            if isinstance(value, list):
                child_blocks.append(value)
        if isinstance(node, ast.Match):
            child_blocks.extend(case.body for case in node.cases)
        for block in child_blocks:
            hits.extend(_scan_statements(block, optional_names, guarded=guarded))
    return hits


def _unguarded_optional_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _scan_statements(tree.body, _optional_import_names())


class OptionalImportGuardTest(unittest.TestCase):
    def test_extras_and_import_map_are_complete(self):
        names = _optional_import_names()
        for required in (
            "pyarrow",
            "sqlalchemy",
            "pymongo",
            "fsspec",
            "networkx",
            "numpyro",
            "sympy",
            "sage",
            "sentence_transformers",
        ):
            self.assertIn(required, names)

    def test_no_unguarded_executable_module_scope_optional_imports(self):
        offenders = []
        for path in sorted(_PKG_ROOT.rglob("*.py")):
            if "tests" in path.relative_to(_PKG_ROOT).parts:
                continue
            for lineno, dependency in _unguarded_optional_imports(path):
                offenders.append(f"{path.relative_to(_ROOT)}:{lineno}: {dependency}")
        self.assertEqual(
            offenders,
            [],
            "optional dependencies execute at module scope without an ImportError/TYPE_CHECKING guard:\n"
            + "\n".join(offenders),
        )

    def test_nested_unconditional_import_is_detected(self):
        tree = ast.parse("if True:\n    import pyarrow\n")
        self.assertEqual(_scan_statements(tree.body, _optional_import_names()), [(2, "pyarrow")])

    def test_real_import_guards_are_recognized(self):
        guarded = ast.parse(
            "try:\n    import pyarrow\nexcept ImportError:\n    pyarrow = None\n"
            "if TYPE_CHECKING:\n    import sqlalchemy\n"
        )
        self.assertEqual(_scan_statements(guarded.body, _optional_import_names()), [])


if __name__ == "__main__":
    unittest.main()
