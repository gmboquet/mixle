#!/usr/bin/env python
"""Generate the public-API manifest (worklist A1.1).

The manifest is the declared public surface of ``mixle``: for every package ``__init__.py`` that
defines ``__all__``, the sorted list of exported names, tagged with that package's maturity tier from
the worklist A1.2 registry (:mod:`mixle.maturity`) -- ``stable``, ``provisional``, or ``experimental``.
It is produced by **static AST parsing**, not by importing anything, so it is deterministic and
independent of which optional dependencies happen to be installed -- the same manifest is produced in a
base env, the full env, or CI. Maturity resolution is likewise dependency-free: see
``_load_maturity_of`` below.

The maturity tag is what lets the drift test (``mixle/tests/public_api_manifest_test.py``) tell a real
freeze violation (a ``stable``/``provisional`` symbol added, removed, or renamed) from expected
experimental churn (``mixle.experimental.*`` -- see ``mixle/experimental/README.md``: "import it
expecting churn", graduation "not yet enforced"). Before this tag existed, a reviewer diffing the
manifest saw every package's changes identically, including the ~170 names under
``mixle.experimental.typed_runtime`` that carry no compatibility guarantee at all. A1.1 itself calls for
recording each entry's maturity; this folds in the A1.2 registry, per the note in
``scripts/gen_maturity_manifest.py``.

The committed ``manifests/api_manifest.json`` is the baseline the drift test compares against. When a PR
intentionally changes the public surface, regenerate it::

    python scripts/gen_api_manifest.py

and commit the result -- so that every change to what users can import from ``mixle`` is a reviewed,
recorded diff, per the 0.8.0 feature freeze.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Resolve runtime-built ``__all__`` values from this checkout, never from an unrelated editable install.
sys.path.insert(0, str(REPO_ROOT))
MANIFEST_PATH = REPO_ROOT / "manifests" / "api_manifest.json"
ARTIFACT_ID = "mixle.api_manifest/v1"


def _load_maturity_of(repo_root: Path) -> Callable[[str], str]:
    """Load ``mixle.maturity.maturity_of`` by file path and return a ``str``-returning wrapper.

    ``mixle/maturity.py`` has zero internal dependencies (stdlib ``enum`` only), but reaching it via a
    normal ``import mixle.maturity`` still runs the parent ``mixle/__init__.py`` first -- which can fail
    in a base/minimal env (a missing optional dependency; the same reason dynamic packages below can come
    back ``unresolved``), and, done at the wrong point relative to the ``mixle.reason`` warmup in
    :func:`build_manifest`, can retrigger the ``mixle.stats.bayes.dirichlet`` <-> ``mixle.reason``
    circular import that warmup exists to avoid. Loading the file directly sidesteps both: maturity
    resolution never depends on the rest of ``mixle`` importing cleanly, matching this script's existing
    no-fragile-imports design.
    """
    spec = importlib.util.spec_from_file_location("_mixle_maturity_standalone", repo_root / "mixle" / "maturity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda name: module.maturity_of(name).value


def _string_literals(node: ast.expr) -> list[str] | None:
    """The list of string constants in a list/tuple literal, or literal-list concatenations of them.

    Returns ``None`` if ``node`` is not a fully-static literal we can resolve (e.g. a comprehension or a
    reference to another name) -- the caller records that package as dynamically-built rather than
    guessing.
    """
    if isinstance(node, (ast.List, ast.Tuple)):
        out: list[str] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
            else:
                return None
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_literals(node.left)
        right = _string_literals(node.right)
        if left is None or right is None:
            return None
        return left + right
    return None


def _extract_all(path: Path) -> tuple[list[str], bool]:
    """Return (sorted unique ``__all__`` names, fully_static). ``fully_static`` is False if any
    statement assigning/extending ``__all__`` used a non-literal value we could not resolve."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    fully_static = True
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AugAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        literals = _string_literals(value)
        if literals is None:
            fully_static = False
        else:
            names.extend(literals)
    return sorted(set(names)), fully_static


def build_manifest(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    """Return ``{"artifact": ..., "packages": {dotted: entry, ...}}`` across every ``mixle`` package
    ``__init__.py`` that declares an ``__all__``.

    Each entry is ``{"maturity": <tier>, "names": [...]}`` for a resolved package, or
    ``{"maturity": <tier>, "unresolved": <ExceptionType>}`` for a runtime-assembled package (``mixle``,
    ``mixle.stats``, ``mixle.utils``) whose import failed because of a missing optional dependency in the
    current env -- the drift test skips those rather than falsely diffing against a manifest generated in
    a fuller env. ``maturity`` (worklist A1.2, via :func:`_load_maturity_of`) is always present regardless
    of resolution status: it is a pure dotted-name lookup against the registry, never an import of the
    package itself.

    A package whose ``__all__`` is a static literal is resolved by AST parse (no import). A package
    that assembles ``__all__`` at runtime (``mixle``, ``mixle.stats``, ``mixle.utils``) is resolved by
    importing it; if that import fails (a missing optional dependency in the current env) it is
    recorded as ``{"unresolved": <ExceptionType>}`` so the drift test can skip it rather than falsely
    diff against a manifest generated in a fuller env.

    ``mixle.stats.bayes.dirichlet`` and ``mixle.reason`` have a real circular import between them
    (``mixle.stats.bayes.dirichlet`` -> ``mixle.inference`` -> ``mixle.analysis`` ->
    ``mixle.reason.posterior_protocol`` -> ... -> ``mixle.stats.latent.mixture`` -> back to
    ``mixle.stats.bayes.dirichlet``). Importing ``mixle.stats`` first re-enters
    ``mixle.stats.bayes.dirichlet`` while it is still mid-init and raises a spurious ImportError;
    importing ``mixle.reason`` first lets that module finish completely before anything re-enters it,
    so warm it here before resolving the runtime packages below.
    """
    try:
        importlib.import_module("mixle.reason")
    except Exception:  # noqa: BLE001 -- a genuinely missing dependency still surfaces per-package below
        pass

    maturity_of = _load_maturity_of(repo_root)
    pkg_root = repo_root / "mixle"
    packages: dict[str, object] = {}
    for init in sorted(pkg_root.rglob("__init__.py")):
        names, fully_static = _extract_all(init)
        if not names and fully_static:
            continue  # no __all__ declared here
        dotted = ".".join(init.relative_to(repo_root).parent.parts)
        maturity = maturity_of(dotted)
        if fully_static:
            packages[dotted] = {"maturity": maturity, "names": names}
        else:
            try:
                module = importlib.import_module(dotted)
                packages[dotted] = {"maturity": maturity, "names": sorted(getattr(module, "__all__", []))}
            except Exception as exc:  # noqa: BLE001 -- a missing optional dep must not break the manifest
                packages[dotted] = {"maturity": maturity, "unresolved": type(exc).__name__}
    return {"artifact": ARTIFACT_ID, "packages": packages}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    manifest = build_manifest()
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if "--check" in argv:
        try:
            current = MANIFEST_PATH.read_text(encoding="utf-8")
        except OSError:
            current = None
        if current != rendered:
            print(
                "manifests/api_manifest.json is stale; run: python scripts/gen_api_manifest.py",
                file=sys.stderr,
            )
            return 1
        return 0
    MANIFEST_PATH.write_text(rendered, encoding="utf-8")
    packages = manifest["packages"]
    total = sum(len(v["names"]) for v in packages.values() if "names" in v)
    experimental = sum(1 for v in packages.values() if v.get("maturity") == "experimental")
    print(
        f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}: {len(packages)} packages, {total} public names "
        f"({experimental} package(s) tagged experimental)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
