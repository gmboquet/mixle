"""Base-install import safety (worklist P2.2).

Every ``mixle`` package module must import cleanly with only the base dependencies installed -- an
optional dependency (torch, numba, mpi4py, ...) may only be imported lazily (inside a function), behind
a ``try/except ImportError``, or through the ``mixle.utils.optional_deps`` shim. A bare top-level
``import torch`` in a package module breaks ``import mixle...`` for every base-install user and silently
regressed the wheel twice before (numba and mpi4py, fixed in the 0.7.0 pass).

This gate catches that statically -- by AST, without importing anything -- so it runs in any
environment and pins the property regardless of which extras happen to be installed in CI. Test files
are excluded: they may import optional deps at top level and are gated by pytest markers/skips, not by
base-install import safety.
"""

import ast
import unittest
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]

# Third-party packages that are optional extras, not base dependencies. Keep in sync with the
# ``[project.optional-dependencies]`` table in pyproject.toml. A top-level import of any of these in a
# non-test package module is a base-install breakage.
_OPTIONAL_TOP_LEVEL = frozenset(
    {
        "numba",
        "pyspark",
        "gmpy2",
        "mpmath",
        "zarr",
        "h5py",
        "pandas",
        "mpi4py",
        "torch",
        "jax",
        "jaxlib",
        "flax",
        "optax",
        "ray",
        "dask",
        "distributed",
        "umap",
        "transformers",
        "peft",
        "lightning",
        "pytorch_lightning",
        "cupy",
        "sklearn",
        "hmmlearn",
        "mamba_ssm",
        "triton",
        "safetensors",
        "datasets",
        "accelerate",
        "sentencepiece",
        "tokenizers",
    }
)


def _unguarded_optional_imports(path: Path) -> list[tuple[int, str]]:
    """Top-level (module-scope) imports of an optional dependency. Imports nested inside a function,
    ``try``, ``if``, ``with``, or class body are not module-scope and so are not flagged."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[tuple[int, str]] = []
    for node in tree.body:  # direct children of the module == true top level
        tops: list[str] = []
        if isinstance(node, ast.Import):
            tops = [alias.name.split(".", 1)[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            tops = [node.module.split(".", 1)[0]]
        for top in tops:
            if top in _OPTIONAL_TOP_LEVEL:
                hits.append((node.lineno, top))
    return hits


class OptionalImportGuardTest(unittest.TestCase):
    def test_no_unguarded_top_level_optional_imports(self):
        offenders: list[str] = []
        for path in sorted(_PKG_ROOT.rglob("*.py")):
            if "tests" in path.relative_to(_PKG_ROOT).parts:
                continue
            for lineno, dep in _unguarded_optional_imports(path):
                offenders.append(f"  {path.relative_to(_PKG_ROOT.parent)}:{lineno}: top-level import of {dep!r}")
        self.assertFalse(
            offenders,
            "optional dependencies imported at module top level break the base install; import them "
            "lazily (inside a function), behind try/except ImportError, or via mixle.utils.optional_deps:\n"
            + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
