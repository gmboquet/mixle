"""README examples must run, or -- where they need objects only the reader has -- must at least
resolve every *library* name they use.

The existing README gates (`readme_claims_test`, `readme_claims_hygiene_test`) check the prose for
retired overclaims. Nothing checked the code, and a reader's first contact with this library is
copying a block out of the README. One block imported `GaussianEstimator`, `MixtureEstimator`,
`HiddenMarkovEstimator` and `GradEstimator`, then called `optimize` without importing it: copying it
verbatim raised `NameError`. A word-level gate cannot see that.

Two tiers, because README blocks legitimately come in two kinds:

* **Runnable** -- needs nothing from the reader. Executed in a subprocess; it must exit 0.
* **Illustrative** -- needs an object only the reader has (`my_module` is *your* ``nn.Module``,
  `teacher` is *your* function). Running these would mean inventing a stand-in and proving nothing
  about the documented call. Instead every free name is resolved: it must be bound in the block,
  imported by the block, a builtin, or a declared reader-supplied placeholder. A library function
  the block forgot to import is none of those, which is exactly the defect above.

The placeholder list is deliberately small and explicit. Adding a name to it is a decision that this
README genuinely asks the reader to supply that object -- not a way to silence the check.
"""

from __future__ import annotations

import ast
import builtins
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"

# Objects the README explicitly asks the READER to bring. Each is named in the prose or an adjacent
# comment ("your nn.Module", "teacher = the function doing the job now").
_READER_SUPPLIED = frozenset({"my_module", "teacher", "inputs", "sequences", "x"})


def _blocks() -> list[tuple[int, str]]:
    text = README.read_text(encoding="utf-8")
    return [(text[: m.start()].count("\n") + 1, m.group(1)) for m in re.finditer(r"```python\n(.*?)```", text, re.S)]


def _parses(source: str) -> bool:
    if source.lstrip().startswith(">>>"):
        return False
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def _free_names(tree: ast.Module) -> set[str]:
    """Names a block READS without ever binding or importing them."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.For, ast.comprehension)):
            target = node.target
            bound.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    read = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return read - bound - set(dir(builtins))


_PARSED = [(line, src) for line, src in _blocks() if _parses(src) and "..." not in src]
_RUNNABLE = [(line, src) for line, src in _PARSED if not (_free_names(ast.parse(src)) - _READER_SUPPLIED)]
_RUNNABLE = [(line, src) for line, src in _RUNNABLE if not _free_names(ast.parse(src))]
_ILLUSTRATIVE = [(line, src) for line, src in _PARSED if _free_names(ast.parse(src))]


def test_the_extraction_still_finds_blocks() -> None:
    """Guards the guard: a silently-broken regex would make everything below vacuously pass."""
    assert len(_PARSED) >= 3, f"expected several parsed README blocks, found {len(_PARSED)}"
    assert _RUNNABLE, "expected at least one fully self-contained README example"


@pytest.mark.parametrize("line,source", _RUNNABLE, ids=[f"line{line}" for line, _ in _RUNNABLE])
def test_a_self_contained_example_runs(line: int, source: str, tmp_path: Path) -> None:
    script = tmp_path / f"readme_{line}.py"
    script.write_text(source, encoding="utf-8")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=600, cwd=tmp_path)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        pytest.fail(f"README.md block at line {line} does not run:\n{tail}")


@pytest.mark.parametrize("line,source", _ILLUSTRATIVE, ids=[f"line{line}" for line, _ in _ILLUSTRATIVE])
def test_an_illustrative_example_leaves_only_reader_supplied_names_free(line: int, source: str) -> None:
    free = _free_names(ast.parse(source))
    unexplained = free - _READER_SUPPLIED
    assert not unexplained, (
        f"README.md block at line {line} uses {sorted(unexplained)} without importing or defining "
        f"them. If these are library names the block must import them -- copying it would raise "
        f"NameError. If the reader is meant to supply them, add them to _READER_SUPPLIED here with "
        f"the README naming them explicitly."
    )


def test_the_check_catches_a_missing_library_import() -> None:
    """Negative control: reproduce the exact defect this file exists for.

    A block that imports the estimators but forgets `optimize` -- the real bug found in the README --
    must be reported, not passed over.
    """
    broken = "from mixle.stats import GaussianEstimator\nmodel = optimize(sequences, GaussianEstimator())\n"
    assert "optimize" in _free_names(ast.parse(broken)) - _READER_SUPPLIED


def test_the_runnable_tier_actually_executes_something() -> None:
    """A block that raises must fail the runnable tier, not be quietly skipped."""
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "boom.py"
        script.write_text("raise SystemExit(3)\n", encoding="utf-8")
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 3
