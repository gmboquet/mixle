"""Worklist X12.2 -- the README quickstart must run exactly as written.

The quickstart previously opened with a placeholder ``records = [...]`` that a new user
could not execute without inventing their own data. It now ships concrete heterogeneous
rows (numeric, category, boolean, and a missing value). This test pulls the *exact*
fenced block out of README.md and executes it, so the published snippet cannot silently
rot: if the API changes, or someone reintroduces a placeholder, this fails. That is the
X12.2 acceptance -- "run the exact fenced block from the clean wheel in CI".

We run only the first quickstart block -- the ``optimize(records ...)`` one -- because
the later snippets reference names (teacher, inputs, my_module, sequences) that are
illustrative and intentionally not self-contained.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent.parent / "README.md"

_FENCE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _quickstart_block() -> str:
    if not README.is_file():
        pytest.skip(f"README not found at {README}")
    text = README.read_text(encoding="utf-8")
    for body in _FENCE_RE.findall(text):
        # The quickstart data block is the self-contained one that builds `records`
        # and fits them with a bare optimize() call.
        if "records = [" in body and "optimize(records" in body:
            return body
    pytest.skip("no self-contained quickstart optimize(records) block found in README")
    raise AssertionError  # unreachable; keeps type checkers happy


def test_quickstart_block_has_no_placeholder() -> None:
    block = _quickstart_block()
    assert "[...]" not in block, (
        "README quickstart still contains a '[...]' placeholder; a new user cannot run "
        "it without inventing data (X12.2)."
    )


def test_quickstart_block_is_heterogeneous_with_missing() -> None:
    """The rows must exercise numeric, category, boolean, and a missing value."""
    block = _quickstart_block()
    assert '"free"' in block or "'free'" in block, "quickstart lost its category field"
    assert "True" in block or "False" in block, "quickstart lost its boolean field"
    assert "None" in block, "quickstart no longer includes a missing value"


def test_quickstart_block_executes_as_written() -> None:
    block = _quickstart_block()
    # Execute the fenced snippet verbatim in a fresh namespace. A new user pasting it
    # must get a fitted model with a working score and sampler -- no exceptions.
    namespace: dict[str, object] = {}
    exec(compile(block, "<README quickstart>", "exec"), namespace)

    model = namespace.get("model")
    assert model is not None, "quickstart block did not bind `model`"

    records = namespace.get("records")
    assert isinstance(records, list) and records, "quickstart block did not bind `records`"

    # The two showcased operations must actually work on the fitted model.
    ld = model.log_density(records[0])  # type: ignore[attr-defined]
    assert isinstance(ld, float) or hasattr(ld, "__float__"), "log_density is not a scalar"
    drawn = model.sampler().sample(5)  # type: ignore[attr-defined]
    assert len(list(drawn)) == 5, "sampler().sample(5) did not return five rows"
