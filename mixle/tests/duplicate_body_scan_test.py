"""Worklist Y4.5 -- gate against new copy-pasted bodies (the sibling-bug prevention).

``scripts/scan_duplicate_bodies.py`` finds functions in ``mixle/stats`` and
``mixle/models`` that share an identical (non-trivial) body -- copy-paste that will bite
when a fix lands on one copy and not the other. This test is the shared prevention
mechanism: it re-runs the scan and fails if a duplicate group appears that is not in the
reviewed manifest.

When it fails, the fix is one of:
  * de-duplicate the logic (preferred, when behavior must stay identical), or
  * if the duplication is deliberate and justified, regenerate the manifest with
    ``python scripts/scan_duplicate_bodies.py --write`` and explain why in review.

The manifest is a ratchet: the current set of duplicates must be a subset of it, so the
count can only go down without an explicit, reviewed manifest update.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCANNER = Path(__file__).resolve().parent.parent.parent / "scripts" / "scan_duplicate_bodies.py"


def _load_scanner():
    if not _SCANNER.is_file():
        pytest.skip(f"scanner not found at {_SCANNER}")
    spec = importlib.util.spec_from_file_location("scan_duplicate_bodies", _SCANNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_new_duplicate_bodies() -> None:
    scanner = _load_scanner()
    current = scanner.group_signatures(scanner.scan())
    baseline = scanner.group_signatures(scanner.load_manifest())

    new = current - baseline
    assert not new, (
        "new copy-pasted function bodies detected in mixle/stats or mixle/models "
        "(Y4.5 sibling-bug risk). De-duplicate them, or if intentional, run "
        "`python scripts/scan_duplicate_bodies.py --write` and justify in review.\n"
        + "\n".join("  - " + " == ".join(sig) for sig in sorted(new))
    )


def test_manifest_is_not_stale() -> None:
    """Every manifest entry should still be a real duplicate (keeps the baseline honest)."""
    scanner = _load_scanner()
    current = scanner.group_signatures(scanner.scan())
    baseline = scanner.group_signatures(scanner.load_manifest())

    resolved = baseline - current
    assert not resolved, (
        "the manifest lists duplicates that no longer exist -- prune them with "
        "`python scripts/scan_duplicate_bodies.py --write`:\n"
        + "\n".join("  - " + " == ".join(sig) for sig in sorted(resolved))
    )


def test_scanner_finds_a_planted_duplicate(tmp_path: Path) -> None:
    """Negative control: two identical non-trivial bodies in a fake tree are detected."""
    scanner = _load_scanner()
    body = (
        "        a = 1\n"
        "        b = a + 1\n"
        "        c = b + 1\n"
        "        d = c + 1\n"
        "        e = d + 1\n"
        "        f = e + 1\n"
        "        return f\n"
    )
    pkg = tmp_path / "mixle" / "stats"
    pkg.mkdir(parents=True)
    (tmp_path / "mixle" / "models").mkdir(parents=True)
    (pkg / "a.py").write_text(f"def foo():\n{body}", encoding="utf-8")
    (pkg / "b.py").write_text(f"def bar():\n{body}", encoding="utf-8")

    groups = scanner.scan(root=tmp_path)
    assert any(len(g["locations"]) == 2 for g in groups), "planted duplicate was not detected"
