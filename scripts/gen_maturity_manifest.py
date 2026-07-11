"""Generate ``maturity_manifest.json``: every top-level public mixle surface with its maturity tier.

Worklist A1.6 -- the machine-readable maturity manifest. It joins the *public surface* (the top-level
public subpackages/modules of ``mixle``) with the *maturity registry* (:mod:`mixle.maturity`, worklist
A1.2), so a tool can answer "what is the maturity of every public entry point?" from one committed file.

Run ``python scripts/gen_maturity_manifest.py`` to regenerate the manifest; ``--check`` exits non-zero if
the committed file is stale (used by the drift test). When the public-API manifest (worklist A1.1) lands,
this maturity field can be folded into it; until then it is a standalone, deterministic artifact.
"""

from __future__ import annotations

import json
import os
import pkgutil
import sys

import mixle
from mixle.maturity import maturity_of, status_of

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(_REPO_ROOT, "maturity_manifest.json")


def _public_surfaces() -> list[str]:
    """Top-level public subpackages/modules of ``mixle`` (dotted names), sorted and deterministic."""
    base = os.path.dirname(mixle.__file__)
    return sorted(f"mixle.{m.name}" for m in pkgutil.iter_modules([base]) if not m.name.startswith("_"))


def build_manifest() -> dict:
    """Return the maturity manifest as a plain dict (public surface -> tier + human status)."""
    return {
        "artifact": "mixle.maturity_manifest/v1",
        "surfaces": {
            name: {"maturity": maturity_of(name).value, "status": status_of(name)} for name in _public_surfaces()
        },
    }


def render() -> str:
    """The exact JSON text the manifest file should contain (sorted, trailing newline)."""
    return json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    text = render()
    if "--check" in argv:
        try:
            current = open(MANIFEST_PATH).read()
        except OSError:
            current = None
        if current != text:
            print("maturity_manifest.json is stale; run: python scripts/gen_maturity_manifest.py", file=sys.stderr)
            return 1
        return 0
    with open(MANIFEST_PATH, "w") as f:
        f.write(text)
    print(f"wrote {MANIFEST_PATH} ({len(build_manifest()['surfaces'])} surfaces)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
