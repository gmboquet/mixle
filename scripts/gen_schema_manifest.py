"""Generate ``manifests/serialization_schema_manifest.json``: the versioned catalog of serializable schema types.

Worklist M11.1 -- a versioned schema manifest. mixle's persistence format is type-tagged JSON: every
serializable class registers a stable ``__pysp_type__`` id (:func:`mixle.utils.serialization.
serializable_class_ids`), and a saved payload names those ids. That set of ids *is* the serialization
schema surface -- adding, renaming, or removing one is a schema change that a saved artifact's
loadability depends on. This manifest records the full set so any such change is visible in a diff and
must be acknowledged (paired with the cross-version load fixtures, worklist M11.2).

Run ``python scripts/gen_schema_manifest.py`` to regenerate. Generate it with the optional backends
installed so it captures the *full* schema surface (torch-backed leaves included); the drift test tolerates
a base environment by requiring the live set to be a subset of the recorded manifest.
"""

from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Pin imports to this checkout rather than whichever editable Mixle install happens to be active.
sys.path.insert(0, _REPO_ROOT)

from mixle.utils.serialization import TAG, serializable_class_ids

# Bump when the manifest's own shape changes, not when the type set changes (that shows in registered_types).
SCHEMA_MANIFEST_VERSION = "1"
MANIFEST_PATH = os.path.join(_REPO_ROOT, "manifests", "serialization_schema_manifest.json")


def build_manifest() -> dict:
    """Return the schema manifest: the type tag, the manifest version, and every registered type id."""
    return {
        "artifact": "mixle.serialization_schema_manifest/v1",
        "schema_manifest_version": SCHEMA_MANIFEST_VERSION,
        "tag": TAG,
        "registered_types": sorted(serializable_class_ids()),
    }


def render() -> str:
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
            print(
                "manifests/serialization_schema_manifest.json is stale; run: python scripts/gen_schema_manifest.py",
                file=sys.stderr,
            )
            return 1
        return 0
    with open(MANIFEST_PATH, "w") as f:
        f.write(text)
    print(f"wrote {MANIFEST_PATH} ({len(build_manifest()['registered_types'])} registered types)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
