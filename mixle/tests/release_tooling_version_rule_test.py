"""A pre-release candidate is the release it rehearses, everywhere the release tooling compares them.

D-0216 rehearses the publication path on a real pre-release cut from the release tree with only the
version string changed. Three tools compare a version against the release: the document-state check,
the reproduction-bundle binding in ``run_repro_entry``, and the example-execution manifest. The first
already resolved ``0.8.2rc1`` to ``0.8.2``; the other two compared literally, so on the rehearsal the
bundle binding called every receipt's candidate record a version mismatch. They must agree, and the
retained records the binding requires must be named for the candidate rather than for the release --
a rehearsal writes ``metadata/mixle-0.8.2rc1-py3-none-any.whl.json``.
"""

import fnmatch
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "release-checklists" / "0.8.2-repro-bundle.json"


def _script(name):
    spec = importlib.util.spec_from_file_location("_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CASES = {
    "0.8.2": "0.8.2",
    "0.8.2rc1": "0.8.2",
    "0.8.2rc12": "0.8.2",
    "0.8.2a3": "0.8.2",
    "0.8.2b1": "0.8.2",
    "0.8.2.post1": "0.8.2",
    "0.8.2rc1.dev4": "0.8.2",
    "0.8.3": "0.8.3",
    "0.9.0rc1": "0.9.0",
}


def test_every_implementation_resolves_a_pre_release_to_its_release():
    resolvers = {
        "check_release_document_state.base_release": _script("check_release_document_state").base_release,
        "run_repro_entry.base_release": _script("run_repro_entry").base_release,
        "build_example_execution_manifest.base_release": _script("build_example_execution_manifest").base_release,
    }
    for name, resolve in resolvers.items():
        for version, expected in CASES.items():
            assert resolve(version) == expected, f"{name}({version!r}) != {expected!r}"


def test_a_pre_release_binds_to_its_release_and_another_release_does_not():
    resolve = _script("run_repro_entry").base_release
    release = json.loads(BUNDLE.read_text(encoding="utf-8"))["release"]
    assert resolve("%src1" % release) == release, "the rehearsal cut must bind to the bundle it rehearses"
    assert resolve("0.9.0rc1") != release, "a different release must not bind to this bundle"
    # Non-strings are returned unchanged so the caller's own type check still reports the problem.
    assert resolve(None) is None


def test_the_bundle_requires_the_wheel_record_by_pattern_not_by_release_name():
    required = json.loads(BUNDLE.read_text(encoding="utf-8"))["candidate_binding"]["required_records"]
    wheel_patterns = [pattern for pattern in required if pattern.endswith(".whl.json")]
    assert len(wheel_patterns) == 1, "exactly one wheel record is required"
    pattern = wheel_patterns[0]
    for version in ("0.8.2", "0.8.2rc1"):
        name = "metadata/mixle-%s-py3-none-any.whl.json" % version
        assert fnmatch.fnmatch(name, pattern), f"{pattern!r} does not match {name!r}"
