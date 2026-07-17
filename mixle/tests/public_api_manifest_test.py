"""Public-API drift gate (worklist A1.1).

``api_manifest.json`` at the repo root is the reviewed, committed record of everything ``mixle``
exports -- the ``__all__`` of every public package, each tagged with its maturity tier from the
worklist A1.2 registry (:mod:`mixle.maturity`). This test regenerates that manifest from the current
tree and asserts the ``stable``/``provisional`` surface is unchanged, so that adding, removing, or
renaming one of those public symbols is never accidental: it forces a manifest diff into the PR, which
is exactly the reviewable surface the 0.8.0 feature freeze requires.

Drift confined to a package tagged ``experimental`` (today, only ``mixle.experimental.typed_runtime``)
does *not* fail this test -- it is printed instead. ``mixle/experimental/README.md`` says to "import it
expecting churn" and its graduation gate is "not yet enforced"; the freeze rule itself (worklist
Sec 1.3.5) exempts experimental work. Folding all ~170 of those names into the same blocking assertion
as the frozen stable core would mean either the gate cries wolf on every expected experimental change,
or (worse) reviewers learn to wave the whole diff through. A package's maturity tier itself changing
(e.g. graduating out of ``mixle.experimental``) is never treated as exempt churn -- see
:func:`_partition_drift`.

When a change intentionally alters the public surface, regenerate and commit::

    python scripts/gen_api_manifest.py

Packages whose ``__all__`` is assembled at runtime are resolved by import; if the current environment
cannot import one (a missing optional dependency), that package is skipped here rather than allowed to
falsely diff against a manifest generated in a fuller environment.
"""

import importlib.util
import json
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _REPO_ROOT / "api_manifest.json"
_GEN_PATH = _REPO_ROOT / "scripts" / "gen_api_manifest.py"

_VALID_MATURITIES = {"stable", "provisional", "experimental"}


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_api_manifest", _GEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolvable(packages, key):
    # A runtime-assembled package the current env cannot import is skipped on both sides, so a missing
    # optional dep never turns into a false public-API diff.
    return not (isinstance(packages.get(key), dict) and "unresolved" in packages[key])


def _maturity(entry):
    return entry.get("maturity") if isinstance(entry, dict) else None


def _partition_drift(committed_packages, current_packages):
    """Diff two ``packages`` maps (the shape ``gen_api_manifest.build_manifest`` returns under its
    ``"packages"`` key) into ``(blocking, experimental_only)`` human-readable message lists.

    A key's drift counts as *experimental-only* -- expected churn, not a freeze violation -- exactly when
    every maturity tier attached to it (whichever of the committed/current side is present) is
    ``"experimental"``. A brand-new or removed package is judged by whichever side exists; a package
    whose tier itself changes (e.g. graduating out of ``mixle.experimental``) is always ``blocking``,
    since that reclassification is exactly the kind of decision ``release-checklists/0.8.0-decisions.md``
    exists to record, not something to wave through silently.
    """
    keys = sorted(
        k
        for k in set(committed_packages) | set(current_packages)
        if _resolvable(committed_packages, k) and _resolvable(current_packages, k)
    )
    blocking, experimental_only = [], []
    for key in keys:
        want = committed_packages.get(key)
        got = current_packages.get(key)
        if want == got:
            continue
        want_names, got_names = set((want or {}).get("names", [])), set((got or {}).get("names", []))
        added, removed = sorted(got_names - want_names), sorted(want_names - got_names)
        tiers = {t for t in (_maturity(want), _maturity(got)) if t is not None}
        line = f"  {key} [{'/'.join(sorted(tiers)) or '?'}]: +{added} -{removed}"
        (experimental_only if tiers == {"experimental"} else blocking).append(line)
    return blocking, experimental_only


class PublicApiManifestTest(unittest.TestCase):
    def test_manifest_files_exist(self):
        self.assertTrue(_MANIFEST_PATH.exists(), "api_manifest.json missing -- run scripts/gen_api_manifest.py")
        self.assertTrue(_GEN_PATH.exists(), "scripts/gen_api_manifest.py missing")

    def test_public_surface_matches_committed_manifest(self):
        committed = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))["packages"]
        current = _load_generator().build_manifest(_REPO_ROOT)["packages"]

        blocking, experimental_only = _partition_drift(committed, current)

        if experimental_only:
            # Expected churn under an experimental-tier package (mixle/experimental/README.md: "import
            # it expecting churn"; graduation "not yet enforced") -- surfaced for visibility, but it does
            # not fail the freeze gate.
            print(
                "\nexperimental-surface drift from api_manifest.json (expected churn, not a freeze "
                "violation; regenerate with `python scripts/gen_api_manifest.py` when convenient):\n"
                + "\n".join(experimental_only)
            )

        self.assertFalse(
            blocking,
            "public API drifted from api_manifest.json on a stable/provisional package (regenerate with "
            "`python scripts/gen_api_manifest.py` and commit; if this adds surface, the freeze rule "
            "requires a written exception in release-checklists/0.8.0-decisions.md):\n" + "\n".join(blocking),
        )

    def test_dynamic_packages_resolve_cleanly(self):
        """``mixle``, ``mixle.stats``, and ``mixle.utils`` assemble ``__all__`` at runtime and must
        actually resolve here -- every dependency they need is installed in this environment, none is
        optional. A prior regression imported ``mixle.stats`` before ``mixle.reason`` had a chance to
        finish initializing, tripping a real circular-import chain (``mixle.stats.bayes.dirichlet`` ->
        ``mixle.inference`` -> ``mixle.analysis`` -> ``mixle.reason`` -> ``mixle.stats.latent.mixture``
        -> back to ``mixle.stats.bayes.dirichlet``) and silently recording ``mixle.stats`` as
        ``{"unresolved": "ImportError"}`` -- which ``_resolvable`` above then treats exactly like a
        legitimately-missing optional dependency and skips, defeating drift coverage for the whole
        package. Assert that does not happen."""
        current = _load_generator().build_manifest(_REPO_ROOT)["packages"]
        for key in ("mixle", "mixle.stats", "mixle.utils"):
            entry = current.get(key)
            names = entry.get("names") if isinstance(entry, dict) else None
            self.assertIsInstance(
                names,
                list,
                f"{key} failed to resolve ({entry!r}) even though its dependencies are all installed "
                "here -- likely an import-ordering regression, not a genuinely missing optional dep",
            )

    def test_every_package_has_a_valid_maturity_tag(self):
        """Every entry -- resolved or not -- carries a maturity tag from the worklist A1.2 registry, and
        the anchor this whole split exists for (``mixle.experimental.typed_runtime``, ~170 churn-expected
        names -- see ``mixle/experimental/README.md``) is specifically tagged ``experimental``."""
        current = _load_generator().build_manifest(_REPO_ROOT)["packages"]
        self.assertGreater(len(current), 0)
        for key, entry in current.items():
            self.assertIn(entry.get("maturity"), _VALID_MATURITIES, f"{key} has no valid maturity tag: {entry!r}")
        if "mixle.experimental.typed_runtime" in current:
            self.assertEqual(current["mixle.experimental.typed_runtime"]["maturity"], "experimental")
        self.assertEqual(current.get("mixle.stats", {}).get("maturity"), "stable")


class DriftPartitionTest(unittest.TestCase):
    """Negative-control coverage (worklist T3.6: "a gate that can't fail is not a gate") for the
    blocking/experimental-churn split itself, against synthetic fixtures rather than real source -- this
    is what actually demonstrates the freeze gate distinguishes a real violation from expected
    experimental churn, independent of whatever happens to be true of the tree on a given day."""

    def test_stable_surface_drift_is_blocking(self):
        committed = {"mixle.stats": {"maturity": "stable", "names": ["A", "B"]}}
        current = {"mixle.stats": {"maturity": "stable", "names": ["A"]}}  # "B" silently removed
        blocking, experimental_only = _partition_drift(committed, current)
        self.assertEqual(len(blocking), 1)
        self.assertEqual(experimental_only, [])

    def test_provisional_surface_drift_is_blocking(self):
        # Only "experimental" is exempt (worklist Sec 1.3.5) -- provisional growth still needs review.
        committed = {"mixle.task": {"maturity": "provisional", "names": ["A"]}}
        current = {"mixle.task": {"maturity": "provisional", "names": ["A", "NewThing"]}}
        blocking, experimental_only = _partition_drift(committed, current)
        self.assertEqual(len(blocking), 1)
        self.assertEqual(experimental_only, [])

    def test_experimental_surface_drift_is_not_blocking(self):
        committed = {"mixle.experimental.typed_runtime": {"maturity": "experimental", "names": ["A", "B"]}}
        current = {"mixle.experimental.typed_runtime": {"maturity": "experimental", "names": ["A", "C"]}}
        blocking, experimental_only = _partition_drift(committed, current)
        self.assertEqual(blocking, [])
        self.assertEqual(len(experimental_only), 1)

    def test_graduation_out_of_experimental_is_still_blocking(self):
        # A package's tier itself changing (e.g. graduating to provisional) must not be swallowed by the
        # experimental exemption -- that reclassification belongs in the decision log, not silently passed.
        committed = {"mixle.experimental.typed_runtime": {"maturity": "experimental", "names": ["A"]}}
        current = {"mixle.experimental.typed_runtime": {"maturity": "provisional", "names": ["A"]}}
        blocking, experimental_only = _partition_drift(committed, current)
        self.assertEqual(len(blocking), 1)
        self.assertEqual(experimental_only, [])

    def test_unresolved_packages_are_skipped_on_both_sides(self):
        committed = {"mixle": {"maturity": "provisional", "names": ["A"]}}
        current = {"mixle": {"maturity": "provisional", "unresolved": "ImportError"}}
        blocking, experimental_only = _partition_drift(committed, current)
        self.assertEqual(blocking, [])
        self.assertEqual(experimental_only, [])

    def test_unchanged_manifest_has_no_drift(self):
        committed = {
            "mixle.stats": {"maturity": "stable", "names": ["A"]},
            "mixle.experimental.typed_runtime": {"maturity": "experimental", "names": ["X"]},
        }
        blocking, experimental_only = _partition_drift(committed, dict(committed))
        self.assertEqual(blocking, [])
        self.assertEqual(experimental_only, [])


if __name__ == "__main__":
    unittest.main()
