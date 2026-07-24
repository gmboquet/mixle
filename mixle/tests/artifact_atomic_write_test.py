"""Artifact writes are atomic: a failed serialization never corrupts an artifact (worklist S13.5 / M11.3).

``save_json`` and the manifest writer used ``open(path, "w")``, which truncates the target *before*
serialization runs. A non-serializable model -- or a crash mid-``json.dump`` -- therefore left a truncated,
unloadable ``model.json`` and, worse, destroyed any previous good artifact at that path. The write now goes
through a temp file swapped in with ``os.replace``, so it is all-or-nothing. These are the failure-injection
tests for that contract: the operation fails loudly, the prior artifact survives, and no temp file leaks.

That single-file guarantee left a second, related gap: ``save_module``/``save_json``/``save_arrays`` each write
a *payload* file (weights/model/arrays) and then, as a separate step, the manifest describing it. A failure
between those two steps -- payload write succeeds, manifest write fails or the process dies -- left the
directory holding a NEW payload paired with the OLD manifest/provenance: individually well-formed files, but a
mismatched pair that ``load_*`` cannot detect as corrupt (it loads the new payload next to stale metadata
describing the run that produced the *previous* payload). ``_atomic_artifact_write`` closes that gap: the whole
generation (payload + manifest) is staged in a sibling directory and published to ``path`` with a single
directory-level swap, so a failure anywhere before publish leaves the OLD generation fully intact, never mixed
with any part of the new one. ``SaveJsonAtomicUpdateTest`` / ``SaveArraysAtomicUpdateTest`` below are the
failure-injection tests for that contract on the two torch-free payloads (the torch payload's equivalent,
``SaveModuleAtomicUpdateTest``, lives in ``task_artifact_test.py`` alongside this module's other torch tests).
"""

import glob
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import mixle.stats as st
from mixle.task.artifact import ARRAYS_NAME as _ARRAYS_NAME
from mixle.task.artifact import (
    JSON_MODEL_NAME,
    MANIFEST_NAME,
    _atomic_json_dump,
    load_arrays,
    load_json,
    register_arrays_builder,
    save_arrays,
    save_json,
)
from mixle.utils.serialization import SerializationError

_TMP_GLOB = ".tmp-artifact-*"


def _rescale_arrays_probe(arrays: dict, scale: float = 1.0) -> dict:
    """A trivial registered arrays-builder used only to exercise save_arrays/load_arrays in these tests.

    Defined at module level (not inside a test method) so repeated registration across test methods passes
    the exact same callable object each time -- register_arrays_builder treats that as the documented no-op,
    rather than raising on what would otherwise look like a conflicting re-registration.
    """
    return {"w": arrays["w"] * scale}


class AtomicJsonDumpTest(unittest.TestCase):
    def test_writes_content_and_leaves_no_temp(self):
        with tempfile.TemporaryDirectory() as d:
            dst = os.path.join(d, "out.json")
            _atomic_json_dump(dst, {"a": 1, "b": [2, 3]}, sort_keys=True)
            with open(dst) as f:
                self.assertEqual(f.read(), '{"a": 1, "b": [2, 3]}')
            self.assertEqual(glob.glob(os.path.join(d, _TMP_GLOB)), [])

    def test_mid_write_failure_creates_no_file_and_no_temp(self):
        with tempfile.TemporaryDirectory() as d:
            dst = os.path.join(d, "out.json")
            with self.assertRaises(TypeError):  # a set is not JSON-serializable -> json.dump raises mid-write
                _atomic_json_dump(dst, {1, 2, 3})
            self.assertFalse(os.path.exists(dst))  # target was never created
            self.assertEqual(glob.glob(os.path.join(d, _TMP_GLOB)), [])  # temp cleaned up

    def test_failure_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            dst = os.path.join(d, "out.json")
            _atomic_json_dump(dst, {"good": True})
            with open(dst, encoding="utf-8") as f:
                original = f.read()
            with self.assertRaises(TypeError):
                _atomic_json_dump(dst, {1, 2, 3})
            with open(dst, encoding="utf-8") as f:
                self.assertEqual(f.read(), original)  # untouched by the failed write
            self.assertEqual(glob.glob(os.path.join(d, _TMP_GLOB)), [])


class SaveJsonAtomicityTest(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "art")
            model = st.GaussianDistribution(1.5, 2.0)
            save_json(path, model)
            loaded, _ = load_json(path)
            self.assertAlmostEqual(loaded.log_density(0.3), model.log_density(0.3), places=12)

    def test_failed_overwrite_keeps_previous_artifact_loadable(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "art")
            save_json(path, st.GaussianDistribution(1.5, 2.0))
            with open(os.path.join(path, JSON_MODEL_NAME), encoding="utf-8") as f:
                before = f.read()

            # object() is not registered for serialization -> save must fail without touching the good artifact.
            with self.assertRaises(SerializationError):
                save_json(path, object())

            with open(os.path.join(path, JSON_MODEL_NAME), encoding="utf-8") as f:
                self.assertEqual(f.read(), before)
            self.assertEqual(glob.glob(os.path.join(path, _TMP_GLOB)), [])
            loaded, _ = load_json(path)  # still loadable
            self.assertAlmostEqual(
                loaded.log_density(0.3), st.GaussianDistribution(1.5, 2.0).log_density(0.3), places=12
            )


class SaveJsonAtomicUpdateTest(unittest.TestCase):
    """save_json's model.json + manifest.json must move together as one atomic update.

    Unlike ``SaveJsonAtomicityTest`` above (which fails during *serialization*, before either file is
    touched), these inject the failure into the manifest write itself, AFTER the payload write already
    succeeded -- the specific window finding (a) identified.
    """

    def test_manifest_write_failure_leaves_old_model_and_manifest_both_intact(self):
        from mixle.task import artifact as A

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "art")
            A.save_json(path, st.GaussianDistribution(1.0, 1.0), task="v1", meta={"data_hash": "AAA"})
            with open(os.path.join(path, JSON_MODEL_NAME), "rb") as f:
                old_model = f.read()
            with open(os.path.join(path, MANIFEST_NAME), "rb") as f:
                old_manifest = f.read()

            with mock.patch.object(A, "_write_manifest", side_effect=RuntimeError("simulated manifest failure")):
                with self.assertRaises(RuntimeError):
                    A.save_json(path, st.GaussianDistribution(9.0, 9.0), task="v2", meta={"data_hash": "BBB"})

            # never a mismatched pairing: since the failure was injected on the manifest side, BOTH files
            # must still be exactly the v1 originals -- not "new model, old manifest".
            with open(os.path.join(path, JSON_MODEL_NAME), "rb") as f:
                self.assertEqual(f.read(), old_model)
            with open(os.path.join(path, MANIFEST_NAME), "rb") as f:
                self.assertEqual(f.read(), old_manifest)

            loaded, manifest = A.load_json(path)
            self.assertEqual(manifest.task, "v1")
            self.assertEqual(manifest.meta["data_hash"], "AAA")
            self.assertAlmostEqual(loaded.log_density(0.3), st.GaussianDistribution(1.0, 1.0).log_density(0.3))

            # no staging/backup directory leaked as a sibling of `path`
            self.assertEqual(os.listdir(d), ["art"])

    def test_publish_failure_restores_backup_path_never_missing(self):
        """A failure during the final directory-rename (not the write itself) must restore the old
        generation from its backup rather than leaving `path` missing or half-swapped."""
        from mixle.task import artifact as A

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "art")
            A.save_json(path, st.GaussianDistribution(1.0, 1.0), task="v1")
            with open(os.path.join(path, JSON_MODEL_NAME), "rb") as f:
                old_model = f.read()

            real_replace = os.replace
            state = {"n": 0}

            def flaky_replace(src, dst, *a, **kw):
                if dst == path:
                    state["n"] += 1
                    if state["n"] == 1:  # fail only the first "staging -> path" commit, not the restore
                        raise OSError("simulated publish failure")
                return real_replace(src, dst, *a, **kw)

            with mock.patch("os.replace", side_effect=flaky_replace):
                with self.assertRaises(OSError):
                    A.save_json(path, st.GaussianDistribution(9.0, 9.0), task="v2")

            self.assertTrue(os.path.exists(path))  # must never be left missing
            with open(os.path.join(path, JSON_MODEL_NAME), "rb") as f:
                self.assertEqual(f.read(), old_model)
            _, manifest = A.load_json(path)
            self.assertEqual(manifest.task, "v1")
            self.assertEqual(os.listdir(d), ["art"])  # backup cleaned up (or restored, not left as debris)

    def test_successful_update_fully_replaces_old_generation_no_leaks(self):
        from mixle.task import artifact as A

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "art")
            A.save_json(path, st.GaussianDistribution(1.0, 1.0), task="v1")
            A.save_json(path, st.GaussianDistribution(9.0, 9.0), task="v2", meta={"data_hash": "CCC"})

            loaded, manifest = A.load_json(path)
            self.assertEqual(manifest.task, "v2")
            self.assertEqual(manifest.meta["data_hash"], "CCC")
            self.assertAlmostEqual(loaded.log_density(0.3), st.GaussianDistribution(9.0, 9.0).log_density(0.3))
            self.assertEqual(os.listdir(d), ["art"])  # no leftover staging/backup directories

    def test_first_save_failure_leaves_no_artifact_at_all(self):
        """A first-time save (no prior artifact) that fails during the write must not leave a half-written
        `path` behind -- it should not exist at all."""
        from mixle.task import artifact as A

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "art")
            with mock.patch.object(A, "_write_manifest", side_effect=RuntimeError("simulated failure")):
                with self.assertRaises(RuntimeError):
                    A.save_json(path, st.GaussianDistribution(1.0, 1.0), task="v1")
            self.assertFalse(os.path.exists(path))
            self.assertEqual(os.listdir(d), [])  # no staging debris either


class SaveArraysAtomicUpdateTest(unittest.TestCase):
    """save_arrays' arrays.npz + manifest.json must move together as one atomic update (torch-free)."""

    def setUp(self):
        register_arrays_builder("test.atomic_arrays_probe", _rescale_arrays_probe)

    def test_manifest_write_failure_leaves_old_arrays_and_manifest_both_intact(self):
        from mixle.task import artifact as A

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "arr")
            save_arrays(path, {"w": np.array([1.0, 2.0])}, "test.atomic_arrays_probe", task="v1")
            with open(os.path.join(path, _ARRAYS_NAME), "rb") as f:
                old_arrays = f.read()
            with open(os.path.join(path, MANIFEST_NAME), "rb") as f:
                old_manifest = f.read()

            with mock.patch.object(A, "_write_manifest", side_effect=RuntimeError("simulated manifest failure")):
                with self.assertRaises(RuntimeError):
                    save_arrays(path, {"w": np.array([9.0, 9.0])}, "test.atomic_arrays_probe", task="v2")

            with open(os.path.join(path, _ARRAYS_NAME), "rb") as f:
                self.assertEqual(f.read(), old_arrays)
            with open(os.path.join(path, MANIFEST_NAME), "rb") as f:
                self.assertEqual(f.read(), old_manifest)

            model, manifest = load_arrays(path)
            self.assertEqual(manifest.task, "v1")
            np.testing.assert_array_equal(model["w"], np.array([1.0, 2.0]))
            self.assertEqual(os.listdir(d), ["arr"])

    def test_successful_update_fully_replaces_old_generation_no_leaks(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "arr")
            save_arrays(path, {"w": np.array([1.0, 2.0])}, "test.atomic_arrays_probe", task="v1")
            save_arrays(path, {"w": np.array([3.0, 4.0])}, "test.atomic_arrays_probe", task="v2")

            model, manifest = load_arrays(path)
            self.assertEqual(manifest.task, "v2")
            np.testing.assert_array_equal(model["w"], np.array([3.0, 4.0]))
            self.assertEqual(os.listdir(d), ["arr"])  # no leftover staging/backup directories


if __name__ == "__main__":
    unittest.main()
