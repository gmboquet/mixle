"""Artifact writes are atomic: a failed serialization never corrupts an artifact (worklist S13.5 / M11.3).

``save_json`` and the manifest writer used ``open(path, "w")``, which truncates the target *before*
serialization runs. A non-serializable model -- or a crash mid-``json.dump`` -- therefore left a truncated,
unloadable ``model.json`` and, worse, destroyed any previous good artifact at that path. The write now goes
through a temp file swapped in with ``os.replace``, so it is all-or-nothing. These are the failure-injection
tests for that contract: the operation fails loudly, the prior artifact survives, and no temp file leaks.
"""

import glob
import os
import tempfile
import unittest

import mixle.stats as st
from mixle.task.artifact import JSON_MODEL_NAME, _atomic_json_dump, load_json, save_json
from mixle.utils.serialization import SerializationError

_TMP_GLOB = ".tmp-artifact-*"


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
            original = open(dst).read()
            with self.assertRaises(TypeError):
                _atomic_json_dump(dst, {1, 2, 3})
            self.assertEqual(open(dst).read(), original)  # untouched by the failed write
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
            before = open(os.path.join(path, JSON_MODEL_NAME)).read()

            # object() is not registered for serialization -> save must fail without touching the good artifact.
            with self.assertRaises(SerializationError):
                save_json(path, object())

            self.assertEqual(open(os.path.join(path, JSON_MODEL_NAME)).read(), before)
            self.assertEqual(glob.glob(os.path.join(path, _TMP_GLOB)), [])
            loaded, _ = load_json(path)  # still loadable
            self.assertAlmostEqual(
                loaded.log_density(0.3), st.GaussianDistribution(1.5, 2.0).log_density(0.3), places=12
            )


if __name__ == "__main__":
    unittest.main()
