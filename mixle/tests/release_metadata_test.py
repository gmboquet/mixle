"""The release-artifact fingerprinter is correct (worklist P2.5).

``scripts/release_metadata.py`` records the SHA-256, size, and filename of a release artifact so the
exact file that passed the release gates can be verified later. This checks the fingerprint against
``hashlib`` on a temporary artifact -- if the digest or size were wrong, the reproducibility record
would be worthless.
"""

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "release_metadata.py"


def _load():
    spec = importlib.util.spec_from_file_location("release_metadata", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseMetadataTest(unittest.TestCase):
    def test_script_exists(self):
        self.assertTrue(_SCRIPT.is_file(), f"missing {_SCRIPT}")

    def test_fingerprint_matches_hashlib(self):
        mod = _load()
        payload = b"mixle-release-artifact-fixture" * 1000  # > the 1 MiB streaming chunk boundary? no, but streamed
        with tempfile.TemporaryDirectory() as d:
            wheel = Path(d) / "mixle-9.9.9-py3-none-any.whl"
            wheel.write_bytes(payload)
            meta = mod.artifact_metadata(wheel)
        self.assertEqual(meta["filename"], "mixle-9.9.9-py3-none-any.whl")
        self.assertEqual(meta["size_bytes"], len(payload))
        self.assertEqual(meta["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertIn("python", meta)
        self.assertIn("platform", meta)

    def test_large_artifact_streams_correctly(self):
        # exceed the 1 MiB read chunk so the streaming digest is exercised across multiple reads
        mod = _load()
        payload = b"\x00\x01\x02\x03" * (400 * 1024)  # 1.6 MiB
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "big.whl"
            f.write_bytes(payload)
            meta = mod.artifact_metadata(f)
        self.assertEqual(meta["size_bytes"], len(payload))
        self.assertEqual(meta["sha256"], hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    unittest.main()
