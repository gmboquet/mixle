"""mixle.__version__ resolves from installed package metadata."""

import unittest

import mixle


class VersionTest(unittest.TestCase):
    def test_version_is_a_non_empty_string(self):
        self.assertIsInstance(mixle.__version__, str)
        self.assertTrue(mixle.__version__)

    def test_version_looks_like_a_version(self):
        # PEP 440 is permissive; just check it starts with a numeric release segment.
        self.assertRegex(mixle.__version__, r"^\d+(\.\d+)*")

    def test_version_is_exported_from_dunder_all(self):
        self.assertIn("__version__", mixle.__all__)


if __name__ == "__main__":
    unittest.main()
