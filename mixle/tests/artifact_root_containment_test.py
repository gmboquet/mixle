"""A caller-supplied deployment name must stay under its declared root (MXR-080-1910).

``Solution.deploy(name, root)`` joined ``name`` onto ``root`` unchecked, and both ``pathlib`` and
``os.path.join`` leave a root when asked to: ``"../../escaped"`` traverses out of it, and an ABSOLUTE
name discards the root entirely -- ``Path("registry") / "/tmp/x"`` is ``/tmp/x``. A serving name
arrives from an API request, a config file, or a CLI argument, so writing outside the artifact root is
the ordinary failure, not an exotic one.

The string check alone is not sufficient either: a symlink already inside the root can redirect a
write that looks contained by its text. ``mixle.inference.production.registry`` has enforced both
rules on its own store since MXR-080-0264; this is that logic made shared.
"""

import os
import tempfile
import unittest

from mixle.utils.paths import contained_path, safe_segment

ESCAPES = ("../../escaped", "..", "a/b", "/tmp/absolute", "", "   ", "x\x00y", ".")


class SafeSegmentTest(unittest.TestCase):
    def test_every_escape_shape_is_refused(self):
        for segment in ESCAPES:
            with self.subTest(segment=repr(segment)):
                with self.assertRaises(ValueError):
                    safe_segment(segment, kind="deployment name")

    def test_an_ordinary_name_passes(self):
        self.assertEqual(safe_segment("my-task"), "my-task")

    def test_a_non_string_is_refused(self):
        for value in (None, 3, b"x"):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    safe_segment(value)


class ContainedPathTest(unittest.TestCase):
    def test_every_escape_shape_is_refused_through_the_join(self):
        root = tempfile.mkdtemp()
        for segment in ESCAPES:
            with self.subTest(segment=repr(segment)):
                with self.assertRaises(ValueError):
                    contained_path(root, "tasks", segment, kind="deployment name")

    def test_an_ordinary_name_lands_under_the_root(self):
        root = tempfile.mkdtemp()
        path = contained_path(root, "tasks", "my-task")
        self.assertTrue(str(path).startswith(root))
        self.assertEqual(path.name, "my-task")

    def test_a_symlink_inside_the_root_cannot_redirect_the_write(self):
        # The string is a single safe component; only resolving it reveals the escape.
        root = tempfile.mkdtemp()
        outside = tempfile.mkdtemp()
        os.makedirs(os.path.join(root, "tasks"), exist_ok=True)
        os.symlink(outside, os.path.join(root, "tasks", "sneaky"))
        self.assertEqual(safe_segment("sneaky"), "sneaky")
        with self.assertRaisesRegex(ValueError, "outside the declared root"):
            contained_path(root, "tasks", "sneaky")

    def test_a_symlink_staying_inside_the_root_is_allowed(self):
        root = tempfile.mkdtemp()
        os.makedirs(os.path.join(root, "tasks", "real"), exist_ok=True)
        os.symlink(os.path.join(root, "tasks", "real"), os.path.join(root, "tasks", "alias"))
        self.assertTrue(str(contained_path(root, "tasks", "alias")).startswith(root))


if __name__ == "__main__":
    unittest.main()
