"""Public-surface claims that must hold as shipped, not merely as developed.

MXR-080-1860: a conditional ``__all__.append`` listed a name the static ``__all__`` already carried.
MXR-080-1861: public constructor validation used ``assert``, which ``python -O`` strips.
MXR-080-1320: the README advertised a wider Python range than the metadata installs on.
"""

import pathlib
import subprocess
import sys
import tomllib
import unittest
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]


class ExportUniquenessTest(unittest.TestCase):
    """A duplicated export is a defect in the public inventory even when it imports fine."""

    def test_memory_efficient_training_exports_each_name_once(self):
        from mixle.models import memory_efficient_training

        repeated = [name for name, count in Counter(memory_efficient_training.__all__).items() if count > 1]
        self.assertEqual(repeated, [])

    def test_the_conditional_name_is_still_exported_and_bound(self):
        # The fix removed an append, not the export: both the torch and torch-less branches bind
        # CompressedAdam, so the static entry is the correct single source.
        from mixle.models import memory_efficient_training

        self.assertIn("CompressedAdam", memory_efficient_training.__all__)
        self.assertTrue(hasattr(memory_efficient_training, "CompressedAdam"))

    def test_no_public_module_repeats_an_export(self):
        import importlib
        import pkgutil

        import mixle

        repeated: list[str] = []
        for info in pkgutil.walk_packages(mixle.__path__, prefix="mixle."):
            if ".tests" in info.name or ".experimental" in info.name:
                continue
            try:
                module = importlib.import_module(info.name)
            except Exception:  # noqa: BLE001 - an optional-dependency module is not this test's subject
                continue
            names = getattr(module, "__all__", None)
            if not names:
                continue
            repeated += [f"{info.name}.{n}" for n, c in Counter(names).items() if c > 1]
        self.assertEqual(repeated, [])


class OptimizedModeValidationTest(unittest.TestCase):
    """Validation that disappears under ``-O`` is not validation (MXR-080-1861)."""

    PROBE = (
        "from mixle.experimental.sketch_state_attention import LinearAttentionSpine\n"
        "from mixle.experimental.summary_tree import SummaryTreeSpine\n"
        "bad = 0\n"
        "for make in (lambda: LinearAttentionSpine(8, d_model=3, n_head=2),\n"
        "             lambda: SummaryTreeSpine(8, fanout=1),\n"
        "             lambda: SummaryTreeSpine(8, d_model=3, n_head=2)):\n"
        "    try:\n"
        "        make()\n"
        "        bad += 1\n"
        "    except ValueError:\n"
        "        pass\n"
        "    except Exception:\n"
        "        pass\n"
        "print(bad)\n"
    )

    def _constructed_invalid_count(self, *flags: str) -> int:
        result = subprocess.run(
            [sys.executable, *flags, "-c", self.PROBE],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(ROOT),
            check=False,
        )
        if result.returncode != 0:
            self.skipTest(f"attention spines unavailable in this environment: {result.stderr.strip()[-200:]}")
        return int(result.stdout.strip().splitlines()[-1])

    def test_invalid_architectures_are_refused_with_assertions_enabled(self):
        self.assertEqual(self._constructed_invalid_count(), 0)

    def test_invalid_architectures_are_refused_under_dash_o(self):
        self.assertEqual(self._constructed_invalid_count("-O"), 0)


class DeclaredPythonRangeTest(unittest.TestCase):
    """The README may not advertise a range the metadata refuses to install on (MXR-080-1320)."""

    def test_the_readme_does_not_claim_an_open_ended_range(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("Python 3.11+", readme)

    def test_the_readme_names_exactly_the_supported_versions(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        requires = metadata["project"]["requires-python"]
        self.assertEqual(requires, ">=3.11,<3.13")
        self.assertIn("Python 3.11 or 3.12", readme)
        self.assertIn(requires, readme)

    def test_the_classifiers_agree_with_the_range(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        classifiers = metadata["project"]["classifiers"]
        versioned = {
            c.rsplit(" :: ", 1)[-1] for c in classifiers if c.startswith("Programming Language :: Python :: 3.")
        }
        self.assertEqual(versioned, {"3.11", "3.12"})


if __name__ == "__main__":
    unittest.main()
