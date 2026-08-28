"""T3-02: the README's second Quickstart example -- ``solve(teacher, inputs)`` -- crashed with an
ImportError on the officially-supported base install (numpy+scipy only) because solve()'s default
student is a torch MLP, and nothing near that example said so. The very next Quickstart example
("A PyTorch module fits in one line") IS explicit about needing torch; this one wasn't. Pins that
the README now discloses the requirement before the code block, and that the disclosure matches
what solve() actually does at runtime.
"""

import unittest
from pathlib import Path

from mixle.task import solve

README = Path(__file__).resolve().parents[2] / "README.md"


class ReadmeSolveTorchDisclosureTest(unittest.TestCase):
    def setUp(self):
        text = README.read_text(encoding="utf-8")
        start = text.index("Distill a slow, expensive model into a cheap one")
        end = text.index("A PyTorch module fits in one line")
        self.section = text[start:end]

    def test_torch_requirement_is_disclosed_before_the_code_block(self):
        # A reader hits the crash the moment they paste and run the fenced block, so the warning
        # has to land in the prose ABOVE it -- disclosing it only after the block is too late.
        code_start = self.section.index("```python")
        prose = self.section[:code_start]
        self.assertIn("torch", prose.lower(), "solve() quickstart no longer names torch before the code")
        self.assertIn("mixle[torch]", prose, "solve() quickstart dropped the torch extra install command")

    def test_torch_free_alternative_is_named(self):
        # The runtime ImportError itself points readers at student="generative"; the README must
        # name the same escape hatch, not invent different terminology.
        self.assertIn('student="generative"', self.section)

    def _teacher(self, record):
        (value,) = record
        return value > 1.0

    def _inputs(self):
        # Same shape as the README's own example: tuple records, solve()'s own kind detection.
        data = [0.2, 1.9, 0.4, 2.1, 0.7, 1.6, 0.3, 1.2, 0.9, 2.4]
        return [(float(v),) for v in data]

    def test_default_student_behavior_matches_the_disclosure(self):
        # Exercises solve() exactly as the Quickstart shows it. Whichever branch this environment
        # takes must match what the README now promises: torch present -> the default student
        # trains; torch absent -> the exact documented ImportError, naming both the extra and the
        # torch-free alternative.
        try:
            import torch  # noqa: F401

            has_torch = True
        except ImportError:
            has_torch = False

        if has_torch:
            assistant = solve(self._teacher, self._inputs(), seed=0)
            self.assertIsNotNone(assistant)
        else:
            with self.assertRaises(ImportError) as ctx:
                solve(self._teacher, self._inputs(), seed=0)
            message = str(ctx.exception)
            self.assertIn("mixle[torch]", message)
            self.assertIn('student="generative"', message)

    def test_generative_student_is_the_working_torch_free_path(self):
        # The README's named alternative must actually work, with or without torch installed --
        # this is what makes the disclosure honest rather than just quieter.
        assistant = solve(self._teacher, self._inputs(), seed=0, student="generative")
        self.assertIsNotNone(assistant)
        report = assistant.report()
        self.assertIn("holdout_agreement", report)


if __name__ == "__main__":
    unittest.main()
