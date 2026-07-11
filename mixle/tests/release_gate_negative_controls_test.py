"""Negative controls for the release gates (worklist T3.6).

A gate that never fails proves nothing. For each defect class the 0.8.0 gates claim to catch, this file
deliberately introduces that defect and asserts the corresponding check *reports it with a useful message*
-- the acceptance bar for T3.6. Where the production gate runs as a separate CI job or script, the check
logic is reproduced here in a few lines so the control is self-contained and the *defect class* is proven
detectable; each such check is also small enough to be usable as a real gate.

The controls also assert the *good* case passes, so a control can never succeed vacuously (a check that
rejects everything is as useless as one that rejects nothing).
"""

import ast
import unittest

import numpy as np

# --- reproduced gate checks (kept tiny and dependency-free) --------------------------------------------------


def _module_has_unguarded_import(source: str, banned: str) -> bool:
    """True if ``source`` imports ``banned`` at module top level (not inside a function/try).

    Mirrors the base-install optional-import guard: a top-level ``import torch`` breaks ``import mixle`` in an
    environment without torch, so it must live inside a function or a guarded ``try``.
    """
    tree = ast.parse(source)
    for node in tree.body:  # only module-level statements
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == banned for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == banned:
            return True
    return False


def _removed_public_names(baseline: set[str], current: set[str]) -> set[str]:
    """Public names present in the recorded baseline but missing now -- an unannounced API removal."""
    return baseline - current


def _version_mismatch(pyproject_version: str, package_version: str) -> str | None:
    """Return a message if the two version strings disagree (stale metadata), else None."""
    if pyproject_version != package_version:
        return f"version metadata is stale: pyproject={pyproject_version!r} but package={package_version!r}"
    return None


def _benchmark_regressed(baseline_s: float, current_s: float, tol: float) -> bool:
    """True if ``current_s`` is slower than ``baseline_s`` by more than the fractional tolerance ``tol``."""
    return current_s > baseline_s * (1.0 + tol)


def _undeclared_imports(imported: set[str], declared: set[str], stdlib_and_first_party: set[str]) -> set[str]:
    """Third-party imports an example uses but the project does not declare as a dependency."""
    return {name for name in imported if name not in declared and name not in stdlib_and_first_party}


class NumericalNaNControlTest(unittest.TestCase):
    def test_nan_in_fit_data_is_rejected_loudly(self):
        # Defect: a non-finite value reaches the EM fit. The gate is the fit's own input validation --
        # it must raise, not silently return a plausible-looking model built on NaN responsibilities.
        from mixle.inference.estimation import optimize
        from mixle.stats import MultivariateGaussianEstimator
        from mixle.stats.latent.gaussian_mixture import GaussianMixtureEstimator

        rng = np.random.RandomState(0)
        data = [list(x) for x in np.vstack([rng.randn(100, 2), rng.randn(100, 2) + 5.0])]
        est = GaussianMixtureEstimator([MultivariateGaussianEstimator(dim=2), MultivariateGaussianEstimator(dim=2)])

        # good case: a clean fit succeeds and yields a finite objective.
        model = optimize(data, estimator=est, max_its=10, out=None)
        self.assertTrue(np.isfinite(float(model.seq_log_density(model.dist_to_encoder().seq_encode(data)).mean())))

        # negative control: inject a NaN -> the fit must fail with a useful message, not propagate silently.
        data[7] = [float("nan"), 0.0]
        with self.assertRaises((ValueError, FloatingPointError)) as ctx:
            optimize(data, estimator=est, max_its=10, out=None)
        self.assertRegex(str(ctx.exception).lower(), r"nan|inf|finite")


class ApiRemovalControlTest(unittest.TestCase):
    def test_public_api_removal_is_detected(self):
        import mixle.stats as st

        current = {n for n in dir(st) if not n.startswith("_")}
        # good case: the recorded baseline (== current) shows no removals.
        self.assertEqual(_removed_public_names(current, current), set())
        # negative control: a manifest recorded when GaussianDistribution was public flags its removal.
        baseline = current | {"GaussianDistribution"}
        pruned = current - {"GaussianDistribution"}
        self.assertEqual(_removed_public_names(baseline, pruned), {"GaussianDistribution"})


class UnguardedImportControlTest(unittest.TestCase):
    def test_top_level_optional_import_is_detected(self):
        guarded = "import numpy\n\ndef f():\n    import torch\n    return torch\n"
        unguarded = "import numpy\nimport torch\n\ndef f():\n    return torch\n"
        self.assertFalse(_module_has_unguarded_import(guarded, "torch"))  # good: torch import is inside f()
        self.assertTrue(_module_has_unguarded_import(unguarded, "torch"))  # negative control
        # a `from torch import ...` at top level is caught too.
        self.assertTrue(_module_has_unguarded_import("from torch.nn import Linear\n", "torch"))


class MissingModuleControlTest(unittest.TestCase):
    def test_missing_module_import_fails(self):
        import importlib

        # good case: a real public module imports.
        self.assertIsNotNone(importlib.import_module("mixle.stats"))
        # negative control: a module removed from the package fails the import sweep with a clear error.
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("mixle.this_module_was_deleted_in_a_bad_refactor")


class VersionMetadataControlTest(unittest.TestCase):
    def test_stale_version_metadata_is_detected(self):
        self.assertIsNone(_version_mismatch("0.8.0", "0.8.0"))  # good: consistent
        msg = _version_mismatch("0.8.0", "0.7.0")  # negative control: pyproject bumped, package didn't
        self.assertIsNotNone(msg)
        self.assertIn("stale", msg)


class BenchmarkRegressionControlTest(unittest.TestCase):
    def test_regression_beyond_threshold_is_detected(self):
        self.assertFalse(_benchmark_regressed(1.0, 1.05, tol=0.10))  # good: 5% < 10% budget
        self.assertTrue(_benchmark_regressed(1.0, 1.50, tol=0.10))  # negative control: 50% > 10% budget


class UndeclaredDependencyControlTest(unittest.TestCase):
    def test_example_with_undeclared_dependency_is_detected(self):
        stdlib_and_first_party = {"os", "sys", "mixle"}
        declared = {"numpy"}
        clean = {"os", "numpy", "mixle"}
        dirty = {"os", "numpy", "sklearn", "mixle"}  # sklearn used but not declared
        self.assertEqual(_undeclared_imports(clean, declared, stdlib_and_first_party), set())  # good
        self.assertEqual(_undeclared_imports(dirty, declared, stdlib_and_first_party), {"sklearn"})  # negative control


if __name__ == "__main__":
    unittest.main()
