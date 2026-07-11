"""The deprecation mechanism and policy (worklist A1.5).

Three things are pinned here:

  * the reusable mechanism in ``mixle.utils.deprecation`` -- one warning category (``DeprecationWarning``),
    one message format, attributed to the caller's line;
  * that the concrete deprecated aliases in the tree actually warn *and* still forward to the canonical
    name (a deprecation that changed behavior would be worse than none);
  * a static guard that no *future* "Deprecated alias" can be added silently -- every such method/function
    in the package source must carry the ``@deprecated_alias`` decorator, which is the exact gap A1.5
    was opened to close ("aliases exist but emit no DeprecationWarning").
"""

import ast
import unittest
import warnings
from pathlib import Path

import numpy as np

import mixle
from mixle.utils.deprecation import deprecated_alias, deprecation_message, warn_deprecated

_PKG_ROOT = Path(mixle.__file__).resolve().parent


class DeprecationMechanismTest(unittest.TestCase):
    def test_message_format(self):
        self.assertEqual(
            deprecation_message("Foo.old", "new", since="0.8.0", removed_in="0.10.0"),
            "Foo.old is deprecated since mixle 0.8.0; use new instead. It will be removed in mixle 0.10.0.",
        )
        # removed_in is optional -- omit the removal sentence when it is unknown.
        self.assertEqual(
            deprecation_message("old", "new", since="0.8.0"),
            "old is deprecated since mixle 0.8.0; use new instead.",
        )

    def test_warn_deprecated_category_and_caller_attribution(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_deprecated("old", "new", since="0.8.0", removed_in="0.10.0")  # <- this line is the caller
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, DeprecationWarning)
        self.assertIn("old is deprecated since mixle 0.8.0", str(caught[0].message))
        self.assertEqual(Path(caught[0].filename).name, "deprecation_test.py")

    def test_deprecated_alias_warns_and_forwards(self):
        class C:
            def canonical(self, a, b):
                return a + b

            @deprecated_alias("canonical", since="0.8.0", removed_in="0.10.0")
            def legacy(self, a, b):
                return self.canonical(a, b)

        c = C()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = c.legacy(2, 3)
        self.assertEqual(result, 5)  # forwarding is transparent
        self.assertIs(caught[0].category, DeprecationWarning)
        self.assertIn("C.legacy is deprecated", str(caught[0].message))
        self.assertEqual(C.legacy.__name__, "legacy")  # functools.wraps preserved


class ConcreteAliasesTest(unittest.TestCase):
    def test_gaussian_mixture_accumulator_factory_alias(self):
        from mixle.stats import MultivariateGaussianEstimator
        from mixle.stats.latent.gaussian_mixture import GaussianMixtureEstimator

        est = GaussianMixtureEstimator([MultivariateGaussianEstimator(dim=2), MultivariateGaussianEstimator(dim=2)])
        with self.assertWarns(DeprecationWarning):
            aliased = est.accumulatorFactory()
        self.assertIs(type(aliased), type(est.accumulator_factory()))

    def test_update_alpha_function_alias(self):
        from mixle.stats.latent.labeled_lda import update_alpha, updateAlpha

        alpha = np.array([[1.0, 1.0, 1.0]])
        mean_log_p = np.array([[-0.5, -0.4, -0.6]])
        with np.errstate(over="ignore"):  # the fixed-point math overflows identically in both paths
            with self.assertWarns(DeprecationWarning):
                aliased = updateAlpha(alpha.copy(), mean_log_p, 1e-6)
            np.testing.assert_allclose(aliased, update_alpha(alpha.copy(), mean_log_p, 1e-6))


class NoSilentDeprecatedAliasTest(unittest.TestCase):
    """Every "Deprecated alias" callable in the package source must carry @deprecated_alias."""

    @staticmethod
    def _has_deprecated_alias_decorator(node):
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
            if name == "deprecated_alias":
                return True
        return False

    def test_all_deprecated_aliases_are_wired(self):
        offenders = []
        for path in _PKG_ROOT.rglob("*.py"):
            if "tests" in path.parts:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                doc = ast.get_docstring(node) or ""
                if doc.startswith("Deprecated alias") and not self._has_deprecated_alias_decorator(node):
                    offenders.append(f"{path.relative_to(_PKG_ROOT)}:{node.lineno} {node.name}")
        self.assertEqual(
            offenders,
            [],
            "these 'Deprecated alias' callables emit no DeprecationWarning (add @deprecated_alias):\n"
            + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
