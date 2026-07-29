"""Focused regressions for the 0.8.0 capability/discovery audit."""

from __future__ import annotations

import unittest

import numpy as np

import mixle.capability as capability


class _BrokenEnumerable:
    def enumerator(self):
        raise RuntimeError("enumerator implementation failed")


class _CountingEnumerable:
    def __init__(self):
        self.calls = 0

    def enumerator(self):
        self.calls += 1
        return iter([(1, 0.0)])


class _VectorSummary:
    def mean(self):
        return np.asarray([1.0, 2.0])

    def variance(self):
        return np.asarray([4.0, 9.0])

    def entropy(self):
        raise RuntimeError("broken entropy")


class CapabilityFailureTest(unittest.TestCase):
    def test_broken_implementation_is_not_reported_as_absent(self):
        broken = _BrokenEnumerable()
        with self.assertRaisesRegex(RuntimeError, "implementation failed"):
            capability.supports(broken, capability.Enumerable)
        with self.assertRaisesRegex(RuntimeError, "implementation failed"):
            capability.capabilities(broken)
        with self.assertRaisesRegex(RuntimeError, "implementation failed"):
            capability.what_supports(capability.Enumerable, [broken])

    def test_top_k_validates_budget_before_capability_work(self):
        value = _CountingEnumerable()
        for bad in (-1, 1.5, True, np.nan):
            with self.subTest(k=repr(bad)), self.assertRaises(ValueError):
                capability.top_k(value, bad)
        self.assertEqual(value.calls, 0)


class SummaryAndCatalogTest(unittest.TestCase):
    def test_summary_preserves_vector_shape_and_reports_method_failure(self):
        summary = capability.summarize(_VectorSummary())
        np.testing.assert_array_equal(summary["mean"], np.asarray([1.0, 2.0]))
        np.testing.assert_array_equal(summary["std"], np.asarray([2.0, 3.0]))
        self.assertEqual(summary["_status"]["mean"]["status"], "available")
        self.assertEqual(summary["_status"]["entropy"]["status"], "failed")

    def test_catalog_is_complete_unique_and_runtime_authoritative(self):
        names = [spec.name for spec in capability.CAPABILITY_CATALOG]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("SupportsBackendComponentScoring", names)
        self.assertEqual(
            [item.__name__ for item in capability.ALL_CAPABILITIES],
            [spec.name for spec in capability.CAPABILITY_CATALOG if spec.kind == "distribution facet"],
        )


class DiscoveryTest(unittest.TestCase):
    def test_lazy_contracts_are_discoverable_and_cached(self):
        import mixle.contracts as contracts

        for name in ("Relation", "ComputeEngine", "Kernel", "Surrogate"):
            self.assertIn(name, dir(contracts))
            self.assertIs(getattr(contracts, name), getattr(contracts, name))

    def test_compatibility_shims_have_explicit_public_star_exports(self):
        registry_namespace: dict[str, object] = {}
        meta_namespace: dict[str, object] = {}
        exec("from mixle.registry import *", registry_namespace)
        exec("from mixle.meta import *", meta_namespace)
        self.assertEqual(
            {name for name in registry_namespace if not name.startswith("__")},
            {"Registry", "RegistryEntry"},
        )
        self.assertEqual(
            {name for name in meta_namespace if not name.startswith("__")},
            {"ImprovementOption", "MetaImprovementReport", "improve_by_regret"},
        )


class CapabilityDispatchContractTest(unittest.TestCase):
    def test_protocol_facets_require_callable_members(self):
        # MXR-080-1682: method-presence facets used runtime_checkable isinstance directly, which
        # establishes attribute PRESENCE only. An object with `condition = 3` cleared
        # supports(obj, Conditionable) and therefore require(), and the routing failure surfaced
        # later as a TypeError inside the operation it had already been dispatched to.
        class NotCallable:
            condition = 3

        class Callable_:
            def condition(self, observed):
                return observed

        self.assertFalse(capability.supports(NotCallable(), capability.Conditionable))
        with self.assertRaises(capability.CapabilityError):
            capability.require(NotCallable(), capability.Conditionable, "condition")
        self.assertTrue(capability.supports(Callable_(), capability.Conditionable))

    def test_boolean_is_not_a_support_size(self):
        # bool subclasses int, so support_size() -> True certified a one-point finite support.
        class BooleanSupport:
            def support_size(self):
                return True

        class RealSupport:
            def support_size(self):
                return 3

        self.assertFalse(capability.supports(BooleanSupport(), capability.FiniteSupport))
        self.assertTrue(capability.supports(RealSupport(), capability.FiniteSupport))


if __name__ == "__main__":
    unittest.main()
