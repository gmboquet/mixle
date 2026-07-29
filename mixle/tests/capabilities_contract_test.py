"""Fail-closed contracts for compute capability metadata and hook lookup."""

import unittest

from mixle.stats.compute.capabilities import (
    CapabilitiesNotApplicable,
    DistributionCapabilities,
    capabilities_for,
    register_capabilities,
    replace_capabilities,
    unregister_capabilities,
)


class CapabilitiesContractTest(unittest.TestCase):
    def test_metadata_rejects_unknown_duplicate_and_incoherent_claims(self):
        invalid = (
            {"engine_ready": ()},
            {"engine_ready": ("numpy", "numpy")},
            {"engine_ready": ("numpy", "invented")},
            {"engine_ready": ("torch", "numpy")},
            {"kernel_status": "invented"},
            {"engine_ready": ("numpy",), "kernel_status": "numpy_only"},
            {"engine_ready": ("numpy", "torch"), "kernel_status": "numpy_only", "numpy_only_reason": "host only"},
            {"engine_ready": ("numpy",), "kernel_status": "generic", "numpy_only_reason": "host only"},
            {"engine_ready": ("numpy", "torch"), "kernel_status": "legacy_numpy"},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises((TypeError, ValueError)):
                DistributionCapabilities(**kwargs)

    def test_hook_defects_are_not_hidden_by_fallbacks(self):
        class BrokenClassHook:
            @classmethod
            def compute_capabilities(cls):
                raise TypeError("defect inside class hook")

        class BrokenInstanceHook:
            def compute_capabilities(self):
                raise TypeError("defect inside instance hook")

        class WrongResult:
            @classmethod
            def compute_capabilities(cls):
                return None

        with self.assertRaisesRegex(TypeError, "defect inside class hook"):
            capabilities_for(BrokenClassHook)
        with self.assertRaisesRegex(TypeError, "defect inside instance hook"):
            capabilities_for(BrokenInstanceHook())
        with self.assertRaisesRegex(TypeError, "must return"):
            capabilities_for(WrongResult)

    def test_explicit_not_applicable_result_uses_registered_fallback(self):
        class ConditionalHook:
            def compute_capabilities(self):
                return CapabilitiesNotApplicable("this instance does not own the execution path")

        fallback = DistributionCapabilities(engine_ready=("numpy", "torch"))
        replacement = DistributionCapabilities(engine_ready=("numpy",), kernel_status="legacy_numpy")
        register_capabilities(ConditionalHook, fallback)
        try:
            self.assertIs(capabilities_for(ConditionalHook()), fallback)
            with self.assertRaisesRegex(KeyError, "already registered"):
                register_capabilities(ConditionalHook, DistributionCapabilities())
            replace_capabilities(ConditionalHook, replacement)
            self.assertIs(capabilities_for(ConditionalHook()), replacement)
        finally:
            self.assertIs(unregister_capabilities(ConditionalHook), replacement)


if __name__ == "__main__":
    unittest.main()
