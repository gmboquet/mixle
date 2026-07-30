"""Fail-closed contracts for compute capability metadata and hook lookup."""

import unittest

from mixle.stats.compute.capabilities import (
    KNOWN_COMPUTE_ENGINES,
    CapabilitiesNotApplicable,
    DistributionCapabilities,
    capabilities_for,
    register_capabilities,
    replace_capabilities,
    unregister_capabilities,
)


class CapabilitiesContractTest(unittest.TestCase):
    def test_metadata_rejects_duplicate_and_incoherent_claims(self):
        # Engine-name *membership* is deliberately not checked here; see
        # DistributionCapabilities.__post_init__ and
        # test_a_custom_engine_name_is_declarable_because_membership_is_not_decidable_here.
        invalid = (
            {"engine_ready": ()},
            {"engine_ready": ("numpy", "numpy")},
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

    def test_a_custom_engine_name_is_declarable_because_membership_is_not_decidable_here(self):
        """A name outside KNOWN_COMPUTE_ENGINES must construct, and this is a decision, not a gap.

        Kernel dispatch routes on an engine's capability *flags*, not on its name -- see
        compute_kernel_test.test_kernel_dispatch_is_capability_based_not_name_based, whose fixture declares
        the locally-defined engine ``custom-accelerator``. Closing this constructor over a name list
        was tried (fe6ca83b) and reverted (87aca6ee): it made a declaration require registration,
        which broke that dispatch guarantee and fused_em_association. A typo and a genuine custom
        backend are the same string here, so no check at this layer can reject one and keep the
        other. Typos in the families mixle itself ships are caught by
        test_shipped_families_declare_only_known_engines, where the set of engines really is closed.
        """
        caps = DistributionCapabilities(engine_ready=("numpy", "custom-accelerator"))
        self.assertEqual(caps.engine_ready, ("numpy", "custom-accelerator"))
        self.assertTrue(caps.supports_engine("custom-accelerator"))
        self.assertFalse(caps.supports_engine("torch"))

    def test_shipped_families_declare_only_known_engines(self):
        """Every capability declaration mixle ships names a real engine.

        This is the typo gate the constructor cannot be: third parties may declare any engine name,
        but mixle's own families draw from a closed set, and that is where a misspelling would
        actually ship. Covers both declaration styles -- register_capabilities() and a
        compute_capabilities()/engine_ready class member.
        """
        import inspect
        import sys

        import mixle.stats  # noqa: F401 - importing registers the shipped families
        from mixle.stats.compute import capabilities as caps_module

        offenders: dict[str, tuple[str, ...]] = {}

        def check(label: str, caps: object) -> None:
            if not isinstance(caps, DistributionCapabilities):
                return
            unknown = tuple(name for name in caps.engine_ready if name not in KNOWN_COMPUTE_ENGINES)
            if unknown:
                offenders[label] = unknown

        for dist_type, caps in list(caps_module._CAPABILITIES.items()):
            check(dist_type.__qualname__, caps)

        for module in [
            m for m in list(sys.modules.values()) if (getattr(m, "__name__", "") or "").startswith("mixle.")
        ]:
            for obj in list(vars(module).values()):
                if not inspect.isclass(obj):
                    continue
                declared = getattr(obj, "__dict__", {})
                if "compute_capabilities" not in declared and "engine_ready" not in declared:
                    continue
                try:
                    check(obj.__qualname__, capabilities_for(obj))
                except Exception:  # noqa: BLE001 - a family that cannot report is other tests' problem
                    continue

        self.assertEqual(offenders, {}, f"shipped families declare engines mixle does not have: {offenders}")

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
