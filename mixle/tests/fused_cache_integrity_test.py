"""Fused template and generated-module caches reject stale semantics."""

import hashlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from mixle.stats.compute import fused_codegen as fc
from mixle.utils.optional_deps import HAS_NUMBA


class _ExtensionDistribution:
    pass


def _template(expression, cache_version="1"):
    return fc.LeafTemplate(
        name="cache-integrity-extension",
        matches=lambda dist: isinstance(dist, _ExtensionDistribution),
        data=lambda encoded: (np.ascontiguousarray(encoded, dtype=np.float64),),
        params=lambda components: {},
        expr=expression,
        acc_names=("sx",),
        acc_stmt=lambda values, accumulators, weight: (
            f"{accumulators['sx']}[k] += {weight} * {values[0]}"
        ),
        to_value=lambda stats, count: (count, stats[0]),
        cache_version=cache_version,
    )


class FusedTemplateRegistryTest(unittest.TestCase):
    def test_hook_code_and_explicit_version_change_fingerprint(self):
        first = _template(lambda values, params: values[0], cache_version="1")
        changed_code = _template(lambda values, params: f"({values[0]} + 1.0)", cache_version="1")
        changed_version = _template(lambda values, params: values[0], cache_version="2")
        self.assertNotEqual(first.fingerprint, changed_code.fingerprint)
        self.assertNotEqual(first.fingerprint, changed_version.fingerprint)

    def test_registration_is_unique_precedes_bridge_and_enters_plan_signature(self):
        first = _template(lambda values, params: values[0])
        replacement = _template(lambda values, params: f"({values[0]} + 1.0)")
        with patch.object(fc, "_TEMPLATES", list(fc._TEMPLATES)):
            fc.register_leaf_template(first)
            names = [template.name for template in fc._TEMPLATES]
            self.assertLess(names.index(first.name), names.index("bridge"))
            with self.assertRaisesRegex(ValueError, "already registered"):
                fc.register_leaf_template(replacement)
            plan = fc.analyze(_ExtensionDistribution())
            self.assertEqual(plan.signature[1], (first.fingerprint,))


@unittest.skipUnless(HAS_NUMBA, "generated fused modules require numba")
class FusedModuleImportTest(unittest.TestCase):
    def test_failed_import_removes_partial_module_before_memory_fallback(self):
        source = "def _cache_probe(x):\n    return x + 1"
        digest = hashlib.sha1(f"v2|parallel=False|{source}".encode()).hexdigest()[:16]  # noqa: S324
        module_name = f"_pysp_fused_{digest}"

        class _FailingLoader:
            def exec_module(self, module):
                module.partial = True
                raise RuntimeError("injected import failure")

        spec = types.SimpleNamespace(loader=_FailingLoader())
        partial = types.ModuleType(module_name)
        sys.modules[module_name] = partial
        try:
            with (
                tempfile.TemporaryDirectory() as cache_dir,
                patch.object(fc, "_CACHE_DIR", cache_dir),
                patch.object(fc.importlib.util, "spec_from_file_location", return_value=spec),
                patch.object(fc.importlib.util, "module_from_spec", return_value=types.ModuleType(module_name)),
            ):
                compiled = fc._njit(source, "_cache_probe")
                self.assertEqual(compiled(2), 3)
                self.assertNotIn(module_name, sys.modules)
        finally:
            sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
