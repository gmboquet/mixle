import unittest

from mixle.stats.compute import declarations as declaration_module
from mixle.stats.compute.declarations import (
    DistributionDeclaration,
    ExponentialFamilySpec,
    GeneratedKernelCompilationError,
    ParameterSpec,
    _build_generic_numba_kernel,
    generated_log_density_diagnostics,
    register_declaration,
)


class GeneratedCompilationFailureContractTest(unittest.TestCase):
    def tearDown(self):
        for dist_type in getattr(self, "_types", ()):
            declaration_module._DECLARATIONS.pop(dist_type, None)
            declaration_module._GENERIC_NUMBA_KERNEL_CACHE.pop(dist_type, None)

    def _register(self, dist_type, *, exponential_family=None):
        declaration = DistributionDeclaration(
            name=dist_type.__name__,
            distribution_type=dist_type,
            parameters=(ParameterSpec("scale", constraint="positive"),),
            statistics=(),
            support="real",
            exponential_family=exponential_family,
        )
        self._types = getattr(self, "_types", ()) + (dist_type,)
        register_declaration(declaration)
        return declaration

    def test_unexpected_numba_trace_failure_is_typed_not_cached_and_retryable(self):
        class TransientDistribution:
            attempts = 0

            def __init__(self, scale):
                self.scale = scale

            @staticmethod
            def backend_log_density_from_params(x, scale, engine):
                TransientDistribution.attempts += 1
                if TransientDistribution.attempts == 1:
                    raise RuntimeError("temporary tracing failure")
                return x * scale

        declaration = self._register(TransientDistribution)
        with self.assertRaisesRegex(GeneratedKernelCompilationError, "temporary tracing failure"):
            _build_generic_numba_kernel(TransientDistribution, declaration)
        self.assertNotIn(TransientDistribution, declaration_module._GENERIC_NUMBA_KERNEL_CACHE)

        built = _build_generic_numba_kernel(TransientDistribution, declaration)
        self.assertIsNotNone(built)
        self.assertIn(TransientDistribution, declaration_module._GENERIC_NUMBA_KERNEL_CACHE)
        self.assertEqual(TransientDistribution.attempts, 2)

    def test_validated_unsupported_lowering_is_the_only_negative_cache_entry(self):
        class UnsupportedDistribution:
            attempts = 0

            def __init__(self, scale):
                self.scale = scale

            @staticmethod
            def backend_log_density_from_params(x, scale, engine):
                UnsupportedDistribution.attempts += 1
                return engine.digamma(x * scale)

        declaration = self._register(UnsupportedDistribution)
        self.assertIsNone(_build_generic_numba_kernel(UnsupportedDistribution, declaration))
        self.assertIn(UnsupportedDistribution, declaration_module._GENERIC_NUMBA_KERNEL_CACHE)
        self.assertIsNone(_build_generic_numba_kernel(UnsupportedDistribution, declaration))
        self.assertEqual(UnsupportedDistribution.attempts, 1)

    def test_diagnostics_do_not_hide_broken_exponential_family_hooks(self):
        class BrokenDistribution:
            def __init__(self, scale):
                self.scale = scale

            @staticmethod
            def backend_log_density_from_params(x, scale, engine):
                return x * scale

        def broken_statistics(x, engine):
            raise RuntimeError("broken sufficient-statistic trace")

        self._register(
            BrokenDistribution,
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=broken_statistics,
                natural_parameters=lambda params, engine: (params["scale"],),
                log_partition=lambda params, engine: 0.0,
            ),
        )
        with self.assertRaisesRegex(
            GeneratedKernelCompilationError,
            "broken sufficient-statistic trace",
        ):
            generated_log_density_diagnostics(BrokenDistribution)


if __name__ == "__main__":
    unittest.main()
