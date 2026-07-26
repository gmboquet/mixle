import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats.compute import declarations as declaration_module
from mixle.stats.compute.declarations import (
    DistributionDeclaration,
    ExponentialFamilySpec,
    ParameterSpec,
    StatisticSpec,
    declaration_for,
    generated_log_density,
    register_declaration,
    validate_declaration,
)


class DeclarationRegistrationContractTest(unittest.TestCase):
    def tearDown(self):
        for dist_type in getattr(self, "_registered_types", ()):
            declaration_module._DECLARATIONS.pop(dist_type, None)

    def _track(self, *dist_types):
        self._registered_types = getattr(self, "_registered_types", ()) + dist_types

    def test_registration_is_validated_atomic_and_conflict_safe(self):
        class ExampleDistribution:
            def __init__(self, scale):
                self.scale = scale

        valid = DistributionDeclaration(
            name="example",
            distribution_type=ExampleDistribution,
            parameters=(ParameterSpec("scale", constraint="positive"),),
            statistics=(StatisticSpec("count"),),
            support="real",
        )
        self._track(ExampleDistribution)
        register_declaration(valid)
        register_declaration(valid)
        self.assertIs(declaration_for(ExampleDistribution), valid)

        conflicting = DistributionDeclaration(
            name="different",
            distribution_type=ExampleDistribution,
            parameters=(ParameterSpec("scale", constraint="positive"),),
            statistics=(StatisticSpec("count"),),
            support="real",
        )
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            register_declaration(conflicting)
        self.assertIs(declaration_for(ExampleDistribution), valid)

        invalid = DistributionDeclaration(
            name="invalid",
            distribution_type=ExampleDistribution,
            parameters=(ParameterSpec("missing", constraint="positive"),),
            statistics=(StatisticSpec("count"),),
            support="real",
        )
        declaration_module._DECLARATIONS.pop(ExampleDistribution)
        with self.assertRaisesRegex(ValueError, "not exposed"):
            register_declaration(invalid)
        self.assertIsNone(declaration_for(ExampleDistribution))

    def test_validation_rejects_child_cycles_and_inconsistent_flags(self):
        class ExampleDistribution:
            pass

        cyclic = DistributionDeclaration(
            name="cycle",
            distribution_type=ExampleDistribution,
            parameters=(),
            statistics=(),
            support="real",
        )
        object.__setattr__(cyclic, "children", (cyclic,))
        object.__setattr__(cyclic, "child_roles", ("self",))
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_declaration(cyclic)

        inconsistent = DistributionDeclaration(
            name="inconsistent",
            distribution_type=ExampleDistribution,
            parameters=(),
            statistics=(),
            support="real",
            exponential_family=ExponentialFamilySpec(
                sufficient_statistics=lambda x, engine: (x,),
                natural_parameters=lambda params, engine: (0.0,),
                log_partition=lambda params, engine: 0.0,
                fixed_base=False,
            ),
        )
        with self.assertRaisesRegex(ValueError, "base_measure_from_params"):
            validate_declaration(inconsistent)

    def test_generated_execution_validates_parameter_domain_and_score_shape(self):
        class ExampleDistribution:
            def __init__(self, scale, mode="valid"):
                self.scale = scale
                self.mode = mode

            @staticmethod
            def backend_log_density_from_params(x, scale, engine):
                return engine.asarray(x) * 0.0 - engine.log(scale)

        declaration = DistributionDeclaration(
            name="runtime_example",
            distribution_type=ExampleDistribution,
            parameters=(ParameterSpec("scale", constraint="positive"),),
            statistics=(),
            support="real",
        )
        self._track(ExampleDistribution)
        register_declaration(declaration)

        np.testing.assert_allclose(
            generated_log_density(ExampleDistribution(2.0), np.asarray([1.0, 2.0]), NUMPY_ENGINE),
            -np.log(2.0),
        )
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            generated_log_density(ExampleDistribution(-1.0), np.asarray([1.0]), NUMPY_ENGINE)
        with self.assertRaisesRegex(ValueError, "per-row"):
            generated_log_density(ExampleDistribution(2.0), np.asarray(1.0), NUMPY_ENGINE)

        ExampleDistribution.backend_log_density_from_params = staticmethod(
            lambda x, scale, engine: np.full(np.asarray(x).shape, np.nan)
        )
        with self.assertRaisesRegex(ValueError, "NaN"):
            generated_log_density(ExampleDistribution(2.0), np.asarray([1.0]), NUMPY_ENGINE)


if __name__ == "__main__":
    unittest.main()
