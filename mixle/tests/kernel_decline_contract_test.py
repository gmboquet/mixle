"""Optimized-kernel fallback must be negotiated, typed, and data-independent."""

import unittest
from unittest.mock import patch

import numpy as np

import mixle.stats as stats
from mixle.engines import NUMPY_ENGINE, NumpyEngine
from mixle.stats.compute.capability_decline import KernelCapabilityDeclinedError
from mixle.stats.compute.kernel import GeneratedNumbaKernel, GeneratedNumbaKernelFactory
from mixle.stats.compute.stacked import StackedMixtureKernelFactory


def _gaussian_mixture():
    return stats.MixtureDistribution(
        [
            stats.GaussianDistribution(-1.0, 1.0),
            stats.GaussianDistribution(1.0, 1.0),
        ],
        [0.5, 0.5],
    )


class KernelDeclineContractTest(unittest.TestCase):
    def test_generated_accumulation_does_not_catch_runtime_value_error(self):
        model = _gaussian_mixture()
        estimator = model.estimator()
        kernel = GeneratedNumbaKernel(model, NUMPY_ENGINE, estimator=estimator)
        enc = model.dist_to_encoder().seq_encode([-1.0, 0.0, 1.0])

        with patch(
            "mixle.stats.compute.kernel._generated_numba_component_stats",
            side_effect=ValueError("malformed generated statistic"),
        ):
            with self.assertRaisesRegex(ValueError, "malformed generated statistic"):
                kernel.accumulate(enc, np.ones(3))

    def test_static_resident_decline_selects_host_before_execution(self):
        model = _gaussian_mixture()
        estimator = model.estimator()
        kernel = GeneratedNumbaKernel(model, NUMPY_ENGINE, estimator=estimator)
        enc = model.dist_to_encoder().seq_encode([-1.0, 0.0, 1.0])

        expected = estimator.accumulator_factory().make()
        expected.seq_update(enc, np.ones(3), model)
        with (
            patch("mixle.stats.compute.kernel._estimator_resident_supported", return_value=False),
            patch(
                "mixle.stats.compute.kernel._generated_numba_component_stats",
                side_effect=AssertionError("resident execution must not start"),
            ),
        ):
            actual = kernel.accumulate(enc, np.ones(3))
        np.testing.assert_allclose(actual[0], expected.value()[0])

    def test_stacked_factory_catches_only_typed_capability_decline(self):
        class TorchNamedEngine(NumpyEngine):
            name = "torch"

        class Fallback:
            def __init__(self):
                self.calls = 0
                self.result = object()

            def build(self, *_args, **_kwargs):
                self.calls += 1
                return self.result

        model = _gaussian_mixture()
        engine = TorchNamedEngine()
        fallback = Fallback()
        factory = StackedMixtureKernelFactory(fallback=fallback)

        with patch("mixle.stats.compute.stacked.StackedMixtureKernel", side_effect=ValueError("bad model data")):
            with self.assertRaisesRegex(ValueError, "bad model data"):
                factory.build(model, engine)
        self.assertEqual(fallback.calls, 0)

        with patch(
            "mixle.stats.compute.stacked.StackedMixtureKernel",
            side_effect=KernelCapabilityDeclinedError("static route mismatch"),
        ):
            self.assertIs(factory.build(model, engine), fallback.result)
        self.assertEqual(fallback.calls, 1)

    def test_generated_factory_does_not_hide_constructor_value_error(self):
        model = _gaussian_mixture()
        factory = GeneratedNumbaKernelFactory()
        with patch(
            "mixle.stats.compute.kernel.GeneratedNumbaKernel",
            side_effect=ValueError("constructor invariant failed"),
        ):
            with self.assertRaisesRegex(ValueError, "constructor invariant failed"):
                factory.build(model, NUMPY_ENGINE)


if __name__ == "__main__":
    unittest.main()
