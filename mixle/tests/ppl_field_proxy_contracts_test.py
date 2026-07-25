"""Release contracts for field proxy schemas and optimization layout."""

import unittest

import numpy as np

from mixle.ppl import (
    RBF,
    CustomProxy,
    GaussianField,
    GaussianProxy,
    LogisticNicheProxy,
    PoissonProxy,
    fit_field,
    free,
)

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ProxySchemaContractTest(unittest.TestCase):
    def test_gaussian_proxy_requires_aligned_finite_data_and_valid_scale(self):
        invalid = [
            lambda: GaussianProxy([[1.0], [2.0]]),
            lambda: GaussianProxy([1.0, np.nan]),
            lambda: GaussianProxy([1.0, 2.0], index=[0]),
            lambda: GaussianProxy([1.0, 2.0], index=[0.0, 1.5]),
            lambda: GaussianProxy([1.0, 2.0], scale=0.0),
            lambda: GaussianProxy([1.0, 2.0], slope=np.inf),
        ]
        for call in invalid:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()

    def test_niche_poisson_and_custom_support_are_strict(self):
        for value in ([0.0, 1.0], [[0.0, 2.0]], [[0.0, np.nan]]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    LogisticNicheProxy(value)
        for value in ([-1], [1.5], [np.inf]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    PoissonProxy(value)
        with self.assertRaises(ValueError):
            PoissonProxy([0, 1], offset=[0.0])
        with self.assertRaises(TypeError):
            CustomProxy(None)
        with self.assertRaises(ValueError):
            CustomProxy(lambda *_: 0.0, param_specs=[("rate", "positive", 0.0)])
        with self.assertRaises(ValueError):
            CustomProxy(lambda *_: 0.0, param_specs=[("x", "real", 0.0), ("x", "real", 1.0)])


@unittest.skipUnless(HAS_TORCH, "fit_field requires PyTorch")
class FieldLayoutContractTest(unittest.TestCase):
    @staticmethod
    def _field():
        return GaussianField(np.arange(3.0), RBF(lengthscale=1.0), name="T")

    def test_proxy_shapes_and_indices_are_checked_against_field(self):
        with self.assertRaisesRegex(ValueError, "3 nodes"):
            fit_field(self._field(), [GaussianProxy([1.0])], how="map")
        with self.assertRaisesRegex(ValueError, "within"):
            fit_field(self._field(), [GaussianProxy([1.0], index=[3])], how="map")
        with self.assertRaisesRegex(ValueError, "field columns"):
            fit_field(self._field(), [LogisticNicheProxy([[0.0, 1.0]])], how="map")
        with self.assertRaisesRegex(ValueError, "3 nodes"):
            fit_field(self._field(), [PoissonProxy([0, 1])], how="map")

    def test_duplicate_names_and_unknown_targets_are_rejected(self):
        first = GaussianProxy([0.0, 0.0, 0.0], slope=free, prefix="sensor")
        second = GaussianProxy([0.0, 0.0, 0.0], slope=free, prefix="sensor")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fit_field(self._field(), [first, second], how="map")
        collision = CustomProxy(
            lambda field, params, torch: -0.0 * params["T"],
            param_specs=[("T", "real", 0.0)],
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fit_field(self._field(), [collision], how="map")
        with self.assertRaisesRegex(ValueError, "unknown field"):
            fit_field(
                self._field(),
                [GaussianProxy([0.0, 0.0, 0.0]).on("missing")],
                how="map",
            )

    def test_positive_initial_values_are_natural_scale_and_receipted(self):
        proxy = CustomProxy(
            lambda field, params, torch: -0.0 * params["rate"],
            param_specs=[("rate", "positive", 2.5)],
        )
        fitted = fit_field(None, [proxy], how="map", max_iter=1)
        self.assertAlmostEqual(fitted.mean("rate"), 2.5)
        self.assertEqual(fitted.initialization_space, "natural")
        self.assertEqual(fitted.parameter_id("rate"), "node:1")

        fitted_override = fit_field(None, [proxy], how="map", max_iter=1, init={"rate": 4.0})
        self.assertAlmostEqual(fitted_override.mean("rate"), 4.0)
        for init in ({"rate": 0.0}, {"rate": [1.0, 2.0]}, {"unknown": 1.0}):
            with self.subTest(init=init):
                with self.assertRaises(ValueError):
                    fit_field(None, [proxy], how="map", max_iter=1, init=init)

    def test_custom_likelihood_must_be_finite_scalar(self):
        vector = CustomProxy(lambda field, params, torch: torch.ones(2))
        with self.assertRaisesRegex(ValueError, "scalar"):
            fit_field(None, [vector], how="map", max_iter=1)
        nonfinite = CustomProxy(lambda field, params, torch: torch.tensor(float("nan")))
        with self.assertRaises(FloatingPointError):
            fit_field(None, [nonfinite], how="map", max_iter=1)


if __name__ == "__main__":
    unittest.main()
