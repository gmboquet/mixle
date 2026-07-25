"""Release contracts for field posterior geometry, optimization, and approximation receipts."""

import unittest

import numpy as np

from mixle.ppl import (
    RBF,
    CustomProxy,
    FieldPosterior,
    FieldSystem,
    GaussianField,
    GaussianProxy,
    fit_field,
    free,
    joint,
    multistart,
)

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PosteriorGeometryContractTest(unittest.TestCase):
    def test_invalid_covariance_is_rejected_instead_of_clipped(self):
        with self.assertRaisesRegex(ValueError, "positive-definite"):
            FieldPosterior(
                {"x": 0.0},
                np.array([[-1.0]]),
                {"x": (0, 1)},
                "x",
                np.array([[1.0]]),
                0.0,
                _field_coordinate_count=1,
            )

    def test_sampling_validates_node_and_conditioning_contracts(self):
        posterior = FieldPosterior(
            {"x": 1.0},
            np.array([[1.0]]),
            {"x": (0, 1)},
            "x",
            np.array([[1.0]]),
            0.0,
            _field_coordinate_count=1,
            _supports={"x": "positive"},
        )
        with self.assertRaises(TypeError):
            posterior.sample(nodes="x")
        with self.assertRaises(ValueError):
            posterior.sample(nodes=["x", "x"])
        with self.assertRaises(TypeError):
            posterior.sample(given=[("x", 1.0)])
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            posterior.sample(given={"x": 0.0})


@unittest.skipUnless(HAS_TORCH, "fit_field requires PyTorch")
class FieldInferenceContractTest(unittest.TestCase):
    @staticmethod
    def _field(name="T"):
        return GaussianField(np.arange(3.0), RBF(lengthscale=1.0), name=name)

    def test_laplace_refuses_unidentifiable_uncertainty_without_jitter(self):
        proxy = CustomProxy(
            lambda field, params, torch: -0.0 * params["x"],
            param_specs=[("x", "real", 0.0)],
        )
        with self.assertRaisesRegex(ValueError, "positive-definite"):
            fit_field(None, [proxy], how="laplace")

    def test_map_exhaustion_is_a_failure_not_a_posterior(self):
        proxy = CustomProxy(
            lambda field, params, torch: -(
                (1.0 - params["x"][0]) ** 2
                + 100.0 * (params["x"][1] - params["x"][0] ** 2) ** 2
            ),
            param_specs=[("x", "real", np.array([-1.2, 1.0]))],
        )
        with self.assertRaisesRegex(RuntimeError, "exhausted max_iter"):
            fit_field(None, [proxy], how="map", max_iter=1)

    def test_gauss_newton_rejects_free_scale_and_reports_approximation(self):
        field = self._field()
        with self.assertRaisesRegex(ValueError, "log-scale curvature"):
            fit_field(field, [GaussianProxy([0.0, 0.1, 0.2], scale=free)], how="gauss_newton")
        posterior = fit_field(
            field,
            [GaussianProxy([0.0, 0.1, 0.2], scale=0.5)],
            how="gauss_newton",
        )
        self.assertEqual(posterior.curvature_method, "gauss-newton-jtj-woodbury")
        self.assertEqual(posterior.uncertainty_status, "approximate")
        self.assertEqual(posterior.regularization, 0.0)

    def test_vi_uses_reproducible_unconstrained_coordinates_and_exact_lognormal_moments(self):
        proxy = CustomProxy(
            lambda field, params, torch: (
                -0.5 * (torch.log(params["x"]) / 0.5) ** 2 - torch.log(params["x"])
            ),
            param_specs=[("x", "positive", 1.0)],
        )
        options = dict(how="vi", vi_steps=40, vi_samples=3, rng=7)
        first = fit_field(None, [proxy], **options)
        second = fit_field(None, [proxy], **options)
        mean_u, variance_u = first._variational_params["x"]
        expected_mean = np.exp(mean_u + 0.5 * variance_u)
        expected_var = np.expm1(variance_u) * np.exp(2.0 * mean_u + variance_u)
        self.assertAlmostEqual(first.mean("x"), float(expected_mean[0]))
        self.assertAlmostEqual(float(first.sd("x")[0]), float(np.sqrt(expected_var[0])))
        self.assertEqual(first.termination_reason, "completed-fixed-steps; convergence-not-certified")
        self.assertFalse(first.converged)
        self.assertAlmostEqual(first.mean("x"), second.mean("x"))
        np.testing.assert_allclose(first._variational_params["x"][1], second._variational_params["x"][1])

    def test_vi_controls_are_strict(self):
        proxy = CustomProxy(
            lambda field, params, torch: -0.5 * params["x"] ** 2,
            param_specs=[("x", "real", 0.0)],
        )
        for option in ({"vi_steps": 0}, {"vi_samples": 0}, {"vi_lr": 0.0}):
            with self.subTest(option=option):
                with self.assertRaises(ValueError):
                    fit_field(None, [proxy], how="vi", **option)

    def test_full_proxy_curvature_retains_cross_field_coupling(self):
        a, b = self._field("A"), self._field("B")
        pa = GaussianProxy([0.0, 0.1, 0.0], scale=0.5, prefix="a").on("A")
        pb = GaussianProxy([0.0, -0.1, 0.0], scale=0.5, prefix="b").on("B")
        coupling = CustomProxy(
            lambda field, params, torch: -0.5 * torch.sum((params["A"] - params["B"]) ** 2),
            prefix="coupling",
        )
        posterior = fit_field(FieldSystem([a, b]), [pa, pb, coupling], how="laplace")
        block = posterior._proxy_info["coupling"]
        self.assertGreater(np.max(np.abs(block[:3, 3:6])), 0.0)
        _, sd = posterior.field_posterior(include=["coupling"])
        self.assertTrue(np.all(np.isfinite(sd)))
        self.assertEqual(posterior.curvature_method, "observed-negative-log-posterior-hessian")

    def test_joint_identity_and_multistart_receipts_are_strict(self):
        first, second = self._field("T"), self._field("T")
        with self.assertRaisesRegex(ValueError, "distinct field objects"):
            joint(
                [
                    (first, GaussianProxy([0.0, 0.0, 0.0])),
                    (second, GaussianProxy([0.0, 0.0, 0.0])),
                ]
            )

        proxy = CustomProxy(
            lambda field, params, torch: -0.5 * (params["x"] - 2.0) ** 2,
            param_specs=[("x", "real", 0.0)],
        )
        model = joint([(None, proxy)])
        fitted = multistart(model, [{"unknown": 1.0}, {"x": 3.0}])
        self.assertEqual([item["success"] for item in fitted.multistart_receipt], [False, True])
        self.assertAlmostEqual(fitted.mean("x"), 2.0, places=4)
        with self.assertRaisesRegex(RuntimeError, "all multistart fits failed"):
            multistart(model, [{"unknown": 1.0}])


if __name__ == "__main__":
    unittest.main()
