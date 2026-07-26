import unittest

import numpy as np

from mixle.stats import (
    CategoricalDistribution,
    DiagonalGaussianDistribution,
    GaussianDistribution,
    MixtureDistribution,
    PoissonDistribution,
)
from mixle.stats.compute.fused_kernels import CompiledEncoding, CompiledMixture


class FusedKernelRuntimeContractTest(unittest.TestCase):
    def setUp(self):
        self.model = MixtureDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(2.0, 3.0)],
            [0.4, 0.6],
        )
        self.compiled = CompiledMixture(self.model)
        self.encoding = self.compiled.encode([-2.0, 0.0, 3.0])

    def test_encoding_preserves_two_item_interface_and_structure(self):
        self.assertIsInstance(self.encoding, CompiledEncoding)
        n, columns = self.encoding
        self.assertEqual(n, 3)
        self.assertIs(columns, self.encoding.columns)
        np.testing.assert_allclose(
            self.compiled.seq_log_density(self.encoding),
            self.model.seq_log_density(self.model.dist_to_encoder().seq_encode([-2.0, 0.0, 3.0])),
        )

        incompatible = CompiledMixture(
            MixtureDistribution(
                [PoissonDistribution(1.0), PoissonDistribution(3.0)],
                [0.5, 0.5],
            )
        ).encode([0, 1, 2])
        with self.assertRaisesRegex(ValueError, "incompatible compiled model structure"):
            self.compiled.seq_log_density(incompatible)
        with self.assertRaisesRegex(TypeError, "must be produced"):
            self.compiled.seq_log_density(tuple(self.encoding))

    def test_replacement_model_must_match_compiled_structure(self):
        compatible = MixtureDistribution(
            [GaussianDistribution(-4.0, 2.0), GaussianDistribution(5.0, 0.5)],
            [0.25, 0.75],
        )
        self.assertEqual(self.compiled.seq_component_log_density(self.encoding, compatible).shape, (3, 2))

        wrong_count = MixtureDistribution([GaussianDistribution(0.0, 1.0)], [1.0])
        with self.assertRaisesRegex(ValueError, "1 components; expected 2"):
            self.compiled.seq_component_log_density(self.encoding, wrong_count)

        wrong_family = MixtureDistribution(
            [PoissonDistribution(1.0), PoissonDistribution(2.0)],
            [0.5, 0.5],
        )
        with self.assertRaisesRegex(ValueError, "incompatible distribution structure"):
            self.compiled.seq_component_log_density(self.encoding, wrong_family)

        categories = MixtureDistribution(
            [
                CategoricalDistribution({"a": 1.0}),
                CategoricalDistribution({"a": 0.5, "b": 0.5}),
            ],
            [0.5, 0.5],
        )
        category_compiled = CompiledMixture(categories)
        category_encoding = category_compiled.encode(["a", "b"])
        changed_vocabulary = MixtureDistribution(
            [
                CategoricalDistribution({"a": 1.0}),
                CategoricalDistribution({"a": 0.5, "c": 0.5}),
            ],
            [0.5, 0.5],
        )
        with self.assertRaisesRegex(ValueError, "incompatible distribution structure"):
            category_compiled.seq_log_density(category_encoding, changed_vocabulary)

    def test_vector_dimension_is_part_of_structure(self):
        compiled = CompiledMixture(DiagonalGaussianDistribution([0.0, 0.0], [1.0, 1.0]))
        encoding = compiled.encode([[0.0, 1.0]])
        replacement = DiagonalGaussianDistribution([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "incompatible distribution structure"):
            compiled.seq_log_density(encoding, replacement)

    def test_gamma_requires_exact_safe_geometry(self):
        valid = np.full((3, 2), 0.5)
        result = self.compiled.weighted_suff_stats(self.encoding, valid)
        self.assertEqual(np.asarray(result[0]).shape, (2,))

        for invalid in (
            np.ones((2, 2)),
            np.ones((3, 1)),
            np.ones(6),
        ):
            with self.subTest(shape=invalid.shape):
                with self.assertRaisesRegex(ValueError, r"gamma must have shape \(3, 2\)"):
                    self.compiled.weighted_suff_stats(self.encoding, invalid)

        for invalid in (
            np.array([[0.5, 0.5], [np.nan, 0.0], [0.5, 0.5]]),
            np.array([[0.5, 0.5], [-0.1, 1.1], [0.5, 0.5]]),
        ):
            with self.subTest(values=invalid):
                with self.assertRaises(ValueError):
                    self.compiled.weighted_suff_stats(self.encoding, invalid)

    def test_external_weights_require_one_finite_non_negative_value_per_row(self):
        for invalid in (
            np.ones(2),
            np.array([1.0, np.nan, 1.0]),
            np.array([1.0, -1.0, 1.0]),
        ):
            with self.subTest(weights=invalid):
                with self.assertRaises(ValueError):
                    self.compiled.em_step(self.encoding, object(), weights=invalid)

    def test_worker_pool_growth_closes_the_superseded_pool(self):
        self.compiled.seq_component_log_density(self.encoding, n_threads=2)
        old_pool = self.compiled._pool
        self.compiled.seq_component_log_density(self.encoding, n_threads=3)
        self.assertIsNot(self.compiled._pool, old_pool)
        with self.assertRaises(RuntimeError):
            old_pool.submit(lambda: None)

    def test_close_and_context_manager_have_idempotent_lifecycle(self):
        with CompiledMixture(self.model) as compiled:
            encoding = compiled.encode([-1.0, 1.0])
            compiled.seq_component_log_density(encoding, n_threads=2)
            self.assertIsNotNone(compiled._pool)

        self.assertIsNone(compiled._pool)
        compiled.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            compiled.seq_component_log_density(encoding)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            compiled.encode([0.0])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            compiled.__enter__()

    def test_thread_count_must_be_a_positive_integer(self):
        for invalid in (0, -1, 1.5, True):
            with self.subTest(n_threads=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    self.compiled.seq_component_log_density(self.encoding, n_threads=invalid)


if __name__ == "__main__":
    unittest.main()
