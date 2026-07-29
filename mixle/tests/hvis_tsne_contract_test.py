import unittest

import numpy as np
import scipy.sparse

from mixle.utils.hvis.embed import htsne, humap
from mixle.utils.hvis.goals import Anchor
from mixle.utils.hvis.tsne import (
    _convergence_update,
    _validate_dense_joint_probability,
    _validate_sparse_joint_probability,
    tsne_exact,
    update_alpha,
)


class JointProbabilityContractTest(unittest.TestCase):
    def setUp(self):
        self.valid = np.array([[0.0, 2.0, 1.0], [2.0, 0.0, 3.0], [1.0, 3.0, 0.0]])

    def test_dense_validation_normalizes_valid_input(self):
        p = _validate_dense_joint_probability(self.valid)
        self.assertAlmostEqual(float(p.sum()), 1.0)
        np.testing.assert_array_equal(np.diag(p), np.zeros(3))

    def test_dense_validation_rejects_invalid_probability_models(self):
        invalid = [
            self.valid[:, :2],
            np.array([[0.0, 1.0], [2.0, 0.0]]),
            np.array([[1.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, -1.0], [-1.0, 0.0]]),
            np.array([[0.0, np.nan], [np.nan, 0.0]]),
            np.zeros((2, 2)),
        ]
        for p in invalid:
            with self.subTest(p=repr(p)), self.assertRaises(ValueError):
                _validate_dense_joint_probability(p)

    def test_sparse_validation_rejects_asymmetry_before_triangle_use(self):
        with self.assertRaises(ValueError):
            _validate_sparse_joint_probability(scipy.sparse.csr_matrix([[0.0, 1.0], [0.0, 0.0]]))

    def test_sparse_validation_normalizes_valid_input(self):
        p = _validate_sparse_joint_probability(scipy.sparse.csr_matrix(self.valid))
        self.assertAlmostEqual(float(p.sum()), 1.0)
        np.testing.assert_array_equal(p.diagonal(), np.zeros(3))


class ConvergenceContractTest(unittest.TestCase):
    def test_small_nonnegative_improvement_converges(self):
        converged, _, _ = _convergence_update(1.0, 1.0 - 1.0e-8, 1.0e-7, 0)
        self.assertTrue(converged)

    def test_worsening_never_counts_as_convergence_and_eventually_diverges(self):
        converged, previous, count = _convergence_update(1.0, 1.1, 1.0e-7, 0)
        self.assertFalse(converged)
        converged, previous, count = _convergence_update(previous, 1.2, 1.0e-7, count)
        self.assertFalse(converged)
        with self.assertRaises(RuntimeError):
            _convergence_update(previous, 1.3, 1.0e-7, count)

    def test_nonfinite_objective_fails(self):
        with self.assertRaises(FloatingPointError):
            _convergence_update(1.0, np.nan, 1.0e-7, 0)


class OptimizerControlContractTest(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[0.0, 0.5], [0.5, 0.0]])
        self.y = np.array([[-1.0, 0.0], [1.0, 0.0]])

    def test_alpha_search_is_bounded_and_never_increases_its_objective(self):
        alpha = update_alpha(self.p, self.y, 1.0, 0.1, 1.0e-128, max_its=20, step=0.25)
        self.assertGreaterEqual(alpha, 0.75)
        self.assertLessEqual(alpha, 1.25)

    def test_alpha_search_rejects_invalid_domains_and_controls(self):
        for kwargs in (
            {"max_its": 0},
            {"step": 0.0},
            {"eps": -1.0},
            {"max_alpha": 0.01},
        ):
            with self.subTest(kwargs=repr(kwargs)), self.assertRaises(ValueError):
                update_alpha(self.p, self.y, 1.0, 0.1, 1.0e-128, **kwargs)

    def test_exact_optimizer_rejects_modulo_zero_controls(self):
        with self.assertRaises(ValueError):
            tsne_exact(self.p, max_its=1, print_iter=0)
        with self.assertRaises(ValueError):
            tsne_exact(self.p, max_its=1, check_every=0)

    def test_kernel_rejects_invalid_alpha_and_coordinates(self):
        with self.assertRaises(ValueError):
            update_alpha(self.p, self.y, 0.0, 0.1, 1.0e-128)
        with self.assertRaises(ValueError):
            update_alpha(self.p, np.array([[np.nan], [0.0]]), 1.0, 0.1, 1.0e-128)


class EntryPointContractTest(unittest.TestCase):
    def setUp(self):
        features = np.eye(4, dtype=np.float64) + 0.1
        self.affinity = [(features, features)]

    def test_exact_entry_point_materializes_generator_once(self):
        data = (float(i) for i in range(4))
        y = htsne(
            data,
            affinity=self.affinity,
            method="exact",
            perplexity=2.0,
            max_its=1,
            print_iter=10,
            seed=2,
        )
        self.assertEqual(y.shape, (4, 2))

    def test_internal_umap_entry_point_materializes_generator_once(self):
        data = (float(i) for i in range(4))
        y = humap(
            data,
            affinity=self.affinity,
            engine="internal",
            n_neighbors=2,
            n_epochs=1,
            seed=2,
        )
        self.assertEqual(y.shape, (4, 2))

    def test_unknown_or_unsupported_tsne_algorithm_is_rejected(self):
        with self.assertRaises(ValueError):
            htsne(range(4), affinity=self.affinity, method="typo")
        with self.assertRaises(ValueError):
            htsne(range(4), affinity=self.affinity, method="barnes_hut", optimize_alpha=True)
        with self.assertRaises(ValueError):
            htsne(range(4), affinity=self.affinity, method="barnes_hut", alpha=2.0)

    def test_umap_rejects_unsupported_goals_before_optional_import(self):
        with self.assertRaises(ValueError):
            humap(
                range(4),
                affinity=self.affinity,
                engine="umap-learn",
                goals=[Anchor([0], [[0.0, 0.0]])],
            )

    def test_umap_does_not_silently_reduce_neighborhood_size(self):
        with self.assertRaises(ValueError):
            humap(range(4), affinity=self.affinity, engine="internal", n_neighbors=4)


if __name__ == "__main__":
    unittest.main()
