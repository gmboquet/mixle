"""Tests for Gabow k-best spanning-tree enumeration (mixle.enumeration.spanning)."""

import unittest

import numpy as np

from mixle.enumeration.spanning import k_best_spanning_trees


class KBestSpanningTreeTestCase(unittest.TestCase):
    def test_matches_hand_computed_mst(self):
        # Kruskal by inspection: edges sorted ascending are (0,1)=1, (1,3)=2, (1,2)=3, (0,3)=4, (0,2)=5,
        # (2,3)=6. Taking (0,1), (1,3), (1,2) unions all four vertices with no cycle -- 3 = n-1 edges, total 6.
        w = np.array(
            [
                [0.0, 1.0, 5.0, 4.0],
                [1.0, 0.0, 3.0, 2.0],
                [5.0, 3.0, 0.0, 6.0],
                [4.0, 2.0, 6.0, 0.0],
            ]
        )
        total, tree = next(k_best_spanning_trees(w))
        self.assertAlmostEqual(total, 6.0)
        self.assertEqual(sorted(tree), [(0, 1), (1, 2), (1, 3)])

    def test_rejects_non_square_matrix(self):
        # MXR-080-0202: a 2x3 matrix used to be silently accepted, with the third vertex/column ignored
        # (n was taken from shape[0], so only a 2-vertex problem was ever solved).
        with self.assertRaises(ValueError):
            list(k_best_spanning_trees(np.array([[0.0, 1.0, 9.0], [1.0, 0.0, 9.0]])))

    def test_rejects_asymmetric_matrix(self):
        # MXR-080-0202: an asymmetric matrix used to be silently solved using only the upper triangle
        # (i < j), discarding the lower triangle's 999.0 entries entirely with no error or warning.
        asymmetric = np.array(
            [
                [0.0, 1.0, 100.0],
                [999.0, 0.0, 2.0],
                [999.0, 999.0, 0.0],
            ]
        )
        with self.assertRaises(ValueError):
            list(k_best_spanning_trees(asymmetric))

    def test_rejects_vector_input(self):
        # MXR-080-0202: a 1-D array used to fail with an incidental IndexError ("too many indices for
        # array") instead of a clear, actionable error.
        with self.assertRaises(ValueError):
            list(k_best_spanning_trees(np.array([1.0, 2.0, 3.0])))

    def test_error_message_is_not_an_incidental_indexerror(self):
        try:
            list(k_best_spanning_trees(np.array([1.0, 2.0, 3.0])))
            self.fail("expected ValueError")
        except IndexError:
            self.fail("vector input must raise a clear ValueError, not an incidental IndexError")
        except ValueError:
            pass

    def test_explicit_symmetrization_of_directed_data_still_works(self):
        # The documented policy for genuinely directed/asymmetric data is to symmetrize explicitly (e.g.
        # average both triangles) before calling -- not to rely on undocumented upper-triangle-only behavior.
        directed = np.array(
            [
                [0.0, 1.0, 100.0],
                [999.0, 0.0, 2.0],
                [999.0, 999.0, 0.0],
            ]
        )
        symmetrized = 0.5 * (directed + directed.T)
        np.testing.assert_allclose(symmetrized, symmetrized.T)  # sanity: fixture is actually symmetric
        total, tree = next(k_best_spanning_trees(symmetrized))
        self.assertTrue(np.isfinite(total))
        self.assertEqual(len(tree), 2)  # n - 1 edges for 3 vertices

    def test_negative_control_normal_square_symmetric_matrix(self):
        # A well-formed square symmetric adjacency matrix must be unaffected by the new validation.
        rng = np.random.RandomState(0)
        n = 6
        a = rng.rand(n, n)
        w = 0.5 * (a + a.T)
        np.fill_diagonal(w, 0.0)
        results = list(k_best_spanning_trees(w, k=5))
        self.assertEqual(len(results), 5)
        self.assertTrue(all(results[i][0] <= results[i + 1][0] + 1e-12 for i in range(4)))
        for _, tree in results:
            self.assertEqual(len(tree), n - 1)


if __name__ == "__main__":
    unittest.main()
