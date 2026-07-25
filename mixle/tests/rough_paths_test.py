"""WS-14: rough paths -- the truncated path signature, validated by its defining algebraic identities."""

import unittest
from math import factorial

import numpy as np

from mixle.ppl.rough_paths import path_signature, signature_norms, signature_tensor_product


def _outer_power(v, k):
    r = np.array(1.0)
    for _ in range(k):
        r = np.tensordot(r, v, axes=0)
    return r


class RoughPathsTest(unittest.TestCase):
    def test_contract_rejects_coercion_nonfinite_paths_and_unbounded_requests(self):
        path = np.zeros((2, 3))
        for depth in (1.5, "2", True):
            with self.subTest(depth=depth), self.assertRaises(TypeError):
                path_signature(path, depth)
        with self.assertRaises(ValueError):
            path_signature(np.array([[0.0], [np.nan]]), 2)
        with self.assertRaises(ValueError):
            path_signature(np.zeros((2, 0)), 2)
        with self.assertRaisesRegex(ValueError, "max_elements"):
            path_signature(path, 10, max_elements=100)
        with self.assertRaisesRegex(ValueError, "max_elements"):
            path_signature(path, 10**9, max_elements=100)
        with self.assertRaisesRegex(ValueError, "max_work_elements"):
            path_signature(np.zeros((100, 2)), 3, max_work_elements=100)

    def test_zero_depth_and_single_point_paths_are_bounded(self):
        zero = path_signature(np.array([[1.0, 2.0]]), 0)
        self.assertEqual(len(zero), 1)
        self.assertEqual(float(zero[0]), 1.0)
        single = path_signature(np.array([[1.0, 2.0]]), 3)
        self.assertEqual([level.shape for level in single], [(), (2,), (2, 2), (2, 2, 2)])
        self.assertTrue(all(np.all(level == 0.0) for level in single[1:]))

    def test_linear_path_closed_form(self):
        # signature of a straight segment a->b: level k == (b-a)^{otimes k}/k!
        a, b = np.array([0.3, -0.2, 0.5]), np.array([1.0, 0.4, -0.1])
        sig = path_signature(np.array([a, b]), 4)
        for k in range(1, 5):
            with self.subTest(k=k):
                self.assertTrue(np.allclose(sig[k], _outer_power(b - a, k) / factorial(k)))

    def test_chen_identity(self):
        # S(X) == S(X[:m]) (x) S(X[m:]) -- concatenation maps to the tensor-algebra product
        rng = np.random.RandomState(0)
        p = np.cumsum(rng.standard_normal((9, 3)), axis=0)
        full = path_signature(p, 4)
        comb = signature_tensor_product(path_signature(p[:5], 4), path_signature(p[4:], 4), 4)
        for k in range(5):
            with self.subTest(k=k):
                self.assertTrue(np.allclose(full[k], comb[k], atol=1e-10))

    def test_factorial_bound(self):
        # ||S_k|| <= L^k / k! with L the piecewise-linear path length
        rng = np.random.RandomState(1)
        p = np.cumsum(0.1 * rng.standard_normal((20, 2)), axis=0)
        length = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        for k, nrm in enumerate(signature_norms(path_signature(p, 5)), start=1):
            with self.subTest(k=k):
                self.assertLessEqual(nrm, length**k / factorial(k) + 1e-12)

    def test_time_reversal_inverse(self):
        # S(X) (x) S(reverse X) == 1 (the reversed path is the group inverse)
        rng = np.random.RandomState(2)
        p = np.cumsum(rng.standard_normal((6, 2)), axis=0)
        prod = signature_tensor_product(path_signature(p, 3), path_signature(p[::-1], 3), 3)
        self.assertAlmostEqual(float(prod[0]), 1.0, places=10)
        for k in range(1, 4):
            with self.subTest(k=k):
                self.assertTrue(np.allclose(prod[k], 0.0, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
