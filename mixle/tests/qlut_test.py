"""Quantized function LUTs (mixle.engines.qlut): every nonlinearity as an integer gather, no transcendental."""

import unittest

import numpy as np

from mixle.engines.qlut import (
    QuantizedFunction,
    error_bound,
    quantized_activation,
    quantized_exp,
    step_for_tolerance,
    table_bytes,
)


class QuantizedActivationTest(unittest.TestCase):
    def test_bounded_activations_meet_the_derivative_bound(self):
        rng = np.random.RandomState(0)
        x = rng.randn(100000) * 4  # within the saturating range
        for name, sup in (("sigmoid", 0.25), ("tanh", 1.0)):
            q = quantized_activation(name, step=0.01)
            self.assertLessEqual(q.max_abs_error(x), error_bound(sup, 0.01) * 1.05)

    def test_unbounded_activations_linear_tail_handles_out_of_range(self):
        # gelu/silu/softplus grow linearly; values FAR beyond the table must still be accurate via the tail
        x = np.array([-60.0, -25.0, 25.0, 60.0, 100.0])
        for name, ref in (
            ("gelu", lambda v: 0.5 * v * (1 + np.tanh(0.7978845608 * (v + 0.044715 * v**3)))),
            ("softplus", lambda v: np.log1p(np.exp(-np.abs(v))) + np.maximum(v, 0)),
            ("relu", lambda v: np.maximum(v, 0.0)),
        ):
            q = quantized_activation(name, step=0.02, span=20.0)
            self.assertLess(float(np.max(np.abs(q(x) - ref(x)))), 0.05, name)

    def test_lookup_from_codes_matches_call(self):
        q = quantized_activation("sigmoid", step=0.01)
        x = np.linspace(-5, 5, 1000)
        codes = np.rint(x / 0.01).astype(np.int64)
        self.assertTrue(np.array_equal(q.lookup(codes), q(x)))

    def test_unknown_activation_raises(self):
        with self.assertRaises(ValueError):
            quantized_activation("frobnicate")


class QuantizedExpTest(unittest.TestCase):
    def test_exp_lns_to_linear_is_a_table(self):
        # the softmax / attention 'back to linear' as a gather over LNS log-codes -- no real exp
        s = 0.01
        qexp = quantized_exp(log_step=s, lo_log=-30.0)
        kcodes = np.arange(-3000, 1)  # log-codes in [-30, 0]
        ref = np.exp(kcodes * s)
        got = qexp.lookup(kcodes)
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)
        self.assertLess(float(np.max(rel)), 1e-9)  # exact: the table IS exp(k*s)


class HelpersTest(unittest.TestCase):
    def test_step_for_tolerance_meets_bound(self):
        s = step_for_tolerance(1e-3, 0.25)  # sigmoid
        self.assertLessEqual(error_bound(0.25, s), 1e-3 + 1e-12)
        with self.assertRaises(ValueError):
            step_for_tolerance(0.0, 0.25)

    def test_table_is_cache_resident(self):
        # a sigmoid table over [-20,20] at step 0.01 is ~32 KB -- fits L1/L2
        self.assertLess(table_bytes(0.01, -20.0, 20.0), 64 * 1024)

    def test_construction_validates(self):
        with self.assertRaises(ValueError):
            QuantizedFunction(np.tanh, step=0.0, lo=-1, hi=1)
        with self.assertRaises(ValueError):
            QuantizedFunction(np.tanh, step=0.1, lo=1.0, hi=-1.0)


if __name__ == "__main__":
    unittest.main()
