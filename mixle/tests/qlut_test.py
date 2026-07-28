"""Quantized function LUTs (mixle.engines.qlut): every nonlinearity as an integer gather, no transcendental."""

import unittest

import numpy as np

from mixle.engines.qlut import (
    QuantizedFunction,
    error_bound,
    lse_error_bound,
    quantized_activation,
    quantized_exp,
    quantized_logsumexp,
    step_for_tolerance,
    table_bytes,
)


class QuantizedActivationTest(unittest.TestCase):
    def test_callback_and_direct_code_contracts(self):
        with self.assertRaises(TypeError):
            QuantizedFunction(3, step=0.1, lo=-1, hi=1)
        with self.assertRaises(ValueError):
            QuantizedFunction(lambda x: 1.0, step=0.1, lo=-1, hi=1)
        q = QuantizedFunction(np.exp, step=0.1, lo=-1, hi=1)
        for bad in ([1.9], [999], [True]):
            with self.assertRaises(ValueError):
                q.lookup(bad)

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


class QuantizedLogsumexpTest(unittest.TestCase):
    def test_error_stays_within_the_grid_bound(self):
        rng = np.random.RandomState(3)
        scores = rng.normal(0, 3, 200000)
        exact = float(np.log(np.sum(np.exp(scores - scores.max()))) + scores.max())
        for bits in (8, 12):
            got = quantized_logsumexp(scores, bits=bits, span=24.0)
            self.assertLessEqual(abs(got - exact), lse_error_bound(bits, 24.0), f"bits={bits}")

    def test_weighted_form_is_the_cell_collapsed_attention_lse(self):
        # LSE over per-cell (score, integer count) == LSE over the expanded token stream: the
        # group-attention identity, computed with 2^bits exps instead of one per token.
        rng = np.random.RandomState(4)
        cell_scores = rng.normal(0, 2, 300)
        counts = rng.randint(1, 500, 300)
        token_scores = np.repeat(cell_scores, counts)
        exact = float(np.log(np.sum(np.exp(token_scores - token_scores.max()))) + token_scores.max())
        got = quantized_logsumexp(cell_scores, bits=12, span=24.0, weights=counts)
        self.assertLessEqual(abs(got - exact), lse_error_bound(12, 24.0))

    def test_deep_tail_clips_without_breaking_the_bound(self):
        scores = np.concatenate([np.array([0.0]), np.full(100000, -100.0)])  # far below span=24
        exact = float(np.log(np.sum(np.exp(scores))))  # ~0: the tail is ~1e-44 mass
        got = quantized_logsumexp(scores, bits=12, span=24.0)
        self.assertLessEqual(abs(got - exact), lse_error_bound(12, 24.0))

    def test_masked_slots_and_degenerate_inputs(self):
        # the max itself lands exactly on the top grid level, so a single score is exact
        self.assertAlmostEqual(quantized_logsumexp([3.7]), 3.7, places=12)
        # -inf scores are masked slots (softmax semantics); all-masked or all-zero-weight is -inf
        self.assertAlmostEqual(quantized_logsumexp([2.0, -np.inf]), 2.0, places=12)
        self.assertEqual(quantized_logsumexp([-np.inf, -np.inf]), -np.inf)
        self.assertEqual(quantized_logsumexp([1.0, 2.0], weights=[0, 0]), -np.inf)

    def test_validates_inputs(self):
        with self.assertRaises(ValueError):
            quantized_logsumexp([])
        with self.assertRaises(ValueError):
            quantized_logsumexp([1.0], bits=0)
        with self.assertRaises(ValueError):
            quantized_logsumexp([1.0], bits=True)
        with self.assertRaises(ValueError):
            quantized_logsumexp([1.0], span=-1.0)
        with self.assertRaises(ValueError):
            quantized_logsumexp([1.0, np.nan])
        with self.assertRaises(ValueError):
            quantized_logsumexp([1.0, 2.0], weights=[1.0])
        with self.assertRaises(ValueError):
            quantized_logsumexp([1.0, 2.0], weights=[1.0, -1.0])
        # MXR-080-0144: non-finite span/weights used to pass validation and silently turn the result
        # into NaN/inf (NaN < 0 is False, so the old `(w < 0).any()` check never caught a NaN weight)
        with self.assertRaises(ValueError):
            quantized_logsumexp([1.0], span=np.nan)
        with self.assertRaises(ValueError):
            quantized_logsumexp([1.0], span=np.inf)
        with self.assertRaises(ValueError):
            quantized_logsumexp([1.0, 2.0], weights=[1.0, np.nan])
        with self.assertRaises(ValueError):
            quantized_logsumexp([1.0, 2.0], weights=[1.0, np.inf])
        # negative control: a normal, well-formed call still works after the added validation (a lone
        # score sits exactly on the top grid point, so this is exact, not just within the grid bound)
        self.assertAlmostEqual(quantized_logsumexp([2.0], weights=[3.0]), float(np.log(3.0)) + 2.0, places=12)

    def test_weight_scaling_prevents_finite_linear_overflow(self):
        scores = [0.0, 0.0]
        weights = [1e308, 1e308]
        got = quantized_logsumexp(scores, weights=weights)
        expected = float(np.log(2.0) + np.log(1e308))
        self.assertTrue(np.isfinite(got))
        self.assertAlmostEqual(got, expected, places=12)
        self.assertTrue(np.isfinite(lse_error_bound(12, 24.0, scores=scores, weights=weights)))


class LseErrorBoundTailTest(unittest.TestCase):
    """MXR-080-0143: the grid-only bound omits clipped mass elevated to the bottom bin, which is not
    generally negligible -- a clipped score paired with a large weight can dominate the sum. Passing
    the same ``scores``/``weights`` to :func:`lse_error_bound` closes that gap with a derived,
    data-dependent tail term.
    """

    ADVERSARIAL_SCORES = [0.0, -25.0]
    ADVERSARIAL_WEIGHTS = [1.0, 1e12]
    BITS, SPAN = 12, 24.0

    def test_grid_only_bound_understates_the_adversarial_case_by_orders_of_magnitude(self):
        # the audit's exact scenario: score -25 sits just outside span=24 (relative to the max at 0),
        # so it gets clipped UP to the bottom bin instead of contributing its true, tiny mass -- but its
        # weight is 1e12, so the elevation error is enormous relative to the total sum
        exact = float(np.log(np.sum(np.asarray(self.ADVERSARIAL_WEIGHTS) * np.exp(self.ADVERSARIAL_SCORES))))
        got = quantized_logsumexp(
            self.ADVERSARIAL_SCORES, bits=self.BITS, span=self.SPAN, weights=self.ADVERSARIAL_WEIGHTS
        )
        actual_error = abs(got - exact)
        grid_only = lse_error_bound(self.BITS, self.SPAN)  # old call convention: bits/span only, no data
        # pin down the concrete figures from the audit so this test fails loudly if the mechanism drifts
        self.assertAlmostEqual(actual_error, 0.9623, places=3)
        self.assertAlmostEqual(grid_only, 0.00293, places=5)
        self.assertGreater(actual_error, 300 * grid_only, "the grid-only bound must be shown understating by >300x")

    def test_tail_aware_bound_covers_the_adversarial_case(self):
        exact = float(np.log(np.sum(np.asarray(self.ADVERSARIAL_WEIGHTS) * np.exp(self.ADVERSARIAL_SCORES))))
        got = quantized_logsumexp(
            self.ADVERSARIAL_SCORES, bits=self.BITS, span=self.SPAN, weights=self.ADVERSARIAL_WEIGHTS
        )
        actual_error = abs(got - exact)
        bound = lse_error_bound(self.BITS, self.SPAN, scores=self.ADVERSARIAL_SCORES, weights=self.ADVERSARIAL_WEIGHTS)
        self.assertGreaterEqual(bound, actual_error, "tail-aware bound must actually cover the measured error")

    def test_tail_aware_bound_matches_grid_only_when_nothing_clips(self):
        # negative control: well within span with unit weights -- the tail term must stay negligible,
        # i.e. the fix must not regress the pre-existing, already-correct grid-only behavior
        rng = np.random.RandomState(3)
        scores = rng.normal(0, 3, 200000)
        exact = float(np.log(np.sum(np.exp(scores - scores.max()))) + scores.max())
        got = quantized_logsumexp(scores, bits=self.BITS, span=self.SPAN)
        actual_error = abs(got - exact)
        grid_only = lse_error_bound(self.BITS, self.SPAN)
        tail_aware = lse_error_bound(self.BITS, self.SPAN, scores=scores)
        self.assertGreaterEqual(tail_aware, actual_error)
        self.assertLess(tail_aware - grid_only, 1e-6, "no meaningful clipping: tail term must be ~0")

    def test_tail_aware_bound_holds_across_randomized_adversarial_weights(self):
        # fuzz sweep: huge-dynamic-range weights on scores that may or may not clip, various bits/spans
        rng = np.random.RandomState(42)
        checked = 0
        for _ in range(200):
            n = rng.randint(2, 6)
            scores = rng.uniform(-2000, 5, size=n)
            scores[0] = 0.0
            weights = np.exp(rng.uniform(0, 30, size=n))
            bits = int(rng.choice([8, 10, 12]))
            span = float(rng.choice([8.0, 16.0, 24.0]))
            exact = float(np.log(np.sum(weights * np.exp(scores - scores.max()))) + scores.max())
            if not np.isfinite(exact):
                continue
            got = quantized_logsumexp(scores, bits=bits, span=span, weights=weights)
            bound = lse_error_bound(bits, span, scores=scores, weights=weights)
            self.assertGreaterEqual(
                bound, abs(got - exact) - 1e-9, f"bits={bits} span={span} scores={scores} weights={weights}"
            )
            checked += 1
        self.assertGreater(checked, 100, "the fuzz sweep must actually exercise a meaningful number of trials")

    def test_weights_without_scores_is_rejected(self):
        with self.assertRaises(ValueError):
            lse_error_bound(self.BITS, self.SPAN, weights=[1.0, 2.0])

    def test_lse_error_bound_validates_bits_and_span(self):
        with self.assertRaises(ValueError):
            lse_error_bound(0, self.SPAN)
        with self.assertRaises(ValueError):
            lse_error_bound(self.BITS, -1.0)
        with self.assertRaises(ValueError):
            lse_error_bound(self.BITS, np.nan)
        with self.assertRaises(ValueError):
            lse_error_bound(self.BITS, np.inf)

    def test_all_masked_or_zero_weight_gives_zero_bound(self):
        # matches quantized_logsumexp's -inf result for the same inputs: no rounding occurs
        self.assertEqual(lse_error_bound(self.BITS, self.SPAN, scores=[-np.inf, -np.inf]), 0.0)
        self.assertEqual(lse_error_bound(self.BITS, self.SPAN, scores=[1.0, 2.0], weights=[0.0, 0.0]), 0.0)


class HelpersTest(unittest.TestCase):
    def test_step_for_tolerance_meets_bound(self):
        s = step_for_tolerance(1e-3, 0.25)  # sigmoid
        self.assertLessEqual(error_bound(0.25, s), 1e-3 + 1e-12)
        with self.assertRaises(ValueError):
            step_for_tolerance(0.0, 0.25)

    def test_table_is_cache_resident(self):
        # a sigmoid table over [-20,20] at step 0.01 is ~32 KB -- fits L1/L2
        self.assertLess(table_bytes(0.01, -20.0, 20.0), 64 * 1024)

    def test_table_bytes_rejects_invalid_grid_and_item_size(self):
        for args in ((0.0, -1.0, 1.0), (-0.1, -1.0, 1.0), (0.1, 1.0, -1.0)):
            with self.assertRaises(ValueError):
                table_bytes(*args)
        for itemsize in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                table_bytes(0.1, -1.0, 1.0, itemsize=itemsize)

    def test_construction_validates(self):
        with self.assertRaises(ValueError):
            QuantizedFunction(np.tanh, step=0.0, lo=-1, hi=1)
        with self.assertRaises(ValueError):
            QuantizedFunction(np.tanh, step=0.1, lo=1.0, hi=-1.0)

    # -- MXR-080-0144 ------------------------------------------------------------------------------

    def test_construction_rejects_a_grid_that_degenerates_to_a_single_code(self):
        # lo < hi genuinely (0.4 > 0.0) so the basic ordering check passes, but at step=1.0 both round
        # to code 0: exactly the "valid-looking lo < hi that rounds to a single code" case that used to
        # reach the boundary-slope construction (self.table[1] on a length-1 table) instead of a clean
        # error.
        with self.assertRaises(ValueError):
            QuantizedFunction(np.tanh, step=1.0, lo=0.0, hi=0.4)
        # negative control: the same span at a fine-enough step still constructs and evaluates normally
        q = QuantizedFunction(np.tanh, step=0.1, lo=0.0, hi=0.4)
        self.assertGreaterEqual(len(q.table), 2)
        self.assertAlmostEqual(float(q(0.2)), float(np.tanh(0.2)), places=2)

    def test_construction_rejects_non_finite_step_lo_hi(self):
        for kwargs in (
            {"step": np.nan, "lo": -1.0, "hi": 1.0},
            {"step": np.inf, "lo": -1.0, "hi": 1.0},
            {"step": 0.1, "lo": np.nan, "hi": 1.0},
            {"step": 0.1, "lo": -1.0, "hi": np.inf},
        ):
            with self.assertRaises(ValueError):
                QuantizedFunction(np.tanh, **kwargs)

    def test_construction_rejects_non_finite_function_output(self):
        bad_func = lambda x: np.where(x == 0.0, np.nan, x)  # noqa: E731
        with self.assertRaises(ValueError):
            QuantizedFunction(bad_func, step=0.1, lo=-1.0, hi=1.0)

    def test_error_bound_rejects_non_positive_or_non_finite_derivative(self):
        for bad in (0.0, -1.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                error_bound(bad, 0.01)
        with self.assertRaises(ValueError):
            error_bound(0.25, 0.0)
        self.assertGreater(error_bound(0.25, 0.01), 0.0)  # negative control

    def test_step_for_tolerance_rejects_non_positive_or_non_finite_derivative(self):
        for bad in (0.0, -1.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                step_for_tolerance(1e-3, bad)
        self.assertGreater(step_for_tolerance(1e-3, 0.25), 0.0)  # negative control


if __name__ == "__main__":
    unittest.main()
