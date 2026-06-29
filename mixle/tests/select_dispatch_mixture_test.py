"""Tests for the weighted SelectDistribution -- the *dispatch mixture* over disjoint-typed data.

A weighted ``SelectDistribution`` models ``p(x) = w_{c(x)} * p_{c(x)}(x)``: the choice function is an
OBSERVED component label (e.g. an observation's type), so a mixture of differently-typed children -- a
mix of strings and numbers, say -- is normalized over the union support, fits in closed form (no EM),
and samples one-value-per-draw. ``weights=None`` keeps the legacy conditional behaviour. These tests
also pin the companion fail-loud guard on ``MixtureDistribution`` for disjoint-typed data.
"""

import math
import unittest

import numpy as np

import mixle.stats as stats
from mixle.inference import estimate
from mixle.stats.combinator.select import SelectDistribution, TypeDispatch
from mixle.stats.latent.mixture import MixtureDistribution
from mixle.utils.serialization import register_serializable_callable


def _route_by_type(x):
    """Strings -> child 0, everything else (numbers) -> child 1."""
    return 0 if isinstance(x, str) else 1


# Registered so a weighted SelectDistribution carrying this routing can round-trip through JSON.
register_serializable_callable(_route_by_type, "mixle.tests.select_dispatch_mixture_test.route_by_type")


def _string_number_mixture(weights):
    return SelectDistribution(
        [stats.CategoricalDistribution({"a": 0.6, "b": 0.4}), stats.PoissonDistribution(3.0)],
        _route_by_type,
        weights=weights,
    )


class SelectDispatchMixtureTestCase(unittest.TestCase):
    def test_weighted_log_density_includes_branch_weight(self):
        sel = _string_number_mixture([0.7, 0.3])
        pois = stats.PoissonDistribution(3.0)
        self.assertAlmostEqual(sel.log_density("a"), math.log(0.7) + math.log(0.6))
        self.assertAlmostEqual(sel.log_density(5), math.log(0.3) + pois.log_density(5))

    def test_vectorized_matches_scalar_on_mixed_batch(self):
        sel = _string_number_mixture([0.7, 0.3])
        data = ["a", 5, "b", 2, "a", 0]
        enc = sel.dist_to_encoder().seq_encode(data)
        seq = np.asarray(sel.seq_log_density(enc))
        scalar = np.asarray([sel.log_density(x) for x in data])
        self.assertTrue(np.allclose(seq, scalar))

    def test_normalized_over_union_support(self):
        # A proper density over strings U non-negative ints must sum to 1.
        sel = _string_number_mixture([0.7, 0.3])
        mass = sum(math.exp(sel.log_density(s)) for s in ("a", "b"))
        mass += sum(math.exp(sel.log_density(k)) for k in range(2000))
        self.assertAlmostEqual(mass, 1.0, places=8)

    def test_closed_form_fit_recovers_weights_and_children(self):
        rng = np.random.RandomState(0)
        strs = list(rng.choice(["a", "b", "c"], size=650, p=[0.5, 0.3, 0.2]))  # 65% strings
        nums = list(rng.poisson(4.0, size=350).astype(int))  # 35% numbers
        train = strs + nums
        rng.shuffle(train)

        est = SelectDistribution(
            [stats.CategoricalDistribution({"a": 0.34, "b": 0.33, "c": 0.33}), stats.PoissonDistribution(1.0)],
            _route_by_type,
        ).estimator(estimate_weights=True)
        fit = estimate(train, est)

        self.assertIsNotNone(fit.weights)
        self.assertTrue(np.allclose(fit.weights, [0.65, 0.35], atol=0.02))
        self.assertAlmostEqual(fit.dists[0].pmap["a"], 0.5, delta=0.05)
        self.assertAlmostEqual(fit.dists[0].pmap["c"], 0.2, delta=0.05)
        self.assertAlmostEqual(fit.dists[1].lam, 4.0, delta=0.2)

    def test_sampling_draws_one_value_with_branch_proportions(self):
        sel = _string_number_mixture([0.65, 0.35])
        draws = sel.sampler(0).sample(5000)
        # each draw is a single value of one type, never a (string, number) tuple
        self.assertTrue(all(isinstance(v, (str, np.str_)) or np.isscalar(v) for v in draws))
        frac_str = float(np.mean([isinstance(v, (str, np.str_)) for v in draws]))
        self.assertAlmostEqual(frac_str, 0.65, delta=0.03)

    def test_constructor_validates_weights(self):
        children = [stats.CategoricalDistribution({"a": 1.0}), stats.PoissonDistribution(3.0)]
        with self.assertRaises(ValueError):
            SelectDistribution(children, _route_by_type, weights=[0.5, 0.3, 0.2])  # wrong length
        with self.assertRaises(ValueError):
            SelectDistribution(children, _route_by_type, weights=[1.0, -0.1])  # negative
        with self.assertRaises(ValueError):
            SelectDistribution(children, _route_by_type, weights=[0.0, 0.0])  # zero total

    def test_weightless_select_unchanged(self):
        # Backward compatibility: no weights -> conditional density p(x) = p_{c(x)}(x), no offset.
        sel = SelectDistribution(
            [stats.CategoricalDistribution({"a": 0.6, "b": 0.4}), stats.PoissonDistribution(3.0)],
            _route_by_type,
        )
        self.assertIsNone(sel.weights)
        self.assertAlmostEqual(sel.log_density("a"), math.log(0.6))
        # a weightless distribution's estimator stays weightless by default
        rng = np.random.RandomState(1)
        train = list(rng.choice(["a", "b"], size=50)) + list(rng.poisson(3.0, size=50).astype(int))
        fit = estimate(train, sel.estimator())
        self.assertIsNone(fit.weights)

    def test_weighted_select_json_round_trip(self):
        sel = _string_number_mixture([0.7, 0.3])
        loaded = SelectDistribution.from_json(sel.to_json())
        self.assertTrue(np.allclose(loaded.weights, [0.7, 0.3]))
        self.assertIs(loaded.choice_function, _route_by_type)
        self.assertAlmostEqual(loaded.log_density("a"), sel.log_density("a"))
        self.assertAlmostEqual(loaded.log_density(5), sel.log_density(5))


class ByTypeAutoRoutingTestCase(unittest.TestCase):
    """SelectDistribution.by_type derives the routing from each child's type -- no hand-written
    (and registered) choice function needed."""

    def test_by_type_routes_and_scores_without_a_hand_written_router(self):
        sel = SelectDistribution.by_type(
            [(str, stats.CategoricalDistribution({"a": 0.6, "b": 0.4})), (int, stats.PoissonDistribution(3.0))]
        )
        pois = stats.PoissonDistribution(3.0)
        # default 'auto' weights are uniform, so each branch carries log(0.5)
        self.assertAlmostEqual(sel.log_density("a"), math.log(0.5) + math.log(0.6))
        self.assertAlmostEqual(sel.log_density(5), math.log(0.5) + pois.log_density(5))

    def test_numpy_scalars_route_correctly(self):
        # isinstance(np.int64(5), int) is False; the router must still send it to the int child.
        sel = SelectDistribution.by_type(
            [(str, stats.CategoricalDistribution({"a": 1.0})), (int, stats.PoissonDistribution(3.0))]
        )
        self.assertEqual(sel.choice_function(np.int64(5)), 1)
        self.assertEqual(sel.choice_function(np.str_("a")), 0)

    def test_friendly_number_name_catches_int_and_float(self):
        sel = SelectDistribution.by_type(
            [("str", stats.CategoricalDistribution({"x": 1.0})), ("number", stats.GaussianDistribution(0.0, 1.0))]
        )
        self.assertEqual(sel.choice_function(3), 1)
        self.assertEqual(sel.choice_function(3.5), 1)
        self.assertEqual(sel.choice_function(np.float64(2.0)), 1)
        self.assertEqual(sel.choice_function("x"), 0)

    def test_by_type_fit_is_zero_ceremony(self):
        rng = np.random.RandomState(0)
        train = list(rng.choice(["a", "b", "c"], size=700, p=[0.5, 0.3, 0.2]))
        train += list(rng.poisson(4.0, size=300).astype(int))
        rng.shuffle(train)
        sel = SelectDistribution.by_type(
            [
                (str, stats.CategoricalDistribution({"a": 0.34, "b": 0.33, "c": 0.33})),
                (int, stats.PoissonDistribution(1.0)),
            ]
        )
        fit = estimate(train, sel.estimator())  # estimate_weights defaults on because weights are set
        self.assertTrue(np.allclose(fit.weights, [0.7, 0.3], atol=0.03))
        self.assertAlmostEqual(fit.dists[1].lam, 4.0, delta=0.2)

    def test_by_type_serializes_without_manual_registration(self):
        sel = SelectDistribution.by_type(
            [(str, stats.CategoricalDistribution({"a": 0.6, "b": 0.4})), (int, stats.PoissonDistribution(3.0))],
            weights=[0.7, 0.3],
        )
        loaded = SelectDistribution.from_json(sel.to_json())
        self.assertEqual(loaded.choice_function, sel.choice_function)
        self.assertTrue(np.allclose(loaded.weights, [0.7, 0.3]))
        self.assertAlmostEqual(loaded.log_density("a"), sel.log_density("a"))
        self.assertAlmostEqual(loaded.log_density(7), sel.log_density(7))

    def test_typedispatch_unknown_type_and_no_match_raise(self):
        with self.assertRaises(ValueError):
            TypeDispatch([dict])  # unsupported type object
        with self.assertRaises(ValueError):
            TypeDispatch(["frobnicate"])  # unknown alias name
        router = TypeDispatch([str, int])
        with self.assertRaises(ValueError):
            router(3.5)  # a float matches neither str nor (integral) int

    def test_weightless_by_type(self):
        sel = SelectDistribution.by_type(
            [(str, stats.CategoricalDistribution({"a": 1.0})), (int, stats.PoissonDistribution(3.0))],
            weights=None,
        )
        self.assertIsNone(sel.weights)
        self.assertAlmostEqual(sel.log_density("a"), 0.0)  # log p(a)=log 1, no weight offset


class MixtureDisjointTypeGuardTestCase(unittest.TestCase):
    def test_mixture_fails_loud_on_disjoint_typed_data(self):
        # A finite mixture treats the component as latent, so it cannot encode mixed string/number
        # data. The error must be actionable and point at SelectDistribution.
        m = MixtureDistribution(
            [stats.CategoricalDistribution({"a": 0.6, "b": 0.4}), stats.PoissonDistribution(3.0)],
            [0.5, 0.5],
        )
        with self.assertRaises(TypeError) as ctx:
            m.dist_to_encoder().seq_encode(["a", 5, "b"])
        self.assertIn("SelectDistribution", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
