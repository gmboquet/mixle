import importlib
import unittest

import numpy as np
import pytest

from pysp.stats import (
    AffineTransform,
    BernoulliDistribution,
    BinomialDistribution,
    CategoricalDistribution,
    CompositeDistribution,
    ConditionalDistribution,
    DiagonalGaussianDistribution,
    ExponentialDistribution,
    GammaDistribution,
    GaussianDistribution,
    GeometricDistribution,
    IntegerCategoricalDistribution,
    LogGaussianDistribution,
    LogisticDistribution,
    MarkovChainDistribution,
    MixtureDistribution,
    NegativeBinomialDistribution,
    OptionalDistribution,
    ParetoDistribution,
    PoissonDistribution,
    RayleighDistribution,
    RecordDistribution,
    SelectDistribution,
    SequenceDistribution,
    StudentTDistribution,
    TransformDistribution,
    UniformDistribution,
    WeibullDistribution,
    field,
)
from pysp.utils.fit import fit_map, fit_mle
from pysp.utils.priors import (
    BetaPrior,
    ConditionalPrior,
    DirichletPrior,
    GammaPrior,
    MarkovChainPrior,
    MixturePrior,
    NormalGammaPrior,
    OptionalPrior,
    RecordPrior,
)

pytestmark = [pytest.mark.torch, pytest.mark.optional]

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _sign_choice(x):
    return 0 if float(x) < 0.0 else 1


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class GradientFitTestCase(unittest.TestCase):
    def assertGradientResultDiagnostics(self, result):
        self.assertEqual(len(result.history), result.iterations + 1)
        self.assertAlmostEqual(result.initial_value, result.history[0])
        self.assertAlmostEqual(result.value, result.history[-1])
        self.assertAlmostEqual(result.final_delta, result.history[-1] - result.history[-2])
        self.assertIsInstance(result.converged, bool)
        self.assertGreater(result.improvement, 0.0)
        self.assertIsNotNone(result.best_value)
        self.assertIsInstance(result.best_iteration, int)
        self.assertGreaterEqual(result.best_iteration, 0)
        self.assertLess(result.best_iteration, len(result.history))
        self.assertAlmostEqual(result.best_value, max(result.history))
        self.assertGreater(result.best_improvement, 0.0)
        self.assertIsNotNone(result.final_gradient_norm)
        self.assertTrue(np.isfinite(result.final_gradient_norm))
        self.assertGreaterEqual(result.final_gradient_norm, 0.0)
        self.assertIsNotNone(result.prior_sensitivity)
        self.assertTrue(np.isfinite(result.prior_sensitivity))
        self.assertGreaterEqual(result.prior_sensitivity, 0.0)
        self.assertLessEqual(result.prior_sensitivity, 1.0)

    def test_fit_mle_improves_gaussian_likelihood(self):
        truth = GaussianDistribution(2.0, 0.5)
        data = truth.sampler(seed=1).sample(size=90)
        start = GaussianDistribution(0.0, 3.0)
        enc = start.dist_to_encoder().seq_encode(data)
        ll0 = float(start.seq_log_density(enc).sum())

        result = fit_mle(enc, start, max_its=90, lr=0.05, print_iter=1000, return_result=True)
        fitted, ll = result.as_tuple()

        self.assertIsInstance(fitted, GaussianDistribution)
        self.assertGreater(ll, ll0)
        self.assertLess(abs(fitted.mu - np.mean(data)), 0.2)
        self.assertLess(abs(fitted.sigma2 - np.var(data)), 0.25)
        self.assertGradientResultDiagnostics(result)
        self.assertAlmostEqual(result.log_prior, 0.0)
        self.assertAlmostEqual(result.log_likelihood, result.value)

    def test_fit_mle_accepts_precision_policy(self):
        start = GaussianDistribution(0.0, 2.0)
        enc = start.dist_to_encoder().seq_encode(np.asarray([-1.0, 0.0, 1.0]))

        fitted, ll = fit_mle(enc, start, max_its=2, lr=0.01, print_iter=1000, precision="float32")

        self.assertIsInstance(fitted, GaussianDistribution)
        self.assertTrue(np.isfinite(ll))

    def test_uniform_ordered_bounds_use_declaration_constraint(self):
        from pysp.stats.compute.declarations import declaration_for

        start = UniformDistribution(-3.0, 4.0)
        declaration = declaration_for(start)
        high_spec = [spec for spec in declaration.parameters if spec.name == "high"][0]
        self.assertEqual(high_spec.constraint, "greater_than:low")

        enc = start.dist_to_encoder().seq_encode(np.asarray([-1.0, 0.0, 2.0]))
        fitted, objective = fit_mle(enc, start, max_its=2, lr=0.01, print_iter=1000)

        self.assertIsInstance(fitted, UniformDistribution)
        self.assertGreater(fitted.high, fitted.low)
        self.assertTrue(np.isfinite(objective))

    def test_fit_mle_improves_composite_likelihood(self):
        truth = CompositeDistribution((GaussianDistribution(-1.0, 0.8), ExponentialDistribution(2.0)))
        data = truth.sampler(seed=2).sample(size=90)
        start = CompositeDistribution((GaussianDistribution(0.5, 2.0), ExponentialDistribution(4.0)))
        enc = start.dist_to_encoder().seq_encode(data)
        ll0 = float(start.seq_log_density(enc).sum())

        fitted, ll = fit_mle(enc, start, max_its=100, lr=0.04, print_iter=1000)

        self.assertIsInstance(fitted, CompositeDistribution)
        self.assertGreater(ll, ll0)
        self.assertEqual(len(fitted.dists), 2)

    def test_fit_mle_improves_mixture_likelihood(self):
        truth = MixtureDistribution(
            [GaussianDistribution(-2.0, 0.5), GaussianDistribution(2.0, 0.7)],
            [0.4, 0.6],
        )
        data = truth.sampler(seed=3).sample(size=120)
        start = MixtureDistribution(
            [GaussianDistribution(-1.5, 1.2), GaussianDistribution(1.5, 1.2)],
            [0.5, 0.5],
        )
        enc = start.dist_to_encoder().seq_encode(data)
        ll0 = float(start.seq_log_density(enc).sum())

        fitted, ll = fit_mle(enc, start, max_its=100, lr=0.03, print_iter=1000)

        self.assertIsInstance(fitted, MixtureDistribution)
        self.assertGreater(ll, ll0)
        self.assertAlmostEqual(float(np.sum(fitted.w)), 1.0)

    def test_fit_mle_improves_sequence_element_and_length_likelihood(self):
        rng = np.random.RandomState(1)
        data = []
        for _ in range(100):
            n = int(rng.choice([0, 1, 2], p=[0.1, 0.2, 0.7]))
            data.append(list(rng.normal(2.0, np.sqrt(0.5), size=n)))

        start = SequenceDistribution(
            GaussianDistribution(0.0, 2.0),
            len_dist=IntegerCategoricalDistribution(0, [1.0 / 3.0] * 3),
        )
        enc = start.dist_to_encoder().seq_encode(data)
        ll0 = float(start.seq_log_density(enc).sum())

        fitted, ll = fit_mle(enc, start, max_its=150, lr=0.05, print_iter=1000)

        values = np.asarray([x for row in data for x in row], dtype=float)
        lengths = np.asarray([len(row) for row in data], dtype=int)
        empirical_lengths = np.bincount(lengths, minlength=3).astype(float)
        empirical_lengths /= empirical_lengths.sum()

        self.assertIsInstance(fitted, SequenceDistribution)
        self.assertGreater(ll, ll0)
        self.assertLess(abs(fitted.dist.mu - np.mean(values)), 0.2)
        self.assertLess(abs(fitted.dist.sigma2 - np.var(values)), 0.25)
        self.assertLess(np.linalg.norm(fitted.len_dist.p_vec - empirical_lengths), 0.1)

    def test_fit_mle_improves_record_with_reused_source_likelihood(self):
        rng = np.random.RandomState(5)
        values = rng.normal(1.5, np.sqrt(0.4), size=90)
        data = [{"x": float(x)} for x in values]
        start = RecordDistribution(
            {
                field("left_view", source="x"): GaussianDistribution(-1.0, 2.0),
                field("right_view", source="x"): GaussianDistribution(3.0, 2.0),
            }
        )
        enc = start.dist_to_encoder().seq_encode(data)
        ll0 = float(start.seq_log_density(enc).sum())

        fitted, ll = fit_mle(enc, start, max_its=120, lr=0.05, print_iter=1000)

        self.assertIsInstance(fitted, RecordDistribution)
        self.assertEqual(fitted.sources, ("x", "x"))
        self.assertGreater(ll, ll0)
        for child in fitted.dists:
            self.assertLess(abs(child.mu - np.mean(values)), 0.2)
            self.assertLess(abs(child.sigma2 - np.var(values)), 0.25)

    def test_fit_mle_improves_transform_child_likelihood(self):
        rng = np.random.RandomState(6)
        base = rng.normal(-0.75, np.sqrt(0.6), size=90)
        transform = AffineTransform(loc=2.0, scale=3.0)
        data = [transform.forward(float(x)) for x in base]
        start = TransformDistribution(
            GaussianDistribution(1.0, 2.0),
            transform=transform,
            density_correction=True,
        )
        enc = start.dist_to_encoder().seq_encode(data)
        ll0 = float(start.seq_log_density(enc).sum())

        fitted, ll = fit_mle(enc, start, max_its=120, lr=0.05, print_iter=1000)

        self.assertIsInstance(fitted, TransformDistribution)
        self.assertGreater(ll, ll0)
        self.assertLess(abs(fitted.dist.mu - np.mean(base)), 0.2)
        self.assertLess(abs(fitted.dist.sigma2 - np.var(base)), 0.25)

    def test_fit_mle_improves_select_child_likelihood(self):
        rng = np.random.RandomState(7)
        left = rng.normal(-2.0, np.sqrt(0.4), size=70)
        right = rng.normal(2.0, np.sqrt(0.5), size=80)
        data = np.concatenate([left, right])
        start = SelectDistribution(
            [GaussianDistribution(-0.5, 2.0), GaussianDistribution(0.5, 2.0)],
            _sign_choice,
        )
        enc = start.dist_to_encoder().seq_encode(data)
        ll0 = float(start.seq_log_density(enc).sum())

        fitted, ll = fit_mle(enc, start, max_its=120, lr=0.05, print_iter=1000)

        grouped = [np.asarray([x for x in data if _sign_choice(x) == choice], dtype=float) for choice in (0, 1)]
        self.assertIsInstance(fitted, SelectDistribution)
        self.assertGreater(ll, ll0)
        for child, values in zip(fitted.dists, grouped):
            self.assertLess(abs(child.mu - np.mean(values)), 0.2)
            self.assertLess(abs(child.sigma2 - np.var(values)), 0.25)

    def test_fit_mle_improves_conditional_children_and_given_likelihood(self):
        rng = np.random.RandomState(8)
        left = rng.normal(-2.0, np.sqrt(0.4), size=70)
        right = rng.normal(2.0, np.sqrt(0.5), size=90)
        data = [("a", float(x)) for x in left] + [("b", float(x)) for x in right]
        start = ConditionalDistribution(
            {"a": GaussianDistribution(-0.5, 2.0), "b": GaussianDistribution(0.5, 2.0)},
            given_dist=CategoricalDistribution({"a": 0.5, "b": 0.5}),
        )
        enc = start.dist_to_encoder().seq_encode(data)
        ll0 = float(start.seq_log_density(enc).sum())

        fitted, ll = fit_mle(enc, start, max_its=140, lr=0.05, print_iter=1000)

        self.assertIsInstance(fitted, ConditionalDistribution)
        self.assertGreater(ll, ll0)
        self.assertLess(abs(fitted.dmap["a"].mu - np.mean(left)), 0.2)
        self.assertLess(abs(fitted.dmap["a"].sigma2 - np.var(left)), 0.25)
        self.assertLess(abs(fitted.dmap["b"].mu - np.mean(right)), 0.2)
        self.assertLess(abs(fitted.dmap["b"].sigma2 - np.var(right)), 0.25)
        self.assertAlmostEqual(fitted.given_dist.pmap["a"], len(left) / float(len(data)), delta=0.08)

    def test_fit_mle_improves_markov_chain_initial_transition_and_length_likelihood(self):
        truth = MarkovChainDistribution(
            {"a": 0.7, "b": 0.3},
            {"a": {"a": 0.2, "b": 0.8}, "b": {"a": 0.6, "b": 0.4}},
            len_dist=IntegerCategoricalDistribution(1, [0.2, 0.3, 0.5]),
        )
        data = truth.sampler(seed=9).sample(size=250)
        start = MarkovChainDistribution(
            {"a": 0.5, "b": 0.5},
            {"a": {"a": 0.5, "b": 0.5}, "b": {"a": 0.5, "b": 0.5}},
            len_dist=IntegerCategoricalDistribution(1, [1.0 / 3.0] * 3),
        )
        enc = start.dist_to_encoder().seq_encode(data)
        ll0 = float(start.seq_log_density(enc).sum())

        fitted, ll = fit_mle(enc, start, max_its=220, lr=0.05, print_iter=1000)

        init_counts = {"a": 0.0, "b": 0.0}
        trans_counts = {"a": {"a": 0.0, "b": 0.0}, "b": {"a": 0.0, "b": 0.0}}
        len_counts = np.zeros(3)
        for row in data:
            len_counts[len(row) - 1] += 1.0
            if row:
                init_counts[row[0]] += 1.0
                for prev, cur in zip(row[:-1], row[1:]):
                    trans_counts[prev][cur] += 1.0
        init_total = sum(init_counts.values())
        empirical_init = {key: value / init_total for key, value in init_counts.items()}
        empirical_trans = {
            key: {sub_key: value / sum(row.values()) for sub_key, value in row.items()}
            for key, row in trans_counts.items()
        }
        empirical_len = len_counts / len_counts.sum()

        self.assertIsInstance(fitted, MarkovChainDistribution)
        self.assertGreater(ll, ll0)
        for key in empirical_init:
            self.assertAlmostEqual(fitted.init_prob_map[key], empirical_init[key], delta=0.04)
        for key, row in empirical_trans.items():
            for sub_key, empirical in row.items():
                self.assertAlmostEqual(fitted.transition_map[key][sub_key], empirical, delta=0.04)
        self.assertLess(np.linalg.norm(fitted.len_dist.p_vec - empirical_len), 0.08)

    def test_zero_strength_map_matches_mle(self):
        truth = GaussianDistribution(1.0, 1.5)
        data = truth.sampler(seed=4).sample(size=80)
        start = GaussianDistribution(-0.5, 3.0)
        enc = start.dist_to_encoder().seq_encode(data)

        mle, ll = fit_mle(enc, start, max_its=75, lr=0.05, print_iter=1000)
        mapped, lp = fit_map(enc, start, prior_strength=0.0, max_its=75, lr=0.05, print_iter=1000)

        self.assertAlmostEqual(ll, lp, places=10)
        self.assertAlmostEqual(mle.mu, mapped.mu, places=10)
        self.assertAlmostEqual(mle.sigma2, mapped.sigma2, places=10)

    def test_fit_map_normal_gamma_prior_helper_regularizes_gaussian(self):
        start = GaussianDistribution(0.0, 2.0)
        data = np.asarray([5.0, 5.2, 4.8, 5.1, 4.9])
        enc = start.dist_to_encoder().seq_encode(data)

        mle, _ = fit_mle(enc, start, max_its=300, lr=0.05, print_iter=1000)
        result = fit_map(
            enc,
            start,
            priors=NormalGammaPrior(mu0=0.0, kappa=80.0, alpha=3.0, beta=2.0),
            prior_strength=0.0,
            max_its=300,
            lr=0.05,
            print_iter=1000,
            return_result=True,
        )
        mapped, lp = result.as_tuple()

        self.assertTrue(np.isfinite(lp))
        self.assertLess(abs(mapped.mu), 0.25 * abs(mle.mu))
        self.assertGreater(mapped.sigma2, mle.sigma2)
        self.assertEqual(result.tag, "MAP")
        self.assertGradientResultDiagnostics(result)
        self.assertTrue(np.isfinite(result.log_likelihood))
        self.assertTrue(np.isfinite(result.log_prior))
        self.assertAlmostEqual(result.log_likelihood + result.log_prior, result.value, places=6)
        self.assertGreater(result.prior_sensitivity, 0.0)

    def test_fit_map_dirichlet_prior_helper_smooths_categorical_labels(self):
        dist = CategoricalDistribution({"a": 0.7, "b": 0.2, "c": 0.1})
        data = ["a"] * 45 + ["b"] * 15
        enc = dist.dist_to_encoder().seq_encode(data)

        mle, _ = fit_mle(enc, dist, max_its=600, lr=0.08, print_iter=1000)
        mapped, _ = fit_map(
            enc,
            dist,
            priors=DirichletPrior({"a": 1.0, "b": 1.0, "c": 8.0}),
            prior_strength=0.0,
            max_its=600,
            lr=0.08,
            print_iter=1000,
        )

        self.assertLess(mle.pmap["c"], 0.001)
        self.assertGreater(mapped.pmap["c"], 0.05)
        self.assertGreater(mapped.pmap["a"], mapped.pmap["b"])

    def test_fit_map_mixture_prior_helper_regularizes_weights(self):
        start = MixtureDistribution(
            [GaussianDistribution(-2.0, 0.5), GaussianDistribution(2.0, 0.5)],
            [0.5, 0.5],
        )
        data = [-2.1, -1.9, -2.0, -2.2, -1.8, -2.05]
        enc = start.dist_to_encoder().seq_encode(data)

        mle, _ = fit_mle(enc, start, max_its=300, lr=0.04, print_iter=1000)
        mapped, lp = fit_map(
            enc,
            start,
            priors=MixturePrior(weights=DirichletPrior([1.0, 20.0])),
            prior_strength=0.0,
            max_its=300,
            lr=0.04,
            print_iter=1000,
        )

        self.assertTrue(np.isfinite(lp))
        self.assertGreater(mle.w[0], 0.9)
        self.assertGreater(mapped.w[1], 0.9)

    def test_fit_map_record_prior_helper_routes_by_field(self):
        start = RecordDistribution(
            {
                "x": GaussianDistribution(0.0, 2.0),
                "y": GaussianDistribution(0.0, 2.0),
            }
        )
        data = [{"x": 5.0, "y": -3.0}, {"x": 5.2, "y": -2.8}, {"x": 4.8, "y": -3.1}]
        enc = start.dist_to_encoder().seq_encode(data)

        mle, _ = fit_mle(enc, start, max_its=250, lr=0.05, print_iter=1000)
        mapped, lp = fit_map(
            enc,
            start,
            priors=RecordPrior({"x": NormalGammaPrior(mu0=0.0, kappa=80.0, alpha=3.0, beta=2.0)}),
            prior_strength=0.0,
            max_its=250,
            lr=0.05,
            print_iter=1000,
        )

        self.assertTrue(np.isfinite(lp))
        self.assertLess(abs(mapped.dists[0].mu), 0.25 * abs(mle.dists[0].mu))
        self.assertAlmostEqual(mapped.dists[1].mu, mle.dists[1].mu, delta=0.25)

    def test_fit_map_conditional_prior_helper_routes_by_condition_key(self):
        start = ConditionalDistribution(
            {
                "a": GaussianDistribution(0.0, 2.0),
                "b": GaussianDistribution(0.0, 2.0),
            }
        )
        data = [("a", 5.0), ("a", 5.2), ("a", 4.8), ("b", -3.0), ("b", -2.8), ("b", -3.1)]
        enc = start.dist_to_encoder().seq_encode(data)

        mle, _ = fit_mle(enc, start, max_its=250, lr=0.05, print_iter=1000)
        mapped, lp = fit_map(
            enc,
            start,
            priors=ConditionalPrior({"a": NormalGammaPrior(mu0=0.0, kappa=80.0, alpha=3.0, beta=2.0)}),
            prior_strength=0.0,
            max_its=250,
            lr=0.05,
            print_iter=1000,
        )

        self.assertTrue(np.isfinite(lp))
        self.assertLess(abs(mapped.dmap["a"].mu), 0.25 * abs(mle.dmap["a"].mu))
        self.assertAlmostEqual(mapped.dmap["b"].mu, mle.dmap["b"].mu, delta=0.25)

    def test_fit_map_markov_chain_prior_helper_routes_by_transition_row(self):
        start = MarkovChainDistribution(
            {"a": 0.5, "b": 0.5},
            {"a": {"a": 0.5, "b": 0.5}, "b": {"a": 0.5, "b": 0.5}},
        )
        data = [["a", "a"], ["a", "a"], ["a", "a"], ["a", "a"], ["b", "b"], ["b", "b"]]
        enc = start.dist_to_encoder().seq_encode(data)

        mle, _ = fit_mle(enc, start, max_its=300, lr=0.05, print_iter=1000)
        mapped, lp = fit_map(
            enc,
            start,
            priors=MarkovChainPrior(transitions={"a": DirichletPrior({"a": 1.0, "b": 15.0})}),
            prior_strength=0.0,
            max_its=300,
            lr=0.05,
            print_iter=1000,
        )

        self.assertTrue(np.isfinite(lp))
        self.assertGreater(mle.transition_map["a"]["a"], 0.9)
        self.assertGreater(mapped.transition_map["a"]["b"], 0.7)
        self.assertAlmostEqual(mapped.transition_map["b"]["b"], mle.transition_map["b"]["b"], delta=0.02)

    def test_fit_map_beta_and_gamma_parameter_priors(self):
        opt = OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.2, missing_value=None)
        data = [None, None, 0.0, 0.1, -0.1, 0.2, -0.2, 0.0, 0.05, -0.05]
        enc = opt.dist_to_encoder().seq_encode(data)

        mle, _ = fit_mle(enc, opt, max_its=300, lr=0.05, print_iter=1000)
        mapped, _ = fit_map(
            enc,
            opt,
            priors=OptionalPrior(missing=BetaPrior(12.0, 1.0)),
            prior_strength=0.0,
            max_its=300,
            lr=0.05,
            print_iter=1000,
        )

        self.assertLess(mle.p, 0.3)
        self.assertGreater(mapped.p, 0.5)

        pois = PoissonDistribution(2.0)
        enc = pois.dist_to_encoder().seq_encode([8, 9, 7, 8, 10])
        mle, _ = fit_mle(enc, pois, max_its=300, lr=0.05, print_iter=1000)
        mapped, _ = fit_map(
            enc,
            pois,
            priors=GammaPrior(shape=30.0, rate=10.0, parameter="lam"),
            prior_strength=0.0,
            max_its=300,
            lr=0.05,
            print_iter=1000,
        )

        self.assertGreater(mle.lam, 7.0)
        self.assertLess(mapped.lam, 6.0)

    def test_fit_map_gamma_prior_regularizes_ordered_bound_delta(self):
        start = UniformDistribution(-5.0, 5.0)
        data = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])
        enc = start.dist_to_encoder().seq_encode(data)

        mle, _ = fit_mle(enc, start, max_its=60, lr=0.01, tol=0.0, print_iter=1000)
        mapped, lp = fit_map(
            enc,
            start,
            priors=GammaPrior(shape=80.0, rate=10.0, parameter="high_minus_low"),
            prior_strength=0.0,
            max_its=60,
            lr=0.01,
            tol=0.0,
            print_iter=1000,
        )

        self.assertTrue(np.isfinite(lp))
        self.assertGreater(mapped.high, float(np.max(data)))
        self.assertGreater(mapped.high - mapped.low, mle.high - mle.low + 1.0)

    def test_fit_mle_improves_categorical_likelihood(self):
        dist = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        data = ["a"] * 70 + ["b"] * 20 + ["c"] * 10
        enc = dist.dist_to_encoder().seq_encode(data)
        ll0 = float(dist.seq_log_density(enc).sum())

        fitted, ll = fit_mle(enc, dist, max_its=150, lr=0.05, print_iter=1000)

        self.assertIsInstance(fitted, CategoricalDistribution)
        self.assertGreater(ll, ll0)
        self.assertGreater(fitted.pmap["a"], fitted.pmap["b"])
        self.assertGreater(fitted.pmap["b"], fitted.pmap["c"])

    def test_one_step_smoke_for_converted_leaves(self):
        cases = [
            (GaussianDistribution(0.5, 1.7), np.asarray([-1.0, 0.0, 2.0])),
            (ExponentialDistribution(2.0), np.asarray([0.2, 1.0, 3.0])),
            (PoissonDistribution(3.0), [0, 2, 5]),
            (BernoulliDistribution(0.4), [False, True, True, False]),
            (GammaDistribution(2.0, 1.5), np.asarray([0.5, 1.0, 2.0])),
            (LogGaussianDistribution(0.1, 0.7), np.asarray([0.5, 1.0, 2.5])),
            (BinomialDistribution(0.4, 5), [0, 2, 4]),
            (NegativeBinomialDistribution(2.0, 0.4), [0, 1, 3]),
            (GeometricDistribution(0.4), [1, 2, 3]),
            (DiagonalGaussianDistribution([0.0, 1.0], [1.0, 2.0]), [[-1.0, 0.5], [0.0, 1.0], [2.0, -1.0]]),
            (StudentTDistribution(5.0, 0.25, 1.5), np.asarray([-1.0, 0.0, 2.0])),
            (LogisticDistribution(0.25, 1.5), np.asarray([-2.0, 0.0, 3.0])),
            (WeibullDistribution(1.5, 2.0), np.asarray([0.2, 1.0, 2.5])),
            (RayleighDistribution(1.2), np.asarray([0.2, 1.0, 2.5])),
            (ParetoDistribution(1.0, 2.5), np.asarray([1.1, 2.0, 4.0])),
            (UniformDistribution(-1.0, 3.0), np.asarray([-0.5, 0.0, 2.5])),
            (IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3]), [0, 1, 2, 1]),
            (CategoricalDistribution({"a": 0.4, "b": 0.35, "c": 0.25}), ["a", "b", "a", "c"]),
        ]

        for dist, data in cases:
            with self.subTest(dist=type(dist).__name__):
                enc = dist.dist_to_encoder().seq_encode(data)
                fitted, objective = fit_mle(enc, dist, max_its=2, lr=0.01, print_iter=1000)
                self.assertIsInstance(fitted, type(dist))
                self.assertTrue(np.isfinite(objective))

    def test_fit_mle_accepts_chunked_seq_encode_format(self):
        # pysp.stats.seq_encode returns the chunked [(size, payload), ...] form;
        # the gradient fitter must accept it, not only an encoder's bare payload
        from pysp.stats import seq_encode, seq_log_density_sum

        model = MixtureDistribution(
            [
                CompositeDistribution((GaussianDistribution(-1.0, 1.0), CategoricalDistribution({"a": 0.7, "b": 0.3}))),
                CompositeDistribution((GaussianDistribution(1.0, 1.0), CategoricalDistribution({"a": 0.3, "b": 0.7}))),
            ],
            [0.5, 0.5],
        )
        data = model.sampler(seed=1).sample(400)
        enc = seq_encode(data, model=model)  # chunked form
        self.assertIsInstance(enc, list)
        self.assertIsInstance(enc[0], tuple)
        fitted, objective = fit_mle(enc, model, max_its=40, lr=0.05, print_iter=1000)
        _, ll_start = seq_log_density_sum(enc, model)
        _, ll_end = seq_log_density_sum(enc, fitted)
        self.assertTrue(np.isfinite(objective))
        self.assertGreater(ll_end, ll_start)

    def test_backend_seq_log_density_is_autograd_safe(self):
        # the converted backend bodies must not use in-place ops that break the
        # torch autograd graph (spec hard rule); gradcheck on the leaf families
        import torch

        from pysp.engines import TorchEngine
        from pysp.stats.compute.backend import backend_seq_log_density

        engine = TorchEngine(dtype=torch.float64)
        cases = [
            (GaussianDistribution(0.3, 1.7), torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64)),
            (LogGaussianDistribution(0.1, 1.2), torch.tensor([0.2, 1.0, 2.5], dtype=torch.float64)),
        ]
        for dist, x in cases:
            with self.subTest(dist=type(dist).__name__):
                x = x.clone().requires_grad_(True)
                out = backend_seq_log_density(dist, x, engine)
                out.sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
