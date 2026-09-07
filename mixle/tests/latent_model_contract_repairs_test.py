"""Latent models: posteriors that exist, models that persist, components that can be mixed.

Pass 02 of the ten adversarial reviews of the 0.8.1 candidate (P02-F01..F09) found a documented
API that returned ``None`` on the default install, a fitted distribution with no persistence path
at all, an encoder contract that made an ordinary two-population mixture unfittable, a disclosure
that stopped at the container boundary, and two entry points that answered malformed input with
internal errors. These tests pin each repair.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

import mixle.stats as S
from mixle.inference import learn_bayesian_network, optimize


def _quiet_fit(*args, **kwargs):
    kwargs.setdefault("print_iter", 1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return optimize(*args, **kwargs)


def _two_state_hmm_corpus(count: int = 40):
    true = S.HiddenMarkovModelDistribution(
        topics=[S.GaussianDistribution(-3.0, 1.0), S.GaussianDistribution(3.0, 1.0)],
        w=[0.5, 0.5],
        transitions=[[0.9, 0.1], [0.2, 0.8]],
        len_dist=S.CategoricalDistribution({20: 1.0}),
    )
    return true.sampler(seed=1).sample(count)


class SeqPosteriorTest(unittest.TestCase):
    """P02-F01: seq_posterior returned None without numba, and always with terminal_states."""

    def setUp(self):
        self.data = _two_state_hmm_corpus()

    def _fit(self, **kwargs):
        return _quiet_fit(
            self.data,
            S.HiddenMarkovModelEstimator([S.GaussianEstimator()] * 2, **kwargs),
            max_its=30,
            rng=np.random.RandomState(1),
        )

    def test_the_plain_route_returns_the_same_marginals_as_the_kernel(self):
        with_kernel = self._fit(use_numba=True)
        without = self._fit(use_numba=False)
        smoothed_kernel = with_kernel.seq_posterior(with_kernel.dist_to_encoder().seq_encode(self.data))
        smoothed_plain = without.seq_posterior(without.dist_to_encoder().seq_encode(self.data))
        self.assertIsNotNone(smoothed_plain)
        self.assertEqual(len(smoothed_plain), len(self.data))
        for kernel_block, plain_block in zip(smoothed_kernel, smoothed_plain):
            np.testing.assert_allclose(kernel_block, plain_block, atol=1.0e-12)

    def test_the_filtered_route_agrees_too_and_both_are_probability_rows(self):
        with_kernel = self._fit(use_numba=True)
        without = self._fit(use_numba=False)
        filtered_kernel = with_kernel.seq_posterior(with_kernel.dist_to_encoder().seq_encode(self.data), filtered=True)
        filtered_plain = without.seq_posterior(without.dist_to_encoder().seq_encode(self.data), filtered=True)
        for kernel_block, plain_block in zip(filtered_kernel, filtered_plain):
            np.testing.assert_allclose(kernel_block, plain_block, atol=1.0e-12)
            np.testing.assert_allclose(plain_block.sum(axis=1), 1.0)

    def test_the_marginals_match_the_single_sequence_chain_posterior(self):
        model = self._fit(use_numba=False)
        blocks = model.seq_posterior(model.dist_to_encoder().seq_encode(self.data))
        np.testing.assert_allclose(blocks[0], model.latent_posterior(self.data[0]).marginals(), atol=1.0e-14)

    def test_terminal_states_put_the_last_position_on_a_terminal_state(self):
        model = _quiet_fit(
            self.data,
            S.HiddenMarkovModelEstimator([S.GaussianEstimator()] * 2, terminal_states=[1], use_numba=True),
            max_its=3,
            rng=np.random.RandomState(1),
        )
        blocks = model.seq_posterior(model.dist_to_encoder().seq_encode(self.data))
        self.assertIsNotNone(blocks)
        self.assertEqual(len(blocks), len(self.data))
        for block in blocks:
            np.testing.assert_allclose(block.sum(axis=1), 1.0)
            self.assertAlmostEqual(float(block[-1, 1]), 1.0)


class ChowLiuSerializationTest(unittest.TestCase):
    """P02-F02: a fitted Chow-Liu tree had no persistence path -- to_json and to_dict both raised."""

    def _fit(self, rows, estimator):
        return _quiet_fit(rows, estimator, max_its=2, rng=np.random.RandomState(1))

    def test_string_leaves_round_trip_to_identical_densities(self):
        rows = [tuple(str(v) for v in row) for row in [(1, 1), (0, 1), (1, 0), (0, 0), (1, 1)] * 20]
        model = self._fit(rows, S.ChowLiuTreeEstimator([S.CategoricalEstimator()] * 2))
        restored = S.ChowLiuTreeDistribution.from_json(model.to_json())
        for row in rows:
            self.assertAlmostEqual(model.log_density(row), restored.log_density(row), places=12)

    def test_integer_leaves_round_trip_too(self):
        rows = [(1, 1), (0, 1), (1, 0), (0, 0)] * 20
        model = self._fit(rows, S.ChowLiuTreeEstimator([S.IntegerCategoricalEstimator()] * 2))
        restored = S.ChowLiuTreeDistribution.from_json(model.to_json())
        for row in rows:
            self.assertAlmostEqual(model.log_density(row), restored.log_density(row), places=12)

    def test_the_frozen_key_type_tag_decodes_only_from_the_closed_table(self):
        from mixle.utils.serialization import SerializationError, from_serializable, to_serializable

        self.assertIs(from_serializable(to_serializable(str)), str)
        self.assertIs(from_serializable(to_serializable(int)), int)
        with self.assertRaises(SerializationError):
            from_serializable({"__mixle__": "value-type", "name": "os.system"})


class SupportLimitedComponentTest(unittest.TestCase):
    """P02-F03: an out-of-support value made a whole heterogeneous batch un-encodable."""

    OUT_OF_SUPPORT = (
        (S.ExponentialDistribution(1.0), -1.0),
        (S.GammaDistribution(2.0, 1.0), -1.0),
        (S.WeibullDistribution(2.0, 1.0), -1.0),
        (S.LogGaussianDistribution(0.0, 1.0), -1.0),
        (S.RayleighDistribution(1.0), -1.0),
        (S.HalfNormalDistribution(1.0), -1.0),
        (S.InverseGammaDistribution(2.0, 1.0), -1.0),
        (S.InverseGaussianDistribution(1.0, 1.0), -1.0),
        (S.BetaDistribution(2.0, 2.0), 1.5),
        (S.PoissonDistribution(2.0), -1),
        (S.GeometricDistribution(0.5), 0),
        (S.LogSeriesDistribution(0.5), 0),
    )

    def test_the_vectorized_scorer_returns_the_scalar_path_s_minus_infinity(self):
        for law, outside in self.OUT_OF_SUPPORT:
            with self.subTest(law=type(law).__name__):
                inside = law.sampler(seed=0).sample()
                encoded = law.dist_to_encoder().seq_encode([outside, inside])
                scores = np.asarray(law.seq_log_density(encoded), dtype=np.float64).ravel()
                self.assertEqual(scores[0], -np.inf)
                self.assertEqual(law.log_density(outside), -np.inf)
                self.assertAlmostEqual(float(scores[1]), law.log_density(inside), places=10)

    def test_fitting_on_an_out_of_support_row_is_still_refused_by_name(self):
        for law, outside in self.OUT_OF_SUPPORT:
            with self.subTest(law=type(law).__name__):
                inside = law.sampler(seed=0).sample()
                encoded = law.dist_to_encoder().seq_encode([outside, inside])
                accumulator = law.estimator().accumulator_factory().make()
                with self.assertRaises(ValueError):
                    accumulator.seq_update(encoded, np.ones(2), law)
                # ... and a zero-weight row is no evidence, so an E-step's zeroed responsibility
                # for a row outside this component's support keeps working.
                accumulator.seq_update(encoded, np.array([0.0, 1.0]), law)

    def test_a_gaussian_exponential_mixture_fits_the_two_populations(self):
        rng = np.random.RandomState(0)
        data = np.concatenate([rng.normal(-2.0, 0.5, 200), rng.exponential(3.0, 200)])
        model = _quiet_fit(
            data,
            S.MixtureEstimator([S.GaussianEstimator(), S.ExponentialEstimator()]),
            max_its=30,
            rng=np.random.RandomState(1),
        )
        gaussian, exponential = model.components
        self.assertAlmostEqual(gaussian.mu, -2.0, delta=0.15)
        self.assertAlmostEqual(exponential.beta, 3.0, delta=0.5)
        np.testing.assert_allclose(model.w, [0.5, 0.5], atol=0.05)

    def test_the_heterogeneous_mixture_and_the_hmm_fit_the_same_data(self):
        rng = np.random.RandomState(0)
        data = np.concatenate([rng.normal(-2.0, 0.5, 200), rng.exponential(3.0, 200)])
        heterogeneous = _quiet_fit(
            data,
            S.HeterogeneousMixtureEstimator([S.GaussianEstimator(), S.ExponentialEstimator()]),
            max_its=20,
            rng=np.random.RandomState(1),
        )
        self.assertAlmostEqual(heterogeneous.components[0].mu, -2.0, delta=0.15)
        model = _quiet_fit(
            [[-1.0, 1.0, 2.0, -2.0]] * 20,
            S.HiddenMarkovModelEstimator([S.GaussianEstimator(), S.ExponentialEstimator()]),
            max_its=5,
            rng=np.random.RandomState(1),
        )
        self.assertEqual(len(model.topics), 2)

    def test_the_mixture_s_scalar_and_batch_paths_agree_on_an_out_of_support_row(self):
        mixture = S.MixtureDistribution(
            [S.GaussianDistribution(-2.0, 0.25), S.ExponentialDistribution(3.0)], [0.5, 0.5]
        )
        rows = [-1.0, 1.0]
        np.testing.assert_allclose(
            mixture.seq_log_density(mixture.dist_to_encoder().seq_encode(rows)),
            [mixture.log_density(row) for row in rows],
        )


class NestedRepairDisclosureTest(unittest.TestCase):
    """P02-F05: a floor applied inside a mixture or an HMM vanished from the receipt."""

    def test_a_mixture_and_an_hmm_report_their_components_floors(self):
        plain = _quiet_fit(np.full(100, 3.0), S.GaussianEstimator(), max_its=3)
        self.assertTrue(plain.fit_provenance().repairs)
        mixture = _quiet_fit(
            np.full(100, 3.0), S.MixtureEstimator([S.GaussianEstimator()] * 2), max_its=10, rng=np.random.RandomState(1)
        )
        self.assertEqual(len(mixture.fit_provenance().repairs), 2)
        for note in mixture.fit_provenance().repairs:
            self.assertTrue(note.startswith("components["), note)
            self.assertIn("variance-floored", note)
        self.assertEqual(tuple(mixture.numerical_repairs()), tuple(mixture.fit_provenance().repairs))
        hmm = _quiet_fit(
            [[1.0, 1.0, 1.0]] * 30,
            S.HiddenMarkovModelEstimator([S.GaussianEstimator()] * 2),
            max_its=10,
            rng=np.random.RandomState(1),
        )
        self.assertTrue(all(note.startswith("topics[") for note in hmm.fit_provenance().repairs))

    def test_a_healthy_fit_reports_nothing(self):
        data = np.random.RandomState(0).normal(size=200)
        model = _quiet_fit(
            data, S.MixtureEstimator([S.GaussianEstimator()] * 2), max_its=5, rng=np.random.RandomState(1)
        )
        self.assertEqual(model.fit_provenance().repairs, ())


class BayesianNetworkInputTest(unittest.TestCase):
    """P02-F07: empty, ragged, and wrong-width records reached internal IndexErrors."""

    @staticmethod
    def _fitted_network():
        rng = np.random.RandomState(2)
        n = 600
        a = rng.randint(0, 3, n)
        b = a * 2.0 + rng.normal(0.0, 0.5, n)
        c = np.where(rng.rand(n) < 0.9, a, rng.randint(0, 3, n))
        d = b + c + rng.normal(0.0, 0.3, n)
        rows = [(int(a[i]), float(b[i]), int(c[i]), float(d[i])) for i in range(n)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return learn_bayesian_network(rows)

    def test_empty_and_ragged_corpora_are_named(self):
        with self.assertRaises(ValueError) as caught:
            learn_bayesian_network([])
        self.assertIn("no records", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            learn_bayesian_network([(1, 2.0), (1, 2.0, 3.0)] * 10)
        self.assertIn("fixed width", str(caught.exception))

    def test_a_wrong_width_record_is_refused_and_a_nan_scores_impossible(self):
        network = self._fitted_network()
        self.assertTrue(np.isfinite(network.log_density((0, 0.0, 0, 0.0))))
        self.assertEqual(network.log_density((0, np.nan, 0, 0.0)), -np.inf)
        for bad in ((0, 0.0, 0, 0.0, 1.0), (0, 0.0, 0)):
            with self.assertRaises(ValueError) as caught:
                network.log_density(bad)
            self.assertIn("field(s)", str(caught.exception))


class HmmComponentsAliasTest(unittest.TestCase):
    """P02-F08: the documented ``components=`` alias did not round-trip to an attribute."""

    def test_components_is_the_emission_list(self):
        model = S.HiddenMarkovModelDistribution(
            components=[S.GaussianDistribution(0.0, 1.0)], w=[1.0], transitions=[[1.0]]
        )
        self.assertIs(model.components, model.topics)


class LdaNonConvergenceTest(unittest.TestCase):
    """P02-F09: an alpha solve that ran out of budget raised instead of returning a warned fit."""

    @staticmethod
    def _corpora():
        from mixle.utils.optsutil import count_by_value

        rng = np.random.RandomState(0)
        vocabulary = 20
        many = [list(count_by_value(list(rng.randint(0, vocabulary, 5000))).items()) for _ in range(50)]
        one = [list(count_by_value(list(rng.randint(0, vocabulary, 60))).items())]
        return vocabulary, {"wide": many, "single": one, "identical": one * 50}

    def test_each_corpus_returns_a_model_carrying_its_own_diagnostics(self):
        vocabulary, corpora = self._corpora()
        estimator = S.LDAEstimator([S.IntegerCategoricalEstimator(min_val=0, max_val=vocabulary - 1)] * 3)
        for label, docs in corpora.items():
            with self.subTest(corpus=label):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = optimize(docs, estimator, max_its=5, rng=np.random.RandomState(1), print_iter=1000)
                self.assertIsInstance(model, S.LDADistribution)
                self.assertFalse(model.fit_diagnostics.converged)
                self.assertTrue(
                    any("alpha solve stopped" in str(warning.message) for warning in caught),
                    [str(warning.message) for warning in caught],
                )
                self.assertTrue(np.all(np.isfinite(model.alpha)))
                self.assertTrue(np.all(np.asarray(model.alpha) > 0.0))


class LookbackLagZeroTest(unittest.TestCase):
    """P02-F04: at lag 0 the sampler emitted what the same distribution could not score."""

    @staticmethod
    def _window_law(mean):
        return S.SequenceDistribution(S.GaussianDistribution(mean, 1.0), len_dist=S.CategoricalDistribution({1: 1.0}))

    def test_a_window_emission_law_samples_what_it_scores(self):
        model = S.LookbackHiddenMarkovModelDistribution(
            [self._window_law(-1.0), self._window_law(1.0)],
            w=np.array([0.5, 0.5]),
            transitions=np.array([[0.9, 0.1], [0.1, 0.9]]),
            lag=0,
            len_dist=S.CategoricalDistribution({5: 1.0}),
        )
        drawn = model.sampler(seed=1).sample(3)
        for sequence in drawn:
            self.assertEqual(len(sequence), 5)
            self.assertTrue(all(isinstance(value, float) for value in sequence))
            self.assertTrue(np.isfinite(model.log_density(sequence)))

    def test_scoring_a_scalar_emission_law_names_the_wrapper_instead_of_crashing(self):
        # A scalar family at lag 0 is a mis-specified model: the topics score length-1 windows.
        # It used to fail with Python's own "float() argument must be ... not 'list'", naming
        # neither this model nor the wrapper that fixes it (P02-F04).
        model = S.LookbackHiddenMarkovModelDistribution(
            [S.GaussianDistribution(-1.0, 1.0), S.GaussianDistribution(1.0, 1.0)],
            w=np.array([0.5, 0.5]),
            transitions=np.array([[0.9, 0.1], [0.1, 0.9]]),
            lag=0,
            len_dist=S.CategoricalDistribution({5: 1.0}),
        )
        drawn = model.sampler(seed=1).sample(3)
        with self.assertRaises(ValueError) as caught:
            model.log_density(drawn[0])
        self.assertIn("SequenceEstimator", str(caught.exception))

    def test_fitting_scalar_emissions_at_lag_zero_names_the_wrapper(self):
        rng = np.random.RandomState(0)
        sequences = [[float(value) for value in rng.normal(0.0, 1.0, 15)] for _ in range(40)]
        for estimator in (S.GaussianEstimator(), S.CategoricalEstimator()):
            with self.subTest(estimator=type(estimator).__name__):
                with self.assertRaises(ValueError) as caught:
                    _quiet_fit(
                        sequences,
                        S.LookbackHiddenMarkovModelEstimator([estimator] * 2, lag=0),
                        max_its=4,
                        rng=np.random.RandomState(1),
                    )
                self.assertIn("SequenceEstimator", str(caught.exception))

    def test_a_sequence_estimator_at_lag_zero_fits(self):
        rng = np.random.RandomState(0)
        sequences = [[float(value) for value in rng.normal(0.0, 1.0, 15)] for _ in range(40)]
        model = _quiet_fit(
            sequences,
            S.LookbackHiddenMarkovModelEstimator([S.SequenceEstimator(S.GaussianEstimator())] * 2, lag=0),
            max_its=4,
            rng=np.random.RandomState(1),
        )
        self.assertEqual(len(model.topics), 2)


if __name__ == "__main__":
    unittest.main()
