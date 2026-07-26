"""Public contracts for automatic estimator factories."""

import unittest

from mixle.utils.automatic.factories import (
    get_categorical_estimator,
    get_dpm_mixture,
    get_gamma_estimator,
    get_gaussian_estimator,
    get_gaussian_mixture_estimator,
    get_integer_categorical_estimator,
    get_lognormal_estimator,
    get_poisson_estimator,
    get_student_t_estimator,
)


class FactoryMassValidationTest(unittest.TestCase):
    def test_categorical_requires_positive_observed_mass(self):
        for values in ({}, {"a": 0.0}):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "positive total mass"):
                get_categorical_estimator(values)

    def test_invalid_masses_are_rejected_consistently(self):
        factories = (
            get_categorical_estimator,
            get_integer_categorical_estimator,
            get_poisson_estimator,
            get_gaussian_estimator,
            get_lognormal_estimator,
            get_gamma_estimator,
            get_student_t_estimator,
            get_gaussian_mixture_estimator,
        )
        for mass in (-1.0, float("nan"), float("inf")):
            for factory in factories:
                key = 1 if factory in (get_integer_categorical_estimator, get_poisson_estimator) else 1.0
                with self.subTest(factory=factory.__name__, mass=mass), self.assertRaisesRegex(
                    ValueError, "masses"
                ):
                    factory({key: mass})

    def test_invalid_pseudo_counts_are_rejected_consistently(self):
        for pseudo_count in (-1.0, float("nan"), float("inf")):
            with self.subTest(pseudo_count=pseudo_count), self.assertRaisesRegex(ValueError, "pseudo_count"):
                get_gaussian_estimator({1.0: 1.0}, pseudo_count=pseudo_count)


class BayesianFactoryContractTest(unittest.TestCase):
    def test_lognormal_bayesian_request_attaches_its_conjugate_prior(self):
        estimator = get_lognormal_estimator({1.0: 1.0, 2.0: 1.0}, use_bstats=True)
        self.assertTrue(estimator.has_conj_prior)
        self.assertIsNotNone(estimator.prior)

    def test_gaussian_mixture_bayesian_request_covers_weights_and_components(self):
        estimator = get_gaussian_mixture_estimator({0.0: 1.0, 1.0: 1.0}, use_bstats=True)
        self.assertTrue(estimator.has_conj_prior)
        self.assertTrue(all(component.has_conj_prior for component in estimator.estimators))

    def test_unsupported_bayesian_families_fail_explicitly(self):
        for factory in (get_gamma_estimator, get_student_t_estimator):
            with self.subTest(factory=factory.__name__), self.assertRaisesRegex(NotImplementedError, "conjugate"):
                factory({1.0: 1.0, 2.0: 1.0}, use_bstats=True)


class DpmControlValidationTest(unittest.TestCase):
    def test_invalid_controls_fail_before_fitting(self):
        controls = (
            {"max_components": 0},
            {"max_its": 0},
            {"print_iter": 0},
            {"delta": -1.0},
            {"delta": float("nan")},
            {"pseudo_count": -1.0},
        )
        for kwargs in controls:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                get_dpm_mixture([0.0, 1.0], out=None, **kwargs)

    def test_empty_data_is_rejected_before_component_construction(self):
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            get_dpm_mixture([], out=None)


if __name__ == "__main__":
    unittest.main()
