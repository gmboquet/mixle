"""Regression test for HiddenMarkovAccumulatorFactory keys handling.

The factory previously replaced any caller-supplied keys tuple with (None, None, None)
due to an inverted None check, silently disabling key-based suff-stat merging for the
HMM initial/transition/emission statistics.
"""

import unittest

from mixle.stats.latent.hidden_markov import HiddenMarkovEstimator
from mixle.stats.univariate.discrete.categorical import CategoricalEstimator


class HiddenMarkovKeysTestCase(unittest.TestCase):
    def test_factory_preserves_keys(self):
        keys = ("init_k", "trans_k", "emis_k")
        est = HiddenMarkovEstimator([CategoricalEstimator(), CategoricalEstimator()], keys=keys)
        factory = est.accumulator_factory()
        self.assertEqual(factory.keys, keys)
        acc = factory.make()
        self.assertEqual(acc.init_key, "init_k")
        self.assertEqual(acc.trans_key, "trans_k")
        self.assertEqual(acc.state_key, "emis_k")

    def test_factory_defaults_keys_when_none(self):
        est = HiddenMarkovEstimator([CategoricalEstimator(), CategoricalEstimator()], keys=None)
        factory = est.accumulator_factory()
        self.assertEqual(factory.keys, (None, None, None))
        acc = factory.make()
        self.assertIsNone(acc.init_key)
        self.assertIsNone(acc.trans_key)
        self.assertIsNone(acc.state_key)


class TiedChainDynamicsTestCase(unittest.TestCase):
    """Mixture components may share chain dynamics while keeping their own emissions.

    A keyed part is pooled across every site sharing the key, so it carries the mass of N sites
    while the unkeyed parts -- and the site's own observation count -- carry the mass of one. Two
    conservation checks assumed all parts came from the same observations ("state mass == initial
    plus transition mass", and initial mass against the effective sample), so keys=('i','t',None)
    raised instead of fitting: the feature's headline use could not run. The checks now apply to
    unkeyed accumulators, where the assumption holds.
    """

    @staticmethod
    def _fit(keys):
        import io

        import numpy as np

        from mixle.inference.estimation import optimize
        from mixle.stats import GaussianEstimator, MixtureEstimator

        rng = np.random.RandomState(0)
        data = [list(rng.randn(6)) for _ in range(60)]

        def chain():
            return HiddenMarkovEstimator([GaussianEstimator() for _ in range(2)], keys=keys)

        return optimize(
            data,
            MixtureEstimator([chain() for _ in range(2)]),
            max_its=3,
            rng=np.random.RandomState(1),
            out=io.StringIO(),
        )

    def test_sharing_dynamics_fits_and_actually_ties_the_parameters(self):
        import numpy as np

        for keys in (("ik", "tk", None), ("ik", "tk", "sk"), (None, "tk", None)):
            with self.subTest(keys=keys):
                model = self._fit(keys)
                first, second = model.components[0], model.components[1]
                # the point of the key: both components must come out with the same matrix
                np.testing.assert_allclose(first.transitions, second.transitions)

    def test_unkeyed_fit_still_works_and_still_validates_mass(self):
        import numpy as np

        self._fit((None, None, None))  # the ordinary path is unaffected

        from mixle.stats import GaussianEstimator

        accumulator = HiddenMarkovEstimator([GaussianEstimator() for _ in range(2)]).accumulator_factory().make()
        corrupt = (
            2,
            np.array([1.0, 1.0]),
            np.array([99.0, 99.0]),  # state mass that cannot come from this initial/transition mass
            np.zeros((2, 2)),
            tuple(a.value() for a in accumulator.accumulators),
            None,
        )
        with self.assertRaisesRegex(ValueError, "state counts must equal initial plus transition"):
            accumulator.combine(corrupt)


if __name__ == "__main__":
    unittest.main()
