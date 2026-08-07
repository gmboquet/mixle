"""Regressions for the five findings of the 2026-08-07 adversarial repair review (STAT-RR5-1..5).

Each finding was a hole in a same-week repair, found by probing the repair's edges from the
installed wheel. Every test here reproduces the reviewer's probe, and where the first repair
narrowed a guard, a negative control pins that the guard still catches the corruption it was
written for.
"""

import pickle
import unittest
from types import MappingProxyType

import numpy as np

from mixle.stats import (
    BernoulliSetDistribution,
    BetaDistribution,
    GaussianEstimator,
    HiddenMarkovEstimator,
    PoissonDistribution,
)


class ConjugateBernoulliSetPickleTest(unittest.TestCase):
    """STAT-RR5-1: the pickle repair converted ``pmap`` by name and missed ``posteriors``.

    A conjugate-fitted model carries BOTH as read-only views, so the conjugate path stayed
    unpicklable at every protocol after the first fix. The hooks now walk the instance and record
    which attributes were views, so any future proxied attribute is covered automatically.
    """

    def test_conjugate_model_round_trips_at_every_protocol(self):
        model = BernoulliSetDistribution(
            {"x": 0.5, "y": 0.3},
            prior=BetaDistribution(1.0, 1.0),
            posteriors={"x": (2.0, 3.0), "y": (1.0, 4.0)},
        )
        for protocol in (0, 2, 5):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(model, protocol=protocol))
                self.assertAlmostEqual(restored.log_density(["x"]), model.log_density(["x"]), places=12)
                self.assertIsInstance(restored.pmap, MappingProxyType)
                self.assertIsInstance(restored.posteriors, MappingProxyType)

    def test_plain_model_still_round_trips_with_no_posteriors(self):
        restored = pickle.loads(pickle.dumps(BernoulliSetDistribution({"x": 0.5})))
        self.assertIsNone(restored.posteriors)
        self.assertIsInstance(restored.pmap, MappingProxyType)


class KeyedMassCorruptionTest(unittest.TestCase):
    """STAT-RR5-2: the tied-dynamics repair skipped mass validation whenever any key was set.

    The reviewer fed a keyed accumulator init=2, transition=0, state=198 and it was accepted. The
    checks now enforce the strongest relation each site preserves -- measured, not assumed:
    ``combine`` receives one accumulator's own statistics (pooling happens later, at key_merge),
    so equality holds there unconditionally; the M-step receives pooled parts, where pooling only
    ADDS initial/transition mass, so local state mass bounded by the pooled total survives.
    """

    @staticmethod
    def _corrupt_statistic(accumulator):
        return (
            2,
            np.array([1.0, 1.0]),
            np.array([99.0, 99.0]),  # state mass 198 against initial+transition mass 2
            np.zeros((2, 2)),
            tuple(a.value() for a in accumulator.accumulators),
            None,
        )

    def test_keyed_combine_rejects_the_reviewers_probe(self):
        for keys in (("ik", "tk", None), ("ik", "tk", "sk"), (None, None, "sk")):
            with self.subTest(keys=keys):
                accumulator = (
                    HiddenMarkovEstimator([GaussianEstimator() for _ in range(2)], keys=keys)
                    .accumulator_factory()
                    .make()
                )
                with self.assertRaisesRegex(ValueError, "initial plus transition"):
                    accumulator.combine(self._corrupt_statistic(accumulator))

    def test_unkeyed_combine_still_rejects_it(self):
        accumulator = HiddenMarkovEstimator([GaussianEstimator() for _ in range(2)]).accumulator_factory().make()
        with self.assertRaisesRegex(ValueError, "state counts must equal initial plus transition"):
            accumulator.combine(self._corrupt_statistic(accumulator))

    def test_keyed_estimate_rejects_state_mass_above_the_pooled_total(self):
        estimator = HiddenMarkovEstimator([GaussianEstimator() for _ in range(2)], keys=("ik", "tk", None))
        accumulator = estimator.accumulator_factory().make()
        corrupt = self._corrupt_statistic(accumulator)
        with self.assertRaisesRegex(ValueError, "exceed the pooled initial plus transition"):
            estimator.estimate(4.0, corrupt)

    def test_every_keying_mode_still_fits(self):
        import io

        from mixle.inference.estimation import optimize
        from mixle.stats import MixtureEstimator

        rng = np.random.RandomState(0)
        data = [list(rng.randn(6)) for _ in range(60)]
        for keys in ((None, None, None), ("ik", "tk", None), ("ik", "tk", "sk"), (None, None, "sk")):
            with self.subTest(keys=keys):

                def chain(tied=keys):
                    return HiddenMarkovEstimator([GaussianEstimator() for _ in range(2)], keys=tied)

                optimize(
                    data,
                    MixtureEstimator([chain() for _ in range(2)]),
                    max_its=3,
                    rng=np.random.RandomState(1),
                    out=io.StringIO(),
                )


class LengthScoreGeometryTest(unittest.TestCase):
    """STAT-RR5-3: the null-length repair keyed off the OUTPUT's shape instead of the MODEL.

    An empty score array for a real batch was treated as "nothing to add" (a silent skip), a
    column matrix was accepted (and (n,1)+(n,) broadcasting would have scored garbage), and a
    scalar leaked an IndexError from the shape probe. The null case is now identified by the model
    being a NullDistribution, and everything else must return exactly one score per sequence.
    """

    class _Estimate:
        def __init__(self, length_model):
            self.len_dist = length_model

    @staticmethod
    def _length_model(returning):
        return type("FakeLength", (), {"seq_log_density": lambda self, enc: returning})()

    def _term(self, returning, rows=5):
        from mixle.stats.latent.hidden_markov import _length_term

        return _length_term(self._Estimate(self._length_model(returning)), object(), rows)

    def test_column_matrix_empty_and_scalar_outputs_all_raise_the_named_error(self):
        for bad in (np.zeros((5, 1)), np.zeros(0), np.float64(1.5), np.zeros((5, 5))):
            with self.subTest(shape=getattr(bad, "shape", "scalar")):
                with self.assertRaisesRegex(ValueError, "one log-density per sequence"):
                    self._term(bad)

    def test_a_correct_vector_passes_and_a_null_model_contributes_nothing(self):
        np.testing.assert_array_equal(self._term(np.arange(5.0)), np.arange(5.0))

        from mixle.stats.combinator.null_dist import NullDistribution
        from mixle.stats.latent.hidden_markov import _length_term

        self.assertIsNone(_length_term(self._Estimate(NullDistribution()), object(), 5))


class DirichletDimensionTest(unittest.TestCase):
    """STAT-RR5-4: folding the symmetric spelling into the Dirichlet family erased the dimension.

    A 2-category component against a declared 3-simplex reference produced a finite, invalid ELBO
    term. Declared dimensions must now agree; only a dimension-agnostic template (no declared
    ``dim``) matches any -- which is what the automatic path's deferred priors are.
    """

    def test_cross_dimension_pairs_are_rejected_and_same_dimension_accepted(self):
        from mixle.stats.bayes.dirichlet import DirichletDistribution
        from mixle.stats.bayes.dirichlet_process_mixture import _prior_structure, _prior_structures_match
        from mixle.stats.bayes.symmetric_dirichlet import SymmetricDirichletDistribution

        symmetric3 = _prior_structure(SymmetricDirichletDistribution(1.0, 3))
        general3 = _prior_structure(DirichletDistribution(np.array([2.0, 3.0, 4.0])))
        general2 = _prior_structure(DirichletDistribution(np.array([2.0, 3.0])))
        self.assertTrue(_prior_structures_match(symmetric3, general3))
        self.assertFalse(_prior_structures_match(symmetric3, general2))

    def test_the_automatic_dp_mixture_still_fits_with_template_priors(self):
        import io

        from mixle.utils.automatic import get_dpm_mixture

        rng = np.random.RandomState(0)
        data = [[int(v) for v in rng.randint(0, 5, size=rng.randint(1, 6))] for _ in range(120)]
        model = get_dpm_mixture(data, rng=np.random.RandomState(1), max_components=4, max_its=5, out=io.StringIO())
        self.assertGreater(model.num_components, 0)


class BackoffDeclarationHonestyTest(unittest.TestCase):
    """STAT-RR5-5: one declaring child stamped its support on the whole backoff.

    A set-collapse let a single declared child supply the mixture's support while the other child
    had proved nothing, and a positional role slice labeled a surviving fallback declaration
    "base". Support now requires BOTH children to declare and agree, and roles travel with their
    declarations.
    """

    class _Undeclared(PoissonDistribution):
        @classmethod
        def compute_declaration(cls):
            return None

    def test_an_undeclared_child_leaves_support_unstated_and_roles_truthful(self):
        from mixle.stats.combinator.backoff import BackoffDistribution
        from mixle.stats.compute.declarations import declaration_for

        declaration = declaration_for(BackoffDistribution(self._Undeclared(3.0), PoissonDistribution(4.0)))
        self.assertIsNone(declaration.support)
        self.assertEqual(declaration.child_roles, ("fallback",))

    def test_agreeing_children_still_declare_their_common_support(self):
        from mixle.stats.combinator.backoff import BackoffDistribution
        from mixle.stats.compute.declarations import declaration_for

        declaration = declaration_for(BackoffDistribution(PoissonDistribution(2.0), PoissonDistribution(6.0)))
        self.assertEqual(declaration.support, "non_negative_integer")
        self.assertEqual(declaration.child_roles, ("base", "fallback"))


if __name__ == "__main__":
    unittest.main()
