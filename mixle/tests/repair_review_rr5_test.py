"""Regressions for the 2026-08-07 adversarial repair reviews (STAT-RR5-1..5 and STAT-RR6-1..2).

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


class MassToleranceScaleSweepTest(unittest.TestCase):
    """STAT-RR6-1: the sqrt-mass tolerance grew without bound and eventually accepted anything.

    The sqrt scaling models sequential float32 accumulation drift, but unbounded it computed rtol
    0.584 at mass 1.5e12 and passed a 50 percent mass mismatch. The tolerance now saturates at the
    mass where float32 accumulation itself stops functioning (2**24 -- adding 1.0 to a float32
    running sum is a no-op past it), about 1.95e-3: beyond that, mass either came from float64
    accumulation (drift far below the ceiling) or is itself broken. The sweep is the reviewer's
    acceptance criterion: supported-precision residuals pass and fixed-fraction corruption fails
    at EVERY scale, at every validation site, keyed and unkeyed.
    """

    _SCALES = (1.0e3, 1.0e6, 1.0e9, 1.0e12)

    @staticmethod
    def _statistic(total_mass, state_factor, accumulator):
        half = total_mass / 2.0
        return (
            2,
            np.array([half / 2.0, half / 2.0]),
            np.array([total_mass * state_factor / 2.0] * 2),
            np.full((2, 2), half / 4.0),
            tuple(a.value() for a in accumulator.accumulators),
            None,
        )

    def _accumulator(self, keys):
        return HiddenMarkovEstimator([GaussianEstimator() for _ in range(2)], keys=keys).accumulator_factory().make()

    def test_supported_precision_residuals_pass_at_every_scale(self):
        from mixle.stats.latent.hidden_markov import _responsibility_mass_tolerance

        for mass in self._SCALES:
            residual = 1.19e-7 * (min(mass, 2.0**24) ** 0.5)  # sequential float32 drift model
            factor = 1.0 + residual
            for keys in ((None, None, None), ("ik", "tk", None)):
                with self.subTest(mass=mass, keys=keys):
                    accumulator = self._accumulator(keys)
                    accumulator.combine(self._statistic(mass, factor, accumulator))
            self.assertLess(residual, _responsibility_mass_tolerance(mass))

    def test_fixed_fraction_corruption_fails_at_every_scale_and_site(self):
        for mass in self._SCALES:
            for keys in ((None, None, None), ("ik", "tk", None), ("ik", "tk", "sk")):
                with self.subTest(mass=mass, keys=keys, site="combine"):
                    accumulator = self._accumulator(keys)
                    with self.assertRaisesRegex(ValueError, "initial plus transition"):
                        accumulator.combine(self._statistic(mass, 1.5, accumulator))
                with self.subTest(mass=mass, keys=keys, site="from_value"):
                    accumulator = self._accumulator(keys)
                    with self.assertRaisesRegex(ValueError, "initial plus transition"):
                        accumulator.from_value(self._statistic(mass, 1.5, accumulator))
                with self.subTest(mass=mass, keys=keys, site="estimate"):
                    estimator = HiddenMarkovEstimator([GaussianEstimator() for _ in range(2)], keys=keys)
                    accumulator = estimator.accumulator_factory().make()
                    with self.assertRaisesRegex(ValueError, "initial plus transition"):
                        estimator.estimate(mass, self._statistic(mass, 1.5, accumulator))

    def test_the_reviewers_exact_probe_is_rejected(self):
        for keys in ((None, None, None), ("ik", "tk", None)):
            with self.subTest(keys=keys):
                accumulator = self._accumulator(keys)
                statistic = (
                    2,
                    np.array([2.5e11, 2.5e11]),
                    np.array([7.5e11, 7.5e11]),  # state mass 1.5e12 against initial+transition 1.0e12
                    np.full((2, 2), 1.25e11),
                    tuple(a.value() for a in accumulator.accumulators),
                    None,
                )
                with self.assertRaisesRegex(ValueError, "initial plus transition"):
                    accumulator.combine(statistic)


class BackoffDifferentiabilityHonestyTest(unittest.TestCase):
    """STAT-RR6-2: a declared differentiable fallback vouched for an undeclared base.

    Differentiability follows the same both-or-nothing rule as support: the mixture's score
    differentiates only if BOTH children are declared and both differentiable.
    """

    class _Undeclared(PoissonDistribution):
        @classmethod
        def compute_declaration(cls):
            return None

    def test_an_undeclared_child_blocks_the_differentiable_claim(self):
        from mixle.stats.combinator.backoff import BackoffDistribution
        from mixle.stats.compute.declarations import declaration_for

        partial = declaration_for(BackoffDistribution(self._Undeclared(3.0), PoissonDistribution(4.0)))
        self.assertFalse(partial.differentiable)

    def test_two_declared_children_keep_their_joint_verdict(self):
        from mixle.stats.combinator.backoff import BackoffDistribution
        from mixle.stats.compute.declarations import declaration_for

        both = declaration_for(BackoffDistribution(PoissonDistribution(2.0), PoissonDistribution(6.0)))
        poisson = declaration_for(PoissonDistribution(2.0))
        self.assertEqual(both.differentiable, poisson.differentiable and poisson.differentiable)


class InfiniteAggregateMassTest(unittest.TestCase):
    """STAT-RR7-1: per-element finiteness is not aggregate finiteness.

    Arrays of finite 8e307 values sum to infinity, and every mass comparison then passes
    vacuously -- np.isclose(inf, inf) is True and inf exceeds no bound -- so the corruption was
    accepted AND retained. An infinite total is refused outright now, at every site and in every
    keying mode; it is not a tolerance question at any scale.
    """

    _BIG = np.array([1.0e308, 1.0e308])  # each element finite; the pair sums past float64 max

    def _statistic(self, accumulator, init, state):
        return (2, init, state, np.zeros((2, 2)), tuple(a.value() for a in accumulator.accumulators), None)

    def test_infinite_totals_are_rejected_at_every_site_and_mode(self):
        for keys in ((None, None, None), ("ik", "tk", None)):
            estimator = HiddenMarkovEstimator([GaussianEstimator() for _ in range(2)], keys=keys)
            for site in ("combine", "from_value", "estimate"):
                for init, state in ((self._BIG, self._BIG), (np.array([1.0, 1.0]), self._BIG)):
                    with self.subTest(keys=keys, site=site, overflow="both" if init is self._BIG else "state"):
                        accumulator = estimator.accumulator_factory().make()
                        statistic = self._statistic(accumulator, init, state)
                        with self.assertRaisesRegex(ValueError, "finite totals"):
                            if site == "combine":
                                accumulator.combine(statistic)
                            elif site == "from_value":
                                accumulator.from_value(statistic)
                            else:
                                estimator.estimate(4.0, statistic)


class RejectedRestorationRollbackTest(unittest.TestCase):
    """SYS-RR7-2: a rejected from_value() must leave the accumulator exactly as it was.

    The previous order assigned the count arrays and then validated, so a refused restoration had
    already replaced the state -- an exception whose side effect is the corruption it refused.
    Validation now precedes every assignment, and child restoration is transactional.
    """

    @staticmethod
    def _valid_statistic(accumulator):
        # state mass 3.0 == initial 2.0 + transition 1.0
        return (
            2,
            np.array([1.0, 1.0]),
            np.array([1.5, 1.5]),
            np.full((2, 2), 0.25),
            tuple(a.value() for a in accumulator.accumulators),
            None,
        )

    def test_a_mass_rejected_restore_preserves_all_counts(self):
        accumulator = HiddenMarkovEstimator([GaussianEstimator() for _ in range(2)]).accumulator_factory().make()
        accumulator.from_value(self._valid_statistic(accumulator))
        before = (
            accumulator.init_counts.copy(),
            accumulator.state_counts.copy(),
            accumulator.trans_counts.copy(),
        )
        corrupt = (
            2,
            np.array([1.0, 1.0]),
            np.array([99.0, 99.0]),
            np.zeros((2, 2)),
            tuple(a.value() for a in accumulator.accumulators),
            None,
        )
        with self.assertRaisesRegex(ValueError, "initial plus transition"):
            accumulator.from_value(corrupt)
        np.testing.assert_array_equal(accumulator.init_counts, before[0])
        np.testing.assert_array_equal(accumulator.state_counts, before[1])
        np.testing.assert_array_equal(accumulator.trans_counts, before[2])

    def test_a_child_failure_mid_restore_rolls_the_counts_and_children_back(self):
        accumulator = HiddenMarkovEstimator([GaussianEstimator() for _ in range(2)]).accumulator_factory().make()
        accumulator.from_value(self._valid_statistic(accumulator))
        counts_before = accumulator.init_counts.copy()
        children_before = [child.value() for child in accumulator.accumulators]
        good_children = tuple(child.value() for child in accumulator.accumulators)
        poisoned = (
            2,
            np.array([2.0, 2.0]),
            np.array([3.0, 3.0]),
            np.full((2, 2), 0.5),
            (good_children[0], None),  # None reliably fails the child's own unpacking
            None,
        )
        with self.assertRaises((ValueError, TypeError)):
            accumulator.from_value(poisoned)
        np.testing.assert_array_equal(accumulator.init_counts, counts_before)
        for child, snapshot in zip(accumulator.accumulators, children_before):
            np.testing.assert_array_equal(
                np.asarray(child.value(), dtype=object).shape, np.asarray(snapshot, dtype=object).shape
            )


class MutatorFinitenessInvariantTest(unittest.TestCase):
    """STAT-RR8-1: finiteness must be an INVARIANT, not an ingestion check.

    Validating what arrives leaves every reduction path free to create the same invalid state:
    two statistics with 4.6e307 elements have finite aggregates individually and an infinite one
    combined; a valid [8e307, 0] scaled by a valid 3.0 overflows outright; keyed pooling overflows
    by the same addition; and key_replace copied [inf, 0] straight in with no validation at all.
    Each mutator now validates its RESULT and rolls back on failure.
    """

    _HALF = np.array([4.6e307, 4.6e307])  # finite alone (9.2e307), infinite when doubled

    @staticmethod
    def _accumulator(keys=(None, None, None)):
        return HiddenMarkovEstimator([GaussianEstimator() for _ in range(2)], keys=keys).accumulator_factory().make()

    @staticmethod
    def _statistic(accumulator, init, state, trans=None):
        return (
            2,
            init,
            state,
            np.zeros((2, 2)) if trans is None else trans,
            tuple(child.value() for child in accumulator.accumulators),
            None,
        )

    def test_combine_refuses_an_overflowing_sum_of_valid_inputs_and_rolls_back(self):
        accumulator = self._accumulator()
        accumulator.combine(self._statistic(accumulator, self._HALF, self._HALF))
        before = accumulator.init_counts.copy()
        with self.assertRaisesRegex(ValueError, "finite"):
            accumulator.combine(self._statistic(accumulator, self._HALF, self._HALF))
        np.testing.assert_array_equal(accumulator.init_counts, before)

    def test_scale_refuses_an_overflowing_product_and_rolls_back(self):
        accumulator = self._accumulator()
        big = np.array([8.0e307, 0.0])
        accumulator.combine(self._statistic(accumulator, big, big))
        before = accumulator.init_counts.copy()
        with self.assertRaisesRegex(ValueError, "finite"):
            accumulator.scale(3.0)
        np.testing.assert_array_equal(accumulator.init_counts, before)

    def test_key_merge_refuses_an_overflowing_pool_and_restores_it(self):
        first = self._accumulator(("ik", "tk", None))
        second = self._accumulator(("ik", "tk", None))
        first.combine(self._statistic(first, self._HALF, self._HALF))
        second.combine(self._statistic(second, self._HALF, self._HALF))
        pool = {}
        first.key_merge(pool)
        before = np.asarray(pool["ik"]).copy()
        with self.assertRaisesRegex(ValueError, "finite"):
            second.key_merge(pool)
        np.testing.assert_array_equal(np.asarray(pool["ik"]), before)

    def test_key_replace_validates_the_incoming_replacement(self):
        accumulator = self._accumulator(("ik", "tk", None))
        for bad in (np.array([np.inf, 0.0]), np.array([np.nan, 1.0]), np.array([1.0, 2.0, 3.0])):
            with self.subTest(replacement=str(bad)):
                with self.assertRaises(ValueError):
                    accumulator.key_replace({"ik": bad, "tk": np.zeros((2, 2))})

    def test_legitimate_reductions_are_unaffected(self):
        accumulator = self._accumulator()
        accumulator.combine(
            self._statistic(accumulator, np.array([1.0, 1.0]), np.array([1.5, 1.5]), np.full((2, 2), 0.25))
        )
        accumulator.scale(2.0)
        np.testing.assert_allclose(accumulator.init_counts, [2.0, 2.0])
        keyed = self._accumulator(("ik", "tk", None))
        keyed.combine(self._statistic(keyed, np.array([1.0, 1.0]), np.array([1.5, 1.5]), np.full((2, 2), 0.25)))
        pool = {}
        keyed.key_merge(pool)
        keyed.key_replace(pool)
        np.testing.assert_allclose(keyed.init_counts, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
