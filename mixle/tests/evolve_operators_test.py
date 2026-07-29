"""The improvement operators + scoped registry (mixle.evolve.operators)."""

import unittest

import numpy as np

from mixle.evolve import (
    AutoSelect,
    Mutate,
    OnlineUpdate,
    Recalibrate,
    Recompose,
    Refit,
    nll_objective,
    register_operator,
    registered_operators,
    unregister_operator,
)
from mixle.evolve.objective import pointwise_log_density
from mixle.ops import mixture
from mixle.stats import GaussianDistribution


class OperatorTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.data = list(rng.normal(3.0, 2.0, 400))
        self.champion = GaussianDistribution(0.0, 1.0)  # deliberately wrong
        self.nll = nll_objective()
        self.ctx = {"parent_hash": "parent0"}

    def test_each_operator_returns_a_valid_fitted_model(self):
        for op in [Refit(), OnlineUpdate(mode="streaming"), OnlineUpdate(mode="incremental"), AutoSelect()]:
            with self.subTest(op=repr(op.name)):
                self.assertTrue(op.applicable(self.champion, self.data, ctx=self.ctx))
                cand = op.propose(self.champion, self.data, ctx=self.ctx)
                self.assertEqual(cand.parent_hash, "parent0")
                # the proposed model must score finite per-observation log densities.
                ld = pointwise_log_density(cand.model, self.data)
                self.assertEqual(ld.shape[0], len(self.data))
                self.assertTrue(np.all(np.isfinite(ld)))
                # and it must be a real improvement over the bad champion.
                self.assertLess(self.nll.scalar(cand.model, self.data), self.nll.scalar(self.champion, self.data))

    def test_recalibrate_returns_valid_model_and_scores_off_split(self):
        op = Recalibrate(seed=0)
        self.assertTrue(op.applicable(GaussianDistribution(3.0, 2.0), self.data, ctx=self.ctx))
        cand = op.propose(GaussianDistribution(3.0, 2.0), self.data, ctx=self.ctx)
        # exact change-of-variables: log density on a *different* split is finite and split-safe.
        rng = np.random.RandomState(99)
        other = list(rng.normal(3.0, 2.0, 50))
        ld = pointwise_log_density(cand.model, other)
        self.assertEqual(ld.shape[0], 50)
        self.assertTrue(np.all(np.isfinite(ld)))
        self.assertIn("temperature", cand.meta)

    def test_recalibrate_density_normalizes(self):
        # the recalibrated density must integrate to ~1 (exactness check on the Jacobian).
        from mixle.evolve.operators import _RecalibratedModel

        base = GaussianDistribution(0.0, 1.0)
        recal = _RecalibratedModel(base, temperature=1.5, center=0.0)
        grid = np.linspace(-12, 12, 4001)
        dens = np.exp([recal.log_density(float(x)) for x in grid])
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        mass = trapezoid(dens, grid)
        self.assertAlmostEqual(mass, 1.0, places=3)

    def test_posterior_carry_applicability_is_honest(self):
        # conjugate Gaussian -> applicable; the streaming mode is always applicable.
        self.assertTrue(OnlineUpdate(mode="posterior_carry").applicable(self.champion, self.data, ctx={}))
        self.assertTrue(OnlineUpdate(mode="streaming").applicable(self.champion, self.data, ctx={}))

    def test_scoped_registry_register_and_unregister(self):
        op = Refit(name="custom_refit")
        register_operator(op)
        self.assertIn("custom_refit", registered_operators())
        unregister_operator("custom_refit")
        self.assertNotIn("custom_refit", registered_operators())


class RecalibratedEncodedScoringTest(unittest.TestCase):
    """A _RecalibratedModel must expose ONE density per observation: its public encoded/batch route
    (``dist_to_encoder().seq_encode(...)`` -> ``seq_log_density(...)``, the contract every other
    distribution in this codebase honors) must agree with its scalar ``log_density`` path and its
    split-safe ``seq_log_density_raw`` path for the same observation. Pre-fix, ``dist_to_encoder()``
    delegated straight to the base model's own encoder, so the encoded path scored the UNTRANSFORMED
    raw observation and only added the Jacobian -- a different (wrong) density than the scalar path."""

    def test_encoded_path_agrees_with_scalar_and_raw_batch_paths(self):
        # Pins the exact repro: standard Gaussian, temperature=2, center=0, observation=2. Pre-fix,
        # the encoded path answered -3.6120857... (Jacobian only, no value transform) while the
        # scalar/raw-batch paths correctly answered -2.1120857....
        from mixle.evolve.operators import _RecalibratedModel

        base = GaussianDistribution(0.0, 1.0)
        recal = _RecalibratedModel(base, temperature=2.0, center=0.0)

        scalar = recal.log_density(2.0)
        raw_batch = recal.seq_log_density_raw([2.0])
        enc = recal.dist_to_encoder().seq_encode([2.0])
        encoded = recal.seq_log_density(enc)

        self.assertAlmostEqual(scalar, -2.112085713764618, places=6)
        self.assertAlmostEqual(float(raw_batch[0]), scalar, places=9)
        self.assertAlmostEqual(float(encoded[0]), scalar, places=9)

    def test_encoded_path_agrees_over_a_grid_of_observations_and_temperatures(self):
        from mixle.evolve.operators import _RecalibratedModel

        base = GaussianDistribution(1.5, 2.0)
        rng = np.random.RandomState(0)
        xs = rng.normal(1.5, 3.0, 25)
        for temperature, center in [(0.6, 1.5), (1.0, 0.0), (3.3, -2.0)]:
            with self.subTest(temperature=repr(temperature), center=repr(center)):
                recal = _RecalibratedModel(base, temperature=temperature, center=center)
                scalars = np.array([recal.log_density(float(x)) for x in xs])
                enc = recal.dist_to_encoder().seq_encode(list(xs))
                encoded = np.asarray(recal.seq_log_density(enc), dtype=float)
                np.testing.assert_allclose(encoded, scalars, rtol=1e-10, atol=1e-10)

    def test_recalibrate_operator_output_honors_the_same_invariant(self):
        # end-to-end: a model produced by the real Recalibrate operator (not a hand-built
        # _RecalibratedModel) must still agree between its scalar and public-encoded scoring paths.
        rng = np.random.RandomState(0)
        data = list(rng.normal(3.0, 2.0, 200))
        cand = Recalibrate(seed=0).propose(GaussianDistribution(3.0, 2.0), data, ctx={"parent_hash": None})
        recal = cand.model

        probe = list(np.random.RandomState(7).normal(3.0, 2.0, 30))
        scalars = np.array([recal.log_density(float(x)) for x in probe])
        enc = recal.dist_to_encoder().seq_encode(probe)
        encoded = np.asarray(recal.seq_log_density(enc), dtype=float)
        np.testing.assert_allclose(encoded, scalars, rtol=1e-9, atol=1e-9)


class RecalibrationFiniteControlsTest(unittest.TestCase):
    """MXR-080-1780: recalibration must never construct a non-finite predictive model.

    ``temperature <= 0.0`` rejected only the negative half-line: NaN and infinity both compared
    False and sailed through, and ``center`` was unchecked entirely. The affine map
    ``y -> c + (y - c) / T`` is undefined for either, so what came out was not a distribution --
    NaN density and NaN samples for a NaN temperature, ``-inf`` density and NaN samples for an
    infinite one -- while presenting itself as an ordinary recalibrated model.
    """

    def setUp(self):
        self.base = GaussianDistribution(0.0, 1.0)
        self.data = list(np.random.RandomState(0).normal(3.0, 2.0, 120))

    def test_non_finite_temperature_is_rejected(self):
        from mixle.evolve.operators import _RecalibratedModel

        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(temperature=repr(bad)), self.assertRaises(ValueError):
                _RecalibratedModel(self.base, temperature=bad, center=0.0)

    def test_non_positive_temperature_is_still_rejected(self):
        from mixle.evolve.operators import _RecalibratedModel

        for bad in (0.0, -1.0):
            with self.subTest(temperature=repr(bad)), self.assertRaises(ValueError):
                _RecalibratedModel(self.base, temperature=bad, center=0.0)

    def test_non_finite_center_is_rejected(self):
        from mixle.evolve.operators import _RecalibratedModel

        for bad in (float("nan"), float("inf")):
            with self.subTest(center=repr(bad)), self.assertRaises(ValueError):
                _RecalibratedModel(self.base, temperature=1.0, center=bad)

    def test_an_empty_grid_cannot_report_an_unevaluated_temperature(self):
        # grid=() used to leave the initial best_t = 1.0 in place with best_err still inf, i.e. a
        # "chosen" temperature no candidate evaluation ever produced.
        with self.assertRaises(ValueError):
            Recalibrate(grid=())

    def test_invalid_grid_and_ensemble_controls_are_rejected_at_declaration(self):
        for kwargs in (
            {"grid": (float("nan"),)},
            {"grid": (float("inf"), 1.0)},
            {"grid": (0.0, 1.0)},
            {"grid": (-1.0,)},
            {"ensemble": 0},
            {"ensemble": -8},
            {"ensemble": True},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                Recalibrate(**kwargs)

    def test_a_valid_search_still_produces_a_finite_recalibrated_model(self):
        cand = Recalibrate(seed=0).propose(self.base, self.data, ctx={"parent_hash": None})
        self.assertTrue(np.isfinite(cand.model.temperature))
        self.assertGreater(cand.model.temperature, 0.0)
        self.assertTrue(np.isfinite(cand.model.center))
        self.assertTrue(np.isfinite(cand.meta["pit_error"]))
        self.assertIn(cand.model.temperature, [float(t) for t in Recalibrate().grid])
        self.assertTrue(np.all(np.isfinite(np.asarray(cand.model.sampler(0).sample(16), dtype=float))))

    def test_non_finite_observations_are_refused_rather_than_propagated(self):
        with self.assertRaises(ValueError):
            Recalibrate(seed=0).propose(self.base, [1.0, float("nan"), 2.0], ctx={"parent_hash": None})


class MutateShrinkWeightsTest(unittest.TestCase):
    """The ``shrink`` move drops one mixture component but must renormalize the SURVIVING
    components' weights to a simplex before handing them to ``mixture(...)`` -- passing the original
    (now sub-1) weights straight through trips MixtureDistribution's own simplex validation."""

    # A fresh RandomState(3)'s first draw is `randint(3) == 2`, i.e. index 2 of
    # `["grow", "perturb", "shrink"]` -- pinning this seed makes every test below deterministically
    # exercise the shrink branch on the first call instead of searching across seeds.
    SHRINK_SEED = 3

    def test_seed_pin_selects_shrink_first_try(self):
        self.assertEqual(int(np.random.RandomState(self.SHRINK_SEED).randint(3)), 2)

    def test_shrink_on_two_component_mixture_no_longer_raises(self):
        # Confirmed pre-fix repro: dropping either member of a [0.5, 0.5] mixture leaves weights
        # [0.5] for the sole survivor, which MixtureDistribution's constructor rejects outright
        # (sum=0.5 != 1.0).
        m = mixture([GaussianDistribution(0.0, 1.0), GaussianDistribution(5.0, 1.0)], [0.5, 0.5])
        data = list(np.random.RandomState(0).normal(0.0, 1.0, 50))
        cand = Mutate().propose(m, data, ctx={"seed": self.SHRINK_SEED})
        self.assertEqual(cand.meta["move"], "shrink")
        w = np.asarray(cand.model.w, dtype=float)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=8)
        self.assertTrue(bool(np.all(w >= 0.0)))

    def test_shrink_renormalizes_a_three_component_mixture(self):
        m = mixture(
            [GaussianDistribution(0.0, 1.0), GaussianDistribution(5.0, 1.0), GaussianDistribution(10.0, 1.0)],
            [0.2, 0.3, 0.5],
        )
        data = list(np.random.RandomState(0).normal(0.0, 1.0, 60))
        cand = Mutate().propose(m, data, ctx={"seed": self.SHRINK_SEED})
        self.assertEqual(cand.meta["move"], "shrink")
        w = np.asarray(cand.model.w, dtype=float)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=8)

    def test_shrink_rejects_zero_residual_mass_instead_of_dividing_by_zero(self):
        # Mutate only requires `.components` / `.w` / `.estimator()` on `model` (duck typing, not a
        # validated MixtureDistribution), so an all-zero weight vector -- unreachable from a
        # legitimately-constructed MixtureDistribution, but not excluded by this operator's own
        # duck typing -- must be rejected with a clear error instead of silently computing 0/0 == nan.
        class FakeMixtureModel:
            def __init__(self, components, w):
                self.components = components
                self.w = w

            def estimator(self):
                raise AssertionError("estimator() must not be reached: the guard should raise first")

        components = [GaussianDistribution(0.0, 1.0), GaussianDistribution(5.0, 1.0), GaussianDistribution(10.0, 1.0)]
        fake = FakeMixtureModel(components, np.array([0.0, 0.0, 0.0]))
        data = list(np.random.RandomState(0).normal(0.0, 1.0, 60))
        with self.assertRaisesRegex(ValueError, "residual mass"):
            Mutate().propose(fake, data, ctx={"seed": self.SHRINK_SEED})

    def test_shrink_rejects_non_finite_residual_mass(self):
        class FakeMixtureModel:
            def __init__(self, components, w):
                self.components = components
                self.w = w

            def estimator(self):
                raise AssertionError("estimator() must not be reached: the guard should raise first")

        components = [GaussianDistribution(0.0, 1.0), GaussianDistribution(5.0, 1.0), GaussianDistribution(10.0, 1.0)]
        # np.argmin sends the FIRST NaN's index to `drop`; the SECOND NaN survives into `keep`, so
        # this exercises the non-finite branch specifically (as opposed to the zero-sum branch above).
        fake = FakeMixtureModel(components, np.array([float("nan"), float("nan"), 1.0]))
        data = list(np.random.RandomState(0).normal(0.0, 1.0, 60))
        with self.assertRaisesRegex(ValueError, "residual mass"):
            Mutate().propose(fake, data, ctx={"seed": self.SHRINK_SEED})


class OperatorGeneratorDataTest(unittest.TestCase):
    """Every built-in ``applicable`` measures ``len(list(data))`` (or otherwise iterates ``data``) --
    on a one-shot iterable (a generator, a DB cursor, ...) that fully consumes it, so a naive
    ``propose`` called right after used to receive an exhausted, empty iterable instead of the batch
    ``applicable`` had just approved."""

    def setUp(self):
        self.rows = list(np.random.RandomState(0).normal(3.0, 2.0, 400))
        self.champion = GaussianDistribution(0.0, 1.0)  # deliberately wrong, as in OperatorTest above
        self.nll = nll_objective()
        self.champion_nll = self.nll.scalar(self.champion, self.rows)

    def _generator(self):
        yield from self.rows

    def _assert_really_used_the_data(self, model):
        # An empty-batch fit/update is not always a loud failure (Refit/AutoSelect/Recompose/Mutate
        # raise "optimize() received empty data", but OnlineUpdate/Recalibrate degrade SILENTLY: an
        # empty streaming update or an empty PIT calibration search just leaves the champion
        # unchanged, or worse -- so "scores finite" alone is too weak a check here. A genuine fit/
        # update on these 400 informative rows must strictly improve on the deliberately-wrong
        # champion's NLL; an exhausted-generator no-op cannot (confirmed pre-fix: online_update's
        # empty update produced a wildly WORSE NLL, and recalibrate's empty search left the NLL
        # exactly equal to the champion's -- neither is an improvement).
        ld = pointwise_log_density(model, self.rows)
        self.assertEqual(ld.shape[0], len(self.rows))
        self.assertTrue(np.all(np.isfinite(ld)))
        self.assertLess(self.nll.scalar(model, self.rows), self.champion_nll)

    def test_applicable_then_propose_on_a_generator_still_sees_the_data(self):
        ops = [Refit(), OnlineUpdate(mode="streaming"), AutoSelect(), Recalibrate(), Recompose(), Mutate()]
        for op in ops:
            with self.subTest(op=repr(op.name)):
                gen = self._generator()
                ctx = {"parent_hash": None, "seed": 0}
                self.assertTrue(op.applicable(self.champion, gen, ctx=ctx))
                cand = op.propose(self.champion, gen, ctx=ctx)
                self._assert_really_used_the_data(cand.model)

    def test_shared_ctx_across_multiple_operators_on_one_generator(self):
        # mirrors mixle.evolve.population.Population.step()'s per-generation loop: ONE `data`
        # argument (here, a one-shot generator) and ONE `ctx` dict threaded through applicable+propose
        # for EVERY operator considered in the loop, not just a single applicable/propose pair.
        gen = self._generator()
        ctx = {"parent_hash": None, "seed": 0}
        for op in [Refit(), OnlineUpdate(mode="streaming"), AutoSelect(), Recalibrate()]:
            with self.subTest(op=repr(op.name)):
                self.assertTrue(op.applicable(self.champion, gen, ctx=ctx))
                cand = op.propose(self.champion, gen, ctx=ctx)
                self._assert_really_used_the_data(cand.model)


if __name__ == "__main__":
    unittest.main()
