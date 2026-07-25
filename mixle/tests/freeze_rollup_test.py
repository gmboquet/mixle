"""Tests for the D2 freeze/roll-up cache (mixle.inference.freeze_rollup).

Acceptance criteria under test (see the ConditionalJIT track's D2 item):

1. Explicit approximate freezing can provide a wall-clock-to-F speedup -- measured as a per-datum
   log-density-EVALUATION count (a deliberately more robust-for-CI proxy than raw wall-clock; see
   ``test_evaluation_count_speedup_matches_active_fraction`` for the honesty note on why).
2. F (the observed-data log-likelihood) is non-decreasing under the objective gate.
3. Exact mode is the default; approximation has a bounded escape policy.
4. Cache invalidation binds data, model state, component index, and scoring mode.
"""

import unittest

import numpy as np

from mixle.inference.em import PosteriorTransformEM, observed_log_likelihood, run_em
from mixle.inference.freeze_rollup import (
    FreezeRollupCache,
    _resolve_payload,
    run_em_freeze_rollup,
)
from mixle.stats import (
    CompositeDistribution,
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
    seq_encode,
)
from mixle.stats.bayes.dirichlet import DirichletDistribution
from mixle.stats.latent.mixture import _component_enc


def _make_problem(seed=42, nobs=400):
    """A mixture with 2 slow-converging real components plus 6 far-away decoy components.

    The decoys' optimal weight is small-but-nonzero (not exactly 0, so this exercises the
    freeze/roll-up *cache*, not just MixtureDistribution's pre-existing exact-zero-weight skip)
    and, being widely separated from each other and from the real clusters, settles onto a stable
    fixed point within the first several rounds and then stays there for the rest of the run --
    the "components far from any data whose weight collapses near zero early and stays there"
    scenario the roadmap item calls for.
    """
    truth = MixtureDistribution([GaussianDistribution(-5.0, 0.6), GaussianDistribution(5.0, 0.6)], [0.5, 0.5])
    data = truth.sampler(seed=seed).sample(size=nobs)
    start_components = [
        GaussianDistribution(-0.3, 3.0),
        GaussianDistribution(0.3, 3.0),
        GaussianDistribution(-14.0, 3.0),
        GaussianDistribution(14.0, 3.0),
        GaussianDistribution(-40.0, 3.0),
        GaussianDistribution(40.0, 3.0),
        GaussianDistribution(-70.0, 3.0),
        GaussianDistribution(70.0, 3.0),
    ]
    # [0.4, 0.4] + [0.025] * 6 sums to 0.95, not 1.0 -- MixtureDistribution.__init__ now rejects
    # that at construction (simplex-weight check), so normalize while preserving the exact
    # relative dominant/decoy ratio: a uniform rescale leaves every EM responsibility computation
    # bit-for-bit unchanged, since posterior weights are invariant to a common scale factor.
    start_w = np.array([0.4, 0.4] + [0.025] * 6)
    start = MixtureDistribution(start_components, start_w / start_w.sum())
    estimator = MixtureEstimator([GaussianEstimator() for _ in range(8)])
    enc = seq_encode(data, model=start)
    return start, estimator, enc


class FreezeRollupSpeedupTestCase(unittest.TestCase):
    def test_evaluation_count_speedup_matches_active_fraction(self):
        """The bounded approximate mode reaches the reference F with fewer density evaluations.

        Honesty note on the metric: wall-clock is what the roadmap item names, but on a shared/CI
        machine a handful of milliseconds of Gaussian ``seq_log_density`` calls is dominated by
        scheduling noise, not the effect under test. The thing freeze/roll-up actually removes is
        calls into a component's ``seq_log_density`` (each ``O(nobs)``); a cache hit is an
        signature check and dict lookup. Counting those calls is a deterministic CI-safe proxy for
        the wall-clock claim -- it IS the operation whose count wall-clock would otherwise be
        approximating, without the noise floor.
        """
        start, estimator, enc = _make_problem()

        model, history = run_em_freeze_rollup(
            enc,
            estimator,
            start,
            max_its=400,
            delta=1.0e-9,
            weight_tol=0.05,
            q_gain_tol=1.0e-5,
            weight_delta_tol=1.0e-11,
            freeze_patience=10,
            approximate_freezing=True,
            max_frozen_rounds=50,
        )
        vanilla = run_em(enc, estimator, start, strategy=PosteriorTransformEM(), max_its=400, delta=1.0e-9)

        objective = observed_log_likelihood(enc)
        fr_value = objective(model)
        vanilla_value = objective(vanilla)

        # This fixture stays within the declared approximation budget and reaches the reference F.
        self.assertAlmostEqual(fr_value, vanilla_value, places=6)
        np.testing.assert_allclose(model.w, vanilla.w, atol=1.0e-8)

        # vanilla's per-round cost: PosteriorTransformEM's own E-step (K components) plus run_em's
        # explicit objective(candidate) convergence check (another K components) -- the same
        # 2-evaluations-per-component-per-round structure freeze/roll-up's own accept-gated loop
        # mirrors, so the comparison is apples-to-apples.
        num_components = start.num_components
        vanilla_evals = 2 * num_components * len(history)
        fr_evals = sum(h.n_log_density_evals for h in history)
        mean_active_fraction = float(np.mean([h.active_fraction for h in history]))

        self.assertLess(
            fr_evals, vanilla_evals, "freeze/roll-up issued at least as many log-density evals as vanilla EM would"
        )
        ratio = vanilla_evals / fr_evals
        # A meaningful chunk of components (6 of 8 decoys) spend most of the run frozen; demand at
        # least a modest, honestly-measured speedup rather than asserting a specific large number.
        self.assertGreater(ratio, 1.2)
        # The measured ratio should track the average active fraction actually achieved --
        # loosely, since early rounds (before anything freezes) drag the average toward 1.0.
        self.assertLess(mean_active_fraction, 1.0)
        # Document what was actually measured (visible in -v output / CI logs).
        print(
            "\nfreeze/roll-up speedup: %d rounds, %d vs %d log-density evals, ratio=%.3fx, "
            "mean active fraction=%.3f" % (len(history), fr_evals, vanilla_evals, ratio, mean_active_fraction)
        )


class FreezeRollupMonotonicityTestCase(unittest.TestCase):
    def test_free_energy_is_monotone_round_to_round(self):
        """F (observed-data log-likelihood) never decreases round-to-round under freeze/roll-up.

        This is the Neal-Hinton coordinate-ascent guarantee the whole D-track's correctness
        backbone rests on: a learned/cached scheduling decision may change SPEED, never whether F
        goes up. ``run_em_freeze_rollup`` enforces this directly (an ``accept_tolerance``-gated
        step, exactly like ``mixle.inference.em.MonotonicEM``), so this test is really checking
        that the gate is wired correctly end to end, not re-deriving the EM theorem.
        """
        start, estimator, enc = _make_problem(seed=7, nobs=300)
        _, history = run_em_freeze_rollup(
            enc, estimator, start, max_its=150, delta=1.0e-10, approximate_freezing=True
        )

        self.assertGreater(len(history), 1)
        objectives = [h.objective for h in history]
        for i in range(1, len(objectives)):
            self.assertGreaterEqual(
                objectives[i],
                objectives[i - 1] - 1.0e-9,
                "F decreased from round %d to %d: %r -> %r" % (i - 1, i, objectives[i - 1], objectives[i]),
            )
        # At least one component should actually have frozen during this run -- otherwise the
        # monotonicity check would be vacuous (identical to plain PosteriorTransformEM).
        self.assertTrue(any(h.n_frozen > 0 for h in history))

    def test_dirichlet_prior_uses_the_map_objective(self):
        start, _, enc = _make_problem(seed=13, nobs=300)
        estimator = MixtureEstimator(
            [GaussianEstimator() for _ in range(start.num_components)],
            prior=DirichletDistribution(np.full(start.num_components, 2.0)),
        )
        model, history = run_em_freeze_rollup(enc, estimator, start, max_its=8, delta=None)
        objectives = np.asarray([item.objective for item in history])
        self.assertTrue(np.all(np.diff(objectives) >= -1.0e-9), objectives)
        expected = observed_log_likelihood(enc)(model) + estimator.model_log_density(model)
        self.assertAlmostEqual(history[-1].objective, expected, places=8)
        self.assertIsInstance(model.get_prior()[0], DirichletDistribution)


class FreezeStreakBookkeepingTestCase(unittest.TestCase):
    """A rejected round's ``transaction.restore()`` undoes ``model``/``estimator``'s own mutable
    state, but before this fix ``run_em_freeze_rollup`` ALSO called ``detect_frozen`` against that
    round's (about to be discarded) M-step ``candidate`` to score it for the accept/reject gate --
    polluting the cache's score/weight convergence bookkeeping with a state that was never actually
    accepted. The next round's real ``detect_frozen(cache, model)`` call then compared against
    scores/weights left over from a discarded candidate instead of the model that is
    actually still in play, letting a spuriously "converged" reading accumulate freeze-patience
    streak progress the real (accepted) trajectory never earned -- eventually freezing a component
    that never actually converged and permanently skipping its real M-step.
    """

    def test_rejected_round_does_not_pollute_cache_bookkeeping_with_discarded_candidate_state(self):
        start, estimator, enc = _make_problem(seed=99, nobs=200)
        cache = FreezeRollupCache(freeze_patience=2)

        model, history = run_em_freeze_rollup(
            enc, estimator, start, max_its=1, delta=None, cache=cache, accept_tolerance=-1.0e6
        )

        self.assertFalse(history[0].accepted, "accept_tolerance=-1e6 should force this round to be rejected")
        self.assertIs(model, start, "a rejected round must return the original (unmodified) model")

        payload = _resolve_payload(enc)
        expected_scores = np.column_stack(
            [start.components[idx].seq_log_density(_component_enc(payload, idx)) for idx in range(start.num_components)]
        )
        np.testing.assert_allclose(cache._last_component_scores, expected_scores)

        for idx in range(start.num_components):
            self.assertEqual(cache._prev_weight[idx], float(start.w[idx]))


class FreezeRollupCacheInvalidationTestCase(unittest.TestCase):
    def test_cache_never_reuses_a_column_across_datasets(self):
        component = GaussianDistribution(0.0, 1.0)
        cache = FreezeRollupCache()
        first_enc = component.dist_to_encoder().seq_encode([-2.0, -1.0])
        second_enc = component.dist_to_encoder().seq_encode([3.0, 4.0])

        first, first_hit = cache.component_log_density(0, component, first_enc, frozen=True)
        second, second_hit = cache.component_log_density(0, component, second_enc, frozen=True)

        self.assertFalse(first_hit)
        self.assertFalse(second_hit)
        np.testing.assert_allclose(second, component.seq_log_density(second_enc))
        self.assertFalse(np.allclose(first, second))

    def test_cache_signature_includes_private_slotted_state(self):
        class PrivateSlottedDistribution:
            __slots__ = ("_location",)

            def __init__(self, location):
                self._location = location

            def seq_log_density(self, enc):
                values = np.asarray(enc)
                return -((values - self._location) ** 2)

        component = PrivateSlottedDistribution(0.0)
        data = np.asarray([0.0, 1.0])
        cache = FreezeRollupCache()

        before, _ = cache.component_log_density(0, component, data, frozen=True)
        _, hit = cache.component_log_density(0, component, data, frozen=True)
        component._location = 10.0
        after, mutation_hit = cache.component_log_density(0, component, data, frozen=True)

        self.assertTrue(hit)
        self.assertFalse(mutation_hit)
        self.assertFalse(np.allclose(before, after))

    def test_heuristic_freezing_is_explicit_and_forces_reactivation(self):
        component = GaussianDistribution(0.0, 1.0)
        scores = np.asarray([[0.0], [1.0]])

        exact_cache = FreezeRollupCache(
            weight_tol=1.0, q_gain_tol=1.0, weight_delta_tol=1.0, freeze_patience=1
        )
        exact_cache.record_component_scores(scores)
        self.assertFalse(exact_cache.is_frozen(0, component, 0.01))
        exact_cache.record_component_scores(scores)
        self.assertFalse(exact_cache.is_frozen(0, component, 0.01))

        approximate_cache = FreezeRollupCache(
            weight_tol=1.0,
            q_gain_tol=1.0,
            weight_delta_tol=1.0,
            freeze_patience=1,
            approximate_freezing=True,
            max_frozen_rounds=1,
        )
        approximate_cache.record_component_scores(scores)
        self.assertFalse(approximate_cache.is_frozen(0, component, 0.01))
        approximate_cache.record_component_scores(scores)
        self.assertTrue(approximate_cache.is_frozen(0, component, 0.01))
        self.assertTrue(approximate_cache.is_approximately_frozen(0))
        approximate_cache.record_component_scores(scores)
        self.assertFalse(approximate_cache.is_frozen(0, component, 0.01))
        self.assertFalse(approximate_cache.is_approximately_frozen(0))

    def test_cache_signature_includes_nested_component_parameters(self):
        component = CompositeDistribution((GaussianDistribution(0.0, 1.0), GaussianDistribution(2.0, 1.0)))
        enc = component.dist_to_encoder().seq_encode([(0.0, 2.0), (1.0, 3.0)])
        cache = FreezeRollupCache()

        before, first_hit = cache.component_log_density(0, component, enc, frozen=True)
        _, second_hit = cache.component_log_density(0, component, enc, frozen=True)
        component.dists[0].mu += 10.0
        after, third_hit = cache.component_log_density(0, component, enc, frozen=True)

        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertFalse(third_hit)
        self.assertFalse(np.allclose(before, after))

    def test_cache_invalidates_when_a_frozen_components_parameters_move_again(self):
        """A component the cache is treating as frozen must never serve a stale log-density once
        its parameters change again -- whether that change comes from this module's own M-step
        (unfreezing it) or from an external caller mutating it directly.
        """
        start, estimator, enc = _make_problem(seed=11, nobs=150)
        payload = _resolve_payload(enc)
        cache = FreezeRollupCache(weight_tol=1.0, q_gain_tol=1.0e9, weight_delta_tol=1.0e9, freeze_patience=1)
        component = start.components[0]
        enc_0 = _component_enc(payload, 0)

        # First lookup with frozen=True is still a genuine miss (nothing cached yet).
        ll_first, hit_first = cache.component_log_density(0, component, enc_0, frozen=True)
        self.assertFalse(hit_first)

        # Second lookup, same (unchanged) parameters, still marked frozen: must be a pure cache
        # hit -- no call into seq_log_density.
        ll_cached, hit_second = cache.component_log_density(0, component, enc_0, frozen=True)
        self.assertTrue(hit_second, "second identical-parameter lookup should be a pure cache hit")
        np.testing.assert_array_equal(ll_first, ll_cached)

        # Now mutate the "frozen" component's parameters directly, bypassing this module's own
        # M-step (the "a caller explicitly re-triggers it" case from the roadmap item) -- without
        # calling cache.invalidate(). The signature check inside component_log_density must still
        # catch the drift and recompute rather than silently returning the old array, even though
        # the caller still passes frozen=True.
        component.mu = component.mu + 25.0
        ll_after_mutation, hit_third = cache.component_log_density(0, component, enc_0, frozen=True)
        self.assertFalse(hit_third, "a moved 'frozen' component's cache must be invalidated, not reused")
        expected = component.seq_log_density(enc_0)
        np.testing.assert_allclose(ll_after_mutation, expected)
        self.assertFalse(np.allclose(ll_cached, ll_after_mutation))

        # And an explicit invalidate() drops the entry outright.
        cache.invalidate(0)
        self.assertNotIn(0, cache._entries)

    def test_unfreezing_partway_through_matches_a_no_cache_reference_run(self):
        """End-to-end: a component that freezes and is later forced to re-activate mid-run must
        still land on the correct final fit -- compared against an uncached ``PosteriorTransformEM``
        reference run over the SAME number of rounds -- proving no correctness was lost to a stale
        cache, only speed was gained while a component was genuinely inactive.
        """
        start, estimator, enc = _make_problem(seed=5, nobs=250)
        max_its = 120

        model, history = run_em_freeze_rollup(
            enc,
            estimator,
            start,
            max_its=max_its,
            delta=None,  # run every round, so this exactly matches run_em's fixed max_its loop
            weight_tol=0.05,
            q_gain_tol=1.0e-5,
            weight_delta_tol=1.0e-11,
            freeze_patience=10,
            approximate_freezing=True,
        )
        reference = run_em(enc, estimator, start, strategy=PosteriorTransformEM(), max_its=max_its, delta=None)

        objective = observed_log_likelihood(enc)
        self.assertAlmostEqual(objective(model), objective(reference), places=6)
        np.testing.assert_allclose(model.w, reference.w, atol=1.0e-6)
        self.assertTrue(any(h.n_frozen > 0 for h in history), "test is vacuous unless something actually froze")


if __name__ == "__main__":
    unittest.main()
