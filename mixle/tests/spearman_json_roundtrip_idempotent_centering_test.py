"""Spearman: the identical idempotent-centering defect fixed on Thurstone, pinned here too.

``SpearmanRankingDistribution.__init__`` canonicalizes its ``sigma`` argument through
``_validate_location``: ``sigma = raw - raw.mean() + (dim - 1) / 2.0`` removes the common-shift
non-identifiability before checking the permutahedron constraints. ``__pysp_getstate__`` serializes
that already-canonicalized ``sigma`` verbatim, and ``__pysp_setstate__`` used to reconstruct via
``self.__init__(state['sigma'], ...)`` -- feeding the already-canonicalized array back through the
SAME shift. That shift is not exactly idempotent in float64 (see
``thurstone_json_roundtrip_idempotent_centering_test.py`` for the identical mechanism on
``ThurstoneDistribution.mu``), so a restored ``sigma`` could differ from the original by 1-few ULP --
invisible to ``log_density``/sampling, but changing the raw bytes
:func:`mixle.data.hashing.model_hash` fingerprints, which could trip ``Model.load``'s "integrity
note" ``UserWarning`` (mixle/lifecycle.py) on a same-process, zero-tampering round trip.

Unlike Thurstone, this does not currently manifest through ``SpearmanRankingEstimator.estimate()``:
its ``sigma`` there is always ``argsort(argsort(rank_sum))`` -- an EXACT integer permutation of
``0..dim-1`` -- and canonicalizing an exact integer permutation is bit-exact idempotent (every
intermediate value is a small integer or half-integer, all exactly representable in float64, so
``raw - raw.mean() + (dim - 1) / 2.0`` reproduces ``raw`` exactly). ``EstimatorPathNeverManifestsTest``
below pins that this stays true. The class explicitly supports a second construction path where it
does NOT hold, though: the module docstring documents ``sigma`` as "a (possibly fractional) mean
rank vector", and direct construction with one -- e.g. a consensus mean-rank vector averaged from
several judges' rankings, a wholly ordinary way to build a prototype distribution -- is exactly the
case this file's main sweeps exercise. The fix is the same shape as Thurstone's: an internal,
keyword-only ``_sigma_already_centered`` flag on ``__init__``/``_validate_location``, used only by
``__pysp_setstate__``, skips the shift on restore while leaving ordinary construction (and the
permutahedron validity check, which only reads ``sigma``) untouched.

Fixing this is defensive rather than a live false-positive: nothing in the estimated path exercises
it today. It is fixed anyway because it is the identical latent defect Thurstone had, and a future
change to how ``estimate()`` derives ``sigma`` (e.g. a tie-breaking or smoothing scheme that no
longer lands on an exact integer permutation) could make it manifest silently, exactly as it did for
Thurstone this release.

An adversarial review of the fix above found it introduced a real regression, Spearman-specific
(Thurstone's ``mu`` has no analogous constraint, so it is unaffected): skipping the shift on the
``already_centered=True`` restore path also skipped the ONLY thing that ever enforced ``sigma``'s
total-sum invariant, ``sum(sigma) == dim*(dim-1)/2``. The majorization loop in
``_validate_location`` only lower-bounds ascending prefix sums through ``dim - 1`` terms and never
inspects the full sum, so a ``sigma`` shifted by any constant -- still strictly ascending, so it
clears every one of those bounds -- restored with a silently corrupted total instead of being
rejected. ``SetstateRejectsNonCanonicalTotalTest`` below reproduces that exact scenario and pins the
follow-up fix: a dedicated equality check for the total, scoped to ``already_centered=True``, that
rejects a bad total rather than silently accepting it or silently re-deriving it (re-deriving would
reintroduce the very ULP drift this file's other tests exist to close).
"""

from __future__ import annotations

import os
import tempfile
import unittest
import warnings

import numpy as np

from mixle.data.hashing import model_hash
from mixle.lifecycle import Model
from mixle.stats.rankings.spearman_rho import SpearmanRankingDistribution, SpearmanRankingEstimator

_DIMS = range(2, 9)
_SEEDS = range(15)
_RHOS = (0.3, 0.8, 2.0)

_DEPLOY_DIMS = range(2, 8)
_DEPLOY_SEEDS = range(8)


def _fractional_mean_rank_sigma(dim: int, seed: int, n_judges: int = 7) -> np.ndarray:
    """A realistic fractional mean-rank vector: the elementwise average rank across several
    independent judges' full rankings -- the direct-construction use case the module docstring
    documents ("a (possibly fractional) mean rank vector"). Guaranteed to lie in the permutahedron
    (no try/except needed here) because it is literally a convex combination of permutations of
    ``0..dim-1``, and the permutahedron is defined as their convex hull.
    """
    rng = np.random.RandomState(seed)
    rank_vectors = [np.argsort(rng.permutation(dim)) for _ in range(n_judges)]
    return np.mean(rank_vectors, axis=0)


class SpearmanPureGetstateSetstateIdempotentCenteringTest(unittest.TestCase):
    """Purest isolation: ``__pysp_getstate__`` -> ``__pysp_setstate__`` in memory, zero JSON/file
    machinery, matching the isolation used for the Thurstone finding."""

    def test_round_trip_is_bit_identical_across_a_seeded_sweep(self):
        checked = 0
        for dim in _DIMS:
            for seed in _SEEDS:
                for rho in _RHOS:
                    with self.subTest(dim=dim, seed=seed, rho=rho):
                        sigma = _fractional_mean_rank_sigma(dim, seed)
                        dist = SpearmanRankingDistribution(sigma, rho=rho)
                        state = dist.__pysp_getstate__()
                        restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
                        restored.__pysp_setstate__(state)
                        self.assertTrue(
                            np.array_equal(dist.sigma, restored.sigma),
                            f"sigma drifted by {restored.sigma - dist.sigma!r} across a pure "
                            f"getstate/setstate round trip (dim={dim!r} seed={seed!r} rho={rho!r})",
                        )
                        self.assertEqual(model_hash(dist), model_hash(restored))
                        self.assertTrue(np.array_equal(dist.log_weights, restored.log_weights))
                        self.assertEqual(dist.log_const, restored.log_const)
                        checked += 1
        self.assertGreater(checked, 100)

    def test_fresh_construction_still_canonicalizes_on_first_construction(self):
        # The fix must not touch ordinary construction: only __pysp_setstate__'s internal
        # _sigma_already_centered=True call may skip the shift.
        raw = np.array([5.0, 6.0, 7.0, 8.0])  # a shifted permutation: valid, with a real shift to undo
        dim = len(raw)
        dist = SpearmanRankingDistribution(raw, rho=0.5)
        self.assertTrue(np.array_equal(dist.sigma, raw - raw.mean() + (dim - 1) / 2.0))

    def test_sigma_stays_read_only_on_both_construction_paths(self):
        dist = SpearmanRankingDistribution(np.array([0.0, 1.0, 2.0]), rho=0.5)
        with self.assertRaises(ValueError):
            dist._sigma[0] = 0.0
        restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
        restored.__pysp_setstate__(dist.__pysp_getstate__())
        with self.assertRaises(ValueError):
            restored._sigma[0] = 0.0


class SpearmanDeployLoadIdempotentCenteringTest(unittest.TestCase):
    """Same defect, exercised through the real artifact path (see mixle/lifecycle.py's
    "integrity note" check, and ``ManifestSwapDisclosureTest`` in campaign2_lifecycle_test.py)."""

    def test_deploy_then_load_preserves_sigma_and_model_hash_across_a_seeded_sweep(self):
        checked = 0
        with tempfile.TemporaryDirectory() as tmp:
            for dim in _DEPLOY_DIMS:
                for seed in _DEPLOY_SEEDS:
                    for rho in _RHOS:
                        with self.subTest(dim=dim, seed=seed, rho=rho):
                            sigma = _fractional_mean_rank_sigma(dim, seed)
                            dist = SpearmanRankingDistribution(sigma, rho=rho)
                            # Same construction Model.load() itself uses to wrap an already-fitted
                            # distribution -- exercises the real deploy()/load() path directly.
                            model = Model(dist)
                            model.fitted = dist
                            out = os.path.join(tmp, f"spearman-{dim}-{seed}-{rho}")
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                model.deploy(out)
                                loaded = Model.load(out)
                            self.assertTrue(
                                np.array_equal(dist.sigma, loaded.fitted.sigma),
                                f"sigma drifted across deploy()/load() (dim={dim!r} seed={seed!r} "
                                f"rho={rho!r})",
                            )
                            self.assertEqual(model_hash(dist), model_hash(loaded.fitted))
                            self.assertFalse(
                                any("integrity note" in note for note in loaded.notes), loaded.notes
                            )
                            user_warnings = [
                                str(w.message) for w in caught if issubclass(w.category, UserWarning)
                            ]
                            self.assertEqual(user_warnings, [])
                            checked += 1
        self.assertGreater(checked, 100)


class EstimatorPathNeverManifestsTest(unittest.TestCase):
    """Documents (and pins) why this defect has never been observable through the estimator: pins
    the invariant this file's docstring relies on, so a future change to how ``estimate()`` derives
    ``sigma`` that broke it would be caught here rather than discovered as a live false positive."""

    def test_estimated_sigma_round_trips_exactly_across_a_seeded_sweep(self):
        for dim in range(2, 8):
            for seed in range(10):
                with self.subTest(dim=dim, seed=seed):
                    rng = np.random.RandomState(seed)
                    estimator = SpearmanRankingEstimator(dim=dim)
                    accumulator = estimator.accumulator_factory().make()
                    for _ in range(30):
                        accumulator.update(list(rng.permutation(dim)), 1.0, None)
                    dist = estimator.estimate(None, accumulator.value())
                    restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
                    restored.__pysp_setstate__(dist.__pysp_getstate__())
                    self.assertTrue(np.array_equal(dist.sigma, restored.sigma))


class SetstateRejectsNonCanonicalTotalTest(unittest.TestCase):
    """Regression test for a real defect an adversarial review found in 25c18167 itself: skipping
    the shift on the ``already_centered=True`` restore path also skipped the ONLY check that ever
    enforced ``sigma``'s total-sum invariant (``sum(sigma) == dim*(dim-1)/2``). The majorization
    loop in ``_validate_location`` only lower-bounds each ascending prefix sum through ``dim - 1``
    terms and never inspects the full sum -- on the ordinary (non-restore) path that total is
    already exact by construction, because it is exactly what the skipped shift computes. A
    ``sigma`` shifted by a positive constant is still strictly ascending, so it clears every one of
    those prefix-sum lower bounds, but its total is wrong: fed through ``__pysp_setstate__`` on the
    code 25c18167 introduced, such a ``sigma`` silently kept the corrupted total instead of being
    rejected. This pins the fix: a dedicated equality check for that invariant, scoped to the
    ``already_centered=True`` path, that rejects rather than either silently accepting a bad total
    or silently re-deriving one (re-shifting would reintroduce the ULP drift 25c18167 fixed).
    """

    def test_reviewer_scenario_shift_by_100_on_dim_5_is_rejected(self):
        # The reviewer's exact repro: canonical sigma=[0,1,2,3,4] (dim=5, sum=10) shifted by +100
        # to [100,101,102,103,104] (sum=510, still strictly ascending).
        dim = 5
        canonical = np.arange(dim, dtype=np.float64)
        dist = SpearmanRankingDistribution(canonical, rho=0.7, name="judges", keys="k")
        state = dist.__pysp_getstate__()
        self.assertTrue(np.array_equal(state["sigma"], canonical))

        corrupted = state["sigma"] + 100.0
        self.assertTrue(np.all(np.diff(corrupted) > 0), "corrupted sigma must stay strictly ascending")
        self.assertEqual(float(corrupted.sum()), 510.0)

        restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
        with self.assertRaisesRegex(ValueError, "already_centered"):
            restored.__pysp_setstate__(dict(state, sigma=corrupted))

    def test_corrupted_sigma_clears_the_majorization_loop_on_its_own(self):
        # Sanity check that the reviewer's scenario is really the gap the new check closes, and not
        # something the pre-existing majorization loop already caught for an unrelated reason: every
        # ascending prefix sum clears its lower bound even though the full total (510) is wrong.
        dim = 5
        corrupted = np.arange(dim, dtype=np.float64) + 100.0
        ordered = np.sort(corrupted)
        for count in range(1, dim):
            minimum = count * (count - 1) / 2.0
            self.assertGreaterEqual(float(np.sum(ordered[:count])), minimum)

    def test_a_sweep_of_positive_constant_shifts_and_dims_is_rejected(self):
        checked = 0
        for dim in range(2, 9):
            canonical = np.arange(dim, dtype=np.float64)
            dist = SpearmanRankingDistribution(canonical, rho=1.0)
            state = dist.__pysp_getstate__()
            for shift in (0.01, 0.5, 1.0, 3.5, 100.0):
                with self.subTest(dim=dim, shift=shift):
                    corrupted = state["sigma"] + shift
                    restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
                    with self.assertRaisesRegex(ValueError, "already_centered"):
                        restored.__pysp_setstate__(dict(state, sigma=corrupted))
                    checked += 1
        self.assertGreater(checked, 30)

    def test_legitimately_centered_sigma_still_restores_without_error(self):
        # The fix must not become a false positive on the untampered path this file's other tests
        # already sweep: a real, once-canonicalized sigma must keep restoring cleanly.
        dim = 6
        canonical = _fractional_mean_rank_sigma(dim, seed=3)
        dist = SpearmanRankingDistribution(canonical, rho=0.4)
        state = dist.__pysp_getstate__()
        restored = SpearmanRankingDistribution.__new__(SpearmanRankingDistribution)
        restored.__pysp_setstate__(state)  # must not raise
        self.assertTrue(np.array_equal(dist.sigma, restored.sigma))


if __name__ == "__main__":
    unittest.main()
